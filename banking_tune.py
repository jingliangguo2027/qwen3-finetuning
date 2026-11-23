import argparse
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
OUTPUTS = PROJECT / "outputs"
QV = ["self_attn.q_proj", "self_attn.v_proj"]
ATTENTION = [f"self_attn.{name}_proj" for name in ("q", "k", "v", "o")]
ALL_LINEAR = ATTENTION + [f"mlp.{name}_proj" for name in ("gate", "up", "down")]
LR = 2e-4
BASELINE = (
    OUTPUTS / "banking77-lr-sweep-20261122" / "banking77-lora-n8999-lr0.0002-r8" / "summary.json"
)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, content):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def macro_f1(summary):
    values = []
    for label in summary["by_label"].values():
        tp, fn, fp = label["correct"], label["wrong"], label["incorrect_predictions_as_this_label"]
        denominator = 2 * tp + fp + fn
        values.append(2 * tp / denominator if denominator else 0)
    if len(values) != 77:
        raise ValueError("Expected results for 77 labels.")
    return statistics.mean(values)


def configuration(target, rank, *, epochs=1, schedule="constant"):
    targets = {"qv": QV, "attention": ATTENTION, "attention-mlp": ALL_LINEAR}
    return {"target": target, "targets": targets[target], "rank": rank,
            "epochs": epochs, "schedule": schedule, "lr": LR}


def score_key(row):
    return row["accuracy"], row["macro_f1"], -row["trainable_parameters"]


def signature(row):
    epoch = row["scored_epoch"]
    # A constant-LR first epoch is equivalent to a one-epoch run with the same seed.
    horizon = epoch if row["schedule"] == "constant" else row["epochs"]
    return row["target"], row["rank"], row["schedule"], horizon, epoch


def config_from(row):
    target, rank, schedule, epochs, _ = signature(row)
    return configuration(target, rank, epochs=epochs, schedule=schedule)


def render(root, state):
    save(root / "status.json", json.dumps(state, indent=2, ensure_ascii=False) + "\n")
    lines = [
        "# BANKING77 LoRA 参数调优", "",
        f"状态 **{state['phase']}**。更新于 {datetime.now().astimezone():%Y-%m-%d %H:%M %Z}。", "",
        "每组沿用同一份 BANKING77 划分与提示词。学习率或峰值学习率 0.0002，有效 batch 4，alpha/r 为 2。",
        "", "| 阶段 | 方法 | 挂载 | Rank | Epoch | 调度 | Seed | 参数量 | 准确率 | Macro-F1 | 用时/分 | 记录 |",
        "| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in state["rows"]:
        ok = row["status"] == "完成"
        score = f"{row['accuracy']:.1%}" if ok else "—"
        f1 = f"{row['macro_f1']:.3f}" if ok else "—"
        params = f"{row['trainable_parameters']:,}" if ok else "—"
        minutes = f"{row['seconds'] / 60:.1f}" if "seconds" in row else "—"
        lines.append(
            f"| {row['stage']} | {row['method']} | {row['target']} | {row['rank']} | "
            f"{row.get('scored_epoch', row['epochs'])}/{row['epochs']} | {row['schedule']} | {row['seed']} | "
            f"{params} | {score} | {f1} | {minutes} | [{row['status']}]({row['link']}) |"
        )
    if state.get("finalists"):
        lines.extend(["", "## 候选复跑", ""])
        for item in state["finalists"]:
            result = (f"平均 **{item['mean_accuracy']:.2%}**，标准差 {item['sd_accuracy']:.2%}"
                      if "mean_accuracy" in item else "复跑数量不足，暂不定胜负")
            lines.append(f"- {item['description']}，{item['run_count']}/3 个种子，{result}。")
    if state.get("winner"):
        winner = state["winner"]
        lines.extend(["", f"**验证集胜出配置** {winner['description']}，三种子平均 {winner['mean_accuracy']:.2%}。"])
        adapter = winner["runs"][0].removesuffix("summary.json") + "adapter/"
        lines.append(f"下阶段可从固定 seed 42 的[适配器]({adapter})继续。")
    save(root / "RESULTS.md", "\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run the planned BANKING77 tuning stages serially.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "status.json").exists():
        parser.error("The tuning run already exists.")
    reference = OUTPUTS / ("overnight-preflight-20261122" if args.quick else "overnight-20261122")
    for file in ("manifest.json", "banking77-data.json"):
        (root / file).symlink_to(reference / file)
    data = load(reference / "banking77-data.json")
    samples, validation = (len(data["train"]), len(data["validation"]))
    if (samples, validation) != ((7, 4) if args.quick else (8999, 1000)):
        raise ValueError("The prepared split size does not match the planned run.")
    del data
    state = {"phase": "挂载位置与 Rank", "quick": args.quick, "rows": []}
    rows = state["rows"]
    render(root, state)

    def run(config, stage, seed=42, method="lora"):
        name = f"{method}-{config['target']}-r{config['rank']}-e{config['epochs']}-{config['schedule']}-s{seed}"
        output = root / name
        output.mkdir()
        spec = output / "job.json"
        job = {**config, "task": "banking77", "method": method, "samples": samples, "seed": seed, "name": name}
        save(spec, json.dumps(job))
        record = {**job, "stage": stage, "status": "运行中", "link": f"{name}/run.log"}
        rows.append(record)
        render(root, state)
        started = time.monotonic()
        with (output / "run.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(
                [sys.executable, "-B", "-u", str(PROJECT / "overnight_worker.py"), str(root), "--job", str(spec)],
                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            )
        rows.pop()
        snapshots = []
        try:
            if result.returncode:
                raise ValueError(f"退出码 {result.returncode}")
            for epoch in range(1, config["epochs"] + 1):
                folder = output if epoch == config["epochs"] else output / f"epoch-{epoch}"
                path = folder / "summary.json"
                result_summary = load(path)
                meta = load(output / "training.json")
                if result_summary["evaluated"] != validation or result_summary["scored_epoch"] != epoch:
                    raise ValueError("Unexpected evaluation coverage.")
                snapshots.append({
                    **job, "stage": stage, "status": "完成", "link": str(path.relative_to(root)),
                    "scored_epoch": epoch, "accuracy": result_summary["accuracy"],
                    "macro_f1": macro_f1(result_summary), "trainable_parameters": meta["trainable_parameters"],
                    **({"seconds": time.monotonic() - started} if epoch == config["epochs"] else {}),
                })
        except (OSError, KeyError, ValueError) as exc:
            rows.append({**record, "status": f"失败（{exc}）"})
        rows.extend(snapshots)
        render(root, state)
        return snapshots

    baseline = configuration("qv", 8)
    if args.quick:
        run(baseline, "基线")
    else:
        prior = load(BASELINE)
        if (prior["samples"], prior["lr"], prior["rank"], prior["evaluated"]) != (8999, LR, 8, 1000):
            raise ValueError("The prior LoRA benchmark does not match the comparison.")
        rows.append({
            **baseline, "status": "完成", "stage": "已有基线", "method": "lora", "seed": 42, "scored_epoch": 1,
            "accuracy": prior["accuracy"], "macro_f1": macro_f1(prior),
            "trainable_parameters": 1_146_880, "link": "../" + str(BASELINE.relative_to(OUTPUTS)),
        })
        render(root, state)
    for target, rank in (("attention", 4), ("attention-mlp", 4), ("qv", 16)):
        run(configuration(target, rank), "挂载与 Rank")
    screening = [row for row in rows if row["status"] == "完成"]
    if not screening:
        state["phase"] = "无法选出挂载配置"
        render(root, state)
        raise SystemExit(1)
    best = max(screening, key=score_key)

    state["phase"] = "Epoch 与调度"
    for schedule in ("constant", "warmup-linear"):
        run(configuration(best["target"], best["rank"], epochs=2, schedule=schedule), "Epoch 与调度")

    candidates = {}
    for row in rows:
        if row["status"] == "完成":
            key = signature(row)
            if key not in candidates or score_key(row) > score_key(candidates[key]):
                candidates[key] = row
    top = sorted(candidates.values(), key=score_key, reverse=True)[:2]
    state["phase"] = "复跑两个领先配置"
    repeated = {}
    for row in top:
        config = config_from(row)
        key = (config["target"], config["rank"], config["epochs"], config["schedule"])
        if key not in repeated:
            repeated[key] = [
                snapshot for seed in (43, 44) for snapshot in run(config, "种子复跑", seed=seed)
            ]

    finalists = []
    for row in top:
        config = config_from(row)
        key = (config["target"], config["rank"], config["epochs"], config["schedule"])
        reps = [row] + [s for s in repeated[key] if s["scored_epoch"] == row["scored_epoch"]]
        item = {"config": config, "scored_epoch": row["scored_epoch"], "run_count": len(reps),
                "trainable_parameters": row["trainable_parameters"], "runs": [r["link"] for r in reps],
                "description": f"{row['target']} rank {row['rank']}，{row['schedule']}，第 {row['scored_epoch']} 轮"}
        if len(reps) == 3:
            item.update(mean_accuracy=statistics.mean(r["accuracy"] for r in reps),
                        sd_accuracy=statistics.stdev(r["accuracy"] for r in reps),
                        mean_macro_f1=statistics.mean(r["macro_f1"] for r in reps))
        finalists.append(item)
    state["finalists"] = finalists
    passed = [item for item in finalists if "mean_accuracy" in item]
    if passed:
        winner = max(passed, key=lambda item: (
            item["mean_accuracy"], item["mean_macro_f1"], -item["trainable_parameters"]
        ))
        state["winner"] = winner
        state["phase"] = "QLoRA 确认"
        for row in run(winner["config"], "QLoRA 确认", method="qlora"):
            if row["scored_epoch"] == winner["scored_epoch"]:
                state["qlora_confirmation"] = row
    failed = sum(row["status"].startswith("失败") for row in rows)
    state["phase"] = f"已结束（{failed} 组失败）" if failed else "已结束"
    render(root, state)
    print(f"Report: {root / 'RESULTS.md'}", flush=True)
    if args.quick and (failed or len(passed) != 2 or "qlora_confirmation" not in state):
        raise SystemExit("Incomplete tuning preflight.")


if __name__ == "__main__":
    main()
