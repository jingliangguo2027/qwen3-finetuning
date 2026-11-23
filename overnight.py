import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from overnight_worker import write_json

PROJECT = Path(__file__).resolve().parent
WORKER = PROJECT / "overnight_worker.py"
TASKS = ("banking77", "longmemeval")
METHODS = ("lora", "qlora")
RATES = (0.0002, 0.0001)
THROUGHPUT = {
    ("banking77", "lora"): 1484, ("banking77", "qlora"): 1204,
    ("longmemeval", "lora"): 1614, ("longmemeval", "qlora"): 988,
}


def job(task, method, samples, lr, rank=8):
    name = f"{task}-{method}-n{samples}-lr{lr:g}-r{rank}" if samples else f"{task}-{method}-base"
    return {"name": name, "task": task, "method": method, "samples": samples, "lr": lr, "rank": rank}


def estimate(item, measured_eval):
    key = item["task"], item["method"]
    typical_tokens = 424 if item["task"] == "banking77" else 1860
    train_seconds = item["samples"] * typical_tokens / THROUGHPUT[key]
    return 45 + 1.30 * train_seconds + 1.30 * measured_eval.get(key, 420)


def process(command, log_path, deadline):
    with log_path.open("w", encoding="utf-8") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        while child.poll() is None:
            if time.time() >= deadline:
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                return "deadline"
            time.sleep(1)
        return "ok" if child.returncode == 0 else f"exit {child.returncode}"


def render(root, state):
    write_json(root / "status.json", state)
    lines = [
        "# 过夜实验结果", "",
        f"状态 **{state['phase']}**。截止时间 {state['deadline_local']}。", "",
        "| 任务 | 方法 | 样本 | 学习率 | Rank | 验证准确率 | 用时 | 峰值训练内存 | 记录 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in state["runs"]:
        if row["status"] == "ok":
            accuracy = f"{row['correct']}/{row['evaluated']}（{row['accuracy']:.1%}）"
            result = f"[结果]({row['name']}/summary.json) · [错题]({row['name']}/errors.jsonl)"
            if row.get("oracle_predictions"):
                result += f" · [Oracle 回答]({row['name']}/oracle_hypotheses.jsonl)"
        else:
            accuracy = "—"
            result = f"{row['status']} · [日志]({row['name']}/run.log)" if row.get("log") else row["status"]
        memory = f"{row['training_peak_bytes'] / 2**30:.2f} GiB" if "training_peak_bytes" in row else "—"
        lines.append(
            f"| {row['task']} | {row['method']} | {row['samples']} | {row['lr'] or '—'} | "
            f"{row['rank'] if row['samples'] else '—'} | {accuracy} | {row.get('seconds', 0) / 60:.1f} 分钟 | "
            f"{memory} | {result} |"
        )
    lines.extend(["", "银行分数来自固定验证集，长对话分数来自独立合成留出题。Oracle 文件保留原始回答，待官方评分。"])
    temp = root / "RESULTS.tmp"
    temp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temp.replace(root / "RESULTS.md")


def common_lr(results, task):
    paired = {
        (row["lr"], row["method"]): row["accuracy"] for row in results
        if row["task"] == task and row["status"] == "ok" and row["samples"]
    }
    rates = [rate for rate in RATES if all((rate, method) in paired for method in METHODS)]
    return max(rates, key=lambda rate: sum(paired[rate, method] for method in METHODS), default=None)


def main():
    parser = argparse.ArgumentParser(description="Run the local MLX overnight comparison serially.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--oracle", required=True, type=Path)
    parser.add_argument("--hours", type=float, default=8)
    parser.add_argument("--quick", action="store_true", help="Run six shortened end-to-end checks.")
    args = parser.parse_args()
    if not 0 < args.hours <= 8:
        parser.error("--hours must be between 0 and 8.")
    root = args.output.resolve()
    root.mkdir(parents=True)
    lease = time.time() + args.hours * 3600
    state = {"phase": "准备数据", "pid": os.getpid(), "quick": args.quick,
             "deadline_local": datetime.fromtimestamp(lease).astimezone().isoformat(), "runs": []}
    render(root, state)
    executable = [sys.executable, "-B", "-u", str(WORKER), str(root)]
    prep = process([*executable, "--prepare", str(args.oracle.resolve()),
                    *(["--quick"] if args.quick else [])], root / "prepare.log", lease)
    if prep != "ok":
        state["phase"] = f"数据准备失败（{prep}）"
        render(root, state)
        raise SystemExit(prep)

    evaluation_time = {}
    finished = []

    def execute(item, *, admit=True):
        now = time.time()
        if not admit or now + estimate(item, evaluation_time) + (0 if args.quick else 900) >= lease:
            state["runs"].append({**item, "status": "剩余预算不足"})
            render(root, state)
            return
        out = root / item["name"]
        out.mkdir()
        spec = out / "job.json"
        write_json(spec, item)
        state["runs"].append({**item, "status": "运行中", "log": True})
        render(root, state)
        status = process([*executable, "--job", str(spec)], out / "run.log", lease)
        row = {**item, "status": status, "seconds": time.time() - now, "log": True}
        if status == "ok":
            try:
                summary = json.loads((out / "summary.json").read_text())
                evaluation_time[item["task"], item["method"]] = summary["evaluation_seconds"]
            except (OSError, ValueError, KeyError) as exc:
                row["status"] = "评测摘要缺失或无效"
                with (out / "run.log").open("a", encoding="utf-8") as log:
                    print(exc, file=log)
            else:
                row.update(summary)
                finished.append(row)
        state["runs"][-1] = row
        render(root, state)

    state["phase"] = "短验证" if args.quick else "基线与学习率对照"
    small = {"banking77": 1024, "longmemeval": 128}
    if args.quick:
        for task in TASKS:
            execute(job(task, "lora", 0, None))
            for method in METHODS:
                execute(job(task, method, 7 if task == "banking77" else 4, RATES[0]))
    else:
        for task in TASKS:
            for method in METHODS:
                execute(job(task, method, 0, None))
        for rate in RATES:
            for task in TASKS:
                for method in METHODS:
                    execute(job(task, method, small[task], rate))
        selected = {task: common_lr(finished, task) for task in TASKS}
        state["selected_lr"] = selected

        def pair(items):
            admits = time.time() + sum(estimate(item, evaluation_time) for item in items) + 900 < lease
            for item in items:
                execute(item, admit=admits)

        state["phase"] = "扩量对照"
        for task, size in (("banking77", 8999), ("longmemeval", 512)):
            if selected[task] is not None:
                pair([job(task, method, size, selected[task]) for method in METHODS])
        state["phase"] = "Rank 对照"
        for task in TASKS:
            if selected[task] is not None:
                pair([job(task, method, small[task], selected[task], rank=16) for method in METHODS])

    state["phase"] = "已结束"
    render(root, state)
    if args.quick and len(finished) != 6:
        raise SystemExit("Quick check did not complete all six jobs.")
    print(f"Report: {root / 'RESULTS.md'}", flush=True)


if __name__ == "__main__":
    main()
