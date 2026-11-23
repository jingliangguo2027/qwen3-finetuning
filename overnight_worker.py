import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

from smoke import PROJECT


def write_json(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def prepare(root, oracle_path, quick):
    import hashlib
    import tomllib

    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    from overnight_data import LONG_LIMIT, fixed_oracle_cases, memory_cases
    from prepare_banking77 import DATASET_REVISION, MODEL_ID, MODEL_REVISION
    from smoke import load_banking77, tokenized_example

    source = snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    bank = load_banking77(tokenizer)
    if (len(bank["train"]), len(bank["validation"])) != (8999, 1000):
        raise ValueError("Unexpected BANKING77 split size.")
    memory = list(memory_cases(tokenizer))
    with oracle_path.open("rb") as oracle_file:
        oracle_hash = hashlib.file_digest(oracle_file, "sha256").hexdigest()
    manifest = {
        "source": source, "model_revision": MODEL_REVISION, "banking_revision": DATASET_REVISION,
        "oracle_sha256": oracle_hash,
        "common": tomllib.loads((PROJECT / "configs/banking77_lora.toml").read_text()),
        "task_data": {},
    }
    for task, train, validation, max_length, batch, budget in (
        ("banking77", bank["train"], bank["validation"], 1024, 2, 24),
        ("longmemeval", memory[:512], memory[512:], LONG_LIMIT, 1, 24),
    ):
        if quick:
            train = train.select(range(7)) if task == "banking77" else train[:4]
        rows = [tokenized_example(tokenizer, row, max_length) for row in train]
        val_rows = []
        for index, row in enumerate(validation):
            if quick and index == 4:
                break
            val_rows.append({
                "id": row.get("id", str(index)),
                "question": row["question"], "expected": row["completion"],
                "task_type": row.get("task_type", row["completion"]),
                "input_ids": tokenizer.encode(row["prompt"], add_special_tokens=False),
            })
        write_json(root / f"{task}-data.json", {"train": rows, "validation": val_rows})
        manifest["task_data"][task] = {"batch": batch, "max_length": max_length,
                                      "evaluation_tokens": budget, "available_train": len(rows),
                                      "labels": sorted({row["completion"] for row in validation})}
        print(f"{task}: {len(rows)} train, {len(val_rows)} validation, "
              f"max train length {max(len(ids) for ids, _ in rows)}", flush=True)
    oracle = fixed_oracle_cases(tokenizer, oracle_path, 2 if quick else 24)
    write_json(root / "oracle-data.json", oracle)
    manifest["oracle_question_ids"] = [row["id"] for row in oracle]
    write_json(root / "manifest.json", manifest)


def train_adapter(root, source, common, examples, settings, job):
    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx_lm.tuner.trainer import TrainingArgs, train

    from smoke_mlx import OneStepLog, completion_batches, prepare_model

    batch = settings["batch"]
    accumulation = 4 // batch
    epochs = job.get("epochs", 1)
    per_epoch = math.ceil(len(examples) / batch)
    iterations = per_epoch * epochs
    if per_epoch % accumulation or epochs < 1:
        raise ValueError("Expected the final microbatch to complete an accumulation group.")
    _, model, layers, lora = prepare_model(
        source, PROJECT / "models", common, quantized=job["method"] == "qlora"
    )
    adapter = root / "adapter"
    adapter.mkdir()
    write_json(adapter / "adapter_config.json", {
        "fine_tune_type": "lora", "num_layers": layers, "lora_parameters": lora,
    })

    def batches(dataset, batch_size, max_seq_length, loop, comm_group):
        # The MLX trainer drops incomplete microbatches. Train the final row separately.
        full = len(dataset) - len(dataset) % batch_size
        for _ in range(epochs):
            for rows, width in ((dataset[:full], batch_size), (dataset[full:], len(dataset) - full)):
                if rows:
                    yield from completion_batches(rows, width, max_seq_length, comm_group=comm_group)

    log = OneStepLog()
    target_tokens = sum(len(ids) - offset for ids, offset in examples) * epochs
    lr = job["lr"]
    updates = iterations // accumulation
    warmup_steps = 0
    schedule = job.get("schedule", "constant")
    if schedule == "warmup-linear":
        if updates < 2:
            raise ValueError("Warmup plus decay needs at least two updates.")
        warmup_steps = min(updates - 1, max(1, math.ceil(updates * 0.03)))
        lr = optim.join_schedules(
            [optim.linear_schedule(lr * 0.1, lr, warmup_steps),
             optim.linear_schedule(lr, 0.0, updates - warmup_steps)], [warmup_steps]
        )
    elif schedule != "constant":
        raise ValueError(f"Unknown learning rate schedule: {schedule}")
    start = time.perf_counter()
    mx.reset_peak_memory()
    train(
        model=model,
        optimizer=optim.AdamW(learning_rate=lr, weight_decay=0.0),
        train_dataset=examples,
        args=TrainingArgs(
            batch_size=batch, iters=iterations, grad_accumulation_steps=accumulation,
            max_seq_length=settings["max_length"], steps_per_report=25 * accumulation,
            steps_per_save=per_epoch if epochs > 1 else iterations + 1,
            adapter_file=adapter / "adapters.safetensors",
        ),
        iterate_batches=batches, training_callback=log,
    )
    if log.step != iterations or log.target_tokens != target_tokens or not math.isfinite(log.loss):
        raise RuntimeError(f"Incomplete training: steps={log.step}, target tokens={log.target_tokens}/{target_tokens}")
    weights = mx.load(str(adapter / "adapters.safetensors"))
    change = max(mx.abs(a).max().item() for name, a in weights.items() if name.endswith(".lora_b"))
    if not math.isfinite(change) or change <= 0:
        raise RuntimeError("No finite nonzero LoRA update was saved.")
    return {
        "training_seconds": time.perf_counter() - start, "training_peak_bytes": mx.get_peak_memory(),
        "last_logged_loss": log.loss, "target_tokens": target_tokens, "optimizer_updates": updates,
        "trainable_parameters": sum(a.size for a in weights.values()),
        "microbatches_per_epoch": per_epoch, "warmup_steps": warmup_steps,
        "lora_b_max_change": change,
    }


def evaluate(model, tokenizer, rows, budget, task):
    from mlx_lm import batch_generate
    from mlx_lm.sample_utils import make_sampler

    group = 64 if task == "banking77" else 16
    for start in range(0, len(rows), group):
        items = rows[start:start + group]
        responses = batch_generate(
            model, tokenizer, [item["input_ids"] for item in items], max_tokens=budget,
            sampler=make_sampler(temp=0.0), prefill_batch_size=4 if task == "banking77" else 1,
            completion_batch_size=8 if task == "banking77" else 2, prefill_step_size=512,
        ).texts
        for item, response in zip(items, responses, strict=True):
            yield item, response
        print(f"Generated {start + len(items)}/{len(rows)}", flush=True)


def run(root, job_path):
    from smoke_mlx import quantized_model

    job = json.loads(job_path.read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    name, task = job["name"], job["task"]
    output = root / name
    source = Path(manifest["source"])
    settings = manifest["task_data"][task]
    data = json.loads((root / f"{task}-data.json").read_text())
    adapter = None
    train_stats = {}
    if job["samples"]:
        common = manifest["common"]
        common["lora"].update(r=job["rank"], lora_alpha=2 * job["rank"])
        if "targets" in job:
            common["lora"]["target_modules"] = job["targets"]
        if "seed" in job:
            common["training"]["seed"] = job["seed"]
        if job["samples"] > len(data["train"]):
            raise ValueError("Requested more training samples than prepared.")
        train_stats = train_adapter(output, source, common, data["train"][:job["samples"]], settings, job)
        write_json(output / "training.json", train_stats)
        adapter = str(output / "adapter")

    base = quantized_model(source, PROJECT / "models") if job["method"] == "qlora" else source
    epochs = job.get("epochs", 1)
    for epoch in range(1, epochs):
        snapshot = output / f"epoch-{epoch}"
        snapshot_adapter = snapshot / "adapter"
        snapshot_adapter.mkdir(parents=True)
        step = epoch * train_stats["microbatches_per_epoch"]
        for name, source_file in (
            ("adapter_config.json", output / "adapter" / "adapter_config.json"),
            ("adapters.safetensors", output / "adapter" / f"{step:07d}_adapters.safetensors"),
        ):
            if not source_file.is_file():
                raise FileNotFoundError(source_file)
            (snapshot_adapter / name).symlink_to(source_file)
        evaluate_run(root, snapshot, base, str(snapshot_adapter), data, settings,
                     {**job, "scored_epoch": epoch}, {})
    evaluate_run(root, output, base, adapter, data, settings, {**job, "scored_epoch": epochs}, train_stats)


def evaluate_run(root, output, base, adapter, data, settings, job, train_stats):
    from mlx_lm import load

    task = job["task"]
    model, tokenizer = load(str(base), adapter_path=adapter)
    by_label = defaultdict(lambda: {"correct": 0, "wrong": 0, "incorrect_predictions_as_this_label": 0})
    for label in settings["labels"] if task == "banking77" else ("recall", "update", "missing"):
        by_label[label]
    count = correct = invalid = 0
    start = time.perf_counter()
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as all_rows, \
            (output / "errors.jsonl").open("w", encoding="utf-8") as errors:
        for item, response in evaluate(model, tokenizer, data["validation"], settings["evaluation_tokens"], task):
            count += 1
            predicted = response.strip()
            ok = predicted == item["expected"]
            correct += ok
            by_label[item["task_type"]]["correct" if ok else "wrong"] += 1
            valid = predicted in settings["labels"] if task == "banking77" else None
            if task == "banking77" and not ok:
                invalid += not valid
                if valid:
                    by_label[predicted]["incorrect_predictions_as_this_label"] += 1
            row = {key: value for key, value in item.items() if key != "input_ids"}
            row.update(predicted=predicted, model_output=response, correct=ok)
            if valid is not None:
                row["valid_label_output"] = valid
            line = json.dumps(row, ensure_ascii=False) + "\n"
            all_rows.write(line)
            if not ok:
                errors.write(line)
    details = {"evaluated": count, "correct": correct, "accuracy": correct / count, "by_label": dict(by_label)}
    if task == "banking77":
        details.update(wrong_valid_label=count - correct - invalid, invalid_label_output=invalid)
    if task == "longmemeval":
        oracle = json.loads((root / "oracle-data.json").read_text())
        # Save the synthetic score before generating the separately scored Oracle predictions.
        write_json(output / "validation.json", details)
        with (output / "oracle_hypotheses.jsonl").open("w", encoding="utf-8") as file:
            for item, response in evaluate(model, tokenizer, oracle, 64, task):
                file.write(json.dumps({"question_id": item["id"], "hypothesis": response},
                                      ensure_ascii=False) + "\n")
        details["oracle_predictions"] = len(oracle)
    write_json(output / "summary.json", {**job, **train_stats, **details,
               "evaluation_seconds": time.perf_counter() - start})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", type=Path, metavar="ORACLE")
    modes.add_argument("--job", type=Path)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare(args.root, args.prepare, args.quick)
    else:
        run(args.root, args.job)


if __name__ == "__main__":
    main()
