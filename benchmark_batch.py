import argparse
import json
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

from smoke import PROJECT, load_banking77, tokenized_example
from smoke_data import synthetic_memory_example

CANDIDATES = {"banking77": (1, 2, 4), "longmemeval": (1, 2)}
METHODS = ("lora", "qlora")
EFFECTIVE_BATCH = 4


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def prepare_task(task, tokenizer, count, max_length, long_distractor_rounds):
    if task == "banking77":
        examples = load_banking77(tokenizer)["train"].select(range(count))
    else:
        examples = [
            synthetic_memory_example(
                tokenizer, f"Person {index}", 5, f"A{index:03d}", f"B{index:03d}",
                distractor_rounds=long_distractor_rounds,
            )
            for index in range(count)
        ]
    return [tokenized_example(tokenizer, row, max_length) for row in examples]


def run_worker(spec_path):
    from smoke_mlx import benchmark

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    examples = json.loads(Path(spec["examples"]).read_text(encoding="utf-8"))
    metrics = benchmark(
        Path(spec["source"]), PROJECT / "models", spec["common"], examples, spec["candidate"],
        spec["max_length"], spec["warmup"], spec["measured"],
        grad_checkpoint=spec.get("grad_checkpoint", False),
    )
    write_json(spec_path.with_suffix(".result.json"), metrics)


def pick_shared(results, task):
    passed = {
        (row["method"], row["batch_size"]): row["tokens_per_second"]
        for row in results if row["task"] == task and row["status"] == "ok"
    }
    shared = [batch for batch in CANDIDATES[task] if all((method, batch) in passed for method in METHODS)]
    if not shared:
        return None
    fastest = {method: max(passed[method, batch] for batch in shared) for method in METHODS}
    # Favor the configuration whose slower relative result is closest to its method's best.
    batch = max(shared, key=lambda size: min(passed[method, size] / fastest[method] for method in METHODS))
    return {"batch_size": batch, "accumulation": EFFECTIVE_BATCH // batch}


def main():
    parser = argparse.ArgumentParser(description="Compare small batches at an effective batch size of four.")
    parser.add_argument("--task", choices=(*CANDIDATES, "all"), default="all")
    parser.add_argument("--warmup-updates", type=int, default=2)
    parser.add_argument("--measure-updates", type=int, default=8)
    parser.add_argument("--long-distractor-rounds", type=int, choices=(1, 2), default=1)
    parser.add_argument("--long-gradient-checkpointing", action="store_true")
    parser.add_argument("--prepare-only", action="store_true", help="Check the comparison inputs without Metal.")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker is not None:
        run_worker(args.worker)
        return
    if min(args.warmup_updates, args.measure_updates) < 1:
        parser.error("Warmup and measurement both require at least one optimizer update.")

    if not args.prepare_only:
        try:
            from mlx.core import default_device

            print(f"MLX device: {default_device()}", flush=True)
        except ImportError as exc:
            raise SystemExit(f"Cannot benchmark Apple GPU: {exc}") from exc

    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from prepare_banking77 import MODEL_ID, MODEL_REVISION

    source = snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    common = tomllib.loads((PROJECT / "configs/banking77_lora.toml").read_text())
    outputs = PROJECT / "outputs"
    outputs.mkdir(exist_ok=True)
    report = Path(tempfile.mkdtemp(prefix="batch-benchmark-", dir=outputs))
    tasks = tuple(CANDIDATES) if args.task == "all" else (args.task,)
    results = []
    plan = {
        "model_revision": MODEL_REVISION,
        "effective_batch": EFFECTIVE_BATCH,
        "warmup_updates": args.warmup_updates,
        "measured_updates": args.measure_updates,
        "tasks": {},
    }
    sample_count = (args.warmup_updates + args.measure_updates) * EFFECTIVE_BATCH
    for task in tasks:
        config = tomllib.loads((PROJECT / "configs" / f"{task}_lora.toml").read_text())
        max_length = config["smoke"]["max_length"]
        cases = prepare_task(task, tokenizer, sample_count, max_length, args.long_distractor_rounds)
        cases_path = report / f"{task}-tokens.json"
        write_json(cases_path, cases)
        lengths = [len(ids) for ids, _ in cases]
        plan["tasks"][task] = {
            "length_min": min(lengths), "length_max": max(lengths), "samples": sample_count,
            "candidate_batch_sizes": CANDIDATES[task], "methods": METHODS,
        }
        grad_checkpoint = task == "longmemeval" and args.long_gradient_checkpointing
        plan["tasks"][task]["gradient_checkpointing"] = grad_checkpoint
        print(f"{task}: {sample_count} samples, {min(lengths)}–{max(lengths)} tokens", flush=True)
        if args.prepare_only:
            continue
        for batch in CANDIDATES[task]:
            for method in METHODS:
                label = f"{task}-{method}-batch{batch}"
                candidate = {
                    "task": task, "method": method, "batch_size": batch,
                    "accumulation": EFFECTIVE_BATCH // batch,
                }
                spec_path = report / f"{label}.json"
                write_json(spec_path, {
                    "source": source,
                    "common": common,
                    "examples": str(cases_path),
                    "candidate": candidate,
                    "max_length": max_length,
                    "warmup": args.warmup_updates,
                    "measured": args.measure_updates,
                    "grad_checkpoint": grad_checkpoint,
                })
                log_path = report / f"{label}.log"
                print(f"Benchmarking {label}", flush=True)
                with log_path.open("w", encoding="utf-8") as log:
                    proc = subprocess.run(
                        [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", str(spec_path)],
                        stdout=log, stderr=subprocess.STDOUT,
                    )
                row = {**candidate, "log": log_path.name}
                if proc.returncode == 0:
                    row.update(status="ok", **json.loads(spec_path.with_suffix(".result.json").read_text()))
                    print(f"{row['tokens_per_second']:.0f} tokens/s, {row['peak_memory_bytes'] / 2**30:.2f} GiB")
                else:
                    row.update(status="failed", exit_code=proc.returncode)
                    print(f"Failed, see {log_path.name}")
                results.append(row)
                write_json(report / "results.json", {
                    "plan": plan, "runs": results,
                    "shared_batch": {task: pick_shared(results, task) for task in tasks},
                })

    write_json(report / "plan.json", plan)
    print(f"Report: {report}")
    if not args.prepare_only:
        for task in tasks:
            print(f"{task}: shared batch {pick_shared(results, task)}")
        if any(pick_shared(results, task) is None for task in tasks):
            raise SystemExit("Some tasks have no batch configuration verified in both LoRA and QLoRA.")


if __name__ == "__main__":
    main()
