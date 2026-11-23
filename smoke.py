import argparse
import json
import tempfile
import tomllib
from pathlib import Path

from datasets import Dataset
from datasets.config import HF_DATASETS_CACHE
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from prepare_banking77 import DATASET_REVISION, MODEL_ID, MODEL_REVISION, prepare_banking77
from smoke_data import oracle_prompts, synthetic_memory_example

PROJECT = Path(__file__).resolve().parent


def load_banking77(tokenizer):
    cache_pattern = f"PolyAI___banking77/**/{DATASET_REVISION}/banking77-train.arrow"
    arrow = next(iter(sorted(HF_DATASETS_CACHE.glob(cache_pattern))), None)
    raw = Dataset.from_file(str(arrow), in_memory=True) if arrow is not None else None
    return prepare_banking77(tokenizer, raw_train=raw)


def load_cases(task, tokenizer):
    if task == "banking77":
        data = load_banking77(tokenizer)
        return data["train"][0], data["validation"][0]

    return (
        synthetic_memory_example(tokenizer, "Ada", 1, "P14", "Q27"),
        synthetic_memory_example(tokenizer, "Lin", 4, "A61", "B93"),
    )


def tokenized_example(tokenizer, example, max_length):
    prompt = example["prompt"]
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    full_ids = tokenizer.encode(
        prompt + example["completion"] + tokenizer.eos_token, add_special_tokens=False
    )
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("Tokenization changed the boundary between prompt and answer.")
    if not len(prompt_ids) < len(full_ids) <= max_length:
        raise ValueError(f"Expected an answer within the {max_length}-token budget.")
    return full_ids, len(prompt_ids)


def main():
    parser = argparse.ArgumentParser(description="Run one tiny Qwen3 fine-tuning experiment.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--backend", choices=("torch", "mlx"), help="Default: PyTorch for LoRA, MLX for QLoRA.")
    parser.add_argument("--oracle", type=Path, help="Optional official LongMemEval Oracle JSON, for evaluation only.")
    parser.add_argument("--prepare-only", action="store_true", help="Validate data and token lengths without training.")
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    task, method = config["task"], config["method"]
    if task not in {"banking77", "longmemeval"} or method not in {"lora", "qlora"}:
        parser.error("Choose a BANKING77 or LongMemEval LoRA/QLoRA config.")
    backend = args.backend or ("mlx" if method == "qlora" else "torch")
    if method == "qlora" and backend != "mlx":
        parser.error("The 4-bit QLoRA experiment uses MLX.")
    if args.oracle is not None and task != "longmemeval":
        parser.error("--oracle applies only to LongMemEval experiments.")

    common = tomllib.loads((PROJECT / "configs/banking77_lora.toml").read_text(encoding="utf-8"))
    settings = config["smoke"]
    max_length = settings["max_length"]
    generation_budget = settings["max_new_tokens"]
    source = Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_files_only=True))
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    training, evaluation = load_cases(task, tokenizer)
    tokens, offset = tokenized_example(tokenizer, training, max_length)
    _, evaluation_offset = tokenized_example(tokenizer, evaluation, max_length)
    oracle = (
        oracle_prompts(args.oracle, tokenizer, max_length - generation_budget, limit=2)
        if args.oracle is not None else []
    )
    print(f"{task}/{method}: {backend}; prompt={offset}, answer={len(tokens) - offset} tokens",
          flush=True)
    if args.prepare_only:
        print(f"Held-out prompt={evaluation_offset} tokens; Oracle questions={len(oracle)}")
        return

    if backend == "mlx":
        try:
            from smoke_mlx import fit
        except ImportError as exc:
            raise SystemExit(f"MLX cannot start in this environment: {exc}") from exc
    else:
        from smoke_torch import fit

    outputs = PROJECT / "outputs"
    outputs.mkdir(exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix=f"{task}-{method}-smoke-", dir=outputs))
    predict, metrics = fit(
        source, tokenizer, training, (tokens, offset), common, max_length, run,
        quantized=method == "qlora",
    )
    response = predict(evaluation["prompt"], generation_budget)
    report = {
        "task": task,
        "method": method,
        "backend": backend,
        "model_revision": MODEL_REVISION,
        "settings": settings,
        "lora": common["lora"],
        "learning_rate": common["training"]["learning_rate"],
        "seed": common["training"]["seed"],
        "train_examples": 1,
        "steps": 1,
        "train_prompt_tokens": offset,
        "train_answer_tokens": len(tokens) - offset,
        "held_out_prompt_tokens": evaluation_offset,
        "evaluation_source": "banking77_validation" if task == "banking77" else "synthetic_holdout",
        "expected": evaluation["completion"],
        "prediction": response,
        **metrics,
    }
    if oracle:
        with (run / "oracle_hypotheses.jsonl").open("w", encoding="utf-8") as file:
            for item in oracle:
                hypothesis = predict(item["prompt"], generation_budget)
                file.write(json.dumps({"question_id": item["question_id"], "hypothesis": hypothesis},
                                      ensure_ascii=False) + "\n")
        report["oracle_cases"] = [
            {"question_id": item["question_id"], "prompt_tokens": item["prompt_tokens"]}
            for item in oracle
        ]
    (run / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")
    print(f"Expected: {report['expected']}\nPrediction: {response!r}\nSaved run: {run}")


if __name__ == "__main__":
    main()
