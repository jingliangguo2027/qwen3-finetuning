import argparse
import json
import tempfile
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from prepare_banking77 import DATASET_ID, DATASET_REVISION, MODEL_ID, MODEL_REVISION, prepare_banking77

PROJECT_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Evaluate BANKING77 on a fixed validation prefix.")
    parser.add_argument("--adapter", type=Path, help="Directory containing saved LoRA weights.")
    parser.add_argument("--limit", type=int, default=1, help="Number of validation examples to score.")
    parser.add_argument("--report-dir", type=Path, help="New directory for error rows and label counts.")
    args = parser.parse_args()
    if args.adapter is not None and not args.adapter.is_dir():
        parser.error(f"Adapter directory does not exist: {args.adapter}")
    if args.limit < 1:
        parser.error("--limit must be positive.")
    if args.report_dir is not None and args.report_dir.exists():
        parser.error(f"Report directory already exists: {args.report_dir}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, padding_side="left")
    validation = prepare_banking77(tokenizer)["validation"]
    if args.limit > len(validation):
        parser.error(f"--limit cannot exceed the {len(validation)} validation examples.")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, dtype=torch.float32
    )
    if args.adapter is not None:
        model = PeftModel.from_pretrained(model, args.adapter)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device).eval()

    generation = GenerationConfig(
        do_sample=False,
        max_new_tokens=24,
        eos_token_id=model.generation_config.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    print(f"Model: {args.adapter if args.adapter is not None else MODEL_ID}")
    print(f"Device: {device}")

    batch_size = 4
    correct = 0
    errors = []
    labels_summary = {
        label: {"correct": 0, "wrong": 0, "incorrect_predictions_as_this_label": 0}
        for label in sorted(set(validation["completion"]))
    }
    with torch.inference_mode():
        for start in range(0, args.limit, batch_size):
            end = min(start + batch_size, args.limit)
            batch = validation[start:end]
            inputs = tokenizer(
                batch["prompt"], padding=True, add_special_tokens=False, return_tensors="pt"
            ).to(device)
            prompt_width = inputs["input_ids"].shape[1]
            sequences = model.generate(**inputs, generation_config=generation)
            # Every sequence begins with the full padded input row.
            new_tokens = sequences[:, prompt_width:]
            predictions = tokenizer.batch_decode(new_tokens.cpu(), skip_special_tokens=True)
            rows = zip(batch["question"], batch["completion"], predictions, strict=True)
            for index, (question, expected, response) in enumerate(rows, start=start):
                prediction = response.strip()
                if prediction == expected:
                    correct += 1
                    labels_summary[expected]["correct"] += 1
                    continue

                labels_summary[expected]["wrong"] += 1
                is_label = prediction in labels_summary
                if is_label:
                    labels_summary[prediction]["incorrect_predictions_as_this_label"] += 1
                errors.append({
                    "validation_index": index,
                    "question": question,
                    "expected_label": expected,
                    "model_output": response,
                    "predicted_label": prediction if is_label else None,
                    "error_type": "wrong_label" if is_label else "invalid_label_output",
                })
            if start == 0:
                print(f"Expected: {batch['completion'][0]}")
                print(f"Predicted: {predictions[0].strip()!r}")

    report_dir = args.report_dir
    if report_dir is None:
        outputs = PROJECT_DIR / "outputs"
        outputs.mkdir(exist_ok=True)
        report_dir = Path(tempfile.mkdtemp(prefix="banking77-validation-", dir=outputs))
    else:
        report_dir.mkdir(parents=True)

    invalid = sum(error["error_type"] == "invalid_label_output" for error in errors)
    report = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "adapter": str(args.adapter.resolve()) if args.adapter is not None else None,
        "split": "validation",
        "split_seed": 42,
        "evaluated": args.limit,
        "correct": correct,
        "wrong": len(errors),
        "accuracy": correct / args.limit,
        "wrong_valid_label": len(errors) - invalid,
        "invalid_label_output": invalid,
        "by_label": labels_summary,
    }
    (report_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (report_dir / "errors.jsonl").open("w", encoding="utf-8") as error_file:
        for error in errors:
            error_file.write(json.dumps(error, ensure_ascii=False) + "\n")

    print(f"Accuracy: {correct}/{args.limit} ({correct / args.limit:.2%})")
    print(f"Report: {report_dir.resolve()}")


if __name__ == "__main__":
    main()
