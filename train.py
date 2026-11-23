import argparse
import json
import math
import tempfile
import tomllib
from pathlib import Path

import torch
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer

from prepare_banking77 import DATASET_REVISION, MODEL_ID, MODEL_REVISION, prepare_banking77

PROJECT_DIR = Path(__file__).resolve().parent


def inspect_batch(trainer, tokenizer, expected_answer):
    batch = trainer.data_collator([trainer.train_dataset[0]])
    print(batch.keys())
    for key, value in batch.items():
        print(key, value.shape, value.dtype)
    labels = batch["labels"][0]
    active = labels != -100
    targets = labels[active]
    decoded = tokenizer.decode(targets.tolist())
    if decoded != expected_answer + tokenizer.eos_token:
        raise RuntimeError(f"Unexpected loss targets: {decoded!r}")

    print(f"input_ids: {list(batch['input_ids'].shape)}, {batch['input_ids'].dtype}")
    print(f"labels: {list(batch['labels'].shape)}, {labels.dtype}")
    print(f"Masked tokens: {(~active).sum().item()}")
    print(f"Target tokens: {targets.numel()} → {decoded!r}")


def main():
    parser = argparse.ArgumentParser(description="Inspect or train BANKING77 LoRA.")
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "configs/banking77_lora.toml")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--one-step", action="store_true", help="Run one optimizer step and save the adapter.")
    modes.add_argument("--pilot", action="store_true", help="Run the multi-example training pilot and save the adapter.")
    modes.add_argument("--full", action="store_true", help="Train one epoch on the full training split.")
    cli = parser.parse_args()
    with cli.config.open("rb") as config_file:
        config = tomllib.load(config_file)
    if (config["task"], config["method"]) != ("banking77", "lora"):
        parser.error("This training script currently supports BANKING77 with LoRA.")

    settings = config["training"]
    max_steps = 1
    accumulation_steps = 1
    if cli.pilot:
        max_steps = config["pilot"]["max_steps"]
        accumulation_steps = config["pilot"]["gradient_accumulation_steps"]
        if max_steps < 1 or accumulation_steps < 1:
            parser.error("Pilot steps and gradient accumulation must both be positive.")
    elif cli.full:
        max_steps = -1
        accumulation_steps = config["full"]["gradient_accumulation_steps"]
        if accumulation_steps < 1:
            parser.error("Gradient accumulation must be positive.")

    set_seed(settings["seed"])
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    train_split = prepare_banking77(tokenizer)["train"]
    if cli.full:
        train_dataset = train_split
        expected_steps = math.ceil(len(train_dataset) / accumulation_steps)
    else:
        # The one-step and pilot modes use a fixed number of full groups.
        sample_count = max_steps * accumulation_steps
        if sample_count > len(train_split):
            parser.error(f"Requested {sample_count} training examples but only {len(train_split)} are available.")
        train_dataset = train_split.select(range(sample_count))
        expected_steps = max_steps
    lora = LoraConfig(task_type=TaskType.CAUSAL_LM, **config["lora"])

    with tempfile.TemporaryDirectory(prefix="banking77-") as work_dir:
        args = SFTConfig(
            output_dir=work_dir,
            model_init_kwargs={"revision": MODEL_REVISION, "dtype": torch.float32},
            completion_only_loss=True, #！！！！！！！！！！！
            max_length=settings["max_length"],
            max_steps=max_steps,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=accumulation_steps,
            gradient_checkpointing=False,
            learning_rate=settings["learning_rate"],
            lr_scheduler_type="constant",
            optim="adamw_torch",
            seed=settings["seed"],
            eos_token="<|im_end|>",
            bf16=False,
            fp16=False,
            save_strategy="no",
            logging_steps=25 if cli.full else 1,
            report_to="none",
        )
        trainer = SFTTrainer(
            model=MODEL_ID,
            args=args,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            peft_config=lora,
        )
        print(f"Training device: {trainer.args.device}")
        trainer.model.print_trainable_parameters()
        if len(trainer.train_dataset) != len(train_dataset):
            raise RuntimeError("Trainer preprocessing changed the number of training examples.")
        inspect_batch(trainer, tokenizer, train_dataset[0]["completion"])

        trainable = {
            name: param for name, param in trainer.model.named_parameters() if param.requires_grad
        }
        if not trainable or any(".lora_" not in name for name in trainable):
            raise RuntimeError("Expected only LoRA parameters to have requires_grad=True.")
        name, weight = next((name, param) for name, param in trainable.items() if ".lora_B." in name)
        print(f"Example trainable parameter: {name}, shape={list(weight.shape)}")
        if not (cli.one_step or cli.pilot or cli.full):
            return

        print(
            f"Training {len(train_dataset)} examples, up to {accumulation_steps} per optimizer update, "
            f"{expected_steps} updates total."
        )
        # Keep an independent copy so the optimizer cannot modify the snapshot.
        before = weight.detach().clone()
        result = trainer.train()
        max_change = (weight.detach() - before).abs().max().item()
        if (
            result.global_step != expected_steps
            or not math.isfinite(result.training_loss)
            or not math.isfinite(max_change)
            or max_change <= 0
        ):
            raise RuntimeError(f"Training verification failed: {result}, parameter change={max_change}")

        outputs = PROJECT_DIR / "outputs"
        outputs.mkdir(exist_ok=True)
        prefix = "banking77-lora-"
        if cli.full:
            prefix += "full-"
        elif cli.pilot:
            prefix += "pilot-"
        run_dir = Path(tempfile.mkdtemp(prefix=prefix, dir=outputs))
        trainer.save_model(str(run_dir / "adapter"))
        summary = {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "dataset_revision": DATASET_REVISION,
            "config": config,
            "device": str(trainer.args.device),
            "train_examples": len(train_dataset),
            "gradient_accumulation_steps": accumulation_steps,
            "steps": result.global_step,
            "loss": result.training_loss,
            "parameter": name,
            "max_parameter_change": max_change,
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"Loss: {result.training_loss:.6f}, max LoRA B change: {max_change:.6g}")
        print(f"Saved run: {run_dir}")


if __name__ == "__main__":
    main()
