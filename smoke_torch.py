import math

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType
from transformers import AutoModelForCausalLM, GenerationConfig, set_seed
from trl import SFTConfig, SFTTrainer


def fit(source, tokenizer, training, tokenized, common, max_length, run, *, quantized):
    if quantized:
        raise ValueError("Use MLX for the 4-bit backend.")
    settings = common["training"]
    set_seed(settings["seed"])
    data = Dataset.from_list([{"prompt": training["prompt"], "completion": training["completion"]}])
    trainer = SFTTrainer(
        model=str(source),
        args=SFTConfig(
            output_dir=str(run),
            model_init_kwargs={"dtype": torch.float32},
            completion_only_loss=True,
            eos_token=tokenizer.eos_token,
            max_length=max_length,
            max_steps=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            gradient_checkpointing=False,
            learning_rate=settings["learning_rate"],
            lr_scheduler_type="constant",
            optim="adamw_torch",
            bf16=False,
            fp16=False,
            save_strategy="no",
            logging_steps=1,
            report_to="none",
        ),
        train_dataset=data,
        processing_class=tokenizer,
        peft_config=LoraConfig(task_type=TaskType.CAUSAL_LM, **common["lora"]),
    )
    batch = trainer.data_collator([trainer.train_dataset[0]])
    labels = batch["labels"][0]
    if labels[labels != -100].tolist() != tokenized[0][tokenized[1]:]:
        raise RuntimeError("Trainer targets do not match the answer tokens.")
    trainable = [(name, weight) for name, weight in trainer.model.named_parameters() if weight.requires_grad]
    if not trainable or any(".lora_" not in name for name, _ in trainable):
        raise RuntimeError("Expected only LoRA parameters to be trainable.")
    _, weight = next((name, weight) for name, weight in trainable if ".lora_B." in name)
    before = weight.detach().clone()
    result = trainer.train()
    change = (weight.detach() - before).abs().max().item()
    if result.global_step != 1 or not math.isfinite(result.training_loss) or not math.isfinite(change) or change <= 0:
        raise RuntimeError("One-step training did not produce a finite LoRA update.")
    device = trainer.args.device
    adapter = run / "adapter"
    trainer.save_model(str(adapter))
    del trainable, weight, trainer

    base = AutoModelForCausalLM.from_pretrained(source, dtype=torch.float32, local_files_only=True)
    model = PeftModel.from_pretrained(base, adapter).to(device).eval()

    def predict(prompt, budget):
        inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(device)
        generation = GenerationConfig(
            do_sample=False, max_new_tokens=budget,
            eos_token_id=model.generation_config.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        with torch.inference_mode():
            output = model.generate(**inputs, generation_config=generation)
        return tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    return predict, {"loss": result.training_loss, "lora_b_max_change": change, "device": str(device)}
