import json
import math
import tempfile
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm import convert, generate, load
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tuner.callbacks import TrainingCallback
from mlx_lm.tuner.trainer import TrainingArgs, iterate_batches, train
from mlx_lm.tuner.utils import linear_to_lora_layers


class OneStepLog(TrainingCallback):
    step = 0
    loss = math.nan
    target_tokens = 0

    def on_train_loss_report(self, train_info):
        self.step = train_info["iteration"]
        self.loss = train_info["train_loss"]
        self.target_tokens = train_info["trained_tokens"]


def completion_batches(dataset, batch_size, max_seq_length, **kwargs):
    for batch, bounds in iterate_batches(dataset, batch_size, max_seq_length, **kwargs):
        # MLX-LM's loss treats the end as inclusive. Sequence lengths are exclusive.
        yield batch, bounds - mx.array([0, 1], dtype=bounds.dtype)


def quantized_model(source, models):
    models.mkdir(exist_ok=True)
    target = models / f"qwen3-0.6b-{source.name}-affine-4bit"
    if not target.exists():
        with tempfile.TemporaryDirectory(dir=models) as temp:
            staged = Path(temp) / "model"
            convert(str(source), str(staged), quantize=True, q_bits=4, q_group_size=64,
                    q_mode="affine")
            staged.rename(target)
    config = json.loads((target / "config.json").read_text(encoding="utf-8"))
    quantization = config.get("quantization", {})
    if quantization.get("bits") != 4 or quantization.get("mode", "affine") != "affine":
        raise RuntimeError("Expected a 4-bit MLX affine model.")
    return target


def prepare_model(source, models, common, *, quantized):
    base = quantized_model(source, models) if quantized else source
    settings = common["training"]
    np.random.seed(settings["seed"])
    mx.random.seed(settings["seed"])
    model, _ = load(str(base))
    lora = common["lora"]
    keys = [name if "." in name else f"self_attn.{name}" for name in lora["target_modules"]]
    for layer in model.layers:
        modules = dict(layer.named_modules())
        for key in keys:
            projection = modules.get(key)
            if quantized:
                valid = isinstance(projection, nn.QuantizedLinear) and projection.bits == 4
            else:
                valid = isinstance(projection, nn.Linear)
            if not valid:
                raise RuntimeError(f"Unexpected base projection at {key}.")
    params = {
        "rank": lora["r"],
        "scale": lora["lora_alpha"] / lora["r"],
        "dropout": 0.0,
        "keys": keys,
    }
    num_layers = len(model.layers)
    model.freeze()
    linear_to_lora_layers(model, num_layers, params)
    trainable = tree_flatten(model.trainable_parameters())
    if not trainable or any(not name.endswith((".lora_a", ".lora_b")) for name, _ in trainable):
        raise RuntimeError("Expected only LoRA parameters to be trainable.")
    return base, model, num_layers, params


def fit(source, tokenizer, _training, tokenized, common, max_length, run, *, quantized):
    base, model, num_layers, params = prepare_model(
        source, run.parent.parent / "models", common, quantized=quantized
    )
    settings = common["training"]

    adapter = run / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({
        "fine_tune_type": "lora", "num_layers": num_layers, "lora_parameters": params,
    }, indent=2) + "\n", encoding="utf-8")
    adapter_file = adapter / "adapters.safetensors"
    log = OneStepLog()
    train(
        model=model,
        optimizer=optim.AdamW(learning_rate=settings["learning_rate"], weight_decay=0.0),
        train_dataset=[tokenized],
        args=TrainingArgs(batch_size=1, iters=1, steps_per_report=1, steps_per_save=2,
                          max_seq_length=max_length, adapter_file=adapter_file),
        iterate_batches=completion_batches,
        training_callback=log,
    )
    del model
    saved = mx.load(str(adapter_file))
    change = max(mx.abs(value).max().item() for name, value in saved.items() if name.endswith(".lora_b"))
    if log.step != 1 or not math.isfinite(log.loss) or not math.isfinite(change) or change <= 0:
        raise RuntimeError("One-step MLX training did not produce a finite LoRA update.")
    expected_targets = len(tokenized[0]) - tokenized[1]
    if log.target_tokens != expected_targets:
        raise RuntimeError(f"Expected {expected_targets} answer tokens, trained on {log.target_tokens}.")
    del saved

    model, mlx_tokenizer = load(str(base), adapter_path=str(adapter))

    def predict(prompt, budget):
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        return generate(model, mlx_tokenizer, prompt=input_ids, max_tokens=budget,
                        sampler=make_sampler(temp=0.0)).strip()

    return predict, {
        "loss": log.loss,
        "lora_b_max_change": change,
        "device": str(mx.default_device()),
        "base_quantization": "mlx-affine-4bit" if quantized else None,
        "trained_target_tokens": log.target_tokens,
    }


def benchmark(source, models, common, examples, candidate, max_length, warmup, measured, *, grad_checkpoint=False):
    batch = candidate["batch_size"]
    accumulation = candidate["accumulation"]
    _, model, _, _ = prepare_model(
        source, models, common, quantized=candidate["method"] == "qlora"
    )
    warmup_rows = warmup * batch * accumulation
    warmup_iterations = warmup * accumulation
    final_iteration = (warmup + measured) * accumulation
    measured_examples = examples[warmup_rows:]

    def batches(dataset, batch_size, max_seq_length, loop, comm_group):
        for rows in (dataset[:warmup_rows], dataset[warmup_rows:]):
            yield from completion_batches(rows, batch_size, max_seq_length, comm_group=comm_group)

    class Timer(TrainingCallback):
        start = None
        end = None
        peak = None

        def on_train_loss_report(self, info):
            if not math.isfinite(info["train_loss"]):
                raise RuntimeError("Benchmark training produced a non-finite loss.")
            if info["iteration"] == warmup_iterations:
                mx.reset_peak_memory()
                self.start = time.perf_counter()
            if info["iteration"] == final_iteration:
                self.end = time.perf_counter()
                self.peak = mx.get_peak_memory()

    timer = Timer()
    with tempfile.TemporaryDirectory() as temp:
        train(
            model=model,
            optimizer=optim.AdamW(learning_rate=common["training"]["learning_rate"], weight_decay=0.0),
            train_dataset=examples,
            args=TrainingArgs(
                batch_size=batch,
                grad_accumulation_steps=accumulation,
                iters=final_iteration,
                max_seq_length=max_length,
                grad_checkpoint=grad_checkpoint,
                steps_per_report=accumulation,
                steps_per_save=final_iteration + 1,
                adapter_file=Path(temp) / "discard.safetensors",
            ),
            iterate_batches=batches,
            training_callback=timer,
        )
    if timer.start is None or timer.end is None or timer.end <= timer.start:
        raise RuntimeError("The requested measurement window did not complete.")
    elapsed = timer.end - timer.start
    tokens = sum(len(ids) for ids, _ in measured_examples)
    return {
        "measured_updates": measured,
        "measured_samples": len(measured_examples),
        "input_tokens": tokens,
        "seconds": elapsed,
        "tokens_per_second": tokens / elapsed,
        "peak_memory_bytes": timer.peak,
    }
