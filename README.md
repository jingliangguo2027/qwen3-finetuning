# Qwen3 fine-tuning on BANKING77

An Apple Silicon experiment on classifying English bank support messages into 77 intents with Qwen3-0.6B. The comparison uses 8,999 training examples and the same stratified 1,000-example validation split. The official test split was held out.

| Method | Validation accuracy | Macro-F1 | Peak training memory (GiB) | Training time (min) |
| --- | ---: | ---: | ---: | ---: |
| Majority label | 1.9% | 0.0005 | — | — |
| TF-IDF logistic regression | 86.4% | 0.855 | — | — |
| Unadapted Qwen3 | 12.6% | 0.086 | — | — |
| Unadapted 4-bit Qwen3 | 7.0% | 0.040 | — | — |
| Tuned LoRA, seed 42 | 89.9% | 0.900 | 5.09 | 97.3 |
| Tuned QLoRA, seed 42 | 91.4% | 0.917 | 4.18 | 223.4 |

MLX QLoRA uses frozen 4-bit affine weights. The table reports training-only time. See [the experiment log](EXPERIMENTS.md) for earlier runs, the hyperparameter sweep, and the three-seed LoRA results.

## Reproduce

Use Python 3.12. The MLX training path requires an Apple Silicon Mac with Metal.

```sh
python3.12 -m venv .venv-mlx
.venv-mlx/bin/python -m pip install -r requirements-mlx.txt
.venv-mlx/bin/python -m pip install scikit-learn==1.9.1
.venv-mlx/bin/python linear_baselines.py
```

To reproduce the two tuned runs, first cache the model at the revision pinned in `prepare_banking77.py`. The shared preparation worker also prepares inputs for the exploratory LongMemEval pipeline, so it takes the public [LongMemEval Oracle file](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json). Only the BANKING77 inputs are used by the two jobs below.

```sh
.venv-mlx/bin/python -c 'from huggingface_hub import snapshot_download; from prepare_banking77 import MODEL_ID, MODEL_REVISION; snapshot_download(MODEL_ID, revision=MODEL_REVISION)'
.venv-mlx/bin/python smoke.py --config configs/banking77_lora.toml --backend mlx
repro_dir="$PWD/outputs/banking77-reproduction"
mkdir -p data "$repro_dir"
curl -L --fail 'https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json' -o data/longmemeval_oracle.json
.venv-mlx/bin/python overnight_worker.py "$repro_dir" --prepare data/longmemeval_oracle.json
.venv-mlx/bin/python overnight_worker.py "$repro_dir" --job configs/banking77-lora-tuned.json
.venv-mlx/bin/python overnight_worker.py "$repro_dir" --job configs/banking77-qlora-tuned.json
```

Validation summaries are written under `outputs/banking77-reproduction/<job-name>/`. For the earlier PyTorch / PEFT route, install `requirements.txt` in a separate virtual environment and run `train.py --help` and `predict.py --help`.

The project source is MIT-licensed. The model and datasets are fetched from their original providers and carry their own terms.
