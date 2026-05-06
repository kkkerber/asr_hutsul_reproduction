# ASR for the Hutsul Dialect of Ukrainian — Reproduction Project

A research-grade reproduction study of the paper **"Building ASR
Resources for the Hutsul Dialect of Ukrainian"**.  The repository
fine-tunes four model families on the
[`KSE-RESEARCH-Group/Dido-Yvanchyk-Audio-Dataset-v2`](https://huggingface.co/datasets/KSE-RESEARCH-Group/Dido-Yvanchyk-Audio-Dataset-v2)
corpus (≈19 h 27 m of single-speaker Hutsul speech, 8 412 segments,
16 kHz) and matches the evaluation pipeline reported in the paper.

## Supported model families

| Model                              | Family       | Strategy                | Reported WER | Reported CER |
|------------------------------------|--------------|-------------------------|--------------|--------------|
| `openai/whisper-small`             | Whisper      | LoRA (q_proj, v_proj)   | —            | —            |
| `openai/whisper-medium`            | Whisper      | LoRA                    | —            | —            |
| `openai/whisper-large-v3`          | Whisper      | LoRA                    | 13.20%       | 3.90%        |
| `arampacha/whisper-large-uk-2`     | Whisper      | LoRA                    | 13.03%       | 3.69%        |
| `facebook/wav2vec2-large-xlsr-53`  | Wav2Vec2     | full CTC fine-tuning    | 13.61%       | 2.43%        |
| `Yehor/w2v-bert-uk-v2.1`           | Wav2Vec2-BERT| adapter tuning          | 18.24%       | 3.47%        |
| OmniASR-300M                       | OmniASR      | tri-stage CTC           | 13.82%       | 2.97%        |
| OmniASR-1B                         | OmniASR      | tri-stage CTC           | 13.09%       | 2.75%        |

---

## Repository layout

```
asr_hutsul_reproduction/
├── requirements.txt
├── README.md
├── config.py                 ← StorageLayout, project constants, ProjectConfig
├── preprocess.py             ← dataset loading, splitting, normalisation
├── metrics.py                ← WER/CER + Hutsul-specific error analysis
├── evaluate.py               ← CLI evaluator (auto-detects model family)
├── train.py                  ← unified CLI training launcher
├── configs/                  ← per-family YAML hyper-parameters
│   ├── whisper.yaml
│   ├── wav2vec2.yaml
│   ├── wav2vec2_bert.yaml
│   └── omniasr.yaml
├── models/                   ← per-family Trainer wrappers
│   ├── whisper_trainer.py
│   ├── wav2vec2_trainer.py
│   ├── wav2vec2_bert_trainer.py
│   └── omniasr_trainer.py
├── utils/                    ← reusable helpers
│   ├── augmentation.py       ← waveform aug + SpecAugment
│   ├── text_normalization.py ← Ukrainian/Hutsul text cleanup
│   ├── collators.py          ← Seq2Seq + CTC data collators
│   └── callbacks.py          ← BestCER / memory / early-stopping
├── outputs/                  ← local-fs fallback root (see Storage layout)
└── notebooks/
    ├── colab_training.ipynb  ← main Colab entry point
    └── inference_demo.ipynb  ← single-file transcription demo
```

## Storage layout

All training artefacts (checkpoints, final models, evaluation
outputs, TensorBoard logs, dataset / HF caches, preprocessed splits)
live under a **single root** resolved by `config.StorageLayout`.

| Source             | Picked when                                                         |
|--------------------|---------------------------------------------------------------------|
| `--storage_root X` | The CLI flag is supplied                                            |
| `$HUTSUL_ASR_ROOT` | The env var is set                                                  |
| Drive root         | Running on Colab AND `/content/drive/MyDrive` is mounted            |
| Local fallback     | Any other case — defaults to `<repo>/outputs/`                      |

The Drive root is `/content/drive/MyDrive/hutsul_asr/`.  Subtree:

```
<storage_root>/
├── checkpoints/<variant>/        ← Trainer output_dir, every checkpoint-N
├── final_models/<variant>/       ← post-training save_model + processor
├── preprocessed/<model_type>/    ← cached normalized DatasetDict
│   └── shared/                   ← preprocess.py default
├── evaluations/
│   ├── csv/<run_name>/           ← predictions.csv
│   ├── json/<run_name>/          ← test_results.json + error_analysis.json
│   └── predictions/<run_name>/   ← raw decoded text
├── tensorboard/<variant>/        ← TensorBoard event files
├── cache/                        ← HF_HOME / TRANSFORMERS_CACHE
└── datasets/                     ← HF_DATASETS_CACHE
```

When running inside Colab with Drive mounted, `config.py` automatically
sets `HF_HOME`, `TRANSFORMERS_CACHE` and `HF_DATASETS_CACHE` env vars
on import so every model download is persisted to Drive — no separate
configuration step.

---

## Installation

### Python / CUDA prerequisites

* **Python**: 3.10 or 3.11
* **CUDA**: 12.1+ (PyTorch 2.x bundles it; install the matching wheel
  from <https://pytorch.org/get-started/locally/> if you need
  pre-CUDA-12 support).
* **GPU**: any NVIDIA card with ≥ 8 GB VRAM for the small models;
  see the *Hardware* table below for the heavier variants.

### Local install

```bash
git clone <this-repo>
cd asr_hutsul_reproduction

python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The pinned versions in `requirements.txt` are tested against:

| Package        | Pinned range          |
|----------------|-----------------------|
| torch          | `>=2.2,<2.6`          |
| transformers   | `>=4.45,<4.50`        |
| datasets       | `>=2.20,<3.3`         |
| evaluate       | `>=0.4.2,<0.5`        |
| peft           | `>=0.12,<0.15`        |
| accelerate     | `>=0.34,<1.5`         |
| jiwer          | `>=3.0.4`             |
| audiomentations| `>=0.36.0`            |

### Hugging Face login (optional)

Some base models are gated.  Either set `HF_TOKEN` as an environment
variable or pass `--hf_token` on every command.

```bash
export HF_TOKEN=hf_xxx...
```

---

## Quick start

### Preprocess the dataset (one-off)

```bash
python preprocess.py
```

This downloads the dataset, resamples audio to 16 kHz, applies the
project's text normaliser and writes an 80/10/10 train/val/test
split to `.cache/preprocessed/`.

### Train

The CLI is unified across all four families.  Pass `--model_type`
plus a `--variant` from the corresponding YAML file:

```bash
# Whisper-small with default LoRA (paper recipe)
python train.py --model_type whisper --variant whisper-small

# Whisper-large-v3, custom output dir, augmentation enabled
python train.py --model_type whisper --variant whisper-large-v3 \
                --output_dir outputs/whisper-large-v3 \
                --use_augmentation

# Wav2Vec2-XLSR
python train.py --model_type wav2vec2 --variant xlsr-300m-uk

# Wav2Vec2-BERT-UK with adapter tuning (default)
python train.py --model_type wav2vec2_bert --variant w2v-bert-uk-v2.1

# OmniASR-1B with tri-stage scheduler
python train.py --model_type omniasr --variant omniasr-1b
```

### Resume

```bash
python train.py --model_type whisper --variant whisper-small \
                --resume_from_checkpoint LATEST
```

`LATEST` auto-detects the most recent `checkpoint-*` subdirectory
under the resolved checkpoint dir
(`<storage_root>/checkpoints/<variant>/`).  HF `Trainer` then
transparently restores the optimizer, learning-rate scheduler, FP16
grad-scaler, global step, `TrainerState` and RNG state.

### Evaluate

```bash
# When --output_dir is omitted, evaluate.py writes to the layout's
# evaluations/{csv,json,predictions}/<run_name>/ subtree.
python evaluate.py \
    --checkpoint /content/drive/MyDrive/hutsul_asr/final_models/whisper-small \
    --run_name whisper-small \
    --split test \
    --batch_size 8
```

`evaluate.py` auto-detects the model family from the checkpoint's
`config.json` (or `adapter_config.json` for PEFT/LoRA adapters).
Outputs:

* `test_results.json` — overall WER / CER, run metadata.
* `predictions.csv`   — `(reference, prediction, wer, cer)` per
                       sample.
* `error_analysis.json` — top character-level substitutions /
                          insertions / deletions plus dialect-pair
                          counts (и↔і, е↔є, …).

### Inference demo

See `notebooks/inference_demo.ipynb` for a single-file transcription
example.

---

## Hyper-parameters (paper-aligned)

### Whisper (LoRA)

| Setting                   | small | medium | large-v3 | large-uk-2 |
|---------------------------|-------|--------|----------|------------|
| `max_steps`               | 8 000 | 8 000  | 8 000    | 4 000      |
| `per_device_train_batch`  | 4     | 8      | 4        | 4          |
| `gradient_accumulation`   | 4     | 4      | 4        | 4          |
| effective batch           | 16    | 32     | 16       | 16         |
| LoRA r / α / dropout      | 16 / 32 / 0.05      ||||
| LoRA target               | `q_proj`, `v_proj`  ||||
| `learning_rate`           | 1e-4              ||||
| `warmup_steps`            | 500               ||||
| `generation_max_length`   | 225               ||||
| metric for best model     | CER               ||||

### Wav2Vec2-XLSR (CTC)

| Setting                   | xlsr-300m-uk |
|---------------------------|--------------|
| `max_steps`               | 5 000        |
| `per_device_train_batch`  | 8            |
| `gradient_accumulation`   | 8            |
| effective batch           | 64           |
| `learning_rate`           | 1e-4         |
| `warmup_steps`            | 1 000        |
| feature encoder           | frozen       |
| metric for best model     | CER          |

### Wav2Vec2-BERT-UK-v2.1

| Setting                              | value          |
|--------------------------------------|----------------|
| `max_steps`                          | 5 000          |
| `per_device_train_batch`             | 8              |
| `gradient_accumulation`              | 4 (eff. 32)    |
| `learning_rate`                      | 5e-5           |
| `warmup_steps`                       | 800            |
| feature encoder                      | frozen         |
| lower transformer layers frozen      | 6              |
| adapters trainable                   | yes            |
| augmentation                         | strong preset  |
| metric for best model                | CER            |

### OmniASR

| Setting                   | 300M    | 1B    |
|---------------------------|---------|-------|
| `max_steps`               | 48 000  | 36 000|
| `per_device_train_batch`  | 8       | 4     |
| `gradient_accumulation`   | 4       | 8     |
| effective batch           | 32      | 32    |
| `learning_rate`           | 5e-5    | 5e-5  |
| scheduler                 | tri-stage (warmup 0.10, hold 0.40, decay 0.50) ||
| metric for best model     | WER     | WER   |

---

## LoRA vs. adapter tuning

* **LoRA (Whisper)** — low-rank update matrices are inserted next to
  the `q_proj` and `v_proj` linear layers.  Trainable parameters
  drop to ≲ 1 % of the base model, but the architecture stays
  intact so `model.generate(...)` can be merged back into the base
  weights at inference time (`merge_and_unload`).

* **Adapter tuning (Wav2Vec2-BERT-UK)** — the model already ships
  with adapter modules per encoder layer plus an output adapter.
  We freeze the convolutional feature encoder and the bottom 6
  transformer layers, then unfreeze every parameter whose name
  contains `"adapter"` plus the LM head.  This matches the
  paper-reported recipe and reproduces WER ≈ 18.24%.

`evaluate.py` auto-detects PEFT-only checkpoints (those containing
`adapter_config.json`) and loads them on top of the base model
declared in the adapter config.

---

## Augmentation pipeline

`utils/augmentation.py` implements two stages:

1. **Waveform** (`audiomentations.Compose`):
   * `AddGaussianNoise`,
   * `PitchShift` (±2 semitones, ±3 in *strong* preset),
   * `TimeStretch` (0.8 × – 1.2 ×),
   * `Gain` (±6 dB / ±8 dB *strong*).

2. **Feature-domain** SpecAugment with independent time and
   frequency masking (Park et al., 2019).

Toggle on/off with `--use_augmentation` / `--no_augmentation`.  The
strong preset is enabled by default for Wav2Vec2-BERT.

---

## Hardware requirements

| Model               | Recommended GPU | VRAM (fp16) | Approx. epoch time |
|---------------------|-----------------|-------------|--------------------|
| Whisper-small       | T4 16 GB        | 6–8 GB      | ~25 min/1k steps   |
| Whisper-medium      | L4 / A100       | 14–16 GB    | ~45 min/1k steps   |
| Whisper-large-v3    | A100 40 GB      | 22–28 GB    | ~75 min/1k steps   |
| Whisper-large-uk-2  | A100 40 GB      | 22–28 GB    | ~75 min/1k steps   |
| Wav2Vec2-XLSR       | T4 16 GB        | 8–10 GB     | ~25 min/1k steps   |
| Wav2Vec2-BERT-UK    | L4 / A100       | 14–18 GB    | ~30 min/1k steps   |
| OmniASR-300M        | L4 / A100       | 14–16 GB    | ~30 min/1k steps   |
| OmniASR-1B          | A100 40 GB      | 28–34 GB    | ~60 min/1k steps   |

If your GPU has less VRAM than recommended, reduce `--batch_size`
and increase `--grad_accum` proportionally to preserve the effective
batch size.

---

## Reproducibility limitations

* **Non-deterministic CUDA kernels.**  Several CTC and attention
  kernels do not have deterministic implementations.  The
  `--deterministic` flag enables `torch.use_deterministic_algorithms(True,
  warn_only=True)` plus `CUBLAS_WORKSPACE_CONFIG=:4096:8`, which
  *significantly* slows training down but bit-for-bit reproduces a
  run on the same hardware.  Without the flag, expect ±0.05–0.20%
  WER variance between identical runs.

* **Tokenizer drift.**  Hugging Face occasionally reissues
  tokenizer files for popular checkpoints.  Pinning
  `transformers` and `tokenizers` versions (see `requirements.txt`)
  guards against this.

* **Dataset version.**  The KSE-RESEARCH-Group dataset is
  versioned (`-v2` suffix).  We pin the dataset name; if the
  upstream maintainers issue a `-v3`, reproductions should re-train
  to compare with the original paper.

* **PEFT adapter format.**  PEFT < 0.12 used a different adapter
  metadata schema.  This project requires PEFT 0.12+; if you need
  to load older adapters, upgrade them with PEFT's migration helper.

* **Mixed precision determinism.**  FP16 reductions are inherently
  non-associative; CER differences below ~0.1% should not be
  considered significant when comparing fp16 runs.

---

## Troubleshooting

### `TypeError: __init__() got an unexpected keyword argument 'evaluation_strategy'`

Your `transformers` install is older than 4.41.  The argument was
renamed to `eval_strategy`.  Upgrade:

```bash
pip install --upgrade 'transformers>=4.45,<4.50'
```

### `ValueError: load_best_model_at_end requires the save and eval strategy to match`

The `metric_for_best_model` is set but `save_strategy` and
`eval_strategy` are different.  All YAML configs ship the
canonical `("steps", "steps")` pair — if you pass overrides via
the CLI, keep them aligned.

### `RuntimeError: Expected to mark a variable ready only once. ...` with PEFT

Set `gradient_checkpointing_kwargs={"use_reentrant": False}`
(already the project default).  The `use_reentrant=True` variant is
incompatible with PEFT.

### `ImportError: cannot import name 'load_metric' from 'datasets'`

`load_metric` was removed.  This project uses `evaluate.load`
exclusively — make sure you pulled all source files cleanly.

### Whisper generates English text instead of Ukrainian

Confirm the run actually fine-tuned and that
`generation_config.language` / `.task` are persisted.  `evaluate.py`
re-applies them at load time, but a custom inference loop must do
the same.

### CUDA out of memory

Lower `--batch_size`, raise `--grad_accum`.  As a last resort,
disable augmentation (`--no_augmentation`) and switch to a smaller
variant.

### Resume does nothing / starts from step 0

* Confirm the output directory you pass with `--output_dir` is the
  same one you trained into.
* Inspect `outputs/<model>/<variant>/checkpoint-*/` — empty
  directories from a killed save can prevent
  `get_last_checkpoint` from returning a valid path.  Delete them
  and retry.

### Wav2Vec2-BERT loss stays at `inf` or `nan`

Set `ctc_zero_infinity=true` in the YAML (already the project
default).  This is required when an audio segment is shorter than
its tokenised transcription.

---

## Citing

If you use this project, please cite the original paper and
acknowledge the
[`KSE-RESEARCH-Group/Dido-Yvanchyk-Audio-Dataset-v2`](https://huggingface.co/datasets/KSE-RESEARCH-Group/Dido-Yvanchyk-Audio-Dataset-v2)
authors.

## License

Code in this repository is released under the MIT License.  Check
the underlying base models and the dataset for their respective
licences.
