"""
models/wav2vec2_trainer.py
==========================

Wav2Vec2-XLSR (CTC) fine-tuning entry point.

Pipeline overview
-----------------

1. **Vocabulary** — built from the project-level Ukrainian alphabet
   (:data:`config.CTC_VOCAB`) and written as ``vocab.json`` next to
   the run output.  A :class:`Wav2Vec2CTCTokenizer` is then loaded
   from that vocab with ``[PAD]`` / ``[UNK]`` / word-delimiter
   ``"|"``.

2. **Feature extractor** — :class:`Wav2Vec2FeatureExtractor` loaded
   from the base checkpoint so that pretraining-time normalisation
   settings (``do_normalize``, ``return_attention_mask``) are
   preserved.

3. **Processor** — wraps the two together as a
   :class:`Wav2Vec2Processor` and is saved alongside checkpoints.

4. **Model** — :class:`Wav2Vec2ForCTC` loaded from the base
   checkpoint with overrides for ``vocab_size``, dropouts and
   masking probabilities pulled from the YAML.  The convolutional
   feature encoder is frozen by default (standard XLSR recipe).

5. **Trainer** — :class:`transformers.Trainer` with the modern
   ``eval_strategy`` argument, ``processing_class=processor`` and
   ``label_names=["labels"]``.  ``compute_metrics`` is built by
   :func:`metrics.build_compute_metrics_ctc`.

6. **Padding** — labels are padded with ``-100`` inside
   :class:`utils.collators.DataCollatorCTCWithPadding` so the CTC
   loss skips padding positions; ``input_values`` are 0-padded with
   an attention mask returned by the feature extractor.

The expected paper results are WER ≈ 13.61% / CER ≈ 2.43%.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)

from config import (
    CTC_PAD_TOKEN,
    CTC_UNK_TOKEN,
    CTC_VOCAB,
    CTC_WORD_DELIMITER,
    PROJECT_ROOT,
    ProjectConfig,
    UKRAINIAN_ALPHABET,
    APOSTROPHE,
    configure_logging,
    resolve_storage_layout,
    set_global_seed,
)
from metrics import (
    MetricCalculator,
    build_compute_metrics_ctc,
)
from preprocess import load_and_prepare, prepare_for_model
from utils.augmentation import build_augmentation_pipeline
from utils.callbacks import build_default_callbacks
from utils.collators import DataCollatorCTCWithPadding

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "wav2vec2.yaml"


# ---------------------------------------------------------------------------
# YAML -> args
# ---------------------------------------------------------------------------


def load_yaml_config(
    path: Union[str, Path] = DEFAULT_CONFIG_PATH,
    *,
    variant: Optional[str] = None,
) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    defaults = dict(raw.get("defaults", {}))
    variants = raw.get("variants", {})
    if not variants:
        raise ValueError(f"{path} declares no variants")

    if variant is None:
        variant = next(iter(variants.keys()))
        logger.info("No --variant given, defaulting to %s", variant)
    if variant not in variants:
        raise KeyError(
            f"Variant {variant!r} not in {path}. "
            f"Available: {sorted(variants)}"
        )

    merged = {**defaults, **variants[variant]}
    merged["variant"] = variant
    return merged


# ---------------------------------------------------------------------------
# Args dataclass
# ---------------------------------------------------------------------------


@dataclass
class Wav2Vec2TrainArgs:
    """Resolved training arguments for the Wav2Vec2-XLSR pipeline."""

    model_name_or_path: str
    variant: str
    dataset_name: str
    dataset_config: Optional[str] = None
    audio_column: Optional[str] = None
    text_column: Optional[str] = None

    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    run_name: Optional[str] = None
    resume_from_checkpoint: Optional[str] = None

    sample_rate: int = 16000

    # Schedule
    learning_rate: float = 1e-4
    max_steps: int = 5000
    warmup_steps: int = 1000
    lr_scheduler_type: str = "linear"
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    optimizer: str = "adamw_torch"

    # Batching
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8

    # Cadence
    eval_steps: int = 500
    save_steps: int = 500
    logging_steps: int = 50
    save_total_limit: int = 3

    # Mixed precision / memory
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True

    # Best-model selection
    metric_for_best_model: str = "cer"
    greater_is_better: bool = False
    load_best_model_at_end: bool = True
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.0

    # CTC / model overrides
    ctc_loss_reduction: str = "mean"
    ctc_zero_infinity: bool = True
    freeze_feature_encoder: bool = True
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    feat_proj_dropout: float = 0.0
    mask_time_prob: float = 0.05
    mask_feature_prob: float = 0.0
    layerdrop: float = 0.0

    # Misc
    use_augmentation: bool = False
    seed: int = 42
    deterministic: bool = False
    dataloader_num_workers: int = 2
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    remove_unused_columns: bool = False
    group_by_length: bool = True
    length_column_name: str = "input_length"
    trust_remote_code: bool = False
    hf_token: Optional[str] = None

    # Audio-length safety (CTC-only) ------------------------------------
    # Drop dataset samples shorter than this many seconds.  Required
    # to prevent ``Wav2Vec2Model._compute_mask_indices`` from raising
    # ``mask_length > sequence_length`` on extreme-short batches.
    # 1.0 s gives ~50 encoder frames at conv stride 320 — well above
    # the default ``mask_time_length=10``.
    min_train_audio_duration_sec: float = 1.0
    # Defence-in-depth: collator floor-pads any batch whose
    # ``input_values`` shorter than this number of samples.  Default
    # 6400 = 0.4 s = 2 × mask_time_length × stride.
    min_collator_input_samples: int = 6400


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------


def build_vocab() -> Dict[str, int]:
    """Build the project-level CTC vocabulary as ``{token: id}``.

    Order:
        0..N-1   — Ukrainian alphabet (deterministic)
        N        — apostrophe
        N+1      — word delimiter "|"
        N+2      — [UNK]
        N+3      — [PAD]
    """
    vocab: Dict[str, int] = {}

    for ch in UKRAINIAN_ALPHABET:
        vocab[ch] = len(vocab)
    vocab[APOSTROPHE] = len(vocab)
    vocab[CTC_WORD_DELIMITER] = len(vocab)
    vocab[CTC_UNK_TOKEN] = len(vocab)
    vocab[CTC_PAD_TOKEN] = len(vocab)

    # Sanity check: every token from CTC_VOCAB must be present.
    for token in CTC_VOCAB:
        if token not in vocab:
            raise RuntimeError(
                f"Token {token!r} from CTC_VOCAB missing from generated vocab"
            )
    return vocab


def write_vocab(vocab: Dict[str, int], output_dir: Path) -> Path:
    """Persist ``vocab.json`` in ``output_dir`` (created if needed)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "vocab.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(vocab, fh, ensure_ascii=False, indent=2)
    logger.info("Wrote CTC vocab (%d tokens) to %s", len(vocab), path)
    return path


# ---------------------------------------------------------------------------
# Tokenizer / processor / model
# ---------------------------------------------------------------------------


def build_processor(
    args: Wav2Vec2TrainArgs, run_dir: Path
) -> Wav2Vec2Processor:
    """Build (vocab + tokenizer + FE + processor) and persist them.

    Returns
    -------
    Wav2Vec2Processor
        Ready to be passed to the Trainer as ``processing_class``.
    """
    vocab = build_vocab()
    vocab_path = write_vocab(vocab, run_dir)

    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_file=str(vocab_path),
        unk_token=CTC_UNK_TOKEN,
        pad_token=CTC_PAD_TOKEN,
        word_delimiter_token=CTC_WORD_DELIMITER,
        do_lower_case=False,
    )

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        args.model_name_or_path,
        token=args.hf_token,
        trust_remote_code=args.trust_remote_code,
    )

    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )
    processor.save_pretrained(str(run_dir))
    return processor


def build_model(
    args: Wav2Vec2TrainArgs, processor: Wav2Vec2Processor
) -> Wav2Vec2ForCTC:
    """Load and configure the Wav2Vec2 model for CTC fine-tuning."""
    model = Wav2Vec2ForCTC.from_pretrained(
        args.model_name_or_path,
        attention_dropout=args.attention_dropout,
        hidden_dropout=args.hidden_dropout,
        feat_proj_dropout=args.feat_proj_dropout,
        mask_time_prob=args.mask_time_prob,
        mask_feature_prob=args.mask_feature_prob,
        layerdrop=args.layerdrop,
        ctc_loss_reduction=args.ctc_loss_reduction,
        ctc_zero_infinity=args.ctc_zero_infinity,
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        token=args.hf_token,
        trust_remote_code=args.trust_remote_code,
    )

    if args.freeze_feature_encoder:
        # Modern API; ``freeze_feature_extractor`` is deprecated and
        # raises a FutureWarning.
        if hasattr(model, "freeze_feature_encoder"):
            model.freeze_feature_encoder()
        else:  # pragma: no cover — fallback
            for p in model.wav2vec2.feature_extractor.parameters():
                p.requires_grad = False

    if args.gradient_checkpointing:
        # ``use_cache`` is not present on Wav2Vec2 configs but setting
        # it does no harm and silences a forward-time warning when the
        # underlying transformer is shared with a generative backbone.
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    return model


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------


def _add_input_length_column(dataset: DatasetDict) -> DatasetDict:
    """Add a numeric ``input_length`` column for ``group_by_length``.

    The Trainer reads this column to bucket examples of similar
    duration into the same batch, which substantially reduces padding
    overhead for variable-length CTC inputs.
    """

    def _len(example: Dict[str, Any]) -> Dict[str, Any]:
        if "input_values" in example:
            example["input_length"] = len(example["input_values"])
        elif "input_features" in example:
            example["input_length"] = len(example["input_features"])
        else:
            example["input_length"] = 0
        return example

    new_splits: Dict[str, Dataset] = {}
    for split, ds in dataset.items():
        new_splits[split] = ds.map(
            _len, desc=f"Adding input_length to {split}"
        )
    return DatasetDict(new_splits)


def prepare_dataset(
    args: Wav2Vec2TrainArgs,
    processor: Wav2Vec2Processor,
    project_config: ProjectConfig,
) -> DatasetDict:
    """Run preprocessing + featurisation for the Wav2Vec2 pipeline."""
    raw, audio_col, text_col = load_and_prepare(
        project_config,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        audio_column=args.audio_column,
        text_column=args.text_column,
        token=args.hf_token,
        trust_remote_code=args.trust_remote_code,
    )

    aug_pipeline = build_augmentation_pipeline(
        use_augmentation=args.use_augmentation,
        sample_rate=args.sample_rate,
        enable_specaugment=False,  # collator handles SpecAugment
    )
    waveform_aug = (
        (lambda samples, sr: aug_pipeline.apply_waveform(samples, sr))
        if aug_pipeline.waveform is not None
        else None
    )

    # Whether the FE returns an attention mask depends on the model.
    # Honour the FE config so we don't accidentally pad without a mask.
    fe = processor.feature_extractor
    feature_kwargs: Dict[str, Any] = {}
    if hasattr(fe, "return_attention_mask"):
        feature_kwargs["return_attention_mask"] = fe.return_attention_mask
    feature_kwargs.setdefault("padding", False)  # pad in collator

    prepared = prepare_for_model(
        raw,
        audio_column=audio_col,
        text_column=text_col,
        feature_extractor=processor.feature_extractor,
        tokenizer=processor.tokenizer,
        sample_rate=args.sample_rate,
        waveform_augmenter=waveform_aug,
        augment_splits=("train",),
        feature_kwargs=feature_kwargs,
        tokenizer_kwargs={},
    )

    if args.group_by_length:
        prepared = _add_input_length_column(prepared)

    return prepared


# ---------------------------------------------------------------------------
# Training arguments
# ---------------------------------------------------------------------------


def build_training_args(
    args: Wav2Vec2TrainArgs,
    *,
    logging_dir: Optional[Path] = None,
) -> TrainingArguments:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if logging_dir is not None:
        Path(logging_dir).mkdir(parents=True, exist_ok=True)

    return TrainingArguments(
        output_dir=str(output_dir),
        logging_dir=str(logging_dir) if logging_dir is not None else None,
        run_name=args.run_name or f"wav2vec2-{args.variant}",
        overwrite_output_dir=False,
        # Schedule
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        optim=args.optimizer,
        # Batching
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        # Cadence
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        # Mixed precision / memory
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Best-model
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        load_best_model_at_end=args.load_best_model_at_end,
        # Misc
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to=args.report_to,
        remove_unused_columns=args.remove_unused_columns,
        group_by_length=args.group_by_length,
        length_column_name=args.length_column_name,
        label_names=["labels"],
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def train_wav2vec2(args: Wav2Vec2TrainArgs) -> Dict[str, float]:
    """Run end-to-end Wav2Vec2-XLSR fine-tuning."""
    configure_logging()
    set_global_seed(args.seed, deterministic=args.deterministic)

    if args.hf_token is None:
        args.hf_token = os.environ.get("HF_TOKEN")

    layout = resolve_storage_layout()
    layout.ensure()

    if not args.output_dir or Path(args.output_dir) == Path("outputs"):
        args.output_dir = layout.checkpoint_dir(args.variant)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = output_dir

    final_dir = layout.final_model_dir(args.variant)
    tensorboard_dir = layout.tensorboard_dir(args.variant)

    project_cfg = ProjectConfig(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        audio_column=args.audio_column,
        text_column=args.text_column,
        sample_rate=args.sample_rate,
        seed=args.seed,
        deterministic=args.deterministic,
        use_augmentation=args.use_augmentation,
        preprocessed_dir=layout.preprocessed_dir("wav2vec2"),
        dataset_cache_dir=layout.datasets_cache,
        cache_dir=layout.cache,
        min_train_audio_duration_sec=args.min_train_audio_duration_sec,
        min_collator_input_samples=args.min_collator_input_samples,
    )
    project_cfg.ensure_dirs()

    logger.info("=" * 70)
    logger.info("Wav2Vec2-XLSR fine-tuning — variant=%s", args.variant)
    logger.info("Model:        %s", args.model_name_or_path)
    logger.info("Checkpoints:  %s", output_dir)
    logger.info("Final model:  %s", final_dir)
    logger.info("TensorBoard:  %s", tensorboard_dir)
    logger.info("Storage root: %s", layout.root)
    logger.info("=" * 70)

    processor = build_processor(args, output_dir)
    model = build_model(args, processor)
    dataset = prepare_dataset(args, processor, project_cfg)

    aug_pipeline = build_augmentation_pipeline(
        use_augmentation=args.use_augmentation,
        sample_rate=args.sample_rate,
        enable_waveform=False,
        enable_specaugment=True,
    )

    collator = DataCollatorCTCWithPadding(
        processor=processor,
        padding=True,
        augmentation=aug_pipeline,
        min_input_samples=args.min_collator_input_samples,
    )

    metric_calculator = MetricCalculator()
    compute_metrics = build_compute_metrics_ctc(
        processor, metric_calculator=metric_calculator
    )

    training_args = build_training_args(args, logging_dir=tensorboard_dir)
    callbacks = build_default_callbacks(
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=args.early_stopping_threshold,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=collator,
        compute_metrics=compute_metrics,
        processing_class=processor,
        callbacks=callbacks,
    )

    train_result = trainer.train(
        resume_from_checkpoint=args.resume_from_checkpoint
    )
    trainer.save_state()
    trainer.save_metrics("train", train_result.metrics)

    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))

    eval_metrics = trainer.evaluate(
        eval_dataset=dataset["validation"],
        metric_key_prefix="eval_final",
    )
    trainer.save_metrics("eval_final", eval_metrics)

    logger.info("Training complete. Final eval metrics: %s", eval_metrics)
    return eval_metrics


# ---------------------------------------------------------------------------
# YAML -> args glue
# ---------------------------------------------------------------------------


def args_from_yaml(
    yaml_path: Union[str, Path],
    variant: Optional[str] = None,
    *,
    overrides: Optional[Dict[str, Any]] = None,
) -> Wav2Vec2TrainArgs:
    cfg = load_yaml_config(yaml_path, variant=variant)
    overrides = dict(overrides or {})
    for key, value in overrides.items():
        if value is None:
            continue
        cfg[key] = value

    valid_keys = set(Wav2Vec2TrainArgs.__dataclass_fields__.keys())
    filtered = {k: v for k, v in cfg.items() if k in valid_keys}

    for required in ("model_name_or_path", "variant", "dataset_name"):
        if required not in filtered:
            raise ValueError(
                f"Missing required Wav2Vec2 config field: {required}"
            )
    return Wav2Vec2TrainArgs(**filtered)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "Wav2Vec2TrainArgs",
    "args_from_yaml",
    "build_model",
    "build_processor",
    "build_training_args",
    "build_vocab",
    "load_yaml_config",
    "prepare_dataset",
    "train_wav2vec2",
    "write_vocab",
]
