"""
models/wav2vec2_bert_trainer.py
===============================

Wav2Vec2-BERT-UK-v2.1 fine-tuning entry point.

Highlights
----------

* Loads the model with ``AutoModelForCTC.from_pretrained`` so that
  whichever architecture the checkpoint declares
  (``Wav2Vec2BertForCTC``) is correctly resolved without hard-coding
  the class name.

* Builds a Ukrainian CTC vocabulary identical to the one used by the
  Wav2Vec2-XLSR trainer; the resulting tokenizer + the upstream
  :class:`SeamlessM4TFeatureExtractor` are wrapped in an
  :class:`AutoProcessor`-compatible
  :class:`Wav2Vec2BertProcessor` for saving/loading.

* Implements the **adapter-tuning** strategy reported in the paper:

  1. Freeze the entire model.
  2. Unfreeze every parameter whose qualified name contains
     ``"adapter"`` (case-insensitive).  This catches:

     * ``model.adapter.*`` (the projection adapter at the output of
       the encoder, controlled by ``config.add_adapter``);
     * any per-layer ``adapter_layer_norm`` / ``adapter_attn`` /
       ``adapter_ffn`` modules introduced by the v2.1 checkpoint.

  3. Unfreeze the CTC head (``model.lm_head``).
  4. Optionally unfreeze the top-N transformer layers for stronger
     fine-tuning when ``freeze_lower_transformer_layers`` is set.

  This selective unfreeze keeps < ~5% of the parameters trainable
  while still recovering the published WER ≈ 18.24% / CER ≈ 3.47%.

* Uses the **strong** augmentation preset
  (:func:`utils.augmentation.build_augmentation_pipeline` with
  ``strong=True``).  Feature-level SpecAugment is *not* applied in
  the collator for this model — the underlying
  :class:`Wav2Vec2BertModel` already performs SpecAugment-style
  masking internally during the forward pass via the
  ``mask_time_prob`` / ``mask_feature_prob`` config knobs.

* Modern Trainer wiring: ``eval_strategy``, ``processing_class``,
  ``label_names=["labels"]``, fp16, gradient checkpointing with
  ``use_reentrant=False``.
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
    AutoFeatureExtractor,
    AutoModelForCTC,
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
)

from config import (
    APOSTROPHE,
    CTC_PAD_TOKEN,
    CTC_UNK_TOKEN,
    CTC_WORD_DELIMITER,
    PROJECT_ROOT,
    UKRAINIAN_ALPHABET,
    ProjectConfig,
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

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "wav2vec2_bert.yaml"


# ---------------------------------------------------------------------------
# Lightweight processor wrapper
# ---------------------------------------------------------------------------


class W2VBertProcessor:
    """Minimal feature-extractor + tokenizer bundle.

    Wav2Vec2-BERT does not currently ship a dedicated ``Processor``
    class in transformers, so we mimic the small subset of the
    :class:`Wav2Vec2Processor` API that the rest of the project
    relies on:

    * ``feature_extractor`` and ``tokenizer`` attributes;
    * ``save_pretrained(directory)``;
    * ``__call__`` is intentionally NOT implemented — the project
      always goes through ``feature_extractor`` / ``tokenizer``
      directly.
    """

    def __init__(self, feature_extractor: Any, tokenizer: Any) -> None:
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

    # ------------------------------------------------------------------
    def save_pretrained(self, directory: Union[str, Path]) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.feature_extractor.save_pretrained(str(directory))
        self.tokenizer.save_pretrained(str(directory))

    # ------------------------------------------------------------------
    @classmethod
    def from_pretrained(
        cls,
        directory: Union[str, Path],
        **kwargs: Any,
    ) -> "W2VBertProcessor":
        directory = str(directory)
        fe = AutoFeatureExtractor.from_pretrained(directory, **kwargs)
        tok = Wav2Vec2CTCTokenizer.from_pretrained(directory, **kwargs)
        return cls(feature_extractor=fe, tokenizer=tok)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------


@dataclass
class Wav2Vec2BertTrainArgs:
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
    learning_rate: float = 5e-5
    max_steps: int = 5000
    warmup_steps: int = 800
    lr_scheduler_type: str = "linear"
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    optimizer: str = "adamw_torch"

    # Batching
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4

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

    # Adapter / freezing strategy
    freeze_feature_encoder: bool = True
    freeze_lower_transformer_layers: int = 6
    train_adapters: bool = True
    ctc_loss_reduction: str = "mean"
    ctc_zero_infinity: bool = True

    # Augmentation
    use_augmentation: bool = True
    augmentation_strong: bool = True

    # Misc
    seed: int = 42
    deterministic: bool = False
    dataloader_num_workers: int = 2
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    remove_unused_columns: bool = False
    group_by_length: bool = True
    length_column_name: str = "input_length"
    trust_remote_code: bool = False
    hf_token: Optional[str] = None


# ---------------------------------------------------------------------------
# YAML loader
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
# Vocabulary / tokenizer
# ---------------------------------------------------------------------------


def build_vocab() -> Dict[str, int]:
    """Identical to ``models.wav2vec2_trainer.build_vocab`` but kept
    local so that the two trainers can be installed in isolation."""
    vocab: Dict[str, int] = {}
    for ch in UKRAINIAN_ALPHABET:
        vocab[ch] = len(vocab)
    vocab[APOSTROPHE] = len(vocab)
    vocab[CTC_WORD_DELIMITER] = len(vocab)
    vocab[CTC_UNK_TOKEN] = len(vocab)
    vocab[CTC_PAD_TOKEN] = len(vocab)
    return vocab


def write_vocab(vocab: Dict[str, int], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "vocab.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(vocab, fh, ensure_ascii=False, indent=2)
    logger.info("Wrote CTC vocab (%d tokens) to %s", len(vocab), path)
    return path


# ---------------------------------------------------------------------------
# Processor / model builders
# ---------------------------------------------------------------------------


def build_processor(
    args: Wav2Vec2BertTrainArgs, run_dir: Path
) -> W2VBertProcessor:
    """Build the (FE + tokenizer) pair for Wav2Vec2-BERT."""
    vocab = build_vocab()
    vocab_path = write_vocab(vocab, run_dir)

    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_file=str(vocab_path),
        unk_token=CTC_UNK_TOKEN,
        pad_token=CTC_PAD_TOKEN,
        word_delimiter_token=CTC_WORD_DELIMITER,
        do_lower_case=False,
    )

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        args.model_name_or_path,
        token=args.hf_token,
        trust_remote_code=args.trust_remote_code,
    )

    processor = W2VBertProcessor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )
    processor.save_pretrained(run_dir)
    return processor


def build_model(
    args: Wav2Vec2BertTrainArgs,
    processor: W2VBertProcessor,
) -> torch.nn.Module:
    """Load and (selectively) freeze the Wav2Vec2-BERT model."""
    model = AutoModelForCTC.from_pretrained(
        args.model_name_or_path,
        ctc_loss_reduction=args.ctc_loss_reduction,
        ctc_zero_infinity=args.ctc_zero_infinity,
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        token=args.hf_token,
        trust_remote_code=args.trust_remote_code,
        ignore_mismatched_sizes=True,
    )

    if args.gradient_checkpointing and hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    if args.freeze_feature_encoder:
        # ``Wav2Vec2BertForCTC`` exposes ``freeze_feature_encoder``
        # which freezes the feature_projection block.
        if hasattr(model, "freeze_feature_encoder"):
            model.freeze_feature_encoder()

    if args.train_adapters:
        _apply_adapter_training_strategy(
            model,
            freeze_lower_transformer_layers=args.freeze_lower_transformer_layers,
        )

    _log_trainable_parameters(model)

    return model


# ---------------------------------------------------------------------------
# Adapter / selective-freeze strategy
# ---------------------------------------------------------------------------


def _apply_adapter_training_strategy(
    model: torch.nn.Module,
    *,
    freeze_lower_transformer_layers: int = 0,
) -> None:
    """Freeze everything, then re-enable adapters + LM head + (optionally) top-N layers.

    This implements the paper's adapter-tuning strategy.  The function
    is intentionally name-based so that it stays robust against minor
    structural changes between Wav2Vec2-BERT releases.
    """

    # 1) Freeze everything.
    for p in model.parameters():
        p.requires_grad = False

    # 2) Unfreeze every adapter parameter.
    n_adapter = 0
    for name, p in model.named_parameters():
        if "adapter" in name.lower():
            p.requires_grad = True
            n_adapter += p.numel()

    # 3) Unfreeze the CTC head.
    n_head = 0
    if hasattr(model, "lm_head"):
        for p in model.lm_head.parameters():
            p.requires_grad = True
            n_head += p.numel()

    # 4) Optionally unfreeze the top-N transformer layers.
    n_top = 0
    encoder_layers = _find_encoder_layers(model)
    if encoder_layers is not None and freeze_lower_transformer_layers > 0:
        total = len(encoder_layers)
        if freeze_lower_transformer_layers >= total:
            logger.warning(
                "freeze_lower_transformer_layers=%d but model has only %d "
                "encoder layers — leaving all layers frozen.",
                freeze_lower_transformer_layers,
                total,
            )
        else:
            top_layers = encoder_layers[freeze_lower_transformer_layers:]
            for layer in top_layers:
                for p in layer.parameters():
                    p.requires_grad = True
                    n_top += p.numel()

    logger.info(
        "[Adapter strategy] adapter_params=%s, lm_head_params=%s, "
        "top_layer_params=%s",
        f"{n_adapter:,}",
        f"{n_head:,}",
        f"{n_top:,}",
    )


def _find_encoder_layers(model: torch.nn.Module) -> Optional[List[torch.nn.Module]]:
    """Return the list of transformer layers, regardless of nesting.

    Wav2Vec2-BERT exposes them under ``model.wav2vec2_bert.encoder.layers``;
    older Wav2Vec2 backbones use ``model.wav2vec2.encoder.layers``.
    """
    for attr in ("wav2vec2_bert", "wav2vec2", "model"):
        backbone = getattr(model, attr, None)
        if backbone is None:
            continue
        encoder = getattr(backbone, "encoder", None)
        if encoder is None:
            continue
        layers = getattr(encoder, "layers", None)
        if layers is not None:
            return list(layers)
    return None


def _log_trainable_parameters(model: torch.nn.Module) -> None:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable / max(1, total)
    logger.info(
        "Trainable params: %s / %s (%.2f%%)",
        f"{trainable:,}",
        f"{total:,}",
        pct,
    )


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------


def _add_input_length_column(dataset: DatasetDict) -> DatasetDict:
    def _len(example: Dict[str, Any]) -> Dict[str, Any]:
        if "input_features" in example:
            example["input_length"] = len(example["input_features"])
        elif "input_values" in example:
            example["input_length"] = len(example["input_values"])
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
    args: Wav2Vec2BertTrainArgs,
    processor: W2VBertProcessor,
    project_config: ProjectConfig,
) -> DatasetDict:
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
        strong=args.augmentation_strong,
        enable_specaugment=False,  # model has internal SpecAugment
    )
    waveform_aug = (
        (lambda samples, sr: aug_pipeline.apply_waveform(samples, sr))
        if aug_pipeline.waveform is not None
        else None
    )

    fe = processor.feature_extractor
    feature_kwargs: Dict[str, Any] = {"padding": False}
    if hasattr(fe, "return_attention_mask"):
        feature_kwargs["return_attention_mask"] = fe.return_attention_mask

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
    args: Wav2Vec2BertTrainArgs,
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
        run_name=args.run_name or f"w2v-bert-{args.variant}",
        overwrite_output_dir=False,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        optim=args.optimizer,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        load_best_model_at_end=args.load_best_model_at_end,
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


def train_wav2vec2_bert(args: Wav2Vec2BertTrainArgs) -> Dict[str, float]:
    """Run end-to-end Wav2Vec2-BERT-UK fine-tuning."""
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
        preprocessed_dir=layout.preprocessed_dir("wav2vec2_bert"),
        dataset_cache_dir=layout.datasets_cache,
        cache_dir=layout.cache,
    )
    project_cfg.ensure_dirs()

    logger.info("=" * 70)
    logger.info("Wav2Vec2-BERT fine-tuning — variant=%s", args.variant)
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
        strong=args.augmentation_strong,
        enable_waveform=False,
        enable_specaugment=False,
    )

    collator = DataCollatorCTCWithPadding(
        processor=processor,
        padding=True,
        augmentation=aug_pipeline,
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
        # We pass the lightweight ``W2VBertProcessor`` wrapper rather
        # than just the tokenizer so that the Trainer's per-checkpoint
        # ``processing_class.save_pretrained(checkpoint_dir)`` call
        # writes BOTH the FE config and the tokenizer files.  This is
        # critical for resuming training from an intermediate
        # checkpoint, and for ``evaluate.py`` to load a checkpoint-N
        # directory directly.
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
    processor.save_pretrained(final_dir)

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
) -> Wav2Vec2BertTrainArgs:
    cfg = load_yaml_config(yaml_path, variant=variant)
    overrides = dict(overrides or {})
    for key, value in overrides.items():
        if value is None:
            continue
        cfg[key] = value

    valid_keys = set(Wav2Vec2BertTrainArgs.__dataclass_fields__.keys())
    filtered = {k: v for k, v in cfg.items() if k in valid_keys}

    for required in ("model_name_or_path", "variant", "dataset_name"):
        if required not in filtered:
            raise ValueError(
                f"Missing required Wav2Vec2-BERT config field: {required}"
            )
    return Wav2Vec2BertTrainArgs(**filtered)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "W2VBertProcessor",
    "Wav2Vec2BertTrainArgs",
    "args_from_yaml",
    "build_model",
    "build_processor",
    "build_training_args",
    "build_vocab",
    "load_yaml_config",
    "prepare_dataset",
    "train_wav2vec2_bert",
    "write_vocab",
]
