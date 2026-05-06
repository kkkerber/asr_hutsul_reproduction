"""
models/whisper_trainer.py
=========================

Whisper-family fine-tuning entry point.

The trainer uses the modern Hugging Face stack (transformers >= 4.45,
PEFT >= 0.12, accelerate >= 0.34):

* :class:`WhisperProcessor` is loaded with ``language=`` /
  ``task=`` so that the prefix tokens ``<|startoftranscript|><|uk|>
  <|transcribe|><|notimestamps|>`` are added during tokenisation.
* The model's :class:`GenerationConfig` is updated *in place* — we do
  not set ``forced_decoder_ids`` (the modern API recommends letting
  the generation-config language/task pair drive the prefix).  We
  also clear ``suppress_tokens`` so the model is free to emit any
  token during evaluation.
* PEFT LoRA is applied with ``target_modules=["q_proj", "v_proj"]``,
  ``r=16``, ``lora_alpha=32``, ``lora_dropout=0.05``, ``bias="none"``
  and **no** ``task_type`` (using the generic :class:`peft.PeftModel`
  wrapper — the seq2seq wrapper would inject ``input_ids`` into
  Whisper's forward and crash it).
* The Trainer is :class:`Seq2SeqTrainer` with
  ``predict_with_generate=True`` and the modern
  ``eval_strategy`` argument (the old ``evaluation_strategy`` was
  renamed in transformers 4.41).  The ``processing_class`` argument
  is used (the old ``tokenizer=`` keyword has been deprecated since
  4.46).
* Labels are padded with ``-100`` inside
  :class:`utils.collators.DataCollatorSpeechSeq2SeqWithPadding` and
  the duplicate BOS token is stripped before training.

Usage
-----

::

    from models.whisper_trainer import train_whisper
    train_whisper(args)            # ``args`` from ``train.py``
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

import torch
from datasets import DatasetDict
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from config import (
    PROJECT_ROOT,
    ProjectConfig,
    configure_logging,
    resolve_storage_layout,
    set_global_seed,
)
from metrics import (
    MetricCalculator,
    build_compute_metrics_seq2seq,
)
from preprocess import load_and_prepare, prepare_for_model
from utils.augmentation import build_augmentation_pipeline
from utils.callbacks import build_default_callbacks
from utils.collators import DataCollatorSpeechSeq2SeqWithPadding

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "whisper.yaml"


def load_yaml_config(
    path: Union[str, Path] = DEFAULT_CONFIG_PATH,
    *,
    variant: Optional[str] = None,
) -> Dict[str, Any]:
    """Load the Whisper YAML config and merge a chosen variant.

    The result is a single flat dict that can be passed straight to
    ``Seq2SeqTrainingArguments`` (after dropping non-Trainer keys).
    """
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    defaults = dict(raw.get("defaults", {}))
    variants = raw.get("variants", {})

    if not variants:
        raise ValueError(f"{path} does not declare any variants")

    if variant is None:
        variant = next(iter(variants.keys()))
        logger.info("No --variant given, defaulting to %s", variant)

    if variant not in variants:
        raise KeyError(
            f"Variant {variant!r} not found in {path}. "
            f"Available: {sorted(variants)}"
        )

    merged = {**defaults, **variants[variant]}
    merged["variant"] = variant
    return merged


# ---------------------------------------------------------------------------
# Argument bundle (filled either from CLI overrides or YAML)
# ---------------------------------------------------------------------------


@dataclass
class WhisperTrainArgs:
    """Resolved training arguments for the Whisper pipeline.

    ``train.py`` builds an instance of this via
    :func:`build_train_args_from_cli`.
    """

    # Model / dataset
    model_name_or_path: str
    variant: str
    dataset_name: str
    dataset_config: Optional[str] = None
    audio_column: Optional[str] = None
    text_column: Optional[str] = None

    # Output
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    run_name: Optional[str] = None
    resume_from_checkpoint: Optional[str] = None

    # Whisper-specific
    language: str = "uk"
    task: str = "transcribe"
    generation_max_length: int = 225
    num_beams: int = 1
    sample_rate: int = 16000

    # Schedule
    learning_rate: float = 1e-4
    max_steps: int = 8000
    warmup_steps: int = 500
    lr_scheduler_type: str = "linear"
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    optimizer: str = "adamw_torch"

    # Batching
    per_device_train_batch_size: int = 4
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
    predict_with_generate: bool = True

    # Best-model selection
    metric_for_best_model: str = "cer"
    greater_is_better: bool = False
    load_best_model_at_end: bool = True
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.0

    # PEFT / LoRA
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )

    # Misc
    use_augmentation: bool = False
    seed: int = 42
    deterministic: bool = False
    dataloader_num_workers: int = 2
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    remove_unused_columns: bool = False
    group_by_length: bool = False
    trust_remote_code: bool = False
    hf_token: Optional[str] = None


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def build_processor(args: WhisperTrainArgs) -> WhisperProcessor:
    """Load the Whisper processor with language/task already wired in."""
    processor = WhisperProcessor.from_pretrained(
        args.model_name_or_path,
        language=args.language,
        task=args.task,
        token=args.hf_token,
        trust_remote_code=args.trust_remote_code,
    )
    return processor


def build_model(args: WhisperTrainArgs) -> WhisperForConditionalGeneration:
    """Load the Whisper model and configure the generation config.

    Modern Whisper (transformers >= 4.36) drives the prefix entirely
    through ``generation_config.language`` / ``.task``.  Setting
    ``forced_decoder_ids`` is no longer necessary (and emits a
    deprecation warning).  We clear it explicitly along with
    ``suppress_tokens`` so PEFT does not leak the legacy values.
    """
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        token=args.hf_token,
        trust_remote_code=args.trust_remote_code,
    )

    # Configure generation.  We update both ``model.config`` (legacy
    # path used by some callers) and ``model.generation_config``.
    if model.generation_config is None:
        from transformers import GenerationConfig

        model.generation_config = GenerationConfig.from_pretrained(
            args.model_name_or_path,
            token=args.hf_token,
        )
    model.generation_config.language = args.language
    model.generation_config.task = args.task
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []
    model.generation_config.max_length = args.generation_max_length
    model.generation_config.num_beams = args.num_beams

    # ``model.config.forced_decoder_ids`` is the legacy field; set to
    # ``None`` so HF's deprecation path doesn't override our
    # generation_config language/task.
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    # ``use_cache`` must be off when gradient checkpointing is on,
    # otherwise we get warnings and incorrect gradients.
    model.config.use_cache = False

    return model


def maybe_apply_lora(
    model: WhisperForConditionalGeneration, args: WhisperTrainArgs
) -> torch.nn.Module:
    """Wrap ``model`` with PEFT LoRA according to ``args``."""
    if not args.use_lora:
        return model

    from peft import LoraConfig, get_peft_model

    # IMPORTANT: do *not* set ``task_type`` here.
    #
    # Setting ``task_type=TaskType.SEQ_2_SEQ_LM`` causes
    # ``peft.get_peft_model`` to wrap the model in
    # :class:`peft.PeftModelForSeq2SeqLM`, which is a text-to-text
    # seq2seq wrapper.  Its ``forward`` hard-codes a text-input
    # signature (``input_ids=None``, ``inputs_embeds=None``, ...) and
    # explicitly passes ``input_ids=...`` and ``inputs_embeds=...``
    # to the base model — both kwargs that
    # :class:`WhisperForConditionalGeneration.forward` does not
    # accept (Whisper expects ``input_features``).  Forward then
    # raises:
    #
    #   TypeError: WhisperForConditionalGeneration.forward() got an
    #              unexpected keyword argument 'input_ids'
    #
    # Leaving ``task_type`` unset (or ``None``) makes
    # ``get_peft_model`` return the generic :class:`peft.PeftModel`
    # whose ``forward`` simply delegates ``*args, **kwargs`` to the
    # base model.  This is what the official HF Whisper-LoRA
    # fine-tuning recipe uses.
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=list(args.lora_target_modules),
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
    )

    # When using PEFT + gradient checkpointing the input embeddings'
    # outputs need ``requires_grad=True``; PEFT exposes a helper for
    # this on the underlying model.
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    peft_model = get_peft_model(model, lora_config)

    # Print parameter stats to the log so it is easy to verify LoRA
    # actually shrinks the trainable footprint.
    try:
        peft_model.print_trainable_parameters()
    except Exception:  # pragma: no cover
        pass

    return peft_model


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------


def prepare_dataset(
    args: WhisperTrainArgs,
    processor: WhisperProcessor,
    project_config: ProjectConfig,
) -> DatasetDict:
    """Run preprocessing + featurisation for the Whisper pipeline."""

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
        enable_specaugment=False,  # SpecAugment is applied in collator
    )

    waveform_aug = (
        (lambda samples, sr: aug_pipeline.apply_waveform(samples, sr))
        if aug_pipeline.waveform is not None
        else None
    )

    prepared = prepare_for_model(
        raw,
        audio_column=audio_col,
        text_column=text_col,
        feature_extractor=processor.feature_extractor,
        tokenizer=processor.tokenizer,
        sample_rate=args.sample_rate,
        waveform_augmenter=waveform_aug,
        augment_splits=("train",),
        feature_kwargs={},  # Whisper's FE handles padding to 30s internally
        tokenizer_kwargs={"return_attention_mask": False},
    )

    return prepared


# ---------------------------------------------------------------------------
# Training arguments
# ---------------------------------------------------------------------------


def build_seq2seq_training_args(
    args: WhisperTrainArgs,
    *,
    logging_dir: Optional[Path] = None,
) -> Seq2SeqTrainingArguments:
    """Map ``WhisperTrainArgs`` to ``Seq2SeqTrainingArguments``.

    The argument names below match transformers >= 4.46.  In
    particular:

    * ``eval_strategy`` (renamed from ``evaluation_strategy``);
    * ``optim`` (string identifier, ``adamw_torch``);
    * ``report_to`` is a list of strings;
    * ``predict_with_generate`` and ``generation_max_length`` are
      Seq2Seq-specific;
    * ``label_names=["labels"]`` is required when using PEFT, otherwise
      the Trainer cannot find the label tensor inside the inputs.
    """
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if logging_dir is not None:
        Path(logging_dir).mkdir(parents=True, exist_ok=True)

    return Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        logging_dir=str(logging_dir) if logging_dir is not None else None,
        run_name=args.run_name or f"whisper-{args.variant}",
        overwrite_output_dir=False,
        # ---- Schedule ------------------------------------------------
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        optim=args.optimizer,
        # ---- Batching ------------------------------------------------
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        # ---- Cadence -------------------------------------------------
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        # ---- Mixed precision / memory --------------------------------
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        predict_with_generate=args.predict_with_generate,
        generation_max_length=args.generation_max_length,
        generation_num_beams=args.num_beams,
        # ---- Best-model ----------------------------------------------
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        load_best_model_at_end=args.load_best_model_at_end,
        # ---- Misc ----------------------------------------------------
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to=args.report_to,
        remove_unused_columns=args.remove_unused_columns,
        group_by_length=args.group_by_length,
        # PEFT requires explicit label names; Whisper labels live in
        # the ``labels`` column produced by the collator.
        label_names=["labels"],
        # ``logging_dir`` defaults to ``output_dir/runs/...`` which is
        # exactly what we want for TensorBoard.
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def train_whisper(args: WhisperTrainArgs) -> Dict[str, float]:
    """Run end-to-end Whisper fine-tuning.

    Returns a dictionary containing the final evaluation metrics
    (post-training).  All artefacts are written under
    ``args.output_dir``.
    """
    configure_logging()
    set_global_seed(args.seed, deterministic=args.deterministic)

    if args.hf_token is None:
        args.hf_token = os.environ.get("HF_TOKEN")

    layout = resolve_storage_layout()
    layout.ensure()

    # Default ``output_dir`` to ``<storage_root>/checkpoints/<variant>/``
    # when the caller (train.py / direct API) did not override it.
    if not args.output_dir or Path(args.output_dir) == Path("outputs"):
        args.output_dir = layout.checkpoint_dir(args.variant)
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

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
        preprocessed_dir=layout.preprocessed_dir("whisper"),
        dataset_cache_dir=layout.datasets_cache,
        cache_dir=layout.cache,
    )
    project_cfg.ensure_dirs()

    logger.info("=" * 70)
    logger.info("Whisper fine-tuning — variant=%s", args.variant)
    logger.info("Model:        %s", args.model_name_or_path)
    logger.info("Checkpoints:  %s", args.output_dir)
    logger.info("Final model:  %s", final_dir)
    logger.info("TensorBoard:  %s", tensorboard_dir)
    logger.info("Storage root: %s", layout.root)
    logger.info("=" * 70)

    # ---- Processor / model / LoRA ---------------------------------------
    processor = build_processor(args)
    model = build_model(args)
    model = maybe_apply_lora(model, args)

    # ---- Dataset ---------------------------------------------------------
    dataset = prepare_dataset(args, processor, project_cfg)

    # ---- Collator + augmentation ----------------------------------------
    aug_pipeline = build_augmentation_pipeline(
        use_augmentation=args.use_augmentation,
        sample_rate=args.sample_rate,
        enable_waveform=False,  # waveform aug already applied in map
        enable_specaugment=True,
    )

    # ``decoder_start_token_id`` lives on the original (non-PEFT) config.
    # For Whisper this is ``<|startoftranscript|>`` (id 50258) — NOT
    # ``bos_token_id`` (id 50257, ``<|endoftext|>``).  The collator
    # uses this id to strip the duplicate prefix token that the
    # Trainer would otherwise produce after ``shift_tokens_right``.
    base_model = (
        model.base_model.model
        if hasattr(model, "base_model")
        else model
    )
    decoder_start_token_id = (
        getattr(base_model.config, "decoder_start_token_id", None)
    )
    if decoder_start_token_id is None:
        # Fallback to bos only when the config really has nothing.
        decoder_start_token_id = processor.tokenizer.bos_token_id

    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=decoder_start_token_id,
        augmentation=aug_pipeline,
    )

    # ---- Metrics ---------------------------------------------------------
    metric_calculator = MetricCalculator()
    compute_metrics = build_compute_metrics_seq2seq(
        processor, metric_calculator=metric_calculator
    )

    # ---- TrainingArguments ----------------------------------------------
    training_args = build_seq2seq_training_args(
        args, logging_dir=tensorboard_dir
    )

    # ---- Callbacks -------------------------------------------------------
    callbacks = build_default_callbacks(
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=args.early_stopping_threshold,
    )

    # ---- Trainer ---------------------------------------------------------
    # ``processing_class`` (transformers 4.46+) replaces the old
    # ``tokenizer=`` kwarg.  Passing the full ``WhisperProcessor`` lets
    # the Trainer save the feature extractor and the tokenizer
    # together at checkpoint time.
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=collator,
        compute_metrics=compute_metrics,
        processing_class=processor,
        callbacks=callbacks,
    )

    # ---- Train -----------------------------------------------------------
    train_result = trainer.train(
        resume_from_checkpoint=args.resume_from_checkpoint
    )
    trainer.save_state()
    trainer.save_metrics("train", train_result.metrics)

    # Save the final (best) model under
    # ``<storage_root>/final_models/<variant>/``.  For PEFT we save
    # the adapters; for full fine-tuning we save the whole model.
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))

    # ---- Final evaluation -----------------------------------------------
    eval_metrics = trainer.evaluate(
        eval_dataset=dataset["validation"],
        metric_key_prefix="eval_final",
    )
    trainer.save_metrics("eval_final", eval_metrics)

    logger.info("Training complete. Final eval metrics: %s", eval_metrics)
    return eval_metrics


# ---------------------------------------------------------------------------
# YAML -> WhisperTrainArgs glue used by ``train.py``
# ---------------------------------------------------------------------------


def args_from_yaml(
    yaml_path: Union[str, Path],
    variant: Optional[str] = None,
    *,
    overrides: Optional[Dict[str, Any]] = None,
) -> WhisperTrainArgs:
    """Build a fully-populated :class:`WhisperTrainArgs` from YAML.

    ``overrides`` is a flat dict of CLI overrides that takes precedence
    over the YAML.  Unknown keys are ignored (so the dispatcher in
    ``train.py`` can pass a single dict to every model trainer).
    """
    cfg = load_yaml_config(yaml_path, variant=variant)
    overrides = dict(overrides or {})

    # YAML-only nested ``lora`` block -> flat ``lora_*`` fields.
    lora_block = cfg.pop("lora", {})
    cfg.setdefault("lora_r", lora_block.get("r", 16))
    cfg.setdefault("lora_alpha", lora_block.get("lora_alpha", 32))
    cfg.setdefault("lora_dropout", lora_block.get("lora_dropout", 0.05))
    cfg.setdefault("lora_bias", lora_block.get("bias", "none"))
    cfg.setdefault(
        "lora_target_modules",
        list(lora_block.get("target_modules", ["q_proj", "v_proj"])),
    )

    # Optimizer name in YAML is ``optimizer`` but the field is the same.
    cfg.setdefault("optimizer", cfg.pop("optimizer", "adamw_torch"))

    # Apply CLI overrides last.
    for key, value in overrides.items():
        if value is None:
            continue
        cfg[key] = value

    # Pick out fields that ``WhisperTrainArgs`` accepts.
    valid_keys = set(WhisperTrainArgs.__dataclass_fields__.keys())
    filtered = {k: v for k, v in cfg.items() if k in valid_keys}

    # Pre-condition: required fields.
    for required in ("model_name_or_path", "variant", "dataset_name"):
        if required not in filtered:
            raise ValueError(
                f"Missing required Whisper config field: {required}"
            )

    return WhisperTrainArgs(**filtered)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "WhisperTrainArgs",
    "args_from_yaml",
    "build_model",
    "build_processor",
    "build_seq2seq_training_args",
    "load_yaml_config",
    "maybe_apply_lora",
    "prepare_dataset",
    "train_whisper",
]
