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


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "whisper.yaml"


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

    model_name_or_path: str
    variant: str
    dataset_name: str
    dataset_config: Optional[str] = None
    audio_column: Optional[str] = None
    text_column: Optional[str] = None

    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    run_name: Optional[str] = None
    resume_from_checkpoint: Optional[str] = None

    language: str = "uk"
    task: str = "transcribe"
    generation_max_length: int = 225
    num_beams: int = 1
    sample_rate: int = 16000

    learning_rate: float = 1e-4
    max_steps: int = 8000
    warmup_steps: int = 500
    lr_scheduler_type: str = "linear"
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    optimizer: str = "adamw_torch"

    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4

    eval_steps: int = 500
    save_steps: int = 500
    logging_steps: int = 50
    save_total_limit: int = 3

    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True
    predict_with_generate: bool = True

    metric_for_best_model: str = "cer"
    greater_is_better: bool = False
    load_best_model_at_end: bool = True
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.0

    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )

    use_augmentation: bool = False
    seed: int = 42
    deterministic: bool = False
    dataloader_num_workers: int = 2
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    remove_unused_columns: bool = False
    group_by_length: bool = False
    trust_remote_code: bool = False
    hf_token: Optional[str] = None


def build_processor(args: WhisperTrainArgs) -> WhisperProcessor:
    processor = WhisperProcessor.from_pretrained(
        args.model_name_or_path,
        language=args.language,
        task=args.task,
        token=args.hf_token,
        trust_remote_code=args.trust_remote_code,
    )
    return processor


def build_model(args: WhisperTrainArgs) -> WhisperForConditionalGeneration:
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        token=args.hf_token,
        trust_remote_code=args.trust_remote_code,
    )

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

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False

    return model


def maybe_apply_lora(
    model: WhisperForConditionalGeneration, args: WhisperTrainArgs
) -> torch.nn.Module:
    if not args.use_lora:
        return model

    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=list(args.lora_target_modules),
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
    )

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    peft_model = get_peft_model(model, lora_config)

    try:
        peft_model.print_trainable_parameters()
    except Exception:
        pass

    return peft_model


def ensure_whisper_generation_metadata(
    model: torch.nn.Module,
    processor: Any,
    model_name_or_path: Optional[str] = None,
    *,
    hf_token: Optional[str] = None,
) -> None:
    """Populate ``lang_to_id`` / ``task_to_id`` / ``no_timestamps_token_id``
    on ``model.generation_config`` if any are missing.

    transformers >= 4.45 requires these mappings inside
    ``Whisper.generate()`` to resolve the language and task prefix
    tokens.  Older checkpoints (e.g. ``arampacha/whisper-large-uk-2``)
    ship a ``generation_config.json`` predating that schema, so we
    seed the missing fields from a freshly loaded GenerationConfig
    first and fall back to the tokenizer vocabulary.
    """
    # ``generation_config`` is owned by the inner Whisper model, but
    # both ``WhisperForConditionalGeneration`` and ``PeftModel`` expose
    # it directly: HF stores it as a plain attribute, and PEFT's
    # ``__getattr__`` proxies missing names through ``base_model`` to
    # the wrapped model.  Mutating the returned object mutates the one
    # underlying ``GenerationConfig`` instance, so we deliberately do
    # NOT walk ``.base_model.model`` — that path is wrapper-specific
    # and crashes on raw Whisper (``WhisperModel`` has no ``.model``).
    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is None:
        return

    needed = ("lang_to_id", "task_to_id", "no_timestamps_token_id")
    if all(getattr(gen_cfg, n, None) for n in needed):
        return

    if model_name_or_path:
        try:
            from transformers import GenerationConfig

            fresh = GenerationConfig.from_pretrained(
                model_name_or_path, token=hf_token
            )
            for n in needed:
                if not getattr(gen_cfg, n, None) and getattr(fresh, n, None):
                    setattr(gen_cfg, n, getattr(fresh, n))
        except Exception as exc:
            logger.warning(
                "Could not reload GenerationConfig from %s (%s); will "
                "build mappings from the tokenizer.",
                model_name_or_path,
                exc,
            )

    if all(getattr(gen_cfg, n, None) for n in needed):
        return

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return

    import re

    unk_id = tokenizer.unk_token_id

    if not getattr(gen_cfg, "lang_to_id", None):
        lang_to_id: Dict[str, int] = {}
        for tok, idx in tokenizer.get_vocab().items():
            m = re.fullmatch(r"<\|([a-z]{2,3})\|>", tok)
            if m:
                lang_to_id[tok] = int(idx)
        if lang_to_id:
            gen_cfg.lang_to_id = lang_to_id

    if not getattr(gen_cfg, "task_to_id", None):
        task_to_id: Dict[str, int] = {}
        for task in ("transcribe", "translate"):
            idx = tokenizer.convert_tokens_to_ids(f"<|{task}|>")
            if isinstance(idx, int) and idx != unk_id:
                task_to_id[task] = idx
        if task_to_id:
            gen_cfg.task_to_id = task_to_id

    if not getattr(gen_cfg, "no_timestamps_token_id", None):
        idx = tokenizer.convert_tokens_to_ids("<|notimestamps|>")
        if isinstance(idx, int) and idx != unk_id:
            gen_cfg.no_timestamps_token_id = idx

    logger.info(
        "Whisper generation_config metadata ready: "
        "lang_to_id=%d langs, task_to_id=%d tasks, no_timestamps_token_id=%s",
        len(getattr(gen_cfg, "lang_to_id", {}) or {}),
        len(getattr(gen_cfg, "task_to_id", {}) or {}),
        getattr(gen_cfg, "no_timestamps_token_id", None),
    )


def prepare_dataset(
    args: WhisperTrainArgs,
    processor: WhisperProcessor,
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
        enable_specaugment=False,
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
        feature_kwargs={},
        tokenizer_kwargs={"return_attention_mask": False},
    )

    return prepared


def build_seq2seq_training_args(
    args: WhisperTrainArgs,
    *,
    logging_dir: Optional[Path] = None,
) -> Seq2SeqTrainingArguments:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if logging_dir is not None:
        Path(logging_dir).mkdir(parents=True, exist_ok=True)

    return Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        logging_dir=str(logging_dir) if logging_dir is not None else None,
        run_name=args.run_name or f"whisper-{args.variant}",
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
        label_names=["labels"],
    )


def train_whisper(args: WhisperTrainArgs) -> Dict[str, float]:
    configure_logging()
    set_global_seed(args.seed, deterministic=args.deterministic)

    if args.hf_token is None:
        args.hf_token = os.environ.get("HF_TOKEN")

    layout = resolve_storage_layout()
    layout.ensure()

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

    logger.info("Whisper fine-tuning — variant=%s", args.variant)
    logger.info("Model:        %s", args.model_name_or_path)
    logger.info("Checkpoints:  %s", args.output_dir)
    logger.info("Final model:  %s", final_dir)
    logger.info("TensorBoard:  %s", tensorboard_dir)

    processor = build_processor(args)
    model = build_model(args)
    # Seed generation_config.lang_to_id / task_to_id / no_timestamps_token_id
    # BEFORE PEFT wrapping so we operate on the bare Whisper config.
    ensure_whisper_generation_metadata(
        model, processor, args.model_name_or_path, hf_token=args.hf_token
    )
    model = maybe_apply_lora(model, args)

    dataset = prepare_dataset(args, processor, project_cfg)

    aug_pipeline = build_augmentation_pipeline(
        use_augmentation=args.use_augmentation,
        sample_rate=args.sample_rate,
        enable_waveform=False,
        enable_specaugment=True,
    )

    base_model = (
        model.base_model.model
        if hasattr(model, "base_model")
        else model
    )
    decoder_start_token_id = (
        getattr(base_model.config, "decoder_start_token_id", None)
    )
    if decoder_start_token_id is None:
        decoder_start_token_id = processor.tokenizer.bos_token_id

    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=decoder_start_token_id,
        augmentation=aug_pipeline,
    )

    metric_calculator = MetricCalculator()
    compute_metrics = build_compute_metrics_seq2seq(
        processor, metric_calculator=metric_calculator
    )

    training_args = build_seq2seq_training_args(
        args, logging_dir=tensorboard_dir
    )

    callbacks = build_default_callbacks(
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=args.early_stopping_threshold,
    )

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


def args_from_yaml(
    yaml_path: Union[str, Path],
    variant: Optional[str] = None,
    *,
    overrides: Optional[Dict[str, Any]] = None,
) -> WhisperTrainArgs:
    cfg = load_yaml_config(yaml_path, variant=variant)
    overrides = dict(overrides or {})

    lora_block = cfg.pop("lora", {})
    cfg.setdefault("lora_r", lora_block.get("r", 16))
    cfg.setdefault("lora_alpha", lora_block.get("lora_alpha", 32))
    cfg.setdefault("lora_dropout", lora_block.get("lora_dropout", 0.05))
    cfg.setdefault("lora_bias", lora_block.get("bias", "none"))
    cfg.setdefault(
        "lora_target_modules",
        list(lora_block.get("target_modules", ["q_proj", "v_proj"])),
    )

    cfg.setdefault("optimizer", cfg.pop("optimizer", "adamw_torch"))

    for key, value in overrides.items():
        if value is None:
            continue
        cfg[key] = value

    valid_keys = set(WhisperTrainArgs.__dataclass_fields__.keys())
    filtered = {k: v for k, v in cfg.items() if k in valid_keys}

    for required in ("model_name_or_path", "variant", "dataset_name"):
        if required not in filtered:
            raise ValueError(
                f"Missing required Whisper config field: {required}"
            )

    return WhisperTrainArgs(**filtered)


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "WhisperTrainArgs",
    "args_from_yaml",
    "build_model",
    "build_processor",
    "ensure_whisper_generation_metadata",
    "build_seq2seq_training_args",
    "load_yaml_config",
    "maybe_apply_lora",
    "prepare_dataset",
    "train_whisper",
]
