from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

import torch
from torch.optim.lr_scheduler import LambdaLR
from datasets import Dataset, DatasetDict
from transformers import (
    AutoFeatureExtractor,
    AutoModelForCTC,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
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
    build_compute_metrics_ctc,
)
from preprocess import load_and_prepare, prepare_for_model
from utils.augmentation import build_augmentation_pipeline
from utils.callbacks import build_default_callbacks
from utils.collators import DataCollatorCTCWithPadding

logger = logging.getLogger(__name__)


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "omniasr.yaml"

class OmniASRProcessor:
    def __init__(self, feature_extractor: Any, tokenizer: Any) -> None:
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

    @property
    def model_input_names(self) -> List[str]:
        # Required by Trainer.get_train_dataloader() (transformers >= 4.44)
        # under group_by_length=True.
        names = getattr(self.feature_extractor, "model_input_names", None)
        return list(names) if names else ["input_features"]

    def save_pretrained(self, directory: Union[str, Path]) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.feature_extractor.save_pretrained(str(directory))
        self.tokenizer.save_pretrained(str(directory))


@dataclass
class OmniASRTrainArgs:
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

    learning_rate: float = 5e-5
    max_steps: int = 48000
    warmup_ratio: float = 0.10
    hold_ratio: float = 0.40
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    optimizer: str = "adamw_torch"

    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4

    eval_steps: int = 1000
    save_steps: int = 1000
    logging_steps: int = 100
    save_total_limit: int = 3

    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True

    metric_for_best_model: str = "wer"
    greater_is_better: bool = False
    load_best_model_at_end: bool = True
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.0

    ctc_loss_reduction: str = "mean"
    ctc_zero_infinity: bool = True

    use_augmentation: bool = False
    seed: int = 42
    deterministic: bool = False
    dataloader_num_workers: int = 2
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    remove_unused_columns: bool = False
    group_by_length: bool = True
    length_column_name: str = "input_length"
    trust_remote_code: bool = True
    hf_token: Optional[str] = None

    min_train_audio_duration_sec: float = 1.0
    min_collator_input_samples: int = 0


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


def tri_stage_lambda(
    *,
    total_steps: int,
    warmup_ratio: float,
    hold_ratio: float,
) -> Any:
    if total_steps <= 0:
        raise ValueError(f"total_steps must be > 0 (got {total_steps})")
    if not 0.0 <= warmup_ratio <= 1.0:
        raise ValueError(f"warmup_ratio must be in [0,1] (got {warmup_ratio})")
    if not 0.0 <= hold_ratio <= 1.0:
        raise ValueError(f"hold_ratio must be in [0,1] (got {hold_ratio})")
    if warmup_ratio + hold_ratio > 1.0:
        raise ValueError(
            f"warmup_ratio + hold_ratio must be <= 1.0 "
            f"(got {warmup_ratio + hold_ratio})"
        )

    warmup_steps = max(1, int(total_steps * warmup_ratio))
    hold_steps = int(total_steps * hold_ratio)
    decay_steps = max(1, total_steps - warmup_steps - hold_steps)

    def _fn(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(warmup_steps)
        if step < warmup_steps + hold_steps:
            return 1.0
        progress = (step - warmup_steps - hold_steps) / float(decay_steps)
        return max(0.0, 1.0 - progress)

    return _fn


def build_tri_stage_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_ratio: float,
    hold_ratio: float,
) -> LambdaLR:
    return LambdaLR(
        optimizer,
        lr_lambda=tri_stage_lambda(
            total_steps=total_steps,
            warmup_ratio=warmup_ratio,
            hold_ratio=hold_ratio,
        ),
    )


def build_processor(args: OmniASRTrainArgs) -> OmniASRProcessor:
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        args.model_name_or_path,
        token=args.hf_token,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        token=args.hf_token,
        trust_remote_code=args.trust_remote_code,
    )
    return OmniASRProcessor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )


def build_model(
    args: OmniASRTrainArgs, processor: OmniASRProcessor
) -> torch.nn.Module:
    model_kwargs: Dict[str, Any] = {
        "token": args.hf_token,
        "trust_remote_code": args.trust_remote_code,
        "ignore_mismatched_sizes": True,
    }

    optional_overrides: Dict[str, Any] = {
        "ctc_loss_reduction": args.ctc_loss_reduction,
        "ctc_zero_infinity": args.ctc_zero_infinity,
    }
    if processor.tokenizer.pad_token_id is not None:
        optional_overrides["pad_token_id"] = processor.tokenizer.pad_token_id

    model = AutoModelForCTC.from_pretrained(
        args.model_name_or_path,
        **optional_overrides,
        **model_kwargs,
    )

    if args.gradient_checkpointing and hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    return model


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
    args: OmniASRTrainArgs,
    processor: OmniASRProcessor,
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
    args: OmniASRTrainArgs,
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
        run_name=args.run_name or f"omniasr-{args.variant}",
        overwrite_output_dir=False,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
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


def build_optimizer_and_scheduler(
    model: torch.nn.Module,
    args: OmniASRTrainArgs,
) -> Tuple[torch.optim.Optimizer, LambdaLR]:
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    decay_params: List[torch.nn.Parameter] = []
    no_decay_params: List[torch.nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(nd in name for nd in no_decay):
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    param_groups = [
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    scheduler = build_tri_stage_scheduler(
        optimizer,
        total_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        hold_ratio=args.hold_ratio,
    )

    return optimizer, scheduler


def train_omniasr(args: OmniASRTrainArgs) -> Dict[str, float]:
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
        preprocessed_dir=layout.preprocessed_dir("omniasr"),
        dataset_cache_dir=layout.datasets_cache,
        cache_dir=layout.cache,
        min_train_audio_duration_sec=args.min_train_audio_duration_sec,
    )
    project_cfg.ensure_dirs()

    logger.info("OmniASR fine-tuning — variant=%s", args.variant)
    logger.info("Model:        %s", args.model_name_or_path)
    logger.info("Checkpoints:  %s", output_dir)
    logger.info("Final model:  %s", final_dir)
    logger.info("TensorBoard:  %s", tensorboard_dir)

    processor = build_processor(args)
    processor.save_pretrained(output_dir)

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

    optimizer, scheduler = build_optimizer_and_scheduler(model, args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=collator,
        compute_metrics=compute_metrics,
        processing_class=processor,
        callbacks=callbacks,
        optimizers=(optimizer, scheduler),
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

def args_from_yaml(
    yaml_path: Union[str, Path],
    variant: Optional[str] = None,
    *,
    overrides: Optional[Dict[str, Any]] = None,
) -> OmniASRTrainArgs:
    cfg = load_yaml_config(yaml_path, variant=variant)
    overrides = dict(overrides or {})
    for key, value in overrides.items():
        if value is None:
            continue
        cfg[key] = value

    valid_keys = set(OmniASRTrainArgs.__dataclass_fields__.keys())
    filtered = {k: v for k, v in cfg.items() if k in valid_keys}

    for required in ("model_name_or_path", "variant", "dataset_name"):
        if required not in filtered:
            raise ValueError(
                f"Missing required OmniASR config field: {required}"
            )
    return OmniASRTrainArgs(**filtered)


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "OmniASRProcessor",
    "OmniASRTrainArgs",
    "args_from_yaml",
    "build_model",
    "build_optimizer_and_scheduler",
    "build_processor",
    "build_training_args",
    "build_tri_stage_scheduler",
    "load_yaml_config",
    "prepare_dataset",
    "train_omniasr",
    "tri_stage_lambda",
]
