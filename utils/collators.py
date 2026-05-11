from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch

from .augmentation import AugmentationPipeline

logger = logging.getLogger(__name__)


def _detect_input_key(sample: Dict[str, Any]) -> str:
    if "input_features" in sample:
        return "input_features"
    if "input_values" in sample:
        return "input_values"
    raise KeyError(
        "Sample does not contain 'input_features' or 'input_values'. "
        f"Available keys: {list(sample.keys())}"
    )


_WHISPER_ALLOWED_KEYS = frozenset(
    {"input_features", "attention_mask", "labels"}
)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:

    processor: Any
    decoder_start_token_id: Optional[int] = None
    augmentation: Optional[AugmentationPipeline] = None
    enforce_keys: bool = True

    def __post_init__(self) -> None:
        if self.decoder_start_token_id is None:
            tokenizer = getattr(self.processor, "tokenizer", None)
            if tokenizer is not None:
                self.decoder_start_token_id = tokenizer.bos_token_id

    def __call__(
        self, features: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        if not features:
            raise ValueError("Empty batch passed to collator")

        input_key = _detect_input_key(features[0])
        input_features: List[Dict[str, Any]] = [
            {input_key: f[input_key]} for f in features
        ]

        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt",
        )

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )


        if (
            self.decoder_start_token_id is not None
            and labels.shape[1] > 0
            and bool((labels[:, 0] == self.decoder_start_token_id).all())
        ):
            labels = labels[:, 1:]

        batch["labels"] = labels

        if self.augmentation is not None and self.augmentation.specaugment is not None:
            try:
                batch[input_key] = self.augmentation.apply_features(
                    batch[input_key]
                )
            except Exception as exc:
                logger.warning(
                    "SpecAugment failed on a batch (%s); using unaugmented "
                    "features.",
                    exc,
                )


        if self.enforce_keys:
            allowed = _WHISPER_ALLOWED_KEYS
            extra = [k for k in list(batch.keys()) if k not in allowed]
            if extra:
                blocking = {"input_ids", "inputs_embeds"}
                hard = [k for k in extra if k in blocking]
                if hard:
                    raise RuntimeError(
                        "Whisper collator produced disallowed keys "
                        f"{hard!r} (full batch keys: {list(batch.keys())}).  "
                        "Construct LoraConfig WITHOUT task_type."
                    )
                for k in extra:
                    batch.pop(k, None)

        return batch


@dataclass
class DataCollatorCTCWithPadding:
    processor: Any
    padding: Union[bool, str] = True
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None
    augmentation: Optional[AugmentationPipeline] = None
    min_input_samples: Optional[int] = None

    def __call__(
        self, features: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        if not features:
            raise ValueError("Empty batch passed to collator")

        input_key = _detect_input_key(features[0])
        input_features = [{input_key: f[input_key]} for f in features]

        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of_labels,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels

        if (
            self.min_input_samples is not None
            and input_key == "input_values"
            and batch[input_key].dim() == 2
        ):
            current = batch[input_key].shape[1]
            target = int(self.min_input_samples)
            if current < target:
                pad_amt = target - current
                pad_zeros = torch.zeros(
                    batch[input_key].shape[0],
                    pad_amt,
                    dtype=batch[input_key].dtype,
                )
                batch[input_key] = torch.cat(
                    [batch[input_key], pad_zeros], dim=1
                )
                if "attention_mask" in batch:
                    pad_mask = torch.zeros(
                        batch["attention_mask"].shape[0],
                        pad_amt,
                        dtype=batch["attention_mask"].dtype,
                    )
                    batch["attention_mask"] = torch.cat(
                        [batch["attention_mask"], pad_mask], dim=1
                    )
                logger.debug(
                    "Floor-padded input_values from %d -> %d samples",
                    current,
                    target,
                )

        if self.augmentation is not None and self.augmentation.specaugment is not None:
            try:
                batch[input_key] = self.augmentation.apply_features(
                    batch[input_key]
                )
            except Exception as exc:
                logger.warning(
                    "SpecAugment failed on a batch (%s); using unaugmented "
                    "features.",
                    exc,
                )

        return batch


__all__ = [
    "DataCollatorCTCWithPadding",
    "DataCollatorSpeechSeq2SeqWithPadding",
]
