"""
utils/collators.py
==================

Data collators for the four model families used in this project.

* :class:`DataCollatorSpeechSeq2SeqWithPadding`
      — Whisper-style encoder/decoder collator.  Pads ``input_features``
        with the feature extractor and ``labels`` with the tokenizer.
        Padding tokens in ``labels`` are replaced with ``-100`` so that
        the cross-entropy loss skips them, and the leading
        ``decoder_start_token_id`` is removed if the tokenizer
        re-prepended it.

* :class:`DataCollatorCTCWithPadding`
      — Wav2Vec2 / Wav2Vec2-BERT / OmniASR collator.  Pads
        ``input_values`` (or ``input_features`` for Wav2Vec2-BERT) with
        the feature extractor and ``labels`` with the tokenizer.  CTC
        ignored-index in labels is ``-100``.

Both collators are dataclasses that take a ``processor`` and operate on
already-feature-extracted samples produced by ``preprocess.py``.

The collators support the optional :class:`AugmentationPipeline` from
:mod:`utils.augmentation` for feature-domain SpecAugment applied at
batch construction time (waveform augmentation runs earlier, inside
the dataset ``map`` step).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch

from .augmentation import AugmentationPipeline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: pick the right key out of an already-feature-extracted sample
# ---------------------------------------------------------------------------


def _detect_input_key(sample: Dict[str, Any]) -> str:
    """Return the key under which the model expects audio features.

    Whisper / Wav2Vec2-BERT / OmniASR use ``input_features``.
    Wav2Vec2 (XLSR-CTC) uses ``input_values``.
    """
    if "input_features" in sample:
        return "input_features"
    if "input_values" in sample:
        return "input_values"
    raise KeyError(
        "Sample does not contain 'input_features' or 'input_values'. "
        f"Available keys: {list(sample.keys())}"
    )


# ---------------------------------------------------------------------------
# Whisper / seq2seq collator
# ---------------------------------------------------------------------------


_WHISPER_ALLOWED_KEYS = frozenset(
    {"input_features", "attention_mask", "labels"}
)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Whisper-compatible padding collator.

    Parameters
    ----------
    processor
        A :class:`transformers.WhisperProcessor` instance.
    decoder_start_token_id
        The decoder's BOS token id.  Required because the Whisper
        tokenizer re-prepends a BOS token when calling ``pad`` on
        already-tokenised sequences; we strip it once so the
        ``Trainer`` (which calls ``shift_tokens_right`` internally)
        doesn't double-shift.  When ``None`` we read the value from
        ``processor.tokenizer.bos_token_id`` (Whisper) or fall back to
        ``model.config.decoder_start_token_id``.
    augmentation
        Optional :class:`AugmentationPipeline`.  When supplied the
        SpecAugment stage is applied to the padded ``input_features``
        tensor.  The waveform stage of the pipeline is *not* executed
        here — that runs earlier in the dataset ``map`` step.
    enforce_keys
        Defence-in-depth.  When ``True`` (the default), the collator
        actively scrubs any key from the returned batch that is not
        accepted by ``WhisperForConditionalGeneration.forward`` —
        most importantly ``input_ids`` and ``inputs_embeds``, which
        a misconfigured ``PeftModelForSeq2SeqLM`` wrapper would
        otherwise inject and crash the forward pass.
    """

    processor: Any
    decoder_start_token_id: Optional[int] = None
    augmentation: Optional[AugmentationPipeline] = None
    enforce_keys: bool = True

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        if self.decoder_start_token_id is None:
            tokenizer = getattr(self.processor, "tokenizer", None)
            if tokenizer is not None:
                # Whisper's BOS is re-prepended on pad; this is the id
                # we want to strip if it appears at position 0.
                self.decoder_start_token_id = tokenizer.bos_token_id

    # ------------------------------------------------------------------
    def __call__(
        self, features: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        if not features:
            raise ValueError("Empty batch passed to collator")

        # 1) Pad audio features ------------------------------------------------
        input_key = _detect_input_key(features[0])
        input_features: List[Dict[str, Any]] = [
            {input_key: f[input_key]} for f in features
        ]

        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt",
        )

        # 2) Pad labels --------------------------------------------------------
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Whisper's tokenizer re-prepends the BOS token in ``pad`` even
        # though our examples already include it.  Strip it once if
        # present so the Trainer's ``shift_tokens_right`` does not
        # double-shift the sequence.
        if (
            self.decoder_start_token_id is not None
            and labels.shape[1] > 0
            and bool((labels[:, 0] == self.decoder_start_token_id).all())
        ):
            labels = labels[:, 1:]

        batch["labels"] = labels

        # 3) Optional SpecAugment ---------------------------------------------
        if self.augmentation is not None and self.augmentation.specaugment is not None:
            try:
                batch[input_key] = self.augmentation.apply_features(
                    batch[input_key]
                )
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "SpecAugment failed on a batch (%s); using unaugmented "
                    "features.",
                    exc,
                )

        # 4) Defence-in-depth: scrub anything Whisper.forward cannot
        #    accept.  In practice this is a safety net — the canonical
        #    cause of stray ``input_ids`` keys in a Whisper batch is a
        #    PEFT misconfiguration (LoraConfig with task_type=SEQ_2_SEQ_LM)
        #    which we already fix in ``models.whisper_trainer``.  But if
        #    a user wires up a custom PEFT wrapper or subclasses this
        #    collator, the assertion below catches the mistake at the
        #    collator boundary rather than letting it surface as a
        #    confusing TypeError inside ``model.forward``.
        if self.enforce_keys:
            allowed = _WHISPER_ALLOWED_KEYS
            extra = [k for k in list(batch.keys()) if k not in allowed]
            if extra:
                # ``input_ids`` / ``inputs_embeds`` are the most common
                # offenders.  Other unknown keys are dropped silently
                # because some FE configs return e.g.
                # ``num_frames`` which the model doesn't need.
                blocking = {"input_ids", "inputs_embeds"}
                hard = [k for k in extra if k in blocking]
                if hard:
                    raise RuntimeError(
                        "Whisper collator produced disallowed keys "
                        f"{hard!r} (full batch keys: {list(batch.keys())}).  "
                        "This is almost always caused by PEFT wrapping the "
                        "Whisper model in PeftModelForSeq2SeqLM — make sure "
                        "LoraConfig is constructed WITHOUT task_type."
                    )
                for k in extra:
                    batch.pop(k, None)

        return batch


# ---------------------------------------------------------------------------
# CTC collator (Wav2Vec2 / Wav2Vec2-BERT / OmniASR)
# ---------------------------------------------------------------------------


@dataclass
class DataCollatorCTCWithPadding:
    """CTC-compatible padding collator.

    Parameters
    ----------
    processor
        A :class:`transformers.Wav2Vec2Processor`,
        :class:`transformers.SeamlessM4TFeatureExtractor` (for
        Wav2Vec2-BERT), or any HF processor that exposes
        ``feature_extractor`` and ``tokenizer``.
    padding
        Padding strategy passed to :meth:`pad`.  ``True`` /
        ``"longest"`` pads to the longest in the batch; an int pads to
        a fixed length.
    pad_to_multiple_of
        If provided, pad sequences to a multiple of this value
        (mirrors ``Trainer`` recommendations for FP16 / Tensor Cores).
    pad_to_multiple_of_labels
        Same as above but for the label tensor.
    augmentation
        Optional :class:`AugmentationPipeline` — SpecAugment is applied
        on the padded feature tensor when present.
    """

    processor: Any
    padding: Union[bool, str] = True
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None
    augmentation: Optional[AugmentationPipeline] = None

    # ------------------------------------------------------------------
    def __call__(
        self, features: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        if not features:
            raise ValueError("Empty batch passed to collator")

        # 1) Audio features ----------------------------------------------------
        input_key = _detect_input_key(features[0])
        input_features = [{input_key: f[input_key]} for f in features]

        # ``feature_extractor.pad`` accepts either ``input_values`` or
        # ``input_features`` — both spellings work because the underlying
        # ``BatchFeature`` only uses keys present in each item.
        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # 2) Labels ------------------------------------------------------------
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

        # 3) Optional SpecAugment ---------------------------------------------
        if self.augmentation is not None and self.augmentation.specaugment is not None:
            try:
                batch[input_key] = self.augmentation.apply_features(
                    batch[input_key]
                )
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "SpecAugment failed on a batch (%s); using unaugmented "
                    "features.",
                    exc,
                )

        return batch


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    "DataCollatorCTCWithPadding",
    "DataCollatorSpeechSeq2SeqWithPadding",
]
