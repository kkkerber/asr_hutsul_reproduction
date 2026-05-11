
from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from config import DIALECT_SUBSTITUTION_PAIRS
from utils.text_normalization import build_default_normalizer

logger = logging.getLogger(__name__)


try:
    import evaluate

    _HAS_EVALUATE = True
except ImportError:
    _HAS_EVALUATE = False
    evaluate = None

try:
    import jiwer

    _HAS_JIWER = True
except ImportError:
    _HAS_JIWER = False
    jiwer = None


class MetricCalculator:

    def __init__(self) -> None:
        self._wer_metric = None
        self._cer_metric = None

        if _HAS_EVALUATE:
            try:
                self._wer_metric = evaluate.load("wer")
                self._cer_metric = evaluate.load("cer")
            except Exception as exc:
                logger.warning(
                    "evaluate.load failed (%s) — falling back to jiwer", exc
                )

        if (self._wer_metric is None or self._cer_metric is None) and not _HAS_JIWER:
            raise ImportError(
                "Neither 'evaluate' nor 'jiwer' is usable. Install at least "
                "one (`pip install evaluate jiwer`)."
            )

    def compute_wer(
        self,
        predictions: Sequence[str],
        references: Sequence[str],
    ) -> float:
        if len(predictions) != len(references):
            raise ValueError(
                f"predictions and references differ in length: "
                f"{len(predictions)} vs {len(references)}"
            )

        preds, refs = _drop_empty_pairs(predictions, references)
        if not refs:
            return 0.0

        if self._wer_metric is not None:
            try:
                return float(
                    self._wer_metric.compute(
                        predictions=preds, references=refs
                    )
                )
            except Exception as exc:
                logger.warning("evaluate WER failed (%s) — using jiwer", exc)

        return float(jiwer.wer(refs, preds))

    def compute_cer(
        self,
        predictions: Sequence[str],
        references: Sequence[str],
    ) -> float:
        if len(predictions) != len(references):
            raise ValueError(
                f"predictions and references differ in length: "
                f"{len(predictions)} vs {len(references)}"
            )
        preds, refs = _drop_empty_pairs(predictions, references)
        if not refs:
            return 0.0

        if self._cer_metric is not None:
            try:
                return float(
                    self._cer_metric.compute(
                        predictions=preds, references=refs
                    )
                )
            except Exception as exc:
                logger.warning("evaluate CER failed (%s) — using jiwer", exc)

        return float(jiwer.cer(refs, preds))

    def compute_both(
        self,
        predictions: Sequence[str],
        references: Sequence[str],
    ) -> Dict[str, float]:
        return {
            "wer": self.compute_wer(predictions, references),
            "cer": self.compute_cer(predictions, references),
        }


def _drop_empty_pairs(
    predictions: Sequence[str],
    references: Sequence[str],
) -> Tuple[List[str], List[str]]:
    preds: List[str] = []
    refs: List[str] = []
    for p, r in zip(predictions, references):
        if r is None or len(r.strip()) == 0:
            continue
        preds.append(p if p is not None else "")
        refs.append(r)
    return preds, refs


def compute_wer(
    predictions: Sequence[str],
    references: Sequence[str],
) -> float:
    return MetricCalculator().compute_wer(predictions, references)


def compute_cer(
    predictions: Sequence[str],
    references: Sequence[str],
) -> float:
    return MetricCalculator().compute_cer(predictions, references)


@dataclass
class ErrorAnalysisResult:
    total_words: int = 0
    total_chars: int = 0

    substitutions: Counter = field(default_factory=Counter)
    insertions: Counter = field(default_factory=Counter)
    deletions: Counter = field(default_factory=Counter)

    dialect_pairs: Dict[str, int] = field(default_factory=dict)

    def to_dict(self, top_k: int = 30) -> Dict[str, Any]:
        return {
            "total_words": self.total_words,
            "total_chars": self.total_chars,
            "top_substitutions": [
                {"ref": r, "hyp": h, "count": int(c)}
                for (r, h), c in self.substitutions.most_common(top_k)
            ],
            "top_insertions": [
                {"hyp": h, "count": int(c)}
                for h, c in self.insertions.most_common(top_k)
            ],
            "top_deletions": [
                {"ref": r, "count": int(c)}
                for r, c in self.deletions.most_common(top_k)
            ],
            "dialect_pairs": dict(self.dialect_pairs),
        }


    def to_json(
        self, path: Union[str, Path], top_k: int = 30, indent: int = 2
    ) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                self.to_dict(top_k=top_k),
                indent=indent,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return path


def _levenshtein_alignment(ref: str, hyp: str) -> List[Tuple[str, str, str]]:
    n = len(ref)
    m = len(hyp)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i in range(n + 1):
        dp[i, 0] = i
    for j in range(m + 1):
        dp[0, j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i, j] = dp[i - 1, j - 1]
            else:
                dp[i, j] = 1 + min(
                    dp[i - 1, j - 1],
                    dp[i - 1, j],
                    dp[i, j - 1],
                )

    ops: List[Tuple[str, str, str]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            ops.append(("match", ref[i - 1], hyp[j - 1]))
            i -= 1
            j -= 1
        elif (
            i > 0
            and j > 0
            and dp[i, j] == dp[i - 1, j - 1] + 1
        ):
            ops.append(("sub", ref[i - 1], hyp[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i, j] == dp[i - 1, j] + 1:
            ops.append(("del", ref[i - 1], ""))
            i -= 1
        else:
            ops.append(("ins", "", hyp[j - 1]))
            j -= 1

    ops.reverse()
    return ops


def analyze_substitutions(
    predictions: Sequence[str],
    references: Sequence[str],
    *,
    dialect_pairs: Sequence[Tuple[str, str]] = DIALECT_SUBSTITUTION_PAIRS,
) -> ErrorAnalysisResult:

    if len(predictions) != len(references):
        raise ValueError(
            f"predictions and references differ in length: "
            f"{len(predictions)} vs {len(references)}"
        )

    result = ErrorAnalysisResult()
    pair_keys = {f"{r}->{h}": (r, h) for (r, h) in dialect_pairs}
    for key in pair_keys:
        result.dialect_pairs[key] = 0

    for ref, hyp in zip(references, predictions):
        if ref is None:
            continue
        ref = ref or ""
        hyp = hyp or ""

        result.total_words += len(ref.split())
        result.total_chars += len(ref)

        ops = _levenshtein_alignment(ref, hyp)
        for op, r, h in ops:
            if op == "sub":
                result.substitutions[(r, h)] += 1
                key = f"{r}->{h}"
                if key in pair_keys:
                    result.dialect_pairs[key] += 1
            elif op == "ins":
                result.insertions[h] += 1
            elif op == "del":
                result.deletions[r] += 1

    return result


def build_compute_metrics_seq2seq(
    processor: Any,
    *,
    metric_calculator: Optional[MetricCalculator] = None,
    normalize: bool = True,
) -> Callable[[Any], Dict[str, float]]:

    metric_calculator = metric_calculator or MetricCalculator()
    normalizer = build_default_normalizer() if normalize else None

    tokenizer = getattr(processor, "tokenizer", processor)
    pad_token_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else 0
    )

    def compute_metrics(eval_pred: Any) -> Dict[str, float]:

        preds = (
            eval_pred.predictions
            if hasattr(eval_pred, "predictions")
            else eval_pred[0]
        )
        labels = (
            eval_pred.label_ids
            if hasattr(eval_pred, "label_ids")
            else eval_pred[1]
        )

        if isinstance(preds, tuple):
            preds = preds[0]

        if preds.ndim == 3:
            preds = np.argmax(preds, axis=-1)

        labels = np.where(labels != -100, labels, pad_token_id)

        pred_str = tokenizer.batch_decode(preds, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)

        if normalizer is not None:
            pred_str = [normalizer(s) for s in pred_str]
            label_str = [normalizer(s) for s in label_str]

        return metric_calculator.compute_both(pred_str, label_str)

    return compute_metrics


def build_compute_metrics_ctc(
    processor: Any,
    *,
    metric_calculator: Optional[MetricCalculator] = None,
    normalize: bool = True,
) -> Callable[[Any], Dict[str, float]]:

    metric_calculator = metric_calculator or MetricCalculator()
    normalizer = build_default_normalizer() if normalize else None

    tokenizer = getattr(processor, "tokenizer", processor)
    pad_token_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else 0
    )

    def compute_metrics(eval_pred: Any) -> Dict[str, float]:
        preds = (
            eval_pred.predictions
            if hasattr(eval_pred, "predictions")
            else eval_pred[0]
        )
        labels = (
            eval_pred.label_ids
            if hasattr(eval_pred, "label_ids")
            else eval_pred[1]
        )

        if isinstance(preds, tuple):
            preds = preds[0]

        if preds.ndim == 3:
            pred_ids = np.argmax(preds, axis=-1)
        else:
            pred_ids = preds

        labels = np.where(labels != -100, labels, pad_token_id)

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

        try:
            label_str = tokenizer.batch_decode(
                labels, skip_special_tokens=True, group_tokens=False
            )
        except TypeError:
            label_str = tokenizer.batch_decode(
                labels, skip_special_tokens=True
            )

        if normalizer is not None:
            pred_str = [normalizer(s) for s in pred_str]
            label_str = [normalizer(s) for s in label_str]

        return metric_calculator.compute_both(pred_str, label_str)

    return compute_metrics


__all__ = [
    "ErrorAnalysisResult",
    "MetricCalculator",
    "analyze_substitutions",
    "build_compute_metrics_ctc",
    "build_compute_metrics_seq2seq",
    "compute_cer",
    "compute_wer",
]
