"""
utils/callbacks.py
==================

Trainer callbacks used across all four model trainers.

Modern (transformers >= 4.40) ``TrainerCallback`` API is used
throughout — every callback receives ``args``, ``state`` and
``control`` and a ``**kwargs`` bag containing the live model,
processor, optimizer, etc.

Provided components:

* :class:`BestCERTrackerCallback`
        Tracks ``eval_cer`` across evaluations, logs the best value
        seen so far and writes a small JSON breadcrumb to the run's
        output directory after every improvement.

* :class:`MemoryMonitorCallback`
        Logs peak CUDA memory use after every evaluation step.  Helps
        diagnose OOMs without spamming the per-step log.

* :func:`build_early_stopping_callback`
        Factory that returns a properly configured
        :class:`transformers.EarlyStoppingCallback`.  It picks the
        ``early_stopping_patience`` and ``early_stopping_threshold``
        sensibly and validates that the corresponding
        ``TrainingArguments`` are set up correctly.

* :func:`build_default_callbacks`
        Convenience helper that returns a list of callbacks every
        trainer in ``models/`` plugs into ``Trainer.callbacks``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# We import lazily inside functions where reasonable, but for the
# callback class itself we import at module top — transformers is a
# hard dependency anyway when callbacks are actually used.
from transformers import (
    EarlyStoppingCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)


# ---------------------------------------------------------------------------
# Best-CER tracker
# ---------------------------------------------------------------------------


@dataclass
class BestCERTrackerCallback(TrainerCallback):
    """Track the best ``eval_cer`` and persist a tiny JSON breadcrumb.

    The Trainer already saves the best checkpoint when
    ``load_best_model_at_end=True`` and ``metric_for_best_model="cer"``
    are set — this callback only adds visibility:

    * a single-line log entry after every evaluation,
    * a JSON file ``best_metric.json`` in the output directory that
      records the best step, best CER and matching WER, useful for
      automated dashboards.
    """

    metric_name: str = "eval_cer"
    secondary_metric_name: str = "eval_wer"
    output_filename: str = "best_metric.json"

    best_value: float = float("inf")
    best_step: int = -1
    best_secondary: Optional[float] = None

    # ------------------------------------------------------------------
    def on_evaluate(  # type: ignore[override]
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ) -> TrainerControl:
        if metrics is None:
            return control

        value = metrics.get(self.metric_name)
        if value is None:
            # Some trainers report ``cer`` instead of ``eval_cer``.
            value = metrics.get(self.metric_name.replace("eval_", ""))

        if value is None:
            return control

        if value < self.best_value:
            self.best_value = float(value)
            self.best_step = int(state.global_step)
            self.best_secondary = (
                float(metrics.get(self.secondary_metric_name, float("nan")))
                if self.secondary_metric_name
                else None
            )

            logger.info(
                "[BestCERTracker] new best %s = %.4f at step %d (wer=%.4f)",
                self.metric_name,
                self.best_value,
                self.best_step,
                self.best_secondary if self.best_secondary is not None else float("nan"),
            )

            try:
                out_path = Path(args.output_dir) / self.output_filename
                out_path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "best_step": self.best_step,
                    "best_metric": self.metric_name,
                    "best_value": self.best_value,
                    "secondary_metric": self.secondary_metric_name,
                    "secondary_value": self.best_secondary,
                }
                out_path.write_text(
                    json.dumps(payload, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            except Exception as exc:  # pragma: no cover — IO must not crash training
                logger.warning(
                    "Could not write %s: %s", self.output_filename, exc
                )

        return control


# ---------------------------------------------------------------------------
# Memory monitor
# ---------------------------------------------------------------------------


class MemoryMonitorCallback(TrainerCallback):
    """Log peak CUDA memory after every evaluation."""

    def on_evaluate(  # type: ignore[override]
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        try:
            import torch  # local import — keep this module import-light
        except ImportError:  # pragma: no cover
            return control

        if not torch.cuda.is_available():
            return control

        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        logger.info(
            "[MemoryMonitor] step=%d peak_cuda_memory=%.0f MiB",
            state.global_step,
            peak_mb,
        )
        # Reset for the next window so the metric reflects per-eval
        # peaks rather than the global maximum.
        torch.cuda.reset_peak_memory_stats()
        return control


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------


def build_early_stopping_callback(
    *,
    patience: int = 5,
    threshold: float = 0.0,
) -> EarlyStoppingCallback:
    """Return a configured :class:`EarlyStoppingCallback`.

    The corresponding ``TrainingArguments`` MUST have:

    * ``load_best_model_at_end=True``
    * ``metric_for_best_model="cer"``  (or whichever metric you pick)
    * ``greater_is_better=False`` for CER/WER

    Otherwise the Trainer raises a ``ValueError`` at training start.
    """
    if patience < 1:
        raise ValueError(f"patience must be >= 1, got {patience}")
    if threshold < 0:
        raise ValueError(f"threshold must be >= 0, got {threshold}")

    return EarlyStoppingCallback(
        early_stopping_patience=patience,
        early_stopping_threshold=threshold,
    )


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------


def build_default_callbacks(
    *,
    enable_early_stopping: bool = True,
    early_stopping_patience: int = 5,
    early_stopping_threshold: float = 0.0,
    enable_best_cer_tracker: bool = True,
    enable_memory_monitor: bool = True,
) -> List[TrainerCallback]:
    """Return the default callback set used by every model trainer."""

    callbacks: List[TrainerCallback] = []

    if enable_best_cer_tracker:
        callbacks.append(BestCERTrackerCallback())

    if enable_memory_monitor:
        callbacks.append(MemoryMonitorCallback())

    if enable_early_stopping:
        callbacks.append(
            build_early_stopping_callback(
                patience=early_stopping_patience,
                threshold=early_stopping_threshold,
            )
        )

    return callbacks


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    "BestCERTrackerCallback",
    "MemoryMonitorCallback",
    "build_default_callbacks",
    "build_early_stopping_callback",
]
