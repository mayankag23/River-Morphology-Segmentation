"""
Structured training logger for Module 14.

TrainingLogger writes epoch summaries to the Python logging system in a
consistent, machine-parseable format. It does not depend on any external
logging framework (TensorBoard, W&B, MLflow) — those integrations belong
to future callback plugins.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from src.training.engine.contracts import EpochResult, TrainingConfig

__all__ = ["TrainingLogger"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


class TrainingLogger:
    """
    Logs training progress at epoch granularity.

    Args:
        config:       TrainingConfig for context (architecture, epochs, etc.).
        architecture: Model architecture name for log lines.
    """

    def __init__(self, config: TrainingConfig, architecture: str = "") -> None:
        self._config       = config
        self._architecture = architecture
        self._start_time   = datetime.now(timezone.utc).isoformat()

    def log_train_begin(self, total_epochs: int, num_parameters: int) -> None:
        """Log a training start banner."""
        _LOGGER.info(
            "Training started | architecture=%s | epochs=%d | "
            "params=%d | device=%s | amp=%s | seed=%d",
            self._architecture,
            total_epochs,
            num_parameters,
            self._config.device,
            self._config.mixed_precision,
            self._config.seed,
        )

    def log_epoch(self, result: EpochResult) -> None:
        """
        Log one epoch summary line.

        Format:
            Epoch N/T | train=X.XXXXXX | val=X.XXXXXX | lr=X.XXe-XX | Xs [BEST]
        """
        best_tag = " [BEST]" if result.is_best else ""
        _LOGGER.info(
            "Epoch %d/%d | train=%.6f | val=%.6f | lr=%.3e | %.1fs%s",
            result.epoch,
            self._config.epochs,
            result.train_loss,
            result.val_loss,
            result.lr,
            result.epoch_time,
            best_tag,
        )

    def log_early_stop(self, epoch: int) -> None:
        """Log early stopping event."""
        _LOGGER.info(
            "Early stopping triggered at epoch %d.", epoch
        )

    def log_train_end(self, best_epoch: int, best_metric: float) -> None:
        """Log training completion summary."""
        _LOGGER.info(
            "Training complete | best_epoch=%d | best_metric=%.6f",
            best_epoch, best_metric,
        )
