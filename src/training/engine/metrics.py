"""
Training metrics accumulator for Module 14.

Responsibility: accumulate per-batch training loss across one epoch and
expose the epoch mean. Does NOT compute IoU, Dice, Precision, Recall, or any
evaluation metric — those belong to Module 15.

Three tracked metrics:
    loss:        Mean cross-entropy / dice / focal loss over training batches.
    learning_rate: LR at end of epoch (retrieved from optimizer).
    epoch_time:   Wall-clock seconds for the epoch.
"""

from __future__ import annotations

import time

__all__ = ["MetricsAccumulator"]


class MetricsAccumulator:
    """
    Accumulates per-batch loss values and computes the epoch mean.

    Usage per epoch:
        acc = MetricsAccumulator()
        acc.start()
        for batch in loader:
            loss = compute_loss(...)
            acc.update(loss_value, batch_size)
        result = acc.compute()
    """

    def __init__(self) -> None:
        self._total_loss:    float = 0.0
        self._total_samples: int   = 0
        self._start_time:    float = 0.0
        self._end_time:      float = 0.0

    def start(self) -> None:
        """Start the epoch timer."""
        self._total_loss    = 0.0
        self._total_samples = 0
        self._start_time    = time.perf_counter()

    def update(self, loss: float, batch_size: int = 1) -> None:
        """
        Accumulate one batch's loss.

        Args:
            loss:       Scalar loss value for this batch.
            batch_size: Number of samples in this batch (for weighted mean).
        """
        self._total_loss    += loss * batch_size
        self._total_samples += batch_size

    def compute(self) -> dict[str, float]:
        """
        Finalise and return epoch metrics.

        Returns:
            Dict with keys "loss" and "epoch_time".
        """
        self._end_time = time.perf_counter()
        mean_loss  = (
            self._total_loss / self._total_samples
            if self._total_samples > 0
            else 0.0
        )
        return {
            "loss":       mean_loss,
            "epoch_time": self._end_time - self._start_time,
        }

    @property
    def num_samples(self) -> int:
        """Number of samples accumulated so far in this epoch."""
        return self._total_samples

    @staticmethod
    def get_lr(optimizer: object) -> float:
        """
        Extract current learning rate from an optimizer.

        Reads the first param_group's lr value. Returns 0.0 when the
        optimizer has no param groups (e.g. a mock in tests).

        Args:
            optimizer: torch.optim.Optimizer instance.

        Returns:
            Current learning rate as float.
        """
        try:
            groups = optimizer.param_groups  # type: ignore[attr-defined]
            if groups:
                return float(groups[0]["lr"])
        except (AttributeError, IndexError, KeyError):
            pass
        return 0.0
