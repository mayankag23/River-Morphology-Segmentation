"""
Training history accumulator for Module 14.

TrainingHistory records one EpochResult per epoch and provides
convenience accessors for the Training Engine and callback system.
"""

from __future__ import annotations

import logging
from typing import Iterator

from src.training.engine.contracts import EpochResult

__all__ = ["TrainingHistory"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


class TrainingHistory:
    """
    Mutable container of EpochResult objects, one per completed epoch.

    Methods are not thread-safe; access from the training loop only.
    """

    def __init__(self) -> None:
        self._epochs: list[EpochResult] = []

    def append(self, result: EpochResult) -> None:
        """Add one epoch result."""
        self._epochs.append(result)

    def __len__(self) -> int:
        return len(self._epochs)

    def __iter__(self) -> Iterator[EpochResult]:
        return iter(self._epochs)

    def __getitem__(self, index: int) -> EpochResult:
        return self._epochs[index]

    @property
    def epochs(self) -> list[EpochResult]:
        """All accumulated EpochResult objects (read-only copy)."""
        return list(self._epochs)

    @property
    def train_losses(self) -> list[float]:
        """Training losses in epoch order."""
        return [e.train_loss for e in self._epochs]

    @property
    def val_losses(self) -> list[float]:
        """Validation losses in epoch order."""
        return [e.val_loss for e in self._epochs]

    @property
    def learning_rates(self) -> list[float]:
        """Learning rates in epoch order."""
        return [e.lr for e in self._epochs]

    @property
    def best_epoch(self) -> int:
        """1-based index of the epoch with best validation loss (0 if empty)."""
        if not self._epochs:
            return 0
        best = min(self._epochs, key=lambda e: e.val_loss if e.val_loss > 0 else float("inf"))
        return best.epoch

    @property
    def best_val_loss(self) -> float:
        """Best (lowest) validation loss seen so far (inf if empty)."""
        if not self._epochs:
            return float("inf")
        return min(
            (e.val_loss for e in self._epochs if e.val_loss > 0),
            default=float("inf"),
        )

    def to_tuple(self) -> tuple[EpochResult, ...]:
        """Return an immutable tuple of all epoch results."""
        return tuple(self._epochs)

    def last(self) -> EpochResult | None:
        """Return the most recently added EpochResult, or None."""
        return self._epochs[-1] if self._epochs else None
