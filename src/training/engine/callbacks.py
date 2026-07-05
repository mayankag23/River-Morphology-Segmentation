"""
Callback system for the Training Engine Framework (Module 14).

The Callback pattern decouples cross-cutting concerns (checkpointing,
logging, early stopping) from the core training loop in Trainer.
TrainingEngine passes a CallbackList to Trainer; Trainer calls hook
methods at precise points in the epoch loop.

Hook execution order per epoch:
    on_epoch_begin(epoch)
    for batch in loader:
        on_batch_begin(batch_idx)
        ... forward / backward ...
        on_batch_end(batch_idx, loss)
    on_epoch_end(epoch, epoch_result)
on_train_end(result)

All hooks receive the Trainer's internal context dict so callbacks can
read and write shared state (e.g. EarlyStopping sets "stop_training").
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from src.training.engine.contracts import EpochResult

__all__ = [
    "Callback",
    "CallbackList",
    "CheckpointCallback",
    "LoggingCallback",
    "EarlyStoppingCallback",
]

_LOGGER: logging.Logger = logging.getLogger(__name__)


# ==============================================================================
# Callback ABC
# ==============================================================================

class Callback(ABC):
    """
    Abstract base for all training callbacks.

    All hooks have default no-op implementations so subclasses only need to
    override the hooks they care about.
    """

    def on_train_begin(self, context: dict) -> None:
        """Called once before the first epoch."""

    def on_train_end(self, context: dict) -> None:
        """Called once after the last epoch (or early stopping)."""

    def on_epoch_begin(self, epoch: int, context: dict) -> None:
        """Called at the start of each epoch."""

    def on_epoch_end(self, epoch: int, result: EpochResult, context: dict) -> None:
        """Called at the end of each epoch with the epoch summary."""

    def on_batch_begin(self, batch_idx: int, context: dict) -> None:
        """Called before each training batch."""

    def on_batch_end(self, batch_idx: int, loss: float, context: dict) -> None:
        """Called after each training batch with the batch loss."""


# ==============================================================================
# CallbackList
# ==============================================================================

class CallbackList:
    """
    Ordered list of Callback instances.

    Calls each hook on all registered callbacks in registration order.
    """

    def __init__(self, callbacks: list[Callback] | None = None) -> None:
        self._callbacks: list[Callback] = list(callbacks or [])

    def append(self, callback: Callback) -> None:
        self._callbacks.append(callback)

    def __len__(self) -> int:
        return len(self._callbacks)

    def on_train_begin(self, context: dict) -> None:
        for cb in self._callbacks:
            cb.on_train_begin(context)

    def on_train_end(self, context: dict) -> None:
        for cb in self._callbacks:
            cb.on_train_end(context)

    def on_epoch_begin(self, epoch: int, context: dict) -> None:
        for cb in self._callbacks:
            cb.on_epoch_begin(epoch, context)

    def on_epoch_end(self, epoch: int, result: EpochResult, context: dict) -> None:
        for cb in self._callbacks:
            cb.on_epoch_end(epoch, result, context)

    def on_batch_begin(self, batch_idx: int, context: dict) -> None:
        for cb in self._callbacks:
            cb.on_batch_begin(batch_idx, context)

    def on_batch_end(self, batch_idx: int, loss: float, context: dict) -> None:
        for cb in self._callbacks:
            cb.on_batch_end(batch_idx, loss, context)


# ==============================================================================
# CheckpointCallback
# ==============================================================================

class CheckpointCallback(Callback):
    """
    Saves best and/or latest checkpoints at the end of each epoch.

    Delegates actual file I/O to CheckpointManager (checkpoint.py).

    Args:
        checkpoint_manager: CheckpointManager instance.
        mode:               "min" (lower metric = better) or "max".
        metric:             Key in context to monitor (e.g. "val_loss").
    """

    def __init__(
        self,
        checkpoint_manager: Any,
        mode:   str = "min",
        metric: str = "val_loss",
    ) -> None:
        self._manager = checkpoint_manager
        self._mode    = mode.lower().strip()
        self._metric  = metric
        self._best:   float = float("inf") if self._mode == "min" else float("-inf")

    def on_epoch_end(self, epoch: int, result: EpochResult, context: dict) -> None:
        current = result.val_loss if self._metric == "val_loss" else result.train_loss
        is_best = (
            current < self._best if self._mode == "min" else current > self._best
        )
        if is_best:
            self._best = current

        self._manager.save(
            context = context,
            epoch   = epoch,
            result  = result,
            is_best = is_best,
        )


# ==============================================================================
# LoggingCallback
# ==============================================================================

class LoggingCallback(Callback):
    """
    Logs epoch-level metrics to the Python logging system.

    Uses TrainingLogger when available in context; falls back to _LOGGER.

    Args:
        log_every: Log every N epochs. Default 1 (every epoch).
    """

    def __init__(self, log_every: int = 1) -> None:
        self._log_every = max(1, log_every)

    def on_epoch_end(self, epoch: int, result: EpochResult, context: dict) -> None:
        if epoch % self._log_every != 0:
            return
        logger = context.get("training_logger")
        if logger is not None:
            logger.log_epoch(result)
        else:
            _LOGGER.info(
                "Epoch %d | train_loss=%.6f | val_loss=%.6f | lr=%.2e | "
                "time=%.1fs | best=%s",
                result.epoch,
                result.train_loss,
                result.val_loss,
                result.lr,
                result.epoch_time,
                result.is_best,
            )


# ==============================================================================
# EarlyStoppingCallback
# ==============================================================================

class EarlyStoppingCallback(Callback):
    """
    Halts training when the monitored metric has not improved for
    `patience` consecutive epochs.

    Sets context["stop_training"] = True when triggered. The Trainer
    checks this flag after on_epoch_end.

    Args:
        patience:  Number of epochs with no improvement before stopping.
                   0 disables early stopping.
        mode:      "min" (lower = better) or "max" (higher = better).
        metric:    Metric to monitor. Only "val_loss" and "train_loss" are
                   supported at Module 14 level (no IoU/Dice until Module 15).
        min_delta: Minimum change to count as an improvement.
    """

    def __init__(
        self,
        patience:  int   = 10,
        mode:      str   = "min",
        metric:    str   = "val_loss",
        min_delta: float = 0.0,
    ) -> None:
        self._patience   = max(0, patience)
        self._mode       = mode.lower().strip()
        self._metric     = metric
        self._min_delta  = float(min_delta)
        self._best:      float = float("inf") if self._mode == "min" else float("-inf")
        self._wait:      int   = 0
        self._triggered: bool  = False

    @property
    def triggered(self) -> bool:
        """True if early stopping has fired."""
        return self._triggered

    @property
    def wait(self) -> int:
        """Current number of epochs since the last improvement."""
        return self._wait

    def on_epoch_end(self, epoch: int, result: EpochResult, context: dict) -> None:
        if self._patience == 0:
            return

        current = result.val_loss if self._metric == "val_loss" else result.train_loss

        if self._mode == "min":
            improved = current < (self._best - self._min_delta)
        else:
            improved = current > (self._best + self._min_delta)

        if improved:
            self._best = current
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self._patience:
                _LOGGER.info(
                    "EarlyStoppingCallback: stopping at epoch %d "
                    "(no improvement for %d epochs).",
                    epoch, self._patience,
                )
                self._triggered = True
                context["stop_training"] = True
