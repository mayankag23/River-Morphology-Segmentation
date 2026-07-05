"""
Checkpoint management for the Training Engine Framework (Module 14).

CheckpointManager saves and loads versioned checkpoint files.

Checkpoint schema (version 1.0)
--------------------------------
{
    "version":          "1.0",
    "epoch":            int,
    "train_loss":       float,
    "val_loss":         float,
    "model_state":      OrderedDict,
    "optimizer_state":  dict,
    "scheduler_state":  dict | None,
    "scaler_state":     dict | None,
    "rng_state":        dict,
    "architecture":     str,
    "num_classes":      int,
    "in_channels":      int,
}

File naming convention:
    checkpoint_latest.pt   — always the most recent epoch
    checkpoint_best.pt     — best monitored metric
    checkpoint_epoch_N.pt  — optional per-epoch snapshots
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.training.engine.contracts import CheckpointConfig, EpochResult
from src.training.engine.seed import SeedManager

__all__ = ["CheckpointManager"]

_LOGGER: logging.Logger = logging.getLogger(__name__)

_CHECKPOINT_VERSION: str = "1.0"
_LATEST_NAME:        str = "checkpoint_latest.pt"
_BEST_NAME:          str = "checkpoint_best.pt"


class CheckpointManager:
    """
    Saves and loads versioned training checkpoints.

    Args:
        config:  CheckpointConfig supplying directory and save flags.
        context: Shared context dict carrying model, optimizer, scheduler,
                 scaler references (populated by TrainingEngine).
    """

    def __init__(self, config: CheckpointConfig) -> None:
        self._config    = config
        self._dir       = Path(config.checkpoint_dir).resolve()
        self._best_path:   Path | None = None
        self._latest_path: Path | None = None

    @property
    def best_path(self) -> Path | None:
        """Path to the best checkpoint file, or None if none saved yet."""
        return self._best_path

    @property
    def latest_path(self) -> Path | None:
        """Path to the latest checkpoint file, or None if none saved yet."""
        return self._latest_path

    def save(
        self,
        context: dict,
        epoch:   int,
        result:  EpochResult,
        is_best: bool,
    ) -> None:
        """
        Save latest and/or best checkpoints.

        Args:
            context: Shared engine context with model/optimizer/scheduler/scaler.
            epoch:   Current epoch number.
            result:  EpochResult for this epoch.
            is_best: True if this epoch achieved the best monitored metric.
        """
        try:
            import torch
        except ImportError:
            _LOGGER.warning("CheckpointManager.save: torch not available; skipping.")
            return

        self._dir.mkdir(parents=True, exist_ok=True)

        model       = context.get("model")
        optimizer   = context.get("optimizer")
        scheduler   = context.get("scheduler")
        scaler      = context.get("scaler")
        model_result = context.get("model_result")

        payload: dict = {
            "version":         _CHECKPOINT_VERSION,
            "epoch":           epoch,
            "train_loss":      result.train_loss,
            "val_loss":        result.val_loss,
            "model_state":     model.state_dict() if model is not None else {},
            "optimizer_state": optimizer.state_dict() if optimizer is not None else {},
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state":    scaler.state_dict()    if scaler    is not None else None,
            "rng_state":       SeedManager.get_rng_state(),
            "architecture":    getattr(model_result, "architecture", "unknown") if model_result else "unknown",
            "num_classes":     getattr(model_result, "num_classes",  0)         if model_result else 0,
            "in_channels":     getattr(model_result, "in_channels",  0)         if model_result else 0,
        }

        if self._config.save_latest:
            latest = self._dir / _LATEST_NAME
            torch.save(payload, latest)
            self._latest_path = latest
            _LOGGER.debug("CheckpointManager: saved latest -> %s", latest.name)

        if self._config.save_best and is_best:
            best = self._dir / _BEST_NAME
            torch.save(payload, best)
            self._best_path = best
            _LOGGER.info("CheckpointManager: saved best -> %s (epoch %d)", best.name, epoch)

    def load(self, path: Path | str) -> dict:
        """
        Load a checkpoint from disk.

        Args:
            path: Path to the checkpoint .pt file.

        Returns:
            Raw checkpoint dict.

        Raises:
            FileNotFoundError: File does not exist.
            ValueError:        Checkpoint version is incompatible.
        """
        try:
            import torch
        except ImportError as exc:
            raise ImportError("torch is required to load checkpoints.") from exc

        path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        payload = torch.load(path, map_location="cpu", weights_only=False)
        version = payload.get("version", "unknown")
        if version != _CHECKPOINT_VERSION:
            _LOGGER.warning(
                "CheckpointManager: checkpoint version mismatch: "
                "file=%s, expected=%s. Loading anyway.",
                version, _CHECKPOINT_VERSION,
            )
        _LOGGER.info("CheckpointManager: loaded checkpoint from %s (epoch %d).",
                     path.name, payload.get("epoch", -1))
        return payload

    def restore(self, path: Path | str, context: dict) -> int:
        """
        Restore model, optimizer, scheduler, scaler and RNG state from a checkpoint.

        Args:
            path:    Checkpoint file path.
            context: Shared engine context dict (will be mutated).

        Returns:
            Epoch number of the restored checkpoint.
        """
        payload = self.load(path)

        model     = context.get("model")
        optimizer = context.get("optimizer")
        scheduler = context.get("scheduler")
        scaler    = context.get("scaler")

        if model is not None and "model_state" in payload:
            model.load_state_dict(payload["model_state"])
            _LOGGER.info("CheckpointManager: model state restored.")

        if optimizer is not None and payload.get("optimizer_state"):
            optimizer.load_state_dict(payload["optimizer_state"])
            _LOGGER.info("CheckpointManager: optimizer state restored.")

        if scheduler is not None and payload.get("scheduler_state"):
            try:
                scheduler.load_state_dict(payload["scheduler_state"])
                _LOGGER.info("CheckpointManager: scheduler state restored.")
            except Exception as exc:
                _LOGGER.warning("CheckpointManager: scheduler restore failed: %s", exc)

        if scaler is not None and payload.get("scaler_state"):
            try:
                scaler.load_state_dict(payload["scaler_state"])
                _LOGGER.info("CheckpointManager: AMP scaler state restored.")
            except Exception as exc:
                _LOGGER.warning("CheckpointManager: scaler restore failed: %s", exc)

        if "rng_state" in payload:
            SeedManager.restore_rng_state(payload["rng_state"])

        return int(payload.get("epoch", 0))
