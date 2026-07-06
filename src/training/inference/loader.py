"""
Checkpoint loader for the Inference Pipeline Framework (Module 16).

CheckpointLoader reads Module 14 checkpoint files (version 1.0) and restores
model weights. It supports three selection strategies:
    "best"     -- loads checkpoint_best.pt from checkpoint_dir
    "latest"   -- loads checkpoint_latest.pt from checkpoint_dir
    "explicit" -- loads the path given in InferenceConfig.checkpoint_path

The loader is read-only: it never modifies, saves, or overwrites checkpoints.
It extracts CheckpointMetadata (epoch, losses, architecture, num_classes,
in_channels) to populate InferenceResult provenance.

Backward compatibility note
----------------------------
The checkpoint version field is read and logged. Mismatches produce a WARNING
but do not halt inference — the loader attempts to restore weights regardless.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.training.inference.contracts import CheckpointMetadata, InferenceConfig

__all__ = ["CheckpointLoader"]

_LOGGER: logging.Logger = logging.getLogger(__name__)

_LATEST_NAME: str = "checkpoint_latest.pt"
_BEST_NAME:   str = "checkpoint_best.pt"
_EXPECTED_VERSION: str = "1.0"


class CheckpointLoader:
    """
    Loads Module 14 checkpoint files and restores model state.

    Args:
        config: InferenceConfig specifying strategy, dir, and path.
    """

    def __init__(self, config: InferenceConfig) -> None:
        self._config = config

    def resolve_path(self) -> Path:
        """
        Determine which checkpoint file to load based on config strategy.

        Returns:
            Absolute Path to the checkpoint file.

        Raises:
            FileNotFoundError: The resolved path does not exist.
            ValueError:        checkpoint_strategy is not recognised.
        """
        strategy = self._config.checkpoint_strategy.lower().strip()

        if strategy == "explicit":
            path = Path(self._config.checkpoint_path).resolve()
        elif strategy == "best":
            path = Path(self._config.checkpoint_dir).resolve() / _BEST_NAME
        elif strategy == "latest":
            path = Path(self._config.checkpoint_dir).resolve() / _LATEST_NAME
        else:
            raise ValueError(
                f"CheckpointLoader: unknown strategy '{strategy}'. "
                f"Choose 'best', 'latest', or 'explicit'."
            )

        if not path.exists():
            raise FileNotFoundError(
                f"CheckpointLoader: checkpoint not found at {path} "
                f"(strategy='{strategy}')."
            )

        _LOGGER.info("CheckpointLoader: resolved checkpoint -> %s", path)
        return path

    def load(self, path: Path) -> dict:
        """
        Load and return the raw checkpoint payload dict.

        Args:
            path: Absolute path to the .pt checkpoint file.

        Returns:
            Raw checkpoint dict (keys: version, epoch, model_state, …).

        Raises:
            FileNotFoundError: File does not exist.
            RuntimeError:      torch.load fails.
        """
        try:
            import torch
        except ImportError as exc:
            raise ImportError("torch is required to load checkpoints.") from exc

        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise RuntimeError(
                f"CheckpointLoader: failed to load {path}: {exc}"
            ) from exc

        version = payload.get("version", "unknown")
        if version != _EXPECTED_VERSION:
            _LOGGER.warning(
                "CheckpointLoader: version mismatch: file=%s, expected=%s. "
                "Attempting to load anyway.",
                version, _EXPECTED_VERSION,
            )
        return payload

    def restore_model(self, model: Any, payload: dict) -> Any:
        """
        Load model weights from checkpoint payload.

        Args:
            model:   torch.nn.Module to restore weights into.
            payload: Raw checkpoint dict.

        Returns:
            The model with restored weights (in-place + returned for chaining).
        """
        state = payload.get("model_state", {})
        if not state:
            _LOGGER.warning(
                "CheckpointLoader.restore_model: payload has no 'model_state'."
            )
            return model

        try:
            model.load_state_dict(state, strict=True)
            _LOGGER.info("CheckpointLoader: model weights restored (strict=True).")
        except RuntimeError as exc:
            _LOGGER.warning(
                "CheckpointLoader: strict restore failed (%s); "
                "retrying with strict=False.",
                exc,
            )
            model.load_state_dict(state, strict=False)
            _LOGGER.warning("CheckpointLoader: model weights restored (strict=False).")

        return model

    @staticmethod
    def extract_metadata(path: Path, payload: dict) -> CheckpointMetadata:
        """
        Build CheckpointMetadata from checkpoint payload.

        Args:
            path:    Absolute path of the checkpoint file.
            payload: Raw checkpoint dict.

        Returns:
            Frozen CheckpointMetadata.
        """
        return CheckpointMetadata(
            checkpoint_path    = str(path),
            checkpoint_version = str(payload.get("version",      "unknown")),
            epoch              = int(payload.get("epoch",         0)),
            train_loss         = float(payload.get("train_loss",  0.0)),
            val_loss           = float(payload.get("val_loss",    0.0)),
            architecture       = str(payload.get("architecture",  "unknown")),
            num_classes        = int(payload.get("num_classes",   0)),
            in_channels        = int(payload.get("in_channels",   0)),
        )
