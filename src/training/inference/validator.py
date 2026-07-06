"""
Inference validation for Module 16.

InferenceValidator checks checkpoint compatibility, tensor dimensions,
NaN/Inf values, class IDs, and device availability before inference begins.
Never raises; accumulates issues for the caller to decide.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.training.inference.contracts import CheckpointMetadata, InferenceConfig

__all__ = ["InferenceValidator", "InferenceValidationResult"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


class InferenceValidationResult:
    """Result of one validation pass."""

    def __init__(self, issues: list[str]) -> None:
        self._issues = list(issues)

    @property
    def is_valid(self) -> bool:
        return len(self._issues) == 0

    @property
    def issues(self) -> list[str]:
        return list(self._issues)


class InferenceValidator:
    """Validates inference configuration and per-batch tensor outputs."""

    def validate_config(
        self,
        config:   InferenceConfig,
        ckpt_meta: CheckpointMetadata | None = None,
    ) -> InferenceValidationResult:
        """
        Pre-flight configuration validation.

        Args:
            config:    InferenceConfig.
            ckpt_meta: CheckpointMetadata if already loaded, else None.

        Returns:
            InferenceValidationResult.
        """
        issues: list[str] = []

        if config.batch_size < 1:
            issues.append(f"batch_size must be >= 1, got {config.batch_size}.")

        if config.probability_mode not in ("softmax", "sigmoid"):
            issues.append(
                f"probability_mode must be 'softmax' or 'sigmoid', "
                f"got '{config.probability_mode}'."
            )

        if config.confidence_strategy not in ("max_probability", "entropy"):
            issues.append(
                f"confidence_strategy must be 'max_probability' or 'entropy', "
                f"got '{config.confidence_strategy}'."
            )

        strategy = config.checkpoint_strategy.lower()
        if strategy == "explicit" and not config.checkpoint_path:
            issues.append(
                "checkpoint_strategy is 'explicit' but checkpoint_path is empty."
            )

        if config.device.startswith("cuda"):
            try:
                import torch
                if not torch.cuda.is_available():
                    issues.append(
                        f"Device '{config.device}' requested but CUDA unavailable; "
                        "will fall back to CPU."
                    )
            except ImportError:
                issues.append("torch is not installed; cannot validate CUDA.")

        if ckpt_meta is not None and ckpt_meta.num_classes < 1:
            issues.append(
                f"Checkpoint num_classes={ckpt_meta.num_classes} is invalid."
            )

        return InferenceValidationResult(issues)

    @staticmethod
    def validate_prediction(
        predicted_mask: np.ndarray,
        probabilities:  np.ndarray,
        confidence:     np.ndarray,
        num_classes:    int,
    ) -> InferenceValidationResult:
        """
        Validate a single-sample prediction output.

        Args:
            predicted_mask: (H, W) uint8.
            probabilities:  (C, H, W) float32.
            confidence:     (H, W) float32.
            num_classes:    Expected number of classes.

        Returns:
            InferenceValidationResult.
        """
        issues: list[str] = []

        if predicted_mask.ndim != 2:
            issues.append(
                f"predicted_mask must be 2-D (H, W); got {predicted_mask.shape}."
            )

        if probabilities.ndim != 3:
            issues.append(
                f"probabilities must be 3-D (C, H, W); got {probabilities.shape}."
            )
        elif probabilities.shape[0] != num_classes:
            issues.append(
                f"probabilities.shape[0]={probabilities.shape[0]} != "
                f"num_classes={num_classes}."
            )

        if np.any(np.isnan(probabilities)):
            issues.append("probabilities contains NaN values.")
        if np.any(np.isinf(probabilities)):
            issues.append("probabilities contains Inf values.")

        if np.any(np.isnan(confidence)):
            issues.append("confidence contains NaN values.")

        # Predicted class IDs must be in [0, num_classes).
        if predicted_mask.ndim == 2 and num_classes > 0:
            unique_ids  = set(int(v) for v in np.unique(predicted_mask))
            invalid_ids = {i for i in unique_ids if i < 0 or i >= num_classes}
            if invalid_ids:
                issues.append(
                    f"predicted_mask contains invalid class IDs: "
                    f"{sorted(invalid_ids)} (expected [0, {num_classes}))."
                )

        return InferenceValidationResult(issues)
