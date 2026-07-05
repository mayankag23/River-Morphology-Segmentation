"""
Evaluation input validator for Module 15.

EvaluationValidator checks prediction and target arrays before they are
fed into the confusion matrix accumulator. It also validates the top-level
configuration and model/data compatibility. Never raises; accumulates issues.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.training.evaluation.contracts import EvaluationConfig

__all__ = ["EvaluationValidator", "EvaluationValidationResult"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


class EvaluationValidationResult:
    """Result of one validation check."""

    def __init__(self, issues: list[str]) -> None:
        self._issues = list(issues)

    @property
    def is_valid(self) -> bool:
        return len(self._issues) == 0

    @property
    def issues(self) -> list[str]:
        return list(self._issues)


class EvaluationValidator:
    """
    Validates evaluation inputs at batch level and configuration level.
    """

    def validate_batch(
        self,
        predictions:  np.ndarray,
        targets:      np.ndarray,
        num_classes:  int,
        ignore_index: int,
    ) -> EvaluationValidationResult:
        """
        Validate one batch of predictions and targets.

        Args:
            predictions:  Integer array of predicted class IDs.
            targets:      Integer array of ground-truth class IDs.
            num_classes:  Expected number of valid class IDs.
            ignore_index: Pixel value excluded from evaluation.

        Returns:
            EvaluationValidationResult with any detected issues.
        """
        issues: list[str] = []

        # Shape match.
        if predictions.shape != targets.shape:
            issues.append(
                f"predictions.shape={predictions.shape} != "
                f"targets.shape={targets.shape}"
            )

        # NaN check (cast to float first for integer arrays).
        if np.any(np.isnan(predictions.astype(np.float32))):
            issues.append("predictions contains NaN values")
        if np.any(np.isnan(targets.astype(np.float32))):
            issues.append("targets contains NaN values")

        # Inf check.
        if np.any(np.isinf(predictions.astype(np.float32))):
            issues.append("predictions contains Inf values")

        # Valid class IDs (excluding ignore_index).
        valid_mask = targets != ignore_index
        if valid_mask.any():
            target_ids  = set(int(v) for v in np.unique(targets[valid_mask]))
            invalid_ids = {i for i in target_ids if i < 0 or i >= num_classes}
            if invalid_ids:
                issues.append(
                    f"targets contains class IDs outside [0, {num_classes}): "
                    f"{sorted(invalid_ids)}"
                )

        return EvaluationValidationResult(issues)

    def validate_config(
        self,
        config:       EvaluationConfig,
        model_result: Any,
        data_result:  Any,
    ) -> EvaluationValidationResult:
        """
        Pre-flight configuration and compatibility checks.

        Args:
            config:       EvaluationConfig.
            model_result: ModelResult from Module 13 (or TrainingResult.model).
            data_result:  TransformPipelineResult from Module 12.

        Returns:
            EvaluationValidationResult.
        """
        issues: list[str] = []

        if config.batch_size < 1:
            issues.append(f"batch_size must be >= 1, got {config.batch_size}.")

        if config.split not in ("train", "validation", "test"):
            issues.append(
                f"split='{config.split}' must be one of "
                f"'train', 'validation', 'test'."
            )

        if model_result is None:
            issues.append("model (or model_result) is None.")

        if data_result is None:
            issues.append("data_result (TransformPipelineResult) is None.")
            return EvaluationValidationResult(issues)

        # num_classes compatibility.
        data_classes = getattr(data_result, "num_classes", None)
        if data_classes is not None and data_classes < 1:
            issues.append(f"data_result.num_classes={data_classes} is invalid.")

        return EvaluationValidationResult(issues)
