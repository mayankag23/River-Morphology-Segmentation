"""
Analytics input validator for Module 17.

AnalyticsValidator checks that InferenceResult and AnalyticsConfig are
consistent before analysis begins. Never raises; accumulates issues.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.morphology.contracts import AnalyticsConfig

__all__ = ["AnalyticsValidator", "AnalyticsValidationResult"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


class AnalyticsValidationResult:
    """Result of one validation pass."""

    def __init__(self, issues: list[str]) -> None:
        self._issues = list(issues)

    @property
    def is_valid(self) -> bool:
        return len(self._issues) == 0

    @property
    def issues(self) -> list[str]:
        return list(self._issues)


class AnalyticsValidator:
    """Pre-flight validation for the analytics engine."""

    def validate(
        self,
        config:           AnalyticsConfig,
        inference_result: Any,
    ) -> AnalyticsValidationResult:
        """
        Validate AnalyticsConfig and InferenceResult compatibility.

        Args:
            config:           AnalyticsConfig.
            inference_result: InferenceResult from Module 16.

        Returns:
            AnalyticsValidationResult.
        """
        issues: list[str] = []

        if inference_result is None:
            issues.append("inference_result is None.")
            return AnalyticsValidationResult(issues)

        num_samples = getattr(inference_result, "num_samples", 0)
        if num_samples == 0:
            issues.append("inference_result has 0 samples; nothing to analyse.")

        class_names = getattr(inference_result, "class_names", ())
        if len(class_names) == 0:
            issues.append("inference_result.class_names is empty.")

        if not (0.0 <= config.low_confidence_threshold <= 1.0):
            issues.append(
                f"low_confidence_threshold must be in [0, 1], "
                f"got {config.low_confidence_threshold}."
            )

        if config.pixel_area_m2 < 0.0:
            issues.append(
                f"pixel_area_m2 must be >= 0, got {config.pixel_area_m2}."
            )

        # Validate individual SamplePrediction masks.
        predictions = getattr(inference_result, "predictions", ())
        for pred in predictions:
            mask = getattr(pred, "predicted_mask", None)
            conf = getattr(pred, "confidence",     None)
            if mask is None:
                issues.append(f"Sample '{pred.sample_id}': predicted_mask is None.")
                continue
            if mask.ndim != 2:
                issues.append(
                    f"Sample '{pred.sample_id}': mask must be 2-D, got {mask.ndim}-D."
                )
            if conf is not None and np.any(np.isnan(conf)):
                issues.append(
                    f"Sample '{pred.sample_id}': confidence contains NaN."
                )

        return AnalyticsValidationResult(issues)
