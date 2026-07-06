"""
Uncertainty analytics for Module 17.

UncertaintyAnalyzer consumes the confidence map produced by Module 16
(MaxProbabilityStrategy or EntropyStrategy) and computes per-class
and overall uncertainty statistics.

Scientific interpretation
--------------------------
- mean_confidence close to 1.0 indicates a decisive model (all pixels
  assigned with high certainty).
- low_conf_fraction > 0.2 (20%) suggests the model is uncertain about a
  significant fraction of the scene; downstream analysis should treat the
  predictions with additional caution.
- per_class_confidence reveals which morphological classes the model finds
  difficult to distinguish (e.g. shallow water vs wet sand).
"""

from __future__ import annotations

import logging

import numpy as np

from src.morphology.contracts import AnalyticsConfig, UncertaintyMetrics

__all__ = ["UncertaintyAnalyzer"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


class UncertaintyAnalyzer:
    """
    Computes uncertainty metrics from a confidence map and class mask.

    Args:
        config:      AnalyticsConfig (supplies low_confidence_threshold).
        class_names: Ordered class names from InferenceResult.
    """

    def __init__(
        self,
        config:      AnalyticsConfig,
        class_names: tuple[str, ...],
    ) -> None:
        self._config      = config
        self._class_names = class_names

    def compute(
        self,
        confidence: np.ndarray,
        mask:       np.ndarray,
    ) -> UncertaintyMetrics:
        """
        Compute UncertaintyMetrics for one sample.

        Args:
            confidence: (H, W) float32 per-pixel confidence in [0, 1].
            mask:       (H, W) uint8 predicted class-ID mask.

        Returns:
            Frozen UncertaintyMetrics.
        """
        total_pixels = int(confidence.size)
        if total_pixels == 0:
            return _zero_uncertainty(self._class_names)

        thresh       = self._config.low_confidence_threshold
        conf_flat    = confidence.ravel().astype(np.float64)

        # Global statistics.
        mean_conf   = float(np.mean(conf_flat))
        std_conf    = float(np.std(conf_flat))
        min_conf    = float(np.min(conf_flat))
        max_conf    = float(np.max(conf_flat))
        low_count   = int((conf_flat < thresh).sum())
        low_frac    = low_count / total_pixels

        # Per-class mean confidence.
        per_class: dict[str, float] = {}
        for class_id, class_name in enumerate(self._class_names):
            cls_mask = (mask == class_id)
            if cls_mask.any():
                per_class[class_name] = float(np.mean(confidence[cls_mask]))
            else:
                per_class[class_name] = 0.0

        return UncertaintyMetrics(
            mean_confidence      = mean_conf,
            std_confidence       = std_conf,
            min_confidence       = min_conf,
            max_confidence       = max_conf,
            low_conf_pixel_count = low_count,
            low_conf_fraction    = low_frac,
            per_class_confidence = per_class,
        )


def _zero_uncertainty(class_names: tuple[str, ...]) -> UncertaintyMetrics:
    return UncertaintyMetrics(
        mean_confidence      = 0.0,
        std_confidence       = 0.0,
        min_confidence       = 0.0,
        max_confidence       = 0.0,
        low_conf_pixel_count = 0,
        low_conf_fraction    = 0.0,
        per_class_confidence = {n: 0.0 for n in class_names},
    )
