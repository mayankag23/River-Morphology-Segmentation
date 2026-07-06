"""
Morphology statistics computation for Module 17 (src.morphology package).

MorphologyStatisticsComputer converts a predicted class-ID mask and its
associated confidence map into ClassMorphologyMetrics for every class.

All operations are fully vectorized using numpy.

Refinements vs original proposal
----------------------------------
- confidence_weighted_area: sum of confidence values over class pixels.
  This is the confidence-weighted pixel count:
      confidence_weighted_area = sum(confidence[mask == class_id])
  It is more reliable than raw pixel_count when model confidence is uneven.
- confidence_weighted_area_m2: confidence_weighted_area * pixel_area_m2.
- pixel_width_m and pixel_height_m from AnalyticsConfig are forwarded
  to support multi-resolution satellite imagery in downstream modules.
"""

from __future__ import annotations

import logging

import numpy as np

from src.morphology.contracts import AnalyticsConfig, ClassMorphologyMetrics

__all__ = ["MorphologyStatisticsComputer"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


class MorphologyStatisticsComputer:
    """
    Computes per-class morphology statistics from a predicted mask.

    Args:
        config:      AnalyticsConfig supplying pixel_area_m2, thresholds.
        class_names: Ordered class names from InferenceResult.class_names.
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
        mask:       np.ndarray,
        confidence: np.ndarray,
    ) -> dict[str, ClassMorphologyMetrics]:
        """
        Compute ClassMorphologyMetrics for every class.

        Args:
            mask:       (H, W) uint8 predicted class-ID mask.
            confidence: (H, W) float32 per-pixel confidence map.

        Returns:
            Dict mapping class_name -> ClassMorphologyMetrics.
        """
        total_pixels = int(mask.size)
        if total_pixels == 0:
            return {}

        low_thresh    = self._config.low_confidence_threshold
        pixel_area_m2 = self._config.pixel_area_m2
        result: dict[str, ClassMorphologyMetrics] = {}

        for class_id, class_name in enumerate(self._class_names):
            class_mask  = (mask == class_id)
            pixel_count = int(class_mask.sum())

            total_fraction = pixel_count / total_pixels
            area_m2        = float(pixel_count * pixel_area_m2) if pixel_area_m2 > 0 else 0.0

            if pixel_count > 0:
                class_conf   = confidence[class_mask]
                mean_conf    = float(np.mean(class_conf))
                low_conf_px  = int((class_conf < low_thresh).sum())
                # Confidence-weighted area: sum of confidence values.
                conf_w_area  = float(class_conf.sum())
            else:
                mean_conf   = 0.0
                low_conf_px = 0
                conf_w_area = 0.0

            conf_w_area_m2 = float(conf_w_area * pixel_area_m2) if pixel_area_m2 > 0 else 0.0

            result[class_name] = ClassMorphologyMetrics(
                class_name                  = class_name,
                class_id                    = class_id,
                pixel_count                 = pixel_count,
                area_fraction               = total_fraction,
                total_fraction              = total_fraction,
                area_m2                     = area_m2,
                mean_confidence             = mean_conf,
                low_conf_pixels             = low_conf_px,
                confidence_weighted_area    = conf_w_area,
                confidence_weighted_area_m2 = conf_w_area_m2,
            )

        return result

    def dataset_mean_fractions(
        self,
        all_metrics: list[dict[str, ClassMorphologyMetrics]],
    ) -> dict[str, float]:
        """
        Compute mean area fraction per class across all samples.

        Args:
            all_metrics: List of per-sample dicts from compute().

        Returns:
            Dict class_name -> mean area fraction.
        """
        if not all_metrics:
            return {name: 0.0 for name in self._class_names}

        fracs: dict[str, list[float]] = {n: [] for n in self._class_names}
        for metrics in all_metrics:
            for name, cm in metrics.items():
                fracs[name].append(cm.area_fraction)

        return {
            name: float(np.mean(vals)) if vals else 0.0
            for name, vals in fracs.items()
        }
