"""
Prediction statistics accumulator for Module 15.

PredictionStatisticsAccumulator tracks per-class ground-truth and prediction
pixel counts across all evaluated samples. Statistics are derived from the
final confusion matrix rather than being accumulated separately to ensure
full consistency.
"""

from __future__ import annotations

import numpy as np

from src.training.evaluation.contracts import PredictionStatistics

__all__ = ["PredictionStatisticsAccumulator"]

_EPS: float = 1e-10


class PredictionStatisticsAccumulator:
    """
    Derives PredictionStatistics from a finalized confusion matrix.

    This class is intentionally stateless beyond the class_names mapping;
    all counts are derived from the confusion matrix to guarantee consistency
    with metrics computed from the same matrix.

    Args:
        class_names: Ordered class names matching matrix rows/columns.
    """

    def __init__(self, class_names: tuple[str, ...]) -> None:
        self._class_names   = class_names
        self._total_samples = 0

    def increment_samples(self, n: int = 1) -> None:
        """Record that n samples have been processed."""
        self._total_samples += n

    def compute(self, cm: np.ndarray) -> PredictionStatistics:
        """
        Build PredictionStatistics from a (C, C) confusion matrix.

        Args:
            cm: (C, C) int64 confusion matrix with rows=true, cols=predicted.

        Returns:
            Frozen PredictionStatistics.
        """
        total = int(cm.sum())
        row_sums = cm.sum(axis=1)   # GT pixel counts per class
        col_sums = cm.sum(axis=0)   # Predicted pixel counts per class

        class_pixel_counts: dict[str, int]   = {}
        pred_pixel_counts:  dict[str, int]   = {}
        class_frequencies:  dict[str, float] = {}
        pred_frequencies:   dict[str, float] = {}

        for i, name in enumerate(self._class_names):
            gt_count           = int(row_sums[i])
            pd_count           = int(col_sums[i])
            class_pixel_counts[name] = gt_count
            pred_pixel_counts[name]  = pd_count
            class_frequencies[name]  = gt_count / total if total > 0 else 0.0
            pred_frequencies[name]   = pd_count / total if total > 0 else 0.0

        return PredictionStatistics(
            total_pixels       = total,
            total_samples      = self._total_samples,
            class_pixel_counts = class_pixel_counts,
            pred_pixel_counts  = pred_pixel_counts,
            class_frequencies  = class_frequencies,
            pred_frequencies   = pred_frequencies,
        )
