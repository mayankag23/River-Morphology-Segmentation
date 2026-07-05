"""
Multi-class confusion matrix accumulator for Module 15.

ConfusionMatrixAccumulator implements a streaming (online) confusion matrix
that can be updated batch-by-batch without loading all predictions into memory.
All operations are fully vectorized using numpy; no Python loops over pixels.

The matrix convention:
    matrix[i, j] = number of pixels where true class is i and predicted class is j

Ignored pixels (matching ignore_index) are excluded before every update.
"""

from __future__ import annotations

import numpy as np

from src.training.evaluation.contracts import ConfusionMatrix

__all__ = ["ConfusionMatrixAccumulator"]

_EPS: float = 1e-10


class ConfusionMatrixAccumulator:
    """
    Streaming multi-class confusion matrix.

    Args:
        num_classes:   Number of segmentation classes.
        ignore_index:  Pixel value to exclude from all updates.
        class_names:   Ordered class names for the output ConfusionMatrix.
    """

    def __init__(
        self,
        num_classes:  int,
        ignore_index: int              = 255,
        class_names:  tuple[str, ...] = (),
    ) -> None:
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}.")
        self._C            = num_classes
        self._ignore_index = ignore_index
        self._class_names  = class_names or tuple(str(i) for i in range(num_classes))
        self._matrix       = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, predictions: np.ndarray, targets: np.ndarray) -> None:
        """
        Accumulate one batch of predictions and targets.

        Args:
            predictions: Integer array of predicted class IDs, any shape.
                         Will be flattened internally.
            targets:     Integer array of ground-truth class IDs, same shape.

        Both arrays are flattened before processing. Pixels where targets ==
        ignore_index are excluded. Predictions outside [0, num_classes) are
        also excluded to guard against invalid inputs.
        """
        preds   = np.asarray(predictions, dtype=np.int64).ravel()
        tgts    = np.asarray(targets,     dtype=np.int64).ravel()

        # Exclude ignore_index pixels.
        valid   = tgts != self._ignore_index

        # Also exclude any out-of-range prediction (defensive).
        valid  &= (preds >= 0) & (preds < self._C)
        valid  &= (tgts  >= 0) & (tgts  < self._C)

        preds   = preds[valid]
        tgts    = tgts[valid]

        if len(tgts) == 0:
            return

        # Vectorized accumulation using linear indexing.
        indices = tgts * self._C + preds
        counts  = np.bincount(indices, minlength=self._C * self._C)
        self._matrix += counts.reshape(self._C, self._C)

    def reset(self) -> None:
        """Clear all accumulated counts."""
        self._matrix[:] = 0

    @property
    def matrix(self) -> np.ndarray:
        """Raw (C, C) int64 confusion matrix (read-only view)."""
        return self._matrix

    @property
    def total_pixels(self) -> int:
        """Total non-ignored pixels accumulated so far."""
        return int(self._matrix.sum())

    def compute(self) -> ConfusionMatrix:
        """
        Build an immutable ConfusionMatrix from accumulated counts.

        Returns:
            ConfusionMatrix with raw counts and row-normalized form.
        """
        raw       = self._matrix.copy()
        row_sums  = raw.sum(axis=1, keepdims=True).astype(np.float64)
        norm      = np.where(row_sums > 0, raw / np.maximum(row_sums, _EPS), 0.0)

        matrix_tuple = tuple(tuple(int(v) for v in row) for row in raw)
        norm_tuple   = tuple(tuple(float(v) for v in row) for row in norm)

        return ConfusionMatrix(
            matrix       = matrix_tuple,
            normalized   = norm_tuple,
            class_names  = self._class_names,
            num_classes  = self._C,
            total_pixels = int(raw.sum()),
        )

    def per_class_counts(self) -> dict[str, dict[str, int]]:
        """
        Compute TP, FP, FN, TN per class from the confusion matrix.

        Returns:
            Dict mapping class_name -> {"tp": int, "fp": int, "fn": int, "tn": int}
        """
        result: dict[str, dict[str, int]] = {}
        for i in range(self._C):
            tp  = int(self._matrix[i, i])
            fp  = int(self._matrix[:, i].sum()) - tp      # predicted i but not true i
            fn  = int(self._matrix[i, :].sum()) - tp      # true i but not predicted i
            tn  = int(self._matrix.sum()) - tp - fp - fn
            result[self._class_names[i]] = {
                "tp": tp, "fp": fp, "fn": fn, "tn": tn
            }
        return result
