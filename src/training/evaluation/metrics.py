"""
Metric functions for the Model Evaluation Framework (Module 15).

All metrics are computed from the confusion matrix (C, C) int64 ndarray.
No loops over classes; all operations are fully vectorized.

Numerical stability
-------------------
All divisions use np.where(denom > 0, num / denom, 0.0) to return 0.0
(not NaN) when a class has zero support or zero predictions.

Registered metrics
------------------
    pixel_accuracy       Overall correctly classified pixel fraction.
    mean_pixel_accuracy  Mean per-class pixel accuracy.
    precision            Per-class and mean precision.
    recall               Per-class and mean recall.
    f1                   Per-class and mean F1 (harmonic mean of P and R).
    dice                 Alias for F1 in segmentation context.
    iou                  Per-class and mean IoU (Jaccard index).
    mean_iou             Mean IoU across all classes.
    fw_iou               Frequency-weighted IoU.
    kappa                Cohen's Kappa coefficient.
    balanced_accuracy    Mean per-class recall (sensitivity).

Adding a new metric
-------------------
    @MetricRegistry.register("my_metric")
    def _my_metric(cm: np.ndarray) -> dict:
        ...
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np

__all__ = ["MetricRegistry", "compute_all_metrics"]

_LOGGER: logging.Logger = logging.getLogger(__name__)
_EPS:    float          = 1e-10


# ==============================================================================
# MetricRegistry
# ==============================================================================

class MetricRegistry:
    """
    Registry mapping metric names to computation functions.

    Each function receives a (C, C) int64 confusion matrix (rows=true, cols=pred)
    and returns a dict of {metric_name: value}.
    """

    _registry: dict[str, Callable[[np.ndarray], dict[str, Any]]] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """Decorator to register a metric function under name."""
        def decorator(fn: Callable) -> Callable:
            cls._registry[name.lower()] = fn
            return fn
        return decorator

    @classmethod
    def get(cls, name: str) -> Callable[[np.ndarray], dict[str, Any]]:
        n = name.lower().strip()
        if n not in cls._registry:
            raise KeyError(
                f"MetricRegistry: '{name}' is not registered. "
                f"Available: {sorted(cls._registry)}"
            )
        return cls._registry[n]

    @classmethod
    def registered_names(cls) -> tuple[str, ...]:
        return tuple(sorted(cls._registry.keys()))

    @classmethod
    def clear(cls) -> None:
        """For test isolation ONLY."""
        cls._registry.clear()


# ==============================================================================
# Utility
# ==============================================================================

def _safe_div(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Element-wise safe division; returns 0.0 where denominator == 0."""
    # return np.where(denominator > _EPS, numerator / denominator, 0.0)
    result = np.zeros_like(numerator, dtype=float)

    np.divide(
        numerator,
        denominator,
        out=result,
        where=denominator > _EPS,
    )

    return result  


def _per_class_tp_fp_fn_tn(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute TP, FP, FN, TN arrays of shape (C,) from (C, C) confusion matrix."""
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp     # col sum - diagonal
    fn = cm.sum(axis=1).astype(np.float64) - tp     # row sum  - diagonal
    tn = cm.sum().astype(np.float64) - tp - fp - fn
    return tp, fp, fn, tn


# ==============================================================================
# Registered metric functions
# ==============================================================================

@MetricRegistry.register("pixel_accuracy")
def _pixel_accuracy(cm: np.ndarray) -> dict[str, Any]:
    """Overall fraction of correctly classified non-ignored pixels."""
    correct = float(np.diag(cm).sum())
    total   = float(cm.sum())
    return {"pixel_accuracy": correct / total if total > 0 else 0.0}


@MetricRegistry.register("mean_pixel_accuracy")
def _mean_pixel_accuracy(cm: np.ndarray) -> dict[str, Any]:
    """Mean of per-class pixel accuracy (diagonal / row_sum)."""
    row_sums = cm.sum(axis=1).astype(np.float64)
    per_cls  = _safe_div(np.diag(cm).astype(np.float64), row_sums)
    # Only average over classes that actually appear.
    valid    = row_sums > 0
    mean     = float(per_cls[valid].mean()) if valid.any() else 0.0
    return {"mean_pixel_accuracy": mean, "_per_class_pixel_accuracy": per_cls}


@MetricRegistry.register("precision")
def _precision(cm: np.ndarray) -> dict[str, Any]:
    """Per-class and mean precision: TP / (TP + FP)."""
    tp, fp, fn, tn = _per_class_tp_fp_fn_tn(cm)
    per_cls = _safe_div(tp, tp + fp)
    valid   = (tp + fp) > 0
    mean    = float(per_cls[valid].mean()) if valid.any() else 0.0
    return {"mean_precision": mean, "_per_class_precision": per_cls}


@MetricRegistry.register("recall")
def _recall(cm: np.ndarray) -> dict[str, Any]:
    """Per-class and mean recall (sensitivity): TP / (TP + FN)."""
    tp, fp, fn, tn = _per_class_tp_fp_fn_tn(cm)
    per_cls = _safe_div(tp, tp + fn)
    valid   = (tp + fn) > 0
    mean    = float(per_cls[valid].mean()) if valid.any() else 0.0
    return {"mean_recall": mean, "_per_class_recall": per_cls}


@MetricRegistry.register("f1")
def _f1(cm: np.ndarray) -> dict[str, Any]:
    """Per-class and mean F1: harmonic mean of precision and recall."""
    tp, fp, fn, tn = _per_class_tp_fp_fn_tn(cm)
    prec    = _safe_div(tp, tp + fp)
    rec     = _safe_div(tp, tp + fn)
    per_cls = _safe_div(2.0 * prec * rec, prec + rec)
    valid   = (tp + fp + fn) > 0
    mean    = float(per_cls[valid].mean()) if valid.any() else 0.0
    return {"mean_f1": mean, "_per_class_f1": per_cls}


@MetricRegistry.register("dice")
def _dice(cm: np.ndarray) -> dict[str, Any]:
    """
    Per-class and mean Dice score.
    Dice = 2*TP / (2*TP + FP + FN) == F1.
    """
    tp, fp, fn, tn = _per_class_tp_fp_fn_tn(cm)
    per_cls = _safe_div(2.0 * tp, 2.0 * tp + fp + fn)
    valid   = (2.0 * tp + fp + fn) > 0
    mean    = float(per_cls[valid].mean()) if valid.any() else 0.0
    return {"mean_dice": mean, "_per_class_dice": per_cls}


@MetricRegistry.register("iou")
def _iou(cm: np.ndarray) -> dict[str, Any]:
    """Per-class and mean Intersection over Union (Jaccard index)."""
    tp, fp, fn, tn = _per_class_tp_fp_fn_tn(cm)
    per_cls = _safe_div(tp, tp + fp + fn)
    valid   = (tp + fp + fn) > 0
    mean    = float(per_cls[valid].mean()) if valid.any() else 0.0
    return {"mean_iou": mean, "_per_class_iou": per_cls}


@MetricRegistry.register("mean_iou")
def _mean_iou(cm: np.ndarray) -> dict[str, Any]:
    """Mean IoU (delegates to iou metric; provided as alias for config convenience)."""
    return _iou(cm)


@MetricRegistry.register("fw_iou")
def _fw_iou(cm: np.ndarray) -> dict[str, Any]:
    """
    Frequency-weighted IoU.

    fw_IoU = sum_c(freq_c * IoU_c) where freq_c = row_sum_c / total_pixels.
    """
    tp, fp, fn, tn = _per_class_tp_fp_fn_tn(cm)
    iou_per_cls    = _safe_div(tp, tp + fp + fn)
    row_sums       = cm.sum(axis=1).astype(np.float64)
    total          = float(cm.sum())
    freq           = row_sums / total if total > 0 else np.zeros_like(row_sums)
    fw             = float((freq * iou_per_cls).sum())
    return {"fw_iou": fw, "_per_class_iou_for_fw": iou_per_cls}


@MetricRegistry.register("kappa")
def _kappa(cm: np.ndarray) -> dict[str, Any]:
    """
    Cohen's Kappa coefficient.

    kappa = (p_o - p_e) / (1 - p_e)
        p_o = observed accuracy
        p_e = expected accuracy (chance agreement)
    """
    total  = float(cm.sum())
    if total == 0:
        return {"kappa": 0.0}
    p_o    = float(np.diag(cm).sum()) / total
    row_s  = cm.sum(axis=1).astype(np.float64) / total
    col_s  = cm.sum(axis=0).astype(np.float64) / total
    p_e    = float((row_s * col_s).sum())
    denom  = 1.0 - p_e
    k      = (p_o - p_e) / denom if abs(denom) > _EPS else 0.0
    return {"kappa": float(np.clip(k, -1.0, 1.0))}


@MetricRegistry.register("balanced_accuracy")
def _balanced_accuracy(cm: np.ndarray) -> dict[str, Any]:
    """
    Balanced accuracy = mean per-class recall (sensitivity).

    Handles class imbalance by averaging recall across all present classes.
    """
    result = _recall(cm)
    return {"balanced_accuracy": result["mean_recall"]}


# ==============================================================================
# Convenience function
# ==============================================================================

def compute_all_metrics(
    cm:           np.ndarray,
    metric_names: tuple[str, ...] = (),
) -> dict[str, Any]:
    """
    Compute all registered (or a specified subset of) metrics from a confusion matrix.

    Args:
        cm:           (C, C) int64 confusion matrix.
        metric_names: Tuple of metric names to compute. Empty = all registered.

    Returns:
        Merged dict of all metric outputs.
    """
    names  = metric_names if metric_names else MetricRegistry.registered_names()
    result: dict[str, Any] = {}
    seen:   set[str]       = set()

    for name in names:
        if name in seen:
            continue
        seen.add(name)
        try:
            fn      = MetricRegistry.get(name)
            partial = fn(cm)
            result.update(partial)
        except KeyError:
            _LOGGER.warning("compute_all_metrics: '%s' is not registered; skipping.", name)

    return result
