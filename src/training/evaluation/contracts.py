"""
Public data contracts for the Model Evaluation Framework (Module 15).

Contract chain:
    TrainingResult            (Module 14) ──┐
    TransformPipelineResult   (Module 12) ──┤──> EvaluationEngine.evaluate() ──> EvaluationResult
    Config                                ──┘

EvaluationResult is the immutable public output consumed by:
    Module 16 (Inference)
    Future visualization modules
    External reporting tools

Design invariants
-----------------
- All public contracts are frozen dataclasses.
- No torch types appear at module level (lazy import policy).
- All floating-point metric values default to 0.0 (never NaN) to guarantee
  safe downstream arithmetic.
- Per-class metrics are stored as dicts keyed by class name (str), not by
  class_id (int), so they remain human-readable in JSON/CSV output.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "EvaluationConfig",
    "ClassMetrics",
    "ConfusionMatrix",
    "PredictionStatistics",
    "EvaluationResult",
]


# ==============================================================================
# EvaluationConfig
# ==============================================================================

@dataclass(frozen=True)
class EvaluationConfig:
    """
    Immutable evaluation configuration.

    Attributes:
        split:             Dataset split to evaluate: "test", "validation", "train".
        batch_size:        Inference batch size.
        num_workers:       DataLoader worker count.
        device:            Compute device ("cpu", "cuda", "cuda:0", …).
        ignore_index:      Pixel class index to exclude from all metrics.
        output_dir:        Directory for JSON/CSV reports. "" = no file output.
        save_json:         Write EvaluationResult as JSON.
        save_csv:          Write per-class metrics as CSV.
        metrics:           Ordered tuple of metric names to compute.
                           Empty tuple = compute all registered metrics.
        pin_memory:        Pin DataLoader memory for faster GPU transfers.
    """

    split:        str              = "test"
    batch_size:   int              = 8
    num_workers:  int              = 4
    device:       str              = "cpu"
    ignore_index: int              = 255
    output_dir:   str              = ""
    save_json:    bool             = False
    save_csv:     bool             = False
    metrics:      tuple[str, ...]  = ()
    pin_memory:   bool             = False

    @classmethod
    def from_config(cls, config: Any) -> EvaluationConfig:
        """Build EvaluationConfig from config.evaluation."""
        eval_cfg = getattr(config, "evaluation", None)
        if eval_cfg is None:
            return cls()
        raw_metrics = getattr(eval_cfg, "metrics", ())
        return cls(
            split        = str(getattr(eval_cfg,   "split",        "test")),
            batch_size   = int(getattr(eval_cfg,   "batch_size",   8)),
            num_workers  = int(getattr(eval_cfg,   "num_workers",  4)),
            device       = str(getattr(eval_cfg,   "device",       "cpu")),
            ignore_index = int(getattr(eval_cfg,   "ignore_index", 255)),
            output_dir   = str(getattr(eval_cfg,   "output_dir",   "")),
            save_json    = bool(getattr(eval_cfg,  "save_json",    False)),
            save_csv     = bool(getattr(eval_cfg,  "save_csv",     False)),
            metrics      = tuple(str(m) for m in raw_metrics),
            pin_memory   = bool(getattr(eval_cfg,  "pin_memory",   False)),
        )


# ==============================================================================
# ClassMetrics
# ==============================================================================

@dataclass(frozen=True)
class ClassMetrics:
    """
    Immutable per-class evaluation metrics.

    All values are in [0.0, 1.0] (except num_pixels and num_predicted which
    are raw counts). Every field defaults to 0.0 so no NaN propagates.

    Attributes:
        class_id:       Integer class label.
        class_name:     Human-readable class name from ClassSchema.
        precision:      TP / (TP + FP).
        recall:         TP / (TP + FN).  Also called sensitivity.
        f1:             2 * precision * recall / (precision + recall).
        dice:           Alias for F1 in segmentation context.
        iou:            TP / (TP + FP + FN).  Intersection over Union.
        pixel_accuracy: Fraction of correctly classified pixels for this class.
        tp:             True positives.
        fp:             False positives.
        fn:             False negatives.
        tn:             True negatives.
        num_pixels:     Total ground-truth pixels for this class (TP + FN).
        num_predicted:  Total predicted pixels for this class (TP + FP).
        support:        Alias for num_pixels (ground-truth frequency).
    """

    class_id:       int
    class_name:     str
    precision:      float = 0.0
    recall:         float = 0.0
    f1:             float = 0.0
    dice:           float = 0.0
    iou:            float = 0.0
    pixel_accuracy: float = 0.0
    tp:             int   = 0
    fp:             int   = 0
    fn:             int   = 0
    tn:             int   = 0
    num_pixels:     int   = 0
    num_predicted:  int   = 0
    support:        int   = 0

    def as_dict(self) -> dict:
        """Return a JSON-serializable dict."""
        return {
            "class_id":       self.class_id,
            "class_name":     self.class_name,
            "precision":      round(self.precision,      6),
            "recall":         round(self.recall,         6),
            "f1":             round(self.f1,             6),
            "dice":           round(self.dice,           6),
            "iou":            round(self.iou,            6),
            "pixel_accuracy": round(self.pixel_accuracy, 6),
            "tp":             self.tp,
            "fp":             self.fp,
            "fn":             self.fn,
            "tn":             self.tn,
            "num_pixels":     self.num_pixels,
            "num_predicted":  self.num_predicted,
            "support":        self.support,
        }


# ==============================================================================
# ConfusionMatrix
# ==============================================================================

@dataclass(frozen=True)
class ConfusionMatrix:
    """
    Immutable multi-class confusion matrix.

    Attributes:
        matrix:       (C, C) row=true, col=predicted, stored as a nested tuple.
                      matrix[i][j] = number of pixels with true class i
                      predicted as class j.
        normalized:   Row-normalized version (each row sums to 1.0).
        class_names:  Ordered class names matching matrix rows/columns.
        num_classes:  Number of classes C.
        total_pixels: Total evaluated pixels (excluding ignore_index).
    """

    matrix:       tuple[tuple[int, ...], ...]
    normalized:   tuple[tuple[float, ...], ...]
    class_names:  tuple[str, ...]
    num_classes:  int
    total_pixels: int

    def as_dict(self) -> dict:
        """Return a JSON-serializable dict."""
        return {
            "class_names":  list(self.class_names),
            "num_classes":  self.num_classes,
            "total_pixels": self.total_pixels,
            "matrix":       [list(row) for row in self.matrix],
            "normalized":   [[round(v, 6) for v in row] for row in self.normalized],
        }


# ==============================================================================
# PredictionStatistics
# ==============================================================================

@dataclass(frozen=True)
class PredictionStatistics:
    """
    Aggregated prediction statistics across the evaluated dataset.

    Attributes:
        total_pixels:       All non-ignored pixels across all samples.
        total_samples:      Number of samples (images) evaluated.
        class_pixel_counts: Dict mapping class_name -> ground-truth pixel count.
        pred_pixel_counts:  Dict mapping class_name -> predicted pixel count.
        class_frequencies:  Dict mapping class_name -> fraction of GT pixels.
        pred_frequencies:   Dict mapping class_name -> fraction of pred pixels.
    """

    total_pixels:       int
    total_samples:      int
    class_pixel_counts: dict[str, int]
    pred_pixel_counts:  dict[str, int]
    class_frequencies:  dict[str, float]
    pred_frequencies:   dict[str, float]

    def as_dict(self) -> dict:
        return {
            "total_pixels":       self.total_pixels,
            "total_samples":      self.total_samples,
            "class_pixel_counts": self.class_pixel_counts,
            "pred_pixel_counts":  self.pred_pixel_counts,
            "class_frequencies":  {k: round(v, 6) for k, v in self.class_frequencies.items()},
            "pred_frequencies":   {k: round(v, 6) for k, v in self.pred_frequencies.items()},
        }


# ==============================================================================
# EvaluationResult
# ==============================================================================

@dataclass(frozen=True)
class EvaluationResult:
    """
    Immutable public output of EvaluationEngine.evaluate().

    Attributes:
        pixel_accuracy:      Fraction of correctly classified non-ignored pixels.
        mean_pixel_accuracy: Mean per-class pixel accuracy.
        mean_iou:            Mean Intersection over Union (mIoU).
        fw_iou:              Frequency-weighted IoU.
        mean_dice:           Mean per-class Dice / F1.
        mean_precision:      Mean per-class precision.
        mean_recall:         Mean per-class recall (sensitivity).
        mean_f1:             Mean per-class F1.
        kappa:               Cohen's Kappa coefficient.
        balanced_accuracy:   Mean per-class recall (balanced accuracy).
        per_class:           Dict class_name -> ClassMetrics.
        confusion_matrix:    ConfusionMatrix with raw and normalized forms.
        statistics:          PredictionStatistics across the full dataset.
        split:               Dataset split evaluated ("test", "validation", …).
        architecture:        Model architecture name.
        num_classes:         Number of classes evaluated.
        ignore_index:        Index excluded from all metrics.
        total_samples:       Number of images evaluated.
        total_pixels:        Total non-ignored pixels evaluated.
        evaluation_time_s:   Wall-clock seconds for the full evaluation.
        operations_log:      Ordered log of evaluation steps.
        class_names:         Ordered class names.
    """

    pixel_accuracy:      float
    mean_pixel_accuracy: float
    mean_iou:            float
    fw_iou:              float
    mean_dice:           float
    mean_precision:      float
    mean_recall:         float
    mean_f1:             float
    kappa:               float
    balanced_accuracy:   float
    per_class:           dict[str, ClassMetrics]
    confusion_matrix:    ConfusionMatrix
    statistics:          PredictionStatistics
    split:               str
    architecture:        str
    num_classes:         int
    ignore_index:        int
    total_samples:       int
    total_pixels:        int
    evaluation_time_s:   float
    operations_log:      tuple[str, ...]
    class_names:         tuple[str, ...]

    def summary_lines(self) -> list[str]:
        """Return ASCII-formatted summary lines (no Unicode box-drawing)."""
        return [
            f"  split:              {self.split}",
            f"  architecture:       {self.architecture}",
            f"  total_samples:      {self.total_samples}",
            f"  total_pixels:       {self.total_pixels:,}",
            f"  pixel_accuracy:     {self.pixel_accuracy:.4f}",
            f"  mean_iou:           {self.mean_iou:.4f}",
            f"  mean_dice:          {self.mean_dice:.4f}",
            f"  mean_f1:            {self.mean_f1:.4f}",
            f"  kappa:              {self.kappa:.4f}",
            f"  balanced_accuracy:  {self.balanced_accuracy:.4f}",
            f"  evaluation_time_s:  {self.evaluation_time_s:.2f}",
        ]

    def as_dict(self) -> dict:
        """Return a fully JSON-serializable dict."""
        return {
            "pixel_accuracy":      round(self.pixel_accuracy,      6),
            "mean_pixel_accuracy": round(self.mean_pixel_accuracy,  6),
            "mean_iou":            round(self.mean_iou,             6),
            "fw_iou":              round(self.fw_iou,               6),
            "mean_dice":           round(self.mean_dice,            6),
            "mean_precision":      round(self.mean_precision,       6),
            "mean_recall":         round(self.mean_recall,          6),
            "mean_f1":             round(self.mean_f1,              6),
            "kappa":               round(self.kappa,                6),
            "balanced_accuracy":   round(self.balanced_accuracy,    6),
            "split":               self.split,
            "architecture":        self.architecture,
            "num_classes":         self.num_classes,
            "ignore_index":        self.ignore_index,
            "total_samples":       self.total_samples,
            "total_pixels":        self.total_pixels,
            "evaluation_time_s":   round(self.evaluation_time_s,   3),
            "class_names":         list(self.class_names),
            "per_class":           {k: v.as_dict() for k, v in self.per_class.items()},
            "confusion_matrix":    self.confusion_matrix.as_dict(),
            "statistics":          self.statistics.as_dict(),
            "operations_log":      list(self.operations_log),
        }
