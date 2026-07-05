"""
Core evaluation loop for Module 15.

Evaluator iterates over a DataLoader, runs the model in eval mode with
torch.no_grad(), accumulates predictions into ConfusionMatrixAccumulator,
and assembles the final set of ClassMetrics and aggregate metrics.

Design rules
------------
- torch is imported lazily (not at module level).
- No training logic, no gradient computation, no checkpointing.
- Fully vectorized metric computation via compute_all_metrics().
- Ignore_index pixels are excluded at the confusion matrix level.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.training.evaluation.confusion import ConfusionMatrixAccumulator
from src.training.evaluation.contracts import ClassMetrics, EvaluationConfig
from src.training.evaluation.metrics import compute_all_metrics
from src.training.evaluation.statistics import PredictionStatisticsAccumulator
from src.training.evaluation.validator import EvaluationValidator

__all__ = ["Evaluator"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


class Evaluator:
    """
    Runs model inference on a DataLoader and accumulates evaluation metrics.

    Args:
        config:       EvaluationConfig controlling evaluation behaviour.
        num_classes:  Number of segmentation classes.
        class_names:  Ordered class names from ClassSchema.
        device:       torch.device to run inference on.
    """

    def __init__(
        self,
        config:      EvaluationConfig,
        num_classes: int,
        class_names: tuple[str, ...],
        device:      Any,
    ) -> None:
        self._config      = config
        self._num_classes = num_classes
        self._class_names = class_names
        self._device      = device
        self._validator   = EvaluationValidator()
        self._logger      = logging.getLogger(__name__)

    def run(
        self,
        model:      Any,
        dataloader: Any,
    ) -> dict[str, Any]:
        """
        Run full evaluation loop.

        Args:
            model:      Trained torch.nn.Module. Moved to device if needed.
            dataloader: DataLoader yielding (images, masks) batches.

        Returns:
            Dict with keys:
                "cm_accumulator"     ConfusionMatrixAccumulator (finalized)
                "stats_accumulator"  PredictionStatisticsAccumulator
                "per_class"          dict[str, ClassMetrics]
                "aggregate"          dict[str, float]  (all aggregate metrics)
        """
        import torch

        model = model.to(self._device)
        model.eval()

        cm_acc    = ConfusionMatrixAccumulator(
            num_classes  = self._num_classes,
            ignore_index = self._config.ignore_index,
            class_names  = self._class_names,
        )
        stats_acc = PredictionStatisticsAccumulator(self._class_names)

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                images, masks = self._unpack_batch(batch, self._device)

                # Forward pass — logits (B, C, H, W).
                logits = model(images)

                # Argmax to get predictions (B, H, W).
                preds = logits.argmax(dim=1)

                # Move to CPU numpy.
                preds_np = preds.cpu().numpy().astype(np.int64)
                masks_np = masks.cpu().numpy().astype(np.int64)

                # Optional batch validation (only log, never crash evaluation).
                result = self._validator.validate_batch(
                    predictions  = preds_np,
                    targets      = masks_np,
                    num_classes  = self._num_classes,
                    ignore_index = self._config.ignore_index,
                )
                if not result.is_valid:
                    self._logger.warning(
                        "Evaluator batch %d validation issues: %s",
                        batch_idx, result.issues,
                    )

                # Accumulate into confusion matrix.
                cm_acc.update(preds_np, masks_np)
                stats_acc.increment_samples(images.shape[0])

        # Compute all metrics from the final confusion matrix.
        cm_np      = cm_acc.matrix
        aggregate  = compute_all_metrics(cm_np, self._config.metrics)
        per_class  = self._build_per_class_metrics(cm_np, aggregate)

        return {
            "cm_accumulator":    cm_acc,
            "stats_accumulator": stats_acc,
            "per_class":         per_class,
            "aggregate":         aggregate,
        }

    def _build_per_class_metrics(
        self,
        cm:        np.ndarray,
        aggregate: dict[str, Any],
    ) -> dict[str, ClassMetrics]:
        """Build per-class ClassMetrics from the confusion matrix and aggregate dict."""
        counts = ConfusionMatrixAccumulator(
            num_classes  = self._num_classes,
            ignore_index = self._config.ignore_index,
            class_names  = self._class_names,
        )
        counts._matrix = cm.copy()
        per_class_counts = counts.per_class_counts()

        # Extract per-class arrays from aggregate dict.
        prec_arr  = aggregate.get("_per_class_precision",      np.zeros(self._num_classes))
        rec_arr   = aggregate.get("_per_class_recall",         np.zeros(self._num_classes))
        f1_arr    = aggregate.get("_per_class_f1",             np.zeros(self._num_classes))
        dice_arr  = aggregate.get("_per_class_dice",           np.zeros(self._num_classes))
        iou_arr   = aggregate.get("_per_class_iou",            np.zeros(self._num_classes))
        acc_arr   = aggregate.get("_per_class_pixel_accuracy",  np.zeros(self._num_classes))

        result: dict[str, ClassMetrics] = {}
        row_sums = cm.sum(axis=1)

        for i, name in enumerate(self._class_names):
            cnts       = per_class_counts.get(name, {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
            num_pixels = int(row_sums[i])

            result[name] = ClassMetrics(
                class_id       = i,
                class_name     = name,
                precision      = float(_safe_scalar(prec_arr, i)),
                recall         = float(_safe_scalar(rec_arr,  i)),
                f1             = float(_safe_scalar(f1_arr,   i)),
                dice           = float(_safe_scalar(dice_arr, i)),
                iou            = float(_safe_scalar(iou_arr,  i)),
                pixel_accuracy = float(_safe_scalar(acc_arr,  i)),
                tp             = cnts["tp"],
                fp             = cnts["fp"],
                fn             = cnts["fn"],
                tn             = cnts["tn"],
                num_pixels     = num_pixels,
                num_predicted  = cnts["tp"] + cnts["fp"],
                support        = num_pixels,
            )
        return result

    @staticmethod
    def _unpack_batch(batch: Any, device: Any) -> tuple[Any, Any]:
        """Extract (images, masks) from DataLoader batch and move to device."""
        import torch
        images = batch[0].to(device, non_blocking=True)
        masks  = batch[1].to(device, non_blocking=True)
        if masks.dtype != torch.long:
            masks = masks.long()
        return images, masks


def _safe_scalar(arr: Any, i: int) -> float:
    """Safely extract arr[i] as float, returning 0.0 on any error."""
    try:
        v = float(arr[i])
        return 0.0 if (np.isnan(v) or np.isinf(v)) else v
    except (IndexError, TypeError):
        return 0.0
