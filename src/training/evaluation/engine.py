"""
EvaluationEngine -- the single public interface for Module 15.

Usage
-----
    from src.training.evaluation import EvaluationEngine

    engine = EvaluationEngine(config)
    result = engine.evaluate(training_result, data_result)

    print(result.mean_iou)          # 0.823
    print(result.per_class["water"].iou)   # 0.91
    result.confusion_matrix.as_dict()      # JSON-serializable

EvaluationEngine orchestrates:
    EvaluationFactory -> Evaluator + DataLoader
    Evaluator.run()   -> confusion matrix + per-class + aggregate metrics
    EvaluationReporter -> JSON / CSV (when output_dir is set)
"""

from __future__ import annotations

import logging
import time
from typing import Any

from src.training.evaluation.contracts import (
    EvaluationConfig,
    EvaluationResult,
)
from src.training.evaluation.factory import EvaluationFactory
from src.training.evaluation.reporter import EvaluationReporter
from src.training.evaluation.validator import EvaluationValidator

__all__ = ["EvaluationEngine"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


class EvaluationEngine:
    """
    Orchestrates a complete model evaluation run.

    Args:
        config: EvaluationConfig or project Config object.
                EvaluationConfig.from_config(config) is called when needed.
    """

    def __init__(self, config: Any) -> None:
        if isinstance(config, EvaluationConfig):
            self._config = config
        else:
            self._config = EvaluationConfig.from_config(config)
        self._validator = EvaluationValidator()
        self._logger    = _LOGGER

    def evaluate(
        self,
        training_result: Any,
        data_result:     Any,
        class_names:     tuple[str, ...] | None = None,
        num_classes:     int | None             = None,
    ) -> EvaluationResult:
        """
        Run a complete evaluation and return an immutable EvaluationResult.

        Args:
            training_result: TrainingResult from Module 14 (or any object
                             with a .model attribute).
            data_result:     TransformPipelineResult from Module 12.
            class_names:     Ordered class names. When None, derived from
                             data_result.num_classes as ("class_0", ...).
            num_classes:     Number of classes. When None, taken from
                             data_result.num_classes.

        Returns:
            Frozen EvaluationResult.
        """
        ops: list[str] = []

        # Step 1: Resolve class schema.
        resolved_num_classes, resolved_class_names = self._resolve_classes(
            data_result, num_classes, class_names
        )
        ops.append(
            f"classes: {resolved_num_classes} "
            f"({', '.join(resolved_class_names)})"
        )

        # Step 2: Pre-flight validation.
        model = getattr(training_result, "model", training_result)
        arch  = getattr(training_result, "architecture", "unknown")
        validation = self._validator.validate_config(
            self._config, model, data_result
        )
        for issue in validation.issues:
            self._logger.warning("EvaluationEngine pre-flight: %s", issue)
        ops.append(f"validation: {len(validation.issues)} issue(s)")

        # Step 3: Build evaluator and DataLoader.
        evaluator, dataloader = EvaluationFactory.build(
            config      = self._config,
            data_result = data_result,
            num_classes = resolved_num_classes,
            class_names = resolved_class_names,
        )
        ops.append(
            f"evaluator: split={self._config.split}, "
            f"device={self._config.device}, "
            f"batch_size={self._config.batch_size}"
        )

        # Step 4: Run evaluation.
        t0        = time.perf_counter()
        eval_data = evaluator.run(model, dataloader)
        elapsed   = time.perf_counter() - t0
        ops.append(f"inference: {elapsed:.2f}s")

        # Step 5: Finalise confusion matrix and statistics.
        cm_acc    = eval_data["cm_accumulator"]
        stats_acc = eval_data["stats_accumulator"]
        cm_obj    = cm_acc.compute()
        stats_obj = stats_acc.compute(cm_acc.matrix)
        aggregate = eval_data["aggregate"]
        per_class = eval_data["per_class"]
        ops.append(
            f"totals: samples={stats_obj.total_samples}, "
            f"pixels={stats_obj.total_pixels:,}"
        )

        # Step 6: Build EvaluationResult.
        num_params = getattr(training_result, "num_parameters", 0)

        result = EvaluationResult(
            pixel_accuracy      = float(aggregate.get("pixel_accuracy",      0.0)),
            mean_pixel_accuracy = float(aggregate.get("mean_pixel_accuracy",  0.0)),
            mean_iou            = float(aggregate.get("mean_iou",             0.0)),
            fw_iou              = float(aggregate.get("fw_iou",               0.0)),
            mean_dice           = float(aggregate.get("mean_dice",            0.0)),
            mean_precision      = float(aggregate.get("mean_precision",       0.0)),
            mean_recall         = float(aggregate.get("mean_recall",          0.0)),
            mean_f1             = float(aggregate.get("mean_f1",              0.0)),
            kappa               = float(aggregate.get("kappa",                0.0)),
            balanced_accuracy   = float(aggregate.get("balanced_accuracy",    0.0)),
            per_class           = per_class,
            confusion_matrix    = cm_obj,
            statistics          = stats_obj,
            split               = self._config.split,
            architecture        = arch,
            num_classes         = resolved_num_classes,
            ignore_index        = self._config.ignore_index,
            total_samples       = stats_obj.total_samples,
            total_pixels        = stats_obj.total_pixels,
            evaluation_time_s   = elapsed,
            operations_log      = tuple(ops),
            class_names         = resolved_class_names,
        )

        # Step 7: Log summary.
        for line in result.summary_lines():
            self._logger.info(line)

        # Step 8: Save reports when configured.
        if self._config.output_dir:
            reporter = EvaluationReporter(self._config.output_dir)
            if self._config.save_json:
                reporter.save_json(result)
            if self._config.save_csv:
                reporter.save_csv(result)

        return result

    @staticmethod
    def _resolve_classes(
        data_result:  Any,
        num_classes:  int | None,
        class_names:  tuple[str, ...] | None,
    ) -> tuple[int, tuple[str, ...]]:
        """Resolve num_classes and class_names from available sources."""
        nc = num_classes or getattr(data_result, "num_classes", 4)
        if class_names:
            return nc, tuple(class_names)[:nc]
        # Try to get names from data_result if available (future-proof).
        names_from_data = getattr(data_result, "class_names", None)
        if names_from_data:
            return nc, tuple(names_from_data)[:nc]
        return nc, tuple(f"class_{i}" for i in range(nc))
