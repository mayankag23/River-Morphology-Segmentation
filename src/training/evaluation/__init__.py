"""
src/training/evaluation -- Model Evaluation Framework (Module 15).

Single public entry point:
    EvaluationEngine.evaluate(training_result, data_result) -> EvaluationResult

Usage
-----
    from src.training.evaluation import EvaluationEngine, EvaluationConfig

    config = EvaluationConfig(split="test", batch_size=16)
    engine = EvaluationEngine(config)
    result = engine.evaluate(training_result, data_result,
                             class_names=("background","water","sand","vegetation"))

    print(result.mean_iou)
    print(result.per_class["water"].f1)
    result.as_dict()  # full JSON-serializable summary
"""

# Primary entry point
from src.training.evaluation.engine import EvaluationEngine

# Public contracts
from src.training.evaluation.contracts import (
    ClassMetrics,
    ConfusionMatrix,
    EvaluationConfig,
    EvaluationResult,
    PredictionStatistics,
)

# Confusion matrix
from src.training.evaluation.confusion import ConfusionMatrixAccumulator

# Metrics
from src.training.evaluation.metrics import MetricRegistry, compute_all_metrics

# Statistics
from src.training.evaluation.statistics import PredictionStatisticsAccumulator

# Validator
from src.training.evaluation.validator import EvaluationValidator, EvaluationValidationResult

# Reporter
from src.training.evaluation.reporter import EvaluationReporter

# Factory
from src.training.evaluation.factory import EvaluationFactory

# Evaluator (for advanced use)
from src.training.evaluation.evaluator import Evaluator

__all__ = [
    # Primary
    "EvaluationEngine",
    # Contracts
    "EvaluationConfig",
    "ClassMetrics",
    "ConfusionMatrix",
    "PredictionStatistics",
    "EvaluationResult",
    # Core
    "ConfusionMatrixAccumulator",
    "MetricRegistry",
    "compute_all_metrics",
    "PredictionStatisticsAccumulator",
    # Validation
    "EvaluationValidator",
    "EvaluationValidationResult",
    # Reporting
    "EvaluationReporter",
    # Factory + Evaluator
    "EvaluationFactory",
    "Evaluator",
]
