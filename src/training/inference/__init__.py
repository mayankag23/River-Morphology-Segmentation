"""
src/training/inference -- Inference Pipeline Framework (Module 16).

Single public entry point:
    InferenceEngine.predict(training_result, data_result) -> InferenceResult

Usage
-----
    from src.training.inference import InferenceEngine, InferenceConfig

    config = InferenceConfig(
        checkpoint_strategy = "best",
        checkpoint_dir      = "checkpoints",
        device              = "cpu",
        export_numpy        = True,
        output_dir          = "predictions",
    )
    engine = InferenceEngine(config)
    result = engine.predict(
        training_result,
        data_result,
        class_names = ("background", "water", "sand", "vegetation"),
    )

    for pred in result.predictions:
        print(pred.sample_id, pred.mean_confidence.mean())
"""

# Primary entry point
from src.training.inference.engine import InferenceEngine

# Public contracts
from src.training.inference.contracts import (
    CheckpointMetadata,
    InferenceConfig,
    InferenceResult,
    SamplePrediction,
)

# Loader
from src.training.inference.loader import CheckpointLoader

# Confidence
from src.training.inference.confidence import (
    ConfidenceRegistry,
    ConfidenceStrategy,
    EntropyStrategy,
    MaxProbabilityStrategy,
)

# Post-processing
from src.training.inference.postprocessing import (
    HoleFiller,
    MaskPostprocessor,
    MorphCloseProcessor,
    MorphOpenProcessor,
    PostprocessorPipeline,
    PostprocessorRegistry,
    SmallObjectRemover,
)

# Predictor
from src.training.inference.predictor import Predictor

# Exporter
from src.training.inference.exporter import PredictionExporter

# Validator
from src.training.inference.validator import InferenceValidator, InferenceValidationResult

# Factory
from src.training.inference.factory import InferenceFactory

__all__ = [
    # Primary
    "InferenceEngine",
    # Contracts
    "InferenceConfig",
    "CheckpointMetadata",
    "SamplePrediction",
    "InferenceResult",
    # Loader
    "CheckpointLoader",
    # Confidence
    "ConfidenceStrategy",
    "ConfidenceRegistry",
    "MaxProbabilityStrategy",
    "EntropyStrategy",
    # Post-processing
    "MaskPostprocessor",
    "PostprocessorRegistry",
    "PostprocessorPipeline",
    "HoleFiller",
    "SmallObjectRemover",
    "MorphOpenProcessor",
    "MorphCloseProcessor",
    # Predictor
    "Predictor",
    # Exporter
    "PredictionExporter",
    # Validator
    "InferenceValidator",
    "InferenceValidationResult",
    # Factory
    "InferenceFactory",
]
