"""
src/training/engine -- Training Engine Framework (Module 14).

Single public entry point:
    TrainingEngine.train(model_result, data_result) -> TrainingResult

Usage
-----
    from src.training.engine import TrainingEngine, TrainingConfig

    config = TrainingConfig.from_config(project_config)
    engine = TrainingEngine(config)
    result = engine.train(model_result, data_result)

    # Trained model is on CPU.
    model      = result.model
    best_ckpt  = result.best_checkpoint
    history    = result.history
"""

# Primary entry point
from src.training.engine.engine import TrainingEngine

# Public contracts
from src.training.engine.contracts import (
    CheckpointConfig,
    EpochResult,
    LossConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
    TrainingResult,
)

# Loss functions and registry
from src.training.engine.losses import (
    LossRegistry,
    CrossEntropyLoss,
    DiceLoss,
    FocalLoss,
    CombinedLoss,
)

# Optimizer / scheduler factories
from src.training.engine.optimizer import OptimizerFactory, OptimizerRegistry
from src.training.engine.scheduler import SchedulerFactory, SchedulerRegistry

# Callback system
from src.training.engine.callbacks import (
    Callback,
    CallbackList,
    CheckpointCallback,
    LoggingCallback,
    EarlyStoppingCallback,
)

# Checkpoint management
from src.training.engine.checkpoint import CheckpointManager

# Seed management
from src.training.engine.seed import SeedManager

# Validation
from src.training.engine.validator import TrainingValidator, TrainingValidationResult

# History
from src.training.engine.history import TrainingHistory

# Trainer (for advanced use / testing)
from src.training.engine.trainer import Trainer

# Factory (for advanced use)
from src.training.engine.factory import TrainingEngineFactory

__all__ = [
    # Primary
    "TrainingEngine",
    # Contracts
    "TrainingConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "LossConfig",
    "CheckpointConfig",
    "EpochResult",
    "TrainingResult",
    # Losses
    "LossRegistry",
    "CrossEntropyLoss",
    "DiceLoss",
    "FocalLoss",
    "CombinedLoss",
    # Optimizer / scheduler
    "OptimizerFactory",
    "OptimizerRegistry",
    "SchedulerFactory",
    "SchedulerRegistry",
    # Callbacks
    "Callback",
    "CallbackList",
    "CheckpointCallback",
    "LoggingCallback",
    "EarlyStoppingCallback",
    # Support
    "CheckpointManager",
    "SeedManager",
    "TrainingValidator",
    "TrainingValidationResult",
    "TrainingHistory",
    "Trainer",
    "TrainingEngineFactory",
]
