"""
src/training package -- Modules 11 and 12.

Module 11 -- PyTorch Dataset & DataLoader (FROZEN)
    Bridges TrainingDatasetResult (Module 10) to the PyTorch training loop.

    Input:   TrainingDatasetResult (Module 10)
    Output:  DataLoaderBundle (immutable)

Module 12 -- Data Transformation & Augmentation Pipeline
    Wraps Module 11 RiverMorphologyDataset objects with configurable
    transform / augmentation / normalization pipelines.

    Input:   RiverMorphologyDataset instances (from Module 11 DataLoaderFactory)
    Output:  TransformPipelineResult (immutable)
"""

# ==============================================================================
# MODULE 11 EXPORTS -- FROZEN -- DO NOT MODIFY
# ==============================================================================

# Transform interface (Module 11 signature: __call__(image, mask) -> (image, mask))
from src.training.transforms import (
    Transform,
    IdentityTransform as M11IdentityTransform,
    AlbumentationsTransform,
    AugmentationConfig,
    AugmentationPipeline,
)

# Dataset (lazy rasterio reads, returns (image_tensor, mask_tensor, SampleMetadata))
from src.training.dataset import (
    SampleMetadata,
    RiverMorphologySample,
    RiverMorphologyDataset,
)

# Normalization (computes and applies per-band statistics from training split)
from src.training.normalizer import (
    NormalizationStrategy,
    NormalizationStats,
    DatasetNormalizer,
)

# Temporal sampling (WeightedRandomSampler based on season / hydro-year)
from src.training.sampler import (
    TemporalSampler,
    SamplerStrategy,
)

# Class loss weights (from training-split class pixel distribution)
from src.training.weights import (
    ClassWeightStrategy,
    ClassWeights,
)

# DataLoader factory and bundle (public output contract of Module 11)
from src.training.dataloader import (
    DataLoaderConfig,
    DataLoaderBundle,
    DataLoaderFactory,
)

# ==============================================================================
# MODULE 12 EXPORTS -- Data Transformation & Augmentation Pipeline
# All imports below are additive. No Module 11 symbol is removed or redefined.
# ==============================================================================

# Contracts (public API consumed by Modules 13 and 14)
from src.training.contracts import (
    NormalizationStatistics,
    TransformSample,
    TransformMetadata,
    TransformPipelineResult,
)

# Core transform interface (Module 12 signature: apply(TransformSample) -> TransformSample)
# Note: SegmentationTransform and IdentityTransform here operate on TransformSample,
# which is distinct from Module 11's Transform which operates on (image, mask) pairs.
# They coexist without collision because they are imported from different modules.
from src.training.transform import (
    SegmentationTransform,
    ComposedTransform,
    IdentityTransform,    # Module 12 variant: apply(TransformSample) -> TransformSample
)

# Augmentation transforms (all multi-band, mask-synchronized, config-driven)
from src.training.augmentation import (
    HorizontalFlipTransform,
    VerticalFlipTransform,
    Rotate90Transform,
    BrightnessTransform,
    ContrastTransform,
    GaussianNoiseTransform,
    RandomCropTransform,
    RandomScaleTransform,
)

# Per-band standardization: output = (image - mean) / std
from src.training.normalization import NormalizationTransform

# Streaming Welford accumulator for computing per-band mean/std
from src.training.statistics import DatasetStatisticsAccumulator

# Transform sample validator (shape, dtype, NaN/Inf, class IDs, metadata)
from src.training.validator import (
    TransformValidator,
    TransformValidationResult,
)

# Plugin registry for augmentation transforms
from src.training.registry import TransformRegistry

# Pipeline orchestrator (entry point for Module 12)
from src.training.pipeline import (
    AugmentedDataset,
    TransformPipeline,
)

# ==============================================================================
# __all__
# ==============================================================================

__all__ = [
    # ------------------------------------------------------------------
    # Module 11 -- FROZEN
    # ------------------------------------------------------------------
    # Transform interface
    "Transform",
    "M11IdentityTransform",
    "AlbumentationsTransform",
    "AugmentationConfig",
    "AugmentationPipeline",
    # Dataset
    "SampleMetadata",
    "RiverMorphologySample",
    "RiverMorphologyDataset",
    # Normalization
    "NormalizationStrategy",
    "NormalizationStats",
    "DatasetNormalizer",
    # Sampling
    "TemporalSampler",
    "SamplerStrategy",
    # Class weights
    "ClassWeightStrategy",
    "ClassWeights",
    # DataLoader
    "DataLoaderConfig",
    "DataLoaderBundle",
    "DataLoaderFactory",

    # ------------------------------------------------------------------
    # Module 12
    # ------------------------------------------------------------------
    # Contracts
    "NormalizationStatistics",
    "TransformSample",
    "TransformMetadata",
    "TransformPipelineResult",
    # Core transform interface (TransformSample-based)
    "SegmentationTransform",
    "ComposedTransform",
    "IdentityTransform",
    # Augmentations
    "HorizontalFlipTransform",
    "VerticalFlipTransform",
    "Rotate90Transform",
    "BrightnessTransform",
    "ContrastTransform",
    "GaussianNoiseTransform",
    "RandomCropTransform",
    "RandomScaleTransform",
    # Normalization transform
    "NormalizationTransform",
    # Statistics accumulator
    "DatasetStatisticsAccumulator",
    # Validator
    "TransformValidator",
    "TransformValidationResult",
    # Registry
    "TransformRegistry",
    # Pipeline
    "AugmentedDataset",
    "TransformPipeline",
]

# ==============================================================================
# MODULE 13 EXPORTS -- Segmentation Model Framework
# All imports below are additive. No Module 11/12 symbol is removed or redefined.
# ==============================================================================

from src.training.models import (
    # Contracts
    EncoderConfig,
    DecoderConfig,
    ModelConfig,
    ModelResult,
    # Abstract base
    SegmentationModel,
    # Registry + Factory
    ModelRegistry,
    ModelFactory,
    # Shared blocks
    ConvBnAct,
    DoubleConv,
    ResidualBlock,
    build_norm,
    build_act,
    # Sub-components
    UNetEncoder,
    EncoderOutput,
    UNetPlusPlusDecoder,
    SegmentationHead,
    # Concrete models
    UNetPlusPlus,
)

_MODULE_13_SYMBOLS: list[str] = [
    "EncoderConfig",
    "DecoderConfig",
    "ModelConfig",
    "ModelResult",
    "SegmentationModel",
    "ModelRegistry",
    "ModelFactory",
    "ConvBnAct",
    "DoubleConv",
    "ResidualBlock",
    "build_norm",
    "build_act",
    "UNetEncoder",
    "EncoderOutput",
    "UNetPlusPlusDecoder",
    "SegmentationHead",
    "UNetPlusPlus",
]

# ==============================================================================
# MODULE 14 EXPORTS -- Training Engine Framework
# All imports below are additive. No Module 11/12/13 symbol is removed or redefined.
# ==============================================================================

from src.training.engine import (
    # Primary
    TrainingEngine,
    # Contracts
    TrainingConfig,
    OptimizerConfig,
    SchedulerConfig,
    LossConfig,
    CheckpointConfig,
    EpochResult,
    TrainingResult,
    # Losses
    LossRegistry,
    CrossEntropyLoss,
    DiceLoss,
    FocalLoss,
    CombinedLoss,
    # Optimizer / scheduler
    OptimizerFactory,
    SchedulerFactory,
    # Callbacks
    Callback,
    CallbackList,
    CheckpointCallback,
    LoggingCallback,
    EarlyStoppingCallback,
    # Support
    CheckpointManager,
    SeedManager,
    TrainingValidator,
    TrainingHistory,
)
