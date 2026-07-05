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



# """
# src/training package -- Modules 11 and 12.

# Module 11 -- PyTorch Dataset & DataLoader (FROZEN)
#     Provides: TorchDatasetResult, RiverMorphologyDataset, and DataLoader utilities.
#     These exports must NOT be removed or modified.

# Module 12 -- Data Transformation & Augmentation Pipeline
#     Provides: TransformPipeline, AugmentedDataset, TransformPipelineResult,
#               and all transform / augmentation / normalization / registry classes.

# When integrating this file into the repository, preserve all existing Module 11
# imports and exports exactly as they appear in the current __init__.py.
# The Module 12 section below is purely additive.
# """

# # ==============================================================================
# # MODULE 11 EXPORTS -- FROZEN -- DO NOT MODIFY
# # Copy the existing Module 11 __init__.py content here exactly.
# # The imports below are placeholders; replace with the actual Module 11 exports.
# # ==============================================================================

# # Example (replace with the real Module 11 imports):
# #   from src.training.dataset import RiverMorphologyDataset, TorchDatasetResult
# #   from src.training.dataloader import DataLoaderFactory, DataLoaderConfig
# #
# # IMPORTANT: Do not remove or rename any Module 11 symbol.
# # Modules 10 and the rest of the pipeline depend on TorchDatasetResult
# # being importable from src.training.

# # ==============================================================================
# # MODULE 12 EXPORTS -- Data Transformation & Augmentation Pipeline
# # All imports below are additive. They do not replace any Module 11 export.
# # ==============================================================================

# # Contracts (public API consumed by Modules 13 and 14)
# from src.training.contracts import (
#     NormalizationStatistics,
#     TransformMetadata,
#     TransformPipelineResult,
#     TransformSample,
# )

# # Core transform interface
# from src.training.transform import (
#     ComposedTransform,
#     IdentityTransform,
#     SegmentationTransform,
# )

# # Augmentation transforms
# from src.training.augmentation import (
#     BrightnessTransform,
#     ContrastTransform,
#     GaussianNoiseTransform,
#     HorizontalFlipTransform,
#     RandomCropTransform,
#     RandomScaleTransform,
#     Rotate90Transform,
#     VerticalFlipTransform,
# )

# # Normalization
# from src.training.normalization import NormalizationTransform

# # Statistics computation
# from src.training.statistics import DatasetStatisticsAccumulator

# # Validation
# from src.training.validator import TransformValidationResult, TransformValidator

# # Registry
# from src.training.registry import TransformRegistry

# # Pipeline (public entry point for Module 12)
# from src.training.pipeline import AugmentedDataset, TransformPipeline

# # ==============================================================================
# # __all__
# # This list must include both Module 11 symbols AND Module 12 symbols.
# # Add Module 11 symbol names to the MODULE_11_SYMBOLS list below.
# # ==============================================================================

# _MODULE_11_SYMBOLS: list[str] = [
#     # Replace this list with the actual Module 11 public symbol names.
#     # Example:
#     #   "TorchDatasetResult",
#     #   "RiverMorphologyDataset",
#     #   "DataLoaderFactory",
# ]

# _MODULE_12_SYMBOLS: list[str] = [
#     # Contracts
#     "NormalizationStatistics",
#     "TransformSample",
#     "TransformMetadata",
#     "TransformPipelineResult",
#     # Interface
#     "SegmentationTransform",
#     "ComposedTransform",
#     "IdentityTransform",
#     # Augmentations
#     "HorizontalFlipTransform",
#     "VerticalFlipTransform",
#     "Rotate90Transform",
#     "BrightnessTransform",
#     "ContrastTransform",
#     "GaussianNoiseTransform",
#     "RandomCropTransform",
#     "RandomScaleTransform",
#     # Normalization
#     "NormalizationTransform",
#     # Statistics
#     "DatasetStatisticsAccumulator",
#     # Validation
#     "TransformValidator",
#     "TransformValidationResult",
#     # Registry
#     "TransformRegistry",
#     # Pipeline
#     "TransformPipeline",
#     "AugmentedDataset",
# ]


# # """
# # PyTorch Dataset & DataLoader pipeline for the River Morphology Segmentation
# # System (Module 11).

# # Bridges the assembled dataset (Module 10's TrainingDatasetResult) to the
# # deep-learning training loop.

# # Input:   TrainingDatasetResult (Module 10)
# # Output:  DataLoaderBundle (immutable)

# # Transform architecture:
# #     Transform (ABC)         -- interface: __call__(image, mask) -> (image, mask)
# #     IdentityTransform       -- no-op default (eval splits, tests)
# #     AlbumentationsTransform -- wraps albumentations.Compose; all albumentations
# #                                coupling is confined to this class; Module 12
# #                                may inject any other Transform subclass
# # """

# # from src.training.dataloader import DataLoaderBundle, DataLoaderConfig, DataLoaderFactory
# # from src.training.dataset import RiverMorphologyDataset, RiverMorphologySample, SampleMetadata
# # from src.training.normalizer import DatasetNormalizer, NormalizationStats, NormalizationStrategy
# # from src.training.sampler import TemporalSampler
# # from src.training.transforms import (
# #     AlbumentationsTransform,
# #     AugmentationConfig,
# #     AugmentationPipeline,
# #     IdentityTransform,
# #     Transform,
# # )
# # from src.training.weights import ClassWeights, ClassWeightStrategy

# # __all__ = [
# #     # Transform interface
# #     "Transform",
# #     "IdentityTransform",
# #     "AlbumentationsTransform",
# #     # Dataset
# #     "SampleMetadata",
# #     "RiverMorphologySample",
# #     "RiverMorphologyDataset",
# #     # Normalization
# #     "NormalizationStrategy",
# #     "NormalizationStats",
# #     "DatasetNormalizer",
# #     # Augmentation config (kept for backward compatibility)
# #     "AugmentationConfig",
# #     "AugmentationPipeline",
# #     # Sampling
# #     "TemporalSampler",
# #     # Class weights
# #     "ClassWeightStrategy",
# #     "ClassWeights",
# #     # DataLoader
# #     "DataLoaderConfig",
# #     "DataLoaderBundle",
# #     "DataLoaderFactory",
# # ]