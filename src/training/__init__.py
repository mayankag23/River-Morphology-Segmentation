"""
PyTorch Dataset & DataLoader pipeline for the River Morphology Segmentation
System (Module 11).

Bridges the assembled dataset (Module 10's TrainingDatasetResult) to the
deep-learning training loop.

Input:   TrainingDatasetResult (Module 10)
Output:  DataLoaderBundle (immutable)

Transform architecture:
    Transform (ABC)         -- interface: __call__(image, mask) -> (image, mask)
    IdentityTransform       -- no-op default (eval splits, tests)
    AlbumentationsTransform -- wraps albumentations.Compose; all albumentations
                               coupling is confined to this class; Module 12
                               may inject any other Transform subclass
"""

from src.training.dataloader import DataLoaderBundle, DataLoaderConfig, DataLoaderFactory
from src.training.dataset import RiverMorphologyDataset, RiverMorphologySample, SampleMetadata
from src.training.normalizer import DatasetNormalizer, NormalizationStats, NormalizationStrategy
from src.training.sampler import TemporalSampler
from src.training.transforms import (
    AlbumentationsTransform,
    AugmentationConfig,
    AugmentationPipeline,
    IdentityTransform,
    Transform,
)
from src.training.weights import ClassWeights, ClassWeightStrategy

__all__ = [
    # Transform interface
    "Transform",
    "IdentityTransform",
    "AlbumentationsTransform",
    # Dataset
    "SampleMetadata",
    "RiverMorphologySample",
    "RiverMorphologyDataset",
    # Normalization
    "NormalizationStrategy",
    "NormalizationStats",
    "DatasetNormalizer",
    # Augmentation config (kept for backward compatibility)
    "AugmentationConfig",
    "AugmentationPipeline",
    # Sampling
    "TemporalSampler",
    # Class weights
    "ClassWeightStrategy",
    "ClassWeights",
    # DataLoader
    "DataLoaderConfig",
    "DataLoaderBundle",
    "DataLoaderFactory",
]