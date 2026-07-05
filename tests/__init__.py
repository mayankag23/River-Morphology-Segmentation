"""Unit tests for the River Morphology Segmentation System."""
"""
Unit tests for the src/training package.

Module 11 -- PyTorch Dataset & DataLoader pipeline (test files):
    test_dataset.py       RiverMorphologyDataset, SampleMetadata, RiverMorphologySample
    test_normalizer.py    NormalizationStats, DatasetNormalizer
    test_transforms.py    Transform, IdentityTransform, AlbumentationsTransform,
                          AugmentationConfig, AugmentationPipeline
    test_sampler.py       TemporalSampler, SamplerStrategy
    test_weights.py       ClassWeights, ClassWeightStrategy
    test_dataloader.py    DataLoaderConfig, DataLoaderBundle, DataLoaderFactory

Module 12 -- Data Transformation & Augmentation Pipeline (test files):
    test_training_contracts.py      NormalizationStatistics, TransformSample,
                                    TransformMetadata, TransformPipelineResult
    test_training_transform.py      SegmentationTransform, ComposedTransform,
                                    IdentityTransform
    test_training_augmentation.py   All eight augmentation transforms
    test_training_normalization.py  NormalizationTransform
    test_training_statistics.py     DatasetStatisticsAccumulator
    test_training_validator.py      TransformValidator, TransformValidationResult
    test_training_registry.py       TransformRegistry
    test_training_pipeline.py       TransformPipeline, AugmentedDataset
"""