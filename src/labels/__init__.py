"""
Label management package for the River Morphology Segmentation System
(Module 9 -- Pseudo-Label Generation).

Generates pseudo-label masks automatically from spectral feature-stack
patch GeoTIFFs produced by Modules 7 and 8. Replaces the previous
filesystem-based manual label discovery with a fully automatic spectral
classification pipeline.

Public contract preserved:
    LabelDatasetResult is identical to the v1 implementation so that
    Modules 10 and 11 require zero changes.

New internal pipeline (automatic pseudo-label generation):
    SpectralClassificationEngine
        -> RuleEngine (RuleRegistry)
        -> ConflictResolver
        -> MorphologyProcessor
        -> QualityAssessment
        -> ConfidenceEstimator
        -> PseudoLabelGenerator
        -> LabelManager
        -> LabelDatasetResult

Extensibility:
    Additional label generation strategies (SAM, ML, Hybrid) can be added
    by subclassing LabelGenerationStrategy and registering via
    LabelStrategyRegistry, without changing the public API.
"""

# Contracts (internal pipeline data transfer objects)
from src.labels.contracts import (
    ClassificationContext,
    ClassificationResult,
    ConfidenceResult,
    MorphologyResult,
    PseudoLabelResult,
    QualityResult,
    ReproducibilityMetadata,
    RuleResult,
    SpectralBandData,
)

# Rules
from src.labels.rules import (
    BackgroundRule,
    ClassificationRule,
    RuleEngine,
    RuleRegistry,
    SandRule,
    VegetationRule,
    WaterRule,
)

# Classifier
from src.labels.classifier import SpectralBandReader, SpectralClassificationEngine

# Conflict resolution
from src.labels.conflicts import ConflictResolver

# Morphology
from src.labels.morphology import MorphologyConfig, MorphologyProcessor, PerClassMorphologyConfig

# Quality
from src.labels.quality import QualityAssessment, QualityConfig

# Confidence
from src.labels.confidence import ConfidenceConfig, ConfidenceEstimator

# Strategy interface and registry
from src.labels.strategy import LabelGenerationStrategy, LabelStrategyRegistry

# Concrete generator (also registers itself in LabelStrategyRegistry)
from src.labels.generator import PseudoLabelGenerator

# Schema (unchanged from v1)
from src.labels.schema import ClassDefinition, ClassSchema

# Temporal (unchanged from v1)
from src.labels.temporal import (
    HydrologicalYearResolver,
    SeasonResolver,
    TemporalMetadata,
    TemporalMetadataBuilder,
    validate_temporal_consistency,
)

# Validator (unchanged from v1)
from src.labels.validator import LabelValidationResult, LabelValidator

# Statistics (unchanged from v1)
from src.labels.statistics import (
    ClassPixelStatistics,
    ClassRatio,
    LabelStatistics,
    LabelStatisticsCalculator,
    SeasonCount,
    YearCount,
)

# Manifest (schema v1.1 -- adds generation_strategy field)
from src.labels.manifest import (
    LABEL_MANIFEST_SCHEMA_VERSION,
    LabelManifest,
    LabelManifestEntry,
    LabelManifestManager,
)

# Manager + result (public contract)
from src.labels.manager import LabelDatasetResult, LabelManager

__all__ = [
    # Contracts
    "ClassificationContext",
    "ReproducibilityMetadata",
    "SpectralBandData",
    "RuleResult",
    "ClassificationResult",
    "MorphologyResult",
    "QualityResult",
    "ConfidenceResult",
    "PseudoLabelResult",
    # Rules
    "ClassificationRule",
    "RuleRegistry",
    "WaterRule",
    "SandRule",
    "VegetationRule",
    "BackgroundRule",
    "RuleEngine",
    # Classifier
    "SpectralBandReader",
    "SpectralClassificationEngine",
    # Conflict resolution
    "ConflictResolver",
    # Morphology
    "PerClassMorphologyConfig",
    "MorphologyConfig",
    "MorphologyProcessor",
    # Quality
    "QualityConfig",
    "QualityAssessment",
    # Confidence
    "ConfidenceConfig",
    "ConfidenceEstimator",
    # Strategy
    "LabelGenerationStrategy",
    "LabelStrategyRegistry",
    # Generator
    "PseudoLabelGenerator",
    # Schema
    "ClassDefinition",
    "ClassSchema",
    # Temporal
    "SeasonResolver",
    "HydrologicalYearResolver",
    "TemporalMetadata",
    "TemporalMetadataBuilder",
    "validate_temporal_consistency",
    # Validator
    "LabelValidationResult",
    "LabelValidator",
    # Statistics
    "ClassPixelStatistics",
    "ClassRatio",
    "SeasonCount",
    "YearCount",
    "LabelStatistics",
    "LabelStatisticsCalculator",
    # Manifest
    "LabelManifestEntry",
    "LabelManifest",
    "LabelManifestManager",
    "LABEL_MANIFEST_SCHEMA_VERSION",
    # Manager + result (public contract)
    "LabelDatasetResult",
    "LabelManager",
]



# """
# Label management package for the River Morphology Segmentation System
# (Module 9 — Pseudo-Label Generation).

# Generates pseudo-label masks automatically from spectral features (patch
# GeoTIFFs produced by Module 8). Replaces the previous filesystem-based
# manual label discovery with a fully automatic spectral classification
# pipeline.

# Public contract preserved:
#     LabelDatasetResult is identical to the v1 implementation so that
#     Modules 10 and 11 require zero changes.

# New internal pipeline:
#     SpectralClassificationEngine -> RuleEngine -> ConflictResolver
#     -> MorphologyProcessor -> QualityAssessment -> ConfidenceEstimator
#     -> PseudoLabelGenerator -> LabelManager -> LabelDatasetResult
# """

# from src.labels.classifier import SpectralBandReader, SpectralClassificationEngine
# from src.labels.confidence import ConfidenceConfig, ConfidenceEstimator
# from src.labels.conflicts import ConflictResolver
# from src.labels.contracts import (
#     ClassificationResult,
#     ConfidenceResult,
#     MorphologyResult,
#     PseudoLabelResult,
#     QualityResult,
#     RuleResult,
#     SpectralBandData,
# )
# from src.labels.generator import PseudoLabelGenerator
# from src.labels.manager import LabelDatasetResult, LabelManager
# from src.labels.manifest import (
#     LABEL_MANIFEST_SCHEMA_VERSION,
#     LabelManifest,
#     LabelManifestEntry,
#     LabelManifestManager,
# )
# from src.labels.morphology import MorphologyConfig, MorphologyProcessor
# from src.labels.quality import QualityAssessment, QualityConfig
# from src.labels.rules import (
#     BackgroundRule,
#     ClassificationRule,
#     RuleEngine,
#     SandRule,
#     VegetationRule,
#     WaterRule,
# )
# from src.labels.schema import ClassDefinition, ClassSchema
# from src.labels.statistics import (
#     ClassPixelStatistics,
#     ClassRatio,
#     LabelStatistics,
#     LabelStatisticsCalculator,
#     SeasonCount,
#     YearCount,
# )
# from src.labels.temporal import (
#     HydrologicalYearResolver,
#     SeasonResolver,
#     TemporalMetadata,
#     TemporalMetadataBuilder,
#     validate_temporal_consistency,
# )
# from src.labels.validator import LabelValidationResult, LabelValidator

# __all__ = [
#     # Contracts (internal pipeline)
#     "SpectralBandData",
#     "RuleResult",
#     "ClassificationResult",
#     "MorphologyResult",
#     "QualityResult",
#     "ConfidenceResult",
#     "PseudoLabelResult",
#     # Rules
#     "ClassificationRule",
#     "WaterRule",
#     "SandRule",
#     "VegetationRule",
#     "BackgroundRule",
#     "RuleEngine",
#     # Classifier
#     "SpectralBandReader",
#     "SpectralClassificationEngine",
#     # Conflict resolution
#     "ConflictResolver",
#     # Morphology
#     "MorphologyConfig",
#     "MorphologyProcessor",
#     # Quality
#     "QualityConfig",
#     "QualityAssessment",
#     # Confidence
#     "ConfidenceConfig",
#     "ConfidenceEstimator",
#     # Generator
#     "PseudoLabelGenerator",
#     # Schema (unchanged)
#     "ClassDefinition",
#     "ClassSchema",
#     # Temporal (unchanged)
#     "SeasonResolver",
#     "HydrologicalYearResolver",
#     "TemporalMetadata",
#     "TemporalMetadataBuilder",
#     "validate_temporal_consistency",
#     # Validator (unchanged)
#     "LabelValidationResult",
#     "LabelValidator",
#     # Statistics (unchanged)
#     "ClassPixelStatistics",
#     "ClassRatio",
#     "SeasonCount",
#     "YearCount",
#     "LabelStatistics",
#     "LabelStatisticsCalculator",
#     # Manifest (unchanged)
#     "LabelManifestEntry",
#     "LabelManifest",
#     "LabelManifestManager",
#     "LABEL_MANIFEST_SCHEMA_VERSION",
#     # Manager + result (public contract)
#     "LabelDatasetResult",
#     "LabelManager",
# ]
