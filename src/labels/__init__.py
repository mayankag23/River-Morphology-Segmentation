"""
Label management package for the River Morphology Segmentation System
(Module 9).

Discovers, validates, organizes, and versions segmentation label masks
against patches produced by Module 8, preserving complete temporal
context. River morphology is never assumed temporally static: sandbars
vegetate, channels migrate, and class composition shifts with monsoon and
flood-recession cycles. Every label retains its own acquisition date,
season, hydrological year, and provenance so multiple independent
observations of the same AOI can coexist without one overwriting another.

Input:   PatchDatasetResult (Module 8) + SceneMetadata (Module 7, reused)
Output:  LabelDatasetResult (immutable)

Architecture:
    LabelSource -> LabelValidator -> TemporalMetadataBuilder ->
    LabelStatisticsCalculator -> LabelManifestManager -> LabelManager
    (orchestrator)

Components:
    LabelSourceRecord / LabelSource / FilesystemLabelSource
        -- Pluggable mask discovery. Future sources (SAM-based annotation,
           GIS-derived masks) implement LabelSource without changing
           LabelManager.
    ClassSchema               -- Class taxonomy sourced from config.classes.
    SeasonResolver              -- Month -> season mapping from config.temporal.
    HydrologicalYearResolver      -- Calendar date -> hydrological year.
    TemporalMetadata                -- Frozen per-label temporal/provenance record.
    TemporalMetadataBuilder           -- Builds TemporalMetadata records.
    LabelValidator                      -- Validates mask vs. patch consistency.
    LabelStatisticsCalculator             -- Aggregates class/seasonal/yearly/
                                          ratio statistics.
    LabelManifestEntry / LabelManifest / LabelManifestManager
        -- Frozen records and persistence for label_manifest.csv/json.
    LabelManager                          -- Orchestrates all of the above.
"""

from src.labels.manager import LabelDatasetResult, LabelManager
from src.labels.manifest import (
    LABEL_MANIFEST_SCHEMA_VERSION,
    LabelManifest,
    LabelManifestEntry,
    LabelManifestManager,
)
from src.labels.schema import ClassDefinition, ClassSchema
from src.labels.source import FilesystemLabelSource, LabelSource, LabelSourceRecord
from src.labels.statistics import (
    ClassPixelStatistics,
    ClassRatio,
    LabelStatistics,
    LabelStatisticsCalculator,
    SeasonCount,
    YearCount,
)
from src.labels.temporal import (
    HydrologicalYearResolver,
    SeasonResolver,
    TemporalMetadata,
    TemporalMetadataBuilder,
    validate_temporal_consistency,
)
from src.labels.validator import LabelValidationResult, LabelValidator

__all__ = [
    "LabelSourceRecord",
    "LabelSource",
    "FilesystemLabelSource",
    "ClassDefinition",
    "ClassSchema",
    "SeasonResolver",
    "HydrologicalYearResolver",
    "TemporalMetadata",
    "TemporalMetadataBuilder",
    "validate_temporal_consistency",
    "LabelValidationResult",
    "LabelValidator",
    "ClassPixelStatistics",
    "ClassRatio",
    "SeasonCount",
    "YearCount",
    "LabelStatistics",
    "LabelStatisticsCalculator",
    "LabelManifestEntry",
    "LabelManifest",
    "LabelManifestManager",
    "LABEL_MANIFEST_SCHEMA_VERSION",
    "LabelDatasetResult",
    "LabelManager",
]