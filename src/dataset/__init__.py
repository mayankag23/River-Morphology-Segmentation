"""
Dataset Assembly & Quality Control pipeline for the River Morphology
Segmentation System (Module 10).

Assembles validated patches (Module 8) and validated labels (Module 9)
into a complete, split, quality-controlled training dataset. Produces
train / validation / test manifests and a quality report ready for
Module 11 (PyTorch Dataset creation).

Input:   list[PatchDatasetResult] + list[LabelDatasetResult]
Output:  TrainingDatasetResult (immutable)

Architecture:
    DatasetAssembler (orchestrator)
        -> DatasetValidator       (QC checks)
        -> DatasetSplitter        (train/val/test, scene-level)
        -> DataLeakageDetector    (verify no cross-split overlap)
        -> DatasetStatisticsCalculator (class distribution per split)
        -> DatasetQualityAnalyzer (quality report)
        -> DatasetManifestManager (writes all CSV/JSON files)
        -> DatasetVersionManager  (writes version.json)

Multi-temporal design:
    Labels are valid only for their acquisition date. Temporal splitting
    ensures earlier scenes are always in training, preventing a model from
    implicitly learning future river states during training.
"""

from src.dataset.assembler import DatasetAssembler, TrainingDatasetResult
from src.dataset.leakage import (
    DataLeakageDetector,
    LeakageDetectionResult,
    LeakageRecord,
)
from src.dataset.manifest import (
    DATASET_MANIFEST_SCHEMA_VERSION,
    DatasetManifest,
    DatasetManifestEntry,
    DatasetManifestManager,
    DatasetSample,
)
from src.dataset.quality import DatasetQualityAnalyzer, QualityIssue, QualityReport
from src.dataset.splitter import DatasetSplitter, SplitResult, SplitStrategy
from src.dataset.statistics import (
    ClassStatistics,
    DatasetStatisticsCalculator,
    SplitStatistics,
)
from src.dataset.validator import (
    DatasetValidationResult,
    DatasetValidator,
    ValidationIssue,
)
from src.dataset.version import DatasetVersionInfo, DatasetVersionManager

__all__ = [
    "DatasetSample",
    "DatasetManifestEntry",
    "DatasetManifest",
    "DatasetManifestManager",
    "DATASET_MANIFEST_SCHEMA_VERSION",
    "SplitStrategy",
    "SplitResult",
    "DatasetSplitter",
    "ValidationIssue",
    "DatasetValidationResult",
    "DatasetValidator",
    "LeakageRecord",
    "LeakageDetectionResult",
    "DataLeakageDetector",
    "ClassStatistics",
    "SplitStatistics",
    "DatasetStatisticsCalculator",
    "QualityIssue",
    "QualityReport",
    "DatasetQualityAnalyzer",
    "DatasetVersionInfo",
    "DatasetVersionManager",
    "TrainingDatasetResult",
    "DatasetAssembler",
]