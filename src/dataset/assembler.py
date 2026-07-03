"""
Dataset assembly orchestrator for the River Morphology Segmentation System
(Module 10).

DatasetAssembler is the single entry point for assembling validated patches
(Module 8) and validated labels (Module 9) into a complete, quality-
controlled, split training dataset ready for Module 11 (PyTorch Dataset).

Responsibility: orchestrate all six components. DatasetAssembler contains
no I/O, split, or statistics logic of its own.

Input:   list[PatchDatasetResult] + list[LabelDatasetResult]
Output:  TrainingDatasetResult (immutable)

Output layout:
    {output_dir}/
        version.json
        dataset_manifest.csv / .json
        train.csv
        validation.csv
        test.csv
        statistics.json
        quality_report.json
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.core.config import Config
from src.core.exceptions import InvalidValueError
from src.dataset.leakage import DataLeakageDetector, LeakageDetectionResult
from src.dataset.manifest import (
    DatasetManifest,
    DatasetManifestEntry,
    DatasetManifestManager,
    DatasetSample,
)
from src.dataset.quality import DatasetQualityAnalyzer, QualityReport
from src.dataset.splitter import DatasetSplitter, SplitResult
from src.dataset.statistics import DatasetStatisticsCalculator, SplitStatistics
from src.dataset.validator import DatasetValidationResult, DatasetValidator
from src.dataset.version import DatasetVersionInfo, DatasetVersionManager
from src.labels.schema import ClassSchema

if TYPE_CHECKING:
    from src.labels.manager import LabelDatasetResult
    from src.labels.manifest import LabelManifestEntry
    from src.patches.generator import PatchDatasetResult
    from src.patches.manifest import PatchManifestEntry

__all__ = ["TrainingDatasetResult", "DatasetAssembler"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


# ==============================================================================
# TrainingDatasetResult
# ==============================================================================

@dataclass(frozen=True)
class TrainingDatasetResult:
    """
    Immutable output of DatasetAssembler.assemble().

    All file paths are absolute. Every structured result from the pipeline
    components is retained for traceability.

    Attributes:
        output_dir:               Root directory of the assembled dataset.
        total_samples:             Total samples assembled (valid + excluded).
        train_samples:              Training split sample count.
        validation_samples:          Validation split sample count.
        test_samples:               Test split sample count.
        excluded_samples:            Samples excluded from all splits.
        split_strategy:              SplitStrategy value used.
        source_scenes:               Number of distinct source scenes.
        manifest:                    DatasetManifest with all file paths.
        train_statistics:             SplitStatistics for training split.
        validation_statistics:         SplitStatistics for validation split.
        test_statistics:              SplitStatistics for test split.
        overall_statistics:            SplitStatistics across all splits.
        quality_report:                DatasetQualityAnalyzer output.
        validation_result:             DatasetValidator output.
        leakage_detection:             DataLeakageDetector output.
        version_info:                   DatasetVersionInfo (lineage).
        version_path:                    Absolute path to version.json.
        statistics_path:                  Absolute path to statistics.json.
        quality_report_path:               Absolute path to quality_report.json.
        is_suitable_for_training:           True if no ERROR-severity QC issues.
        operations_log:                      Ordered tuple of operation descriptions.
    """

    output_dir:               Path
    total_samples:             int
    train_samples:              int
    validation_samples:          int
    test_samples:               int
    excluded_samples:            int
    split_strategy:              str
    source_scenes:               int
    manifest:                    DatasetManifest
    train_statistics:             SplitStatistics
    validation_statistics:         SplitStatistics
    test_statistics:              SplitStatistics
    overall_statistics:            SplitStatistics
    quality_report:                QualityReport
    validation_result:             DatasetValidationResult
    leakage_detection:             LeakageDetectionResult
    version_info:                   DatasetVersionInfo
    version_path:                    Path
    statistics_path:                  Path
    quality_report_path:               Path
    is_suitable_for_training:           bool
    operations_log:                      tuple[str, ...]

    def summary_lines(self) -> list[str]:
        """Return ASCII-formatted summary lines."""
        status = "[OK]  " if self.is_suitable_for_training else "[FAIL]"
        return [
            f"  {status} total_samples:   {self.total_samples}",
            f"         train:          {self.train_samples}",
            f"         validation:      {self.validation_samples}",
            f"         test:            {self.test_samples}",
            f"         excluded:        {self.excluded_samples}",
            f"         strategy:        {self.split_strategy}",
            f"         quality_score:   {self.quality_report.overall_quality_score:.2f}",
        ]


# ==============================================================================
# DatasetAssembler
# ==============================================================================

class DatasetAssembler:
    """
    Assembles validated patches and labels into a training-ready dataset.

    Orchestrates six components: DatasetValidator, DatasetSplitter,
    DataLeakageDetector, DatasetStatisticsCalculator, DatasetQualityAnalyzer,
    DatasetManifestManager, DatasetVersionManager. Contains no I/O, split,
    or statistics logic itself.

    Args:
        config: Fully initialized Config object.
    """

    def __init__(self, config: Config) -> None:
        self._config  = config
        self._logger: logging.Logger = logging.getLogger(__name__)

        dataset_cfg = getattr(config, "dataset", None)
        quality_cfg = getattr(dataset_cfg, "quality", None)

        min_samples_per_split = int(
            getattr(quality_cfg, "min_samples_per_split", 5)
        )
        self._output_formats = [
            str(f).lower()
            for f in getattr(dataset_cfg, "output_formats", ["csv", "json"])
        ]
        self._read_masks = True  # compute pixel-level class statistics

        self._class_schema   = ClassSchema.from_config(config)
        self._validator      = DatasetValidator(config)
        self._splitter       = DatasetSplitter(config)
        self._leakage        = DataLeakageDetector()
        self._stats_calc     = DatasetStatisticsCalculator.from_config(
            self._class_schema, config
        )
        self._quality_analyzer = DatasetQualityAnalyzer(min_samples_per_split)
        self._version_mgr    = DatasetVersionManager(config)

        self._logger.debug("DatasetAssembler initialized.")

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def assemble(
        self,
        patch_results:  list[PatchDatasetResult],
        label_results:  list[LabelDatasetResult],
        output_dir:     Path,
        read_masks:     bool = True,
    ) -> TrainingDatasetResult:
        """
        Build and quality-control the complete training dataset.

        Args:
            patch_results:  One PatchDatasetResult per exported scene
                            (from Module 8).
            label_results:   One LabelDatasetResult per labeled scene
                             (from Module 9).
            output_dir:      Root directory for all output files. Created
                             if absent.
            read_masks:      When True, read mask files for per-class pixel
                             statistics. When False, pixel counts are omitted
                             (faster but incomplete statistics).

        Returns:
            Frozen TrainingDatasetResult with all results and file paths.

        Raises:
            InvalidValueError: No valid samples remain after QC.
        """
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        operations: list[str] = []

        # Step 1: Build DatasetSample objects by joining patches and labels.
        samples = self._build_samples(patch_results, label_results)
        source_scenes = len({s.scene_id for s in samples})
        operations.append(f"assemble: {len(samples)} samples from {source_scenes} scene(s)")
        self._logger.info(
            "Assembled %d samples from %d scene(s).", len(samples), source_scenes
        )

        if not samples:
            raise InvalidValueError(
                field="samples",
                value=0,
                reason=(
                    "No valid samples could be assembled. Ensure that "
                    "LabelManifestEntry.is_valid == True for at least one "
                    "patch_id present in PatchManifestEntry."
                ),
            )

        # Step 2: Validate the assembled dataset.
        validation_result = self._validator.validate(samples, check_files=False)
        operations.append(
            f"validate: {validation_result.valid_samples}/{validation_result.total_samples} valid"
        )

        # Step 3: Split into train / validation / test.
        split_result = self._splitter.split(samples)
        operations.append(
            f"split({split_result.strategy}): "
            f"train={split_result.train_count}, "
            f"val={split_result.validation_count}, "
            f"test={split_result.test_count}"
        )

        # Step 4: Verify no leakage.
        leakage_result = self._leakage.detect(
            split_result.train_samples,
            split_result.validation_samples,
            split_result.test_samples,
        )
        operations.append(f"leakage_check: {'PASS' if not leakage_result.has_leakage else 'FAIL'}")

        # Step 5: Compute per-split statistics.
        train_stats  = self._stats_calc.compute(
            list(split_result.train_samples), "train", read_masks=read_masks
        )
        val_stats    = self._stats_calc.compute(
            list(split_result.validation_samples), "validation", read_masks=read_masks
        )
        test_stats   = self._stats_calc.compute(
            list(split_result.test_samples), "test", read_masks=read_masks
        )
        overall_stats = self._stats_calc.compute(samples, "overall", read_masks=read_masks)
        operations.append("statistics: computed for all splits")

        stats_path = self._stats_calc.save_statistics(
            {"train": train_stats, "validation": val_stats,
             "test": test_stats, "overall": overall_stats},
            output_dir,
        )

        # Step 6: Quality report.
        quality_report = self._quality_analyzer.analyze(
            validation_result, leakage_result, train_stats, val_stats, test_stats
        )
        quality_path = self._quality_analyzer.save_report(quality_report, output_dir)
        operations.append(
            f"quality: score={quality_report.overall_quality_score:.2f}, "
            f"suitable={quality_report.is_suitable_for_training}"
        )

        # Step 7: Version info.
        version_info = self._version_mgr.generate(
            total_samples=len(samples),
            train_samples=split_result.train_count,
            validation_samples=split_result.validation_count,
            test_samples=split_result.test_count,
            excluded_samples=0,
            split_strategy=split_result.strategy,
            source_scenes=source_scenes,
        )
        version_path = self._version_mgr.save(version_info, output_dir)
        operations.append(f"version: {version_info.dataset_version}")

        # Step 8: Build manifest entries and write files.
        manifest_manager = DatasetManifestManager()
        for sample in split_result.train_samples:
            manifest_manager.add_entry(
                DatasetManifestEntry.from_sample(sample, "train")
            )
        for sample in split_result.validation_samples:
            manifest_manager.add_entry(
                DatasetManifestEntry.from_sample(sample, "validation")
            )
        for sample in split_result.test_samples:
            manifest_manager.add_entry(
                DatasetManifestEntry.from_sample(sample, "test")
            )

        manifest = manifest_manager.save(output_dir, formats=self._output_formats)
        operations.append(f"manifest: {manifest.entry_count} entries written")

        result = TrainingDatasetResult(
            output_dir=output_dir,
            total_samples=len(samples),
            train_samples=split_result.train_count,
            validation_samples=split_result.validation_count,
            test_samples=split_result.test_count,
            excluded_samples=0,
            split_strategy=split_result.strategy,
            source_scenes=source_scenes,
            manifest=manifest,
            train_statistics=train_stats,
            validation_statistics=val_stats,
            test_statistics=test_stats,
            overall_statistics=overall_stats,
            quality_report=quality_report,
            validation_result=validation_result,
            leakage_detection=leakage_result,
            version_info=version_info,
            version_path=version_path,
            statistics_path=stats_path,
            quality_report_path=quality_path,
            is_suitable_for_training=quality_report.is_suitable_for_training,
            operations_log=tuple(operations),
        )

        for line in result.summary_lines():
            self._logger.info(line)

        return result

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _build_samples(
        self,
        patch_results: list[PatchDatasetResult],
        label_results: list[LabelDatasetResult],
    ) -> list[DatasetSample]:
        """
        Join patch manifest entries and label manifest entries by patch_id.

        Only entries where LabelManifestEntry.is_valid == True are included.
        If a patch_id has a patch entry but no valid label entry, it is silently
        excluded (logged at debug level).

        Args:
            patch_results: PatchDatasetResult objects from Module 8.
            label_results:  LabelDatasetResult objects from Module 9.

        Returns:
            List of DatasetSample objects, one per valid (patch, label) pair.
        """
        # Index label entries by patch_id (valid only)
        label_index: dict[str, Any] = {}
        for lr in label_results:
            for entry in lr.manifest.entries:
                if entry.is_valid:
                    label_index[entry.patch_id] = entry

        samples: list[DatasetSample] = []
        for pr in patch_results:
            for patch_entry in pr.manifest.entries:
                label_entry = label_index.get(patch_entry.patch_id)
                if label_entry is None:
                    self._logger.debug(
                        "No valid label for patch_id=%s -- excluded.",
                        patch_entry.patch_id,
                    )
                    continue

                sample = DatasetSample(
                    sample_id=patch_entry.patch_id,
                    patch_id=patch_entry.patch_id,
                    scene_id=patch_entry.scene_id,
                    patch_path=patch_entry.patch_path,
                    mask_path=label_entry.mask_path,
                    crs=patch_entry.crs,
                    width=patch_entry.width,
                    height=patch_entry.height,
                    num_bands=patch_entry.num_bands,
                    row_index=patch_entry.row_index,
                    col_index=patch_entry.col_index,
                    patch_valid_pixel_ratio=patch_entry.valid_pixel_ratio,
                    label_valid_pixel_ratio=label_entry.valid_pixel_ratio,
                    num_classes_present=label_entry.num_classes_present,
                    acquisition_date=label_entry.acquisition_date,
                    year=label_entry.year,
                    month=label_entry.month,
                    season=label_entry.season,
                    hydrological_year=label_entry.hydrological_year,
                    sensor=label_entry.sensor,
                    river_name=label_entry.river_name,
                    reach_id=label_entry.reach_id,
                    basin_id=label_entry.basin_id,
                    aoi_id=label_entry.aoi_id,
                    label_version=label_entry.label_version,
                    annotator=label_entry.annotator,
                    confidence=label_entry.confidence,
                    confidence_source=label_entry.confidence_source,
                )
                samples.append(sample)

        return samples