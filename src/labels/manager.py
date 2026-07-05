"""
Label dataset orchestration for Module 9 (pseudo-label generation).

LabelManager generates pseudo-label masks for each patch in a
PatchDatasetResult by delegating to a LabelGenerationStrategy (selected
via LabelStrategyRegistry), then validates each mask, builds temporal
metadata, accumulates statistics, and writes the label manifest.

Public contract preserved:
    Input:  PatchDatasetResult + SceneMetadata
    Output: LabelDatasetResult  (identical to the v1 implementation)

source_type = "pseudo_label" is the stable public identifier for
automatically generated labels. The specific generation implementation is
recorded separately in LabelManifestEntry.generation_strategy.

Output layout:
    {output_dir}/
        label_manifest.csv
        label_manifest.json
        scenes/
            {scene_id}/
                labels/
                    {patch_id}_mask.tif   <-- generated pseudo-label
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.core.config import Config
from src.core.exceptions import InvalidValueError
from src.labels.contracts import ClassificationContext
from src.labels.manifest import (
    LabelManifest,
    LabelManifestEntry,
    LabelManifestManager,
)
from src.labels.schema import ClassSchema
from src.labels.statistics import LabelStatisticsCalculator
from src.labels.strategy import LabelGenerationStrategy, LabelStrategyRegistry
from src.labels.temporal import (
    HydrologicalYearResolver,
    SeasonResolver,
    TemporalMetadataBuilder,
    validate_temporal_consistency,
)
from src.labels.validator import LabelValidator

# Import concrete strategy so it is registered in LabelStrategyRegistry.
import src.labels.generator  # noqa: F401  -- registers PseudoLabelGenerator

if TYPE_CHECKING:
    from src.export.metadata import SceneMetadata
    from src.patches.generator import PatchDatasetResult

__all__ = ["LabelDatasetResult", "LabelManager"]

_LOGGER: logging.Logger = logging.getLogger(__name__)

_SCENES_SUBDIR: str = "scenes"
_LABELS_SUBDIR: str = "labels"

# Stable public identifier for automatically generated labels.
_SOURCE_TYPE: str = "pseudo_label"


# ==============================================================================
# LabelDatasetResult  (public contract -- unchanged from Module 9 v1)
# ==============================================================================

@dataclass(frozen=True)
class LabelDatasetResult:
    """
    Immutable output of LabelManager.generate().

    Public contract is identical to the previous Module 9 implementation
    so that Module 10 (DatasetAssembler) requires zero changes.

    Attributes:
        scene_id:            Source scene identifier.
        output_dir:            Root directory for the label dataset.
        scene_labels_dir:        Directory containing generated mask files.
        manifest:                Frozen LabelManifest snapshot.
        statistics:               Aggregated LabelStatistics.
        class_schema:             ClassSchema used for generation / validation.
        source_type:               Always "pseudo_label" for this implementation.
        labels_processed:          Total patches processed (excludes duplicates).
        labels_valid:              Count of masks that passed all checks.
        labels_rejected:            Count of masks that failed QC / validation.
        labels_missing:              Always 0 (masks are generated, not discovered).
        labels_duplicate:             Count of duplicate patch_ids skipped.
        operations_log:               Ordered tuple of operation descriptions.
    """

    scene_id:            str
    output_dir:            Path
    scene_labels_dir:        Path
    manifest:                 LabelManifest
    statistics:                Any   # LabelStatistics at runtime
    class_schema:               ClassSchema
    source_type:                 str
    labels_processed:            int
    labels_valid:                 int
    labels_rejected:               int
    labels_missing:                 int
    labels_duplicate:                int
    operations_log:                   tuple[str, ...]

    def summary_lines(self) -> list[str]:
        """Return ASCII-formatted summary lines (ASCII only, no Unicode)."""
        return [
            f"  scene_id:         {self.scene_id}",
            f"  source_type:       {self.source_type}",
            f"  labels processed:   {self.labels_processed}",
            f"  labels valid:        {self.labels_valid}",
            f"  labels rejected:      {self.labels_rejected}",
            f"  labels missing:        {self.labels_missing}",
            f"  labels duplicate:       {self.labels_duplicate}",
        ]


# ==============================================================================
# LabelManager
# ==============================================================================

class LabelManager:
    """
    Orchestrates pseudo-label generation, validation, and manifest writing.

    Delegates actual mask generation to a LabelGenerationStrategy selected
    by LabelStrategyRegistry (default: PseudoLabelGenerator / spectral_rules).
    Does not perform classification or morphological operations itself.

    Args:
        config: Fully initialized Config object.
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._logger: logging.Logger = logging.getLogger(__name__)

        self._class_schema        = ClassSchema.from_config(config)
        self._season_resolver     = SeasonResolver.from_config(config)
        self._hydro_year_resolver = HydrologicalYearResolver.from_config(config)
        self._temporal_builder    = TemporalMetadataBuilder(
            self._season_resolver, self._hydro_year_resolver, config
        )
        self._strategy: LabelGenerationStrategy = LabelStrategyRegistry.create(
            config, self._class_schema
        )

        labels_cfg = getattr(config, "labels", None)
        self._nodata_value           = int(getattr(labels_cfg, "nodata_value",           255))
        self._max_nodata_ratio       = float(getattr(labels_cfg, "max_nodata_ratio",     0.5))
        self._min_distinct_classes   = int(getattr(labels_cfg, "min_distinct_classes",   1))
        self._reject_single_class    = bool(getattr(labels_cfg, "reject_single_class_masks", False))
        self._mask_filename_pattern  = str(getattr(labels_cfg, "mask_filename_pattern",  "{patch_id}_mask.tif"))
        self._manifest_formats       = [
            str(f).lower()
            for f in getattr(labels_cfg, "output_formats", ["csv", "json"])
        ]

        gen_cfg = getattr(labels_cfg, "generation", None)
        self._pseudo_label_version = str(getattr(gen_cfg, "pseudo_label_version", "1.0.0"))

        self._validator = LabelValidator(
            class_schema               = self._class_schema,
            nodata_value               = self._nodata_value,
            max_nodata_ratio            = self._max_nodata_ratio,
            min_distinct_classes        = self._min_distinct_classes,
            reject_single_class_masks  = self._reject_single_class,
        )

        self._logger.debug(
            "LabelManager initialized. strategy=%s, classes=%d, nodata=%d",
            self._strategy.strategy_type,
            self._class_schema.num_classes,
            self._nodata_value,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        patch_dataset_result: PatchDatasetResult,
        scene_metadata:        SceneMetadata,
        output_dir:              Path,
        aoi_id:                  str,
        river_name:              str | None = None,
        reach_id:                 str | None = None,
        basin_id:                 str | None = None,
        label_version:           str | None = None,
        append_to_manifest:      bool = True,
    ) -> LabelDatasetResult:
        """
        Generate pseudo-labels for all patches in one scene.

        For each PatchManifestEntry:
            1. Build ClassificationContext from temporal metadata.
            2. Call LabelGenerationStrategy.generate() to classify and write mask.
            3. Validate the written mask with LabelValidator.
            4. Build TemporalMetadata and check temporal consistency.
            5. Accumulate statistics.
            6. Record LabelManifestEntry (including generation_strategy).
        Then persist the manifest and return LabelDatasetResult.

        Args:
            patch_dataset_result:  PatchDatasetResult from Module 8.
            scene_metadata:          SceneMetadata from Module 7.
            output_dir:               Root label dataset directory.
            aoi_id:                   AOI identifier embedded in metadata.
            river_name:               Optional river name.
            reach_id:                  Optional river reach identifier.
            basin_id:                  Optional drainage basin identifier.
            label_version:             Override label version string.
            append_to_manifest:         Load and extend existing manifest.

        Returns:
            Frozen LabelDatasetResult.

        Raises:
            InvalidValueError: patch_dataset_result has no patch entries.
        """
        scene_id      = patch_dataset_result.scene_id
        patch_entries = patch_dataset_result.manifest.entries
        if not patch_entries:
            raise InvalidValueError(
                field="patch_dataset_result.manifest.entries",
                value=0,
                reason="must contain at least one patch entry",
            )

        output_dir       = Path(output_dir).resolve()
        scene_labels_dir = output_dir / _SCENES_SUBDIR / scene_id / _LABELS_SUBDIR
        scene_labels_dir.mkdir(parents=True, exist_ok=True)

        operations: list[str] = []
        self._logger.info(
            "Generating pseudo-labels. scene_id=%s, patches=%d, strategy=%s",
            scene_id, len(patch_entries), self._strategy.strategy_type,
        )

        manifest_manager = LabelManifestManager()
        if append_to_manifest:
            manifest_manager.load_existing(output_dir)
        existing_ids = {e.patch_id for e in manifest_manager.entries}

        stats_calc = LabelStatisticsCalculator.from_config(self._class_schema, self._config)

        valid_count    = 0
        rejected_count  = 0
        duplicate_count = 0
        seen_this_run:  set[str] = set()

        for patch_entry in patch_entries:
            patch_id = patch_entry.patch_id

            # Duplicate check.
            if patch_id in existing_ids or patch_id in seen_this_run:
                duplicate_count += 1
                self._logger.warning("Skipping duplicate patch_id: %s", patch_id)
                continue
            seen_this_run.add(patch_id)

            # Build a ClassificationContext from scene metadata so that future
            # rules can use temporal and geographic information.
            context = self._build_context(
                scene_metadata=scene_metadata,
                aoi_id=aoi_id,
                river_name=river_name,
            )

            # Generate pseudo-label mask.
            mask_filename    = self._mask_filename_pattern.format(patch_id=patch_id)
            output_mask_path = scene_labels_dir / mask_filename

            try:
                pseudo_result = self._strategy.generate(
                    patch_path=Path(patch_entry.patch_path),
                    patch_id=patch_id,
                    output_mask_path=output_mask_path,
                    context=context,
                )
            except OSError as exc:
                self._logger.error(
                    "Failed to generate mask for '%s': %s", patch_id, exc
                )
                rejected_count += 1
                continue

            # Validate the generated mask against the source patch.
            validation = self._validator.validate(
                patch_path=Path(patch_entry.patch_path),
                mask_path=output_mask_path if output_mask_path.exists() else None,
            )

            # Temporal metadata.
            temporal = self._temporal_builder.build(
                scene_id=scene_id,
                patch_id=patch_id,
                scene_start_date=scene_metadata.start_date,
                scene_end_date=scene_metadata.end_date,
                sensors=scene_metadata.sensors,
                aoi_id=aoi_id,
                river_name=river_name,
                reach_id=reach_id,
                basin_id=basin_id,
                label_version=label_version or self._pseudo_label_version,
                annotator="spectral_rule_engine",
                confidence=pseudo_result.mask_confidence,
                confidence_source="automatic",
                processing_history=(
                    "spectral_classification",
                    "conflict_resolution",
                    "morphological_processing",
                    "quality_assessment",
                    "validated",
                ),
            )
            temporal_ok, temporal_issues = validate_temporal_consistency(
                temporal, self._season_resolver, self._hydro_year_resolver,
            )

            combined_issues = (
                validation.issues
                + temporal_issues
                + pseudo_result.issues
            )
            is_valid = (
                validation.is_valid
                and temporal_ok
                and pseudo_result.is_acceptable
            )

            # Read mask array for pixel-level statistics.
            mask_array = self._read_mask_array(output_mask_path)
            stats_calc.accumulate(
                mask_data=mask_array,
                validation_result=validation,
                temporal_metadata=temporal,
            )

            if is_valid:
                valid_count += 1
            else:
                rejected_count += 1

            entry = LabelManifestEntry(
                patch_id=patch_id,
                scene_id=scene_id,
                patch_path=str(patch_entry.patch_path),
                mask_path=str(output_mask_path),
                crs=patch_entry.crs,
                width=patch_entry.width,
                height=patch_entry.height,
                is_valid=is_valid,
                validation_issues="; ".join(combined_issues),
                num_classes_present=pseudo_result.num_classes_present,
                valid_pixel_ratio=pseudo_result.valid_pixel_ratio,
                source_type=_SOURCE_TYPE,
                acquisition_date=temporal.acquisition_date,
                year=temporal.year,
                month=temporal.month,
                season=temporal.season,
                hydrological_year=temporal.hydrological_year,
                sensor=temporal.sensor,
                river_name=temporal.river_name or "",
                reach_id=temporal.reach_id or "",
                basin_id=temporal.basin_id or "",
                aoi_id=temporal.aoi_id,
                label_version=temporal.label_version,
                annotator=temporal.annotator,
                confidence=pseudo_result.mask_confidence,
                confidence_source=temporal.confidence_source,
                processing_history=",".join(temporal.processing_history),
                created_at=datetime.now(timezone.utc).isoformat(),
                generation_strategy=self._strategy.strategy_type,
            )
            manifest_manager.add_entry(entry)

        operations.append(
            f"generated: {valid_count + rejected_count}, "
            f"duplicates_skipped: {duplicate_count}"
        )
        operations.append(
            f"valid: {valid_count}, rejected: {rejected_count}"
        )

        manifest = manifest_manager.save(output_dir, formats=self._manifest_formats)
        operations.append(f"write_manifest: {manifest.entry_count} total entries")

        statistics = stats_calc.compute()

        result = LabelDatasetResult(
            scene_id=scene_id,
            output_dir=output_dir,
            scene_labels_dir=scene_labels_dir,
            manifest=manifest,
            statistics=statistics,
            class_schema=self._class_schema,
            source_type=_SOURCE_TYPE,
            labels_processed=len(seen_this_run),
            labels_valid=valid_count,
            labels_rejected=rejected_count,
            labels_missing=0,
            labels_duplicate=duplicate_count,
            operations_log=tuple(operations),
        )

        for line in result.summary_lines():
            self._logger.info(line)

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_context(
        self,
        scene_metadata: SceneMetadata,
        aoi_id:          str,
        river_name:      str | None,
    ) -> ClassificationContext:
        """
        Build a ClassificationContext from SceneMetadata.

        The context is passed to LabelGenerationStrategy.generate() and
        forwarded to each ClassificationRule. Current rules ignore all fields;
        future rules may use them for temporal/sensor adaptation.
        """
        # Compute representative acquisition date (midpoint of scene range).
        try:
            from datetime import date, datetime as dt
            start = dt.strptime(scene_metadata.start_date, "%Y-%m-%d").date()
            end   = dt.strptime(scene_metadata.end_date,   "%Y-%m-%d").date()
            mid   = start + (end - start) // 2
            acq_date = mid.strftime("%Y-%m-%d")
            season   = self._season_resolver.resolve(mid.month)
            hydro_yr = self._hydro_year_resolver.resolve(mid.year, mid.month)
        except Exception:
            acq_date = scene_metadata.start_date
            season   = None
            hydro_yr = None

        sensor = ",".join(scene_metadata.sensors) if scene_metadata.sensors else None

        return ClassificationContext(
            acquisition_date=acq_date,
            season=season,
            hydrological_year=hydro_yr,
            sensor=sensor,
            sensor_generation=None,
            river_name=river_name,
            river_type=None,
            aoi_id=aoi_id,
            previous_class_map=None,
            prior_confidence_map=None,
            num_prior_observations=None,
        )

    @staticmethod
    def _read_mask_array(mask_path: Path) -> Any | None:
        """Read band 1 of a mask GeoTIFF. Returns None on any failure."""
        try:
            import rasterio
            with rasterio.open(mask_path) as ds:
                return ds.read(1)
        except Exception:
            return None


# """
# Label dataset orchestration for Module 9 (pseudo-label generation).

# LabelManager generates pseudo-label masks for each patch in a
# PatchDatasetResult by calling PseudoLabelGenerator, then validates each
# mask, builds temporal metadata, accumulates statistics, and writes the
# label manifest.

# Public contract preserved:
#     Input:  PatchDatasetResult + SceneMetadata
#     Output: LabelDatasetResult

# Internal change:
#     Labels are generated automatically from spectral features rather than
#     discovered from pre-existing mask files. No LabelSource abstraction
#     is required.

# Output layout:
#     {output_dir}/
#         label_manifest.csv
#         label_manifest.json
#         scenes/
#             {scene_id}/
#                 labels/
#                     {patch_id}_mask.tif   <-- generated pseudo-label
# """

# from __future__ import annotations

# import logging
# from dataclasses import dataclass
# from datetime import datetime, timezone
# from pathlib import Path
# from typing import TYPE_CHECKING, Any

# from src.core.config import Config
# from src.core.exceptions import InvalidValueError
# from src.labels.generator import PseudoLabelGenerator
# from src.labels.manifest import (
#     LabelManifest,
#     LabelManifestEntry,
#     LabelManifestManager,
# )
# from src.labels.schema import ClassSchema
# from src.labels.statistics import LabelStatisticsCalculator
# from src.labels.temporal import (
#     HydrologicalYearResolver,
#     SeasonResolver,
#     TemporalMetadataBuilder,
#     validate_temporal_consistency,
# )
# from src.labels.validator import LabelValidator

# if TYPE_CHECKING:
#     from src.export.metadata import SceneMetadata
#     from src.patches.generator import PatchDatasetResult

# __all__ = ["LabelDatasetResult", "LabelManager"]

# _LOGGER: logging.Logger = logging.getLogger(__name__)

# _SCENES_SUBDIR: str = "scenes"
# _LABELS_SUBDIR: str = "labels"

# _SOURCE_TYPE:         str = "pseudo_label"
# _GENERATION_METHOD:   str = "spectral_rules"


# # ==============================================================================
# # LabelDatasetResult  (public contract — unchanged from Module 9 v1)
# # ==============================================================================

# @dataclass(frozen=True)
# class LabelDatasetResult:
#     """
#     Immutable output of LabelManager.generate().

#     Public contract is identical to the previous Module 9 implementation
#     so that Module 10 (DatasetAssembler) requires zero changes.

#     Attributes:
#         scene_id:            Source scene identifier.
#         output_dir:            Root directory for the label dataset.
#         scene_labels_dir:        Directory containing generated mask files.
#         manifest:                Frozen LabelManifest snapshot.
#         statistics:               Aggregated LabelStatistics.
#         class_schema:             ClassSchema used for generation/validation.
#         source_type:               Always "pseudo_label" in this implementation.
#         labels_processed:          Total patches processed.
#         labels_valid:              Count of masks that passed all checks.
#         labels_rejected:            Count of masks that failed QC/validation.
#         labels_missing:              Always 0 (masks are generated, not discovered).
#         labels_duplicate:             Count of duplicate patch_ids skipped.
#         operations_log:               Ordered tuple of operation descriptions.
#     """

#     scene_id:            str
#     output_dir:            Path
#     scene_labels_dir:        Path
#     manifest:                 LabelManifest
#     statistics:                Any   # LabelStatistics
#     class_schema:               ClassSchema
#     source_type:                 str
#     labels_processed:            int
#     labels_valid:                 int
#     labels_rejected:               int
#     labels_missing:                 int
#     labels_duplicate:                int
#     operations_log:                   tuple[str, ...]

#     def summary_lines(self) -> list[str]:
#         """Return ASCII-formatted summary lines."""
#         return [
#             f"  scene_id:          {self.scene_id}",
#             f"  source_type:        {self.source_type}",
#             f"  labels processed:    {self.labels_processed}",
#             f"  labels valid:         {self.labels_valid}",
#             f"  labels rejected:       {self.labels_rejected}",
#             f"  labels missing:         {self.labels_missing}",
#             f"  labels duplicate:        {self.labels_duplicate}",
#         ]


# # ==============================================================================
# # LabelManager
# # ==============================================================================

# class LabelManager:
#     """
#     Orchestrates pseudo-label generation, validation, and manifest writing.

#     Calls PseudoLabelGenerator for each patch, applies LabelValidator,
#     builds TemporalMetadata, accumulates statistics, and writes the
#     label manifest. Does not perform classification or morphological
#     operations itself — those are delegated to PseudoLabelGenerator.

#     Args:
#         config: Fully initialized Config object.
#     """

#     def __init__(self, config: Config) -> None:
#         self._config = config
#         self._logger: logging.Logger = logging.getLogger(__name__)

#         self._class_schema       = ClassSchema.from_config(config)
#         self._season_resolver    = SeasonResolver.from_config(config)
#         self._hydro_year_resolver = HydrologicalYearResolver.from_config(config)
#         self._temporal_builder   = TemporalMetadataBuilder(
#             self._season_resolver, self._hydro_year_resolver, config
#         )
#         self._pseudo_generator   = PseudoLabelGenerator.from_config(config, self._class_schema)

#         labels_cfg = getattr(config, "labels", None)
#         self._nodata_value = int(getattr(labels_cfg, "nodata_value", 255))
#         self._max_nodata_ratio = float(getattr(labels_cfg, "max_nodata_ratio", 0.5))
#         self._min_distinct_classes = int(getattr(labels_cfg, "min_distinct_classes", 1))
#         self._reject_single_class_masks = bool(
#             getattr(labels_cfg, "reject_single_class_masks", False)
#         )
#         self._mask_filename_pattern = str(
#             getattr(labels_cfg, "mask_filename_pattern", "{patch_id}_mask.tif")
#         )
#         self._manifest_formats = [
#             str(f).lower()
#             for f in getattr(labels_cfg, "output_formats", ["csv", "json"])
#         ]

#         gen_cfg = getattr(labels_cfg, "generation", None)
#         self._pseudo_label_version   = str(getattr(gen_cfg, "pseudo_label_version",  "1.0.0"))
#         self._rule_engine_version    = str(getattr(gen_cfg, "rule_engine_version",   "1.0.0"))
#         self._generation_method      = str(getattr(gen_cfg, "generation_method",     _GENERATION_METHOD))

#         self._validator = LabelValidator(
#             class_schema                = self._class_schema,
#             nodata_value                = self._nodata_value,
#             max_nodata_ratio             = self._max_nodata_ratio,
#             min_distinct_classes          = self._min_distinct_classes,
#             reject_single_class_masks    = self._reject_single_class_masks,
#         )

#         self._logger.debug(
#             "LabelManager (pseudo-label mode) initialized. "
#             "classes=%d, nodata=%d, method=%s",
#             self._class_schema.num_classes,
#             self._nodata_value,
#             self._generation_method,
#         )

#     # ------------------------------------------------------------------
#     # Public API
#     # ------------------------------------------------------------------

#     def generate(
#         self,
#         patch_dataset_result: PatchDatasetResult,
#         scene_metadata:        SceneMetadata,
#         output_dir:              Path,
#         aoi_id:                  str,
#         river_name:              str | None = None,
#         reach_id:                 str | None = None,
#         basin_id:                 str | None = None,
#         label_version:           str | None = None,
#         append_to_manifest:      bool = True,
#     ) -> LabelDatasetResult:
#         """
#         Generate pseudo-labels for all patches in one scene.

#         For each PatchManifestEntry:
#             1. Run PseudoLabelGenerator to classify and write mask.
#             2. Validate mask with LabelValidator.
#             3. Build TemporalMetadata and check consistency.
#             4. Accumulate statistics.
#             5. Record LabelManifestEntry.

#         Args:
#             patch_dataset_result:  PatchDatasetResult from Module 8.
#             scene_metadata:          SceneMetadata from Module 7 (provides dates,
#                                     sensors for temporal metadata).
#             output_dir:               Root label dataset directory.
#             aoi_id:                   AOI identifier embedded in metadata.
#             river_name:               Optional river name.
#             reach_id:                  Optional reach identifier.
#             basin_id:                  Optional basin identifier.
#             label_version:             Override label version string.
#             append_to_manifest:         Load and extend existing manifest.

#         Returns:
#             Frozen LabelDatasetResult.

#         Raises:
#             InvalidValueError: patch_dataset_result has no patch entries.
#         """
#         scene_id      = patch_dataset_result.scene_id
#         patch_entries = patch_dataset_result.manifest.entries
#         if not patch_entries:
#             raise InvalidValueError(
#                 field="patch_dataset_result.manifest.entries",
#                 value=0,
#                 reason="must contain at least one patch entry",
#             )

#         output_dir        = Path(output_dir).resolve()
#         scene_labels_dir  = output_dir / _SCENES_SUBDIR / scene_id / _LABELS_SUBDIR
#         scene_labels_dir.mkdir(parents=True, exist_ok=True)

#         operations: list[str] = []
#         self._logger.info(
#             "Generating pseudo-labels. scene_id=%s, patches=%d",
#             scene_id, len(patch_entries),
#         )

#         manifest_manager = LabelManifestManager()
#         if append_to_manifest:
#             manifest_manager.load_existing(output_dir)
#         existing_ids = {e.patch_id for e in manifest_manager.entries}

#         stats_calc = LabelStatisticsCalculator.from_config(self._class_schema, self._config)

#         valid_count     = 0
#         rejected_count   = 0
#         duplicate_count  = 0
#         seen_this_run:   set[str] = set()

#         for patch_entry in patch_entries:
#             patch_id = patch_entry.patch_id

#             # Duplicate check
#             if patch_id in existing_ids or patch_id in seen_this_run:
#                 duplicate_count += 1
#                 self._logger.warning("Skipping duplicate patch_id: %s", patch_id)
#                 continue
#             seen_this_run.add(patch_id)

#             # Generate pseudo-label mask
#             mask_filename = self._mask_filename_pattern.format(patch_id=patch_id)
#             output_mask_path = scene_labels_dir / mask_filename

#             try:
#                 pseudo_result = self._pseudo_generator.generate(
#                     patch_path=Path(patch_entry.patch_path),
#                     patch_id=patch_id,
#                     output_mask_path=output_mask_path,
#                 )
#             except OSError as exc:
#                 self._logger.error(
#                     "Failed to generate mask for %s: %s", patch_id, exc
#                 )
#                 rejected_count += 1
#                 continue

#             # Validate generated mask against source patch
#             validation = self._validator.validate(
#                 patch_path=Path(patch_entry.patch_path),
#                 mask_path=output_mask_path if output_mask_path.exists() else None,
#             )

#             # Temporal metadata
#             temporal = self._temporal_builder.build(
#                 scene_id=scene_id,
#                 patch_id=patch_id,
#                 scene_start_date=scene_metadata.start_date,
#                 scene_end_date=scene_metadata.end_date,
#                 sensors=scene_metadata.sensors,
#                 aoi_id=aoi_id,
#                 river_name=river_name,
#                 reach_id=reach_id,
#                 basin_id=basin_id,
#                 label_version=label_version or self._pseudo_label_version,
#                 annotator="spectral_rule_engine",
#                 confidence=pseudo_result.mask_confidence,
#                 confidence_source="automatic",
#                 processing_history=(
#                     "spectral_classification",
#                     "conflict_resolution",
#                     "morphological_processing",
#                     "quality_assessment",
#                     "validated",
#                 ),
#             )
#             temporal_ok, temporal_issues = validate_temporal_consistency(
#                 temporal, self._season_resolver, self._hydro_year_resolver,
#             )

#             combined_issues = (
#                 validation.issues
#                 + temporal_issues
#                 + pseudo_result.issues
#             )
#             is_valid = (
#                 validation.is_valid
#                 and temporal_ok
#                 and pseudo_result.is_acceptable
#             )

#             # Read mask array for statistics
#             mask_array = self._read_mask_array(output_mask_path)
#             stats_calc.accumulate(
#                 mask_data=mask_array,
#                 validation_result=validation,
#                 temporal_metadata=temporal,
#             )

#             if is_valid:
#                 valid_count += 1
#             else:
#                 rejected_count += 1

#             entry = LabelManifestEntry(
#                 patch_id=patch_id,
#                 scene_id=scene_id,
#                 patch_path=str(patch_entry.patch_path),
#                 mask_path=str(output_mask_path),
#                 crs=patch_entry.crs,
#                 width=patch_entry.width,
#                 height=patch_entry.height,
#                 is_valid=is_valid,
#                 validation_issues="; ".join(combined_issues),
#                 num_classes_present=pseudo_result.num_classes_present,
#                 valid_pixel_ratio=pseudo_result.valid_pixel_ratio,
#                 source_type=_SOURCE_TYPE,
#                 acquisition_date=temporal.acquisition_date,
#                 year=temporal.year,
#                 month=temporal.month,
#                 season=temporal.season,
#                 hydrological_year=temporal.hydrological_year,
#                 sensor=temporal.sensor,
#                 river_name=temporal.river_name or "",
#                 reach_id=temporal.reach_id or "",
#                 basin_id=temporal.basin_id or "",
#                 aoi_id=temporal.aoi_id,
#                 label_version=temporal.label_version,
#                 annotator=temporal.annotator,
#                 confidence=pseudo_result.mask_confidence,
#                 confidence_source=temporal.confidence_source,
#                 processing_history=",".join(temporal.processing_history),
#                 created_at=datetime.now(timezone.utc).isoformat(),
#             )
#             manifest_manager.add_entry(entry)

#         operations.append(
#             f"generated: {valid_count + rejected_count}, "
#             f"duplicates_skipped: {duplicate_count}"
#         )
#         operations.append(
#             f"valid: {valid_count}, rejected: {rejected_count}"
#         )

#         manifest = manifest_manager.save(output_dir, formats=self._manifest_formats)
#         operations.append(f"write_manifest: {manifest.entry_count} total entries")

#         statistics = stats_calc.compute()

#         result = LabelDatasetResult(
#             scene_id=scene_id,
#             output_dir=output_dir,
#             scene_labels_dir=scene_labels_dir,
#             manifest=manifest,
#             statistics=statistics,
#             class_schema=self._class_schema,
#             source_type=_SOURCE_TYPE,
#             labels_processed=len(seen_this_run),
#             labels_valid=valid_count,
#             labels_rejected=rejected_count,
#             labels_missing=0,
#             labels_duplicate=duplicate_count,
#             operations_log=tuple(operations),
#         )

#         for line in result.summary_lines():
#             self._logger.info(line)

#         return result

#     # ------------------------------------------------------------------
#     # Private helpers
#     # ------------------------------------------------------------------

#     @staticmethod
#     def _read_mask_array(mask_path: Path) -> Any | None:
#         """Read band 1 of a mask GeoTIFF. Returns None on failure."""
#         try:
#             import rasterio
#             with rasterio.open(mask_path) as ds:
#                 return ds.read(1)
#         except Exception:
#             return None
