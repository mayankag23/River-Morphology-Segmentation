"""
Pseudo-label generation orchestrator for Module 9.

PseudoLabelGenerator implements LabelGenerationStrategy and is registered
in LabelStrategyRegistry under the key "spectral_rules". It coordinates the
full per-patch classification pipeline:

    SpectralClassificationEngine  (classifier.py)
            |
    ConflictResolver               (conflicts.py)
            |
    MorphologyProcessor            (morphology.py)
            |
    QualityAssessment              (quality.py)
            |
    ConfidenceEstimator            (confidence.py)
            |
    write mask GeoTIFF
            |
    PseudoLabelResult

The mask GeoTIFF is written as a uint8 single-band file to:
    output_mask_path  (provided by LabelManager)

CRS and affine transform are copied from the source patch. Pixels that
could not be classified are written as nodata_value (default 255).

source_type = "pseudo_label" is the stable public identifier for this
category of automatically generated labels; it is NOT defined here.
generation_strategy = "spectral_rules" is this class's specific identifier,
recorded in LabelManifestEntry.generation_strategy.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.labels.classifier import SpectralClassificationEngine
from src.labels.confidence import ConfidenceEstimator
from src.labels.conflicts import ConflictResolver
from src.labels.contracts import (
    ClassificationContext,
    PseudoLabelResult,
    ReproducibilityMetadata,
)
from src.labels.morphology import MorphologyProcessor
from src.labels.quality import QualityAssessment
from src.labels.strategy import LabelStrategyRegistry

__all__ = ["PseudoLabelGenerator"]

_LOGGER: logging.Logger = logging.getLogger(__name__)

_STRATEGY_TYPE: str = "spectral_rules"


@LabelStrategyRegistry.register
class PseudoLabelGenerator:
    """
    Generates a pseudo-label mask for one patch by applying spectral rules.

    Implements LabelGenerationStrategy. Stateless after construction and
    reusable across multiple patches and calls to generate(). Each call reads
    one patch GeoTIFF, applies the full classification pipeline, writes a
    mask GeoTIFF, and returns a PseudoLabelResult.

    Registered in LabelStrategyRegistry under strategy_type="spectral_rules".

    Args:
        classification_engine:  SpectralClassificationEngine.
        conflict_resolver:       ConflictResolver.
        morphology_processor:    MorphologyProcessor.
        quality_assessment:       QualityAssessment.
        confidence_estimator:     ConfidenceEstimator.
        nodata_value:              Sentinel for unclassified/nodata pixels.
        rule_engine_version:       Version string for reproducibility metadata.
        feature_stack_version:     Feature stack schema version string.
        pipeline_version:          Full pipeline version string.
    """

    _STRATEGY_TYPE: str = _STRATEGY_TYPE

    def __init__(
        self,
        classification_engine: SpectralClassificationEngine,
        conflict_resolver:      ConflictResolver,
        morphology_processor:   MorphologyProcessor,
        quality_assessment:     QualityAssessment,
        confidence_estimator:   ConfidenceEstimator,
        nodata_value:           int = 255,
        rule_engine_version:     str = "1.0.0",
        feature_stack_version:   str = "1.0.0",
        pipeline_version:        str = "1.0.0",
    ) -> None:
        self._engine              = classification_engine
        self._resolver            = conflict_resolver
        self._morphology          = morphology_processor
        self._quality             = quality_assessment
        self._confidence          = confidence_estimator
        self._nodata_value        = int(nodata_value)
        self._rule_engine_version  = rule_engine_version
        self._feature_stack_version = feature_stack_version
        self._pipeline_version     = pipeline_version
        self._logger: logging.Logger = logging.getLogger(__name__)

    @property
    def strategy_type(self) -> str:
        """Stable strategy identifier. Recorded in LabelManifestEntry."""
        return self._STRATEGY_TYPE

    @classmethod
    def from_config(
        cls,
        config:       Any,
        class_schema:  Any,
    ) -> PseudoLabelGenerator:
        """
        Build a PseudoLabelGenerator from Config and ClassSchema.

        Args:
            config:       Fully initialized Config object.
            class_schema: ClassSchema from src.labels.schema.

        Returns:
            Configured PseudoLabelGenerator.
        """
        labels_cfg = getattr(config, "labels", None)
        gen_cfg    = getattr(labels_cfg, "generation", None)
        nodata     = int(getattr(labels_cfg, "nodata_value", 255))

        rule_engine_version  = str(getattr(gen_cfg, "rule_engine_version",   "1.0.0"))
        feature_stack_ver    = str(getattr(
            getattr(config, "export", None), "feature_schema_version", "1.0.0"
        ))
        pipeline_version     = str(getattr(
            getattr(config, "export", None), "pipeline_version", "1.0.0"
        ))

        return cls(
            classification_engine = SpectralClassificationEngine.from_config(config),
            conflict_resolver      = ConflictResolver.from_config(config),
            morphology_processor   = MorphologyProcessor.from_config(
                config, class_schema=class_schema
            ),
            quality_assessment     = QualityAssessment.from_config(config, class_schema),
            confidence_estimator   = ConfidenceEstimator.from_config(config),
            nodata_value           = nodata,
            rule_engine_version    = rule_engine_version,
            feature_stack_version  = feature_stack_ver,
            pipeline_version       = pipeline_version,
        )

    def generate(
        self,
        patch_path:       Path,
        patch_id:          str,
        output_mask_path:  Path,
        context:           ClassificationContext | None = None,
    ) -> PseudoLabelResult:
        """
        Run the full pseudo-label generation pipeline for one patch.

        Sequence:
            1. Classify spectral bands -> ClassificationResult (Level 2).
            2. Resolve conflicts       -> resolved_confidence_map (Level 3).
            3. Morphological cleanup   -> MorphologyResult.
            4. Quality assessment      -> QualityResult.
            5. Confidence estimation   -> ConfidenceResult (Level 3 metrics).
            6. Write mask GeoTIFF.
            7. Return PseudoLabelResult.

        Args:
            patch_path:        Path to the source feature-stack GeoTIFF.
            patch_id:           Unique patch identifier.
            output_mask_path:   Destination path for the generated mask GeoTIFF.
            context:            Optional ClassificationContext forwarded to
                               SpectralClassificationEngine and each rule.

        Returns:
            Immutable PseudoLabelResult.

        Raises:
            OSError: Patch cannot be read or mask cannot be written.
        """
        self._logger.debug("Generating pseudo-label for patch '%s'", patch_id)

        # Step 1: Classify (Level 2).
        classification = self._engine.classify(patch_path, context=context)

        # Step 2: Resolve conflicts; populates resolved_confidence_map (Level 3).
        resolved = self._resolver.resolve(classification)

        # Step 3: Morphological cleanup (class map only, no spectral access).
        morphed = self._morphology.process(resolved)

        # Step 4: Quality assessment.
        quality = self._quality.assess(morphed)

        # Step 5: Confidence estimation (uses Level 3).
        confidence = self._confidence.estimate(resolved, morphed.class_map)

        # Step 6: Write mask GeoTIFF.
        # Apply nodata_value to pixels that are zero AND were originally
        # unclassified/nodata in the input.
        final_mask = morphed.class_map.copy()
        src_nodata  = resolved.nodata_mask | resolved.unclassified_mask
        final_mask[(morphed.class_map == 0) & src_nodata] = self._nodata_value

        self._write_mask(
            mask_array=final_mask,
            output_path=output_mask_path,
            source_patch_path=patch_path,
        )

        # Build summary fields.
        indices_used: set[str] = set()
        for rr in classification.rule_results:
            indices_used.update(rr.bands_used)

        valid_classes = {
            int(v) for v in np.unique(final_mask)
            if int(v) != self._nodata_value
        }

        total_px  = int(final_mask.size)
        valid_px  = int(np.sum(final_mask != self._nodata_value))
        valid_ratio = valid_px / total_px if total_px > 0 else 0.0

        # Combine quality and confidence issues.
        all_issues = list(quality.issues)
        if confidence.mask_confidence < self._confidence.min_mask_confidence:
            all_issues.append(
                f"mask_confidence {confidence.mask_confidence:.3f} below "
                f"threshold {self._confidence.min_mask_confidence:.3f}"
            )
        is_acceptable = (
            quality.is_acceptable
            and confidence.mask_confidence >= self._confidence.min_mask_confidence
        )

        # Build reproducibility metadata.
        reproducibility = self._build_reproducibility(patch_path)

        # Read CRS from the written mask (ground truth after write).
        crs = self._read_crs(output_mask_path)

        result = PseudoLabelResult(
            patch_id=patch_id,
            mask_path=output_mask_path.resolve(),
            mask_confidence=confidence.mask_confidence,
            quality_score=quality.quality_score,
            is_acceptable=is_acceptable,
            num_classes_present=len(valid_classes),
            valid_pixel_ratio=valid_ratio,
            unclassified_ratio=quality.unclassified_ratio,
            spectral_indices_used=tuple(sorted(indices_used)),
            issues=tuple(all_issues),
            crs=crs,
            reproducibility=reproducibility,
        )

        self._logger.debug(
            "PseudoLabel '%s': quality=%.3f, confidence=%.3f, "
            "classes=%d, acceptable=%s",
            patch_id, quality.quality_score,
            confidence.mask_confidence,
            len(valid_classes), is_acceptable,
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _write_mask(
        self,
        mask_array:        np.ndarray,
        output_path:        Path,
        source_patch_path:  Path,
    ) -> None:
        """
        Write a uint8 mask as a single-band GeoTIFF.

        Copies CRS and affine transform from the source patch GeoTIFF.

        Args:
            mask_array:       (H, W) uint8 class map.
            output_path:       Destination path.
            source_patch_path: Source patch GeoTIFF for CRS / transform.

        Raises:
            OSError: rasterio is not installed or the write fails.
        """
        try:
            import rasterio
        except ImportError as exc:
            raise OSError("rasterio is not installed.") from exc

        try:
            with rasterio.open(source_patch_path) as src:
                crs       = src.crs
                transform = src.transform
                h, w      = src.height, src.width

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(
                output_path, "w",
                driver="GTiff", dtype="uint8",
                width=w, height=h, count=1,
                crs=crs, transform=transform,
                nodata=self._nodata_value,
                compress="LZW",
            ) as dst:
                dst.write(mask_array.astype(np.uint8), 1)

            self._logger.debug(
                "Mask written: %s (%dx%d)", output_path.name, w, h
            )
        except Exception as exc:
            raise OSError(
                f"Failed to write mask GeoTIFF '{output_path}': {exc}"
            ) from exc

    @staticmethod
    def _read_crs(path: Path) -> str:
        """Read CRS string from a GeoTIFF. Returns empty string on failure."""
        try:
            import rasterio
            with rasterio.open(path) as ds:
                return ds.crs.to_string() if ds.crs else ""
        except Exception:
            return ""

    def _build_reproducibility(
        self, patch_path: Path
    ) -> ReproducibilityMetadata | None:
        """Build reproducibility metadata. Returns None on any failure."""
        try:
            timestamp = datetime.now(timezone.utc).isoformat()
            # Configuration hash is intentionally omitted here because Config
            # is not stored on self (to avoid circular state). The rule_engine
            # version is sufficient for reproducibility tracing.
            return ReproducibilityMetadata(
                rule_engine_version=self._rule_engine_version,
                feature_stack_version=self._feature_stack_version,
                processing_pipeline_version=self._pipeline_version,
                configuration_hash=None,
                rule_configuration_hash=None,
                generation_timestamp=timestamp,
            )
        except Exception as exc:
            self._logger.warning(
                "Could not build ReproducibilityMetadata: %s", exc
            )
            return None

# """
# Pseudo-label generation orchestrator for Module 9.

# PseudoLabelGenerator coordinates the full spectral classification pipeline
# for one patch:

#     SpectralClassificationEngine  (classifier.py)
#             |
#     ConflictResolver               (conflicts.py)
#             |
#     MorphologyProcessor            (morphology.py)
#             |
#     QualityAssessment              (quality.py)
#             |
#     ConfidenceEstimator            (confidence.py)
#             |
#     write mask GeoTIFF             (rasterio direct)
#             |
#     PseudoLabelResult

# The mask GeoTIFF is written to:
#     {scene_labels_dir}/{patch_id}_mask.tif

# as uint8 single-band GeoTIFF with the same CRS and affine transform as
# the source patch. Unclassified pixels are written as nodata_value (255).
# """

# from __future__ import annotations

# import logging
# from pathlib import Path
# from typing import Any

# import numpy as np

# from src.labels.classifier import SpectralClassificationEngine
# from src.labels.confidence import ConfidenceEstimator
# from src.labels.conflicts import ConflictResolver
# from src.labels.contracts import PseudoLabelResult
# from src.labels.morphology import MorphologyProcessor
# from src.labels.quality import QualityAssessment

# __all__ = ["PseudoLabelGenerator"]

# _LOGGER: logging.Logger = logging.getLogger(__name__)


# class PseudoLabelGenerator:
#     """
#     Generates a pseudo-label mask for one patch by applying spectral rules.

#     The generator is stateless after construction and reusable across
#     multiple patches and calls to generate(). Each call reads one patch
#     GeoTIFF, applies the full classification pipeline, and writes a mask.

#     Args:
#         classification_engine:  SpectralClassificationEngine.
#         conflict_resolver:       ConflictResolver.
#         morphology_processor:    MorphologyProcessor.
#         quality_assessment:       QualityAssessment.
#         confidence_estimator:     ConfidenceEstimator.
#         nodata_value:              Sentinel for unclassified/nodata pixels.
#     """

#     def __init__(
#         self,
#         classification_engine: SpectralClassificationEngine,
#         conflict_resolver:      ConflictResolver,
#         morphology_processor:   MorphologyProcessor,
#         quality_assessment:     QualityAssessment,
#         confidence_estimator:   ConfidenceEstimator,
#         nodata_value:           int = 255,
#     ) -> None:
#         self._engine      = classification_engine
#         self._resolver    = conflict_resolver
#         self._morphology  = morphology_processor
#         self._quality     = quality_assessment
#         self._confidence  = confidence_estimator
#         self._nodata_value = int(nodata_value)
#         self._logger: logging.Logger = logging.getLogger(__name__)

#     @classmethod
#     def from_config(
#         cls,
#         config:       Any,
#         class_schema:  Any,
#     ) -> PseudoLabelGenerator:
#         """
#         Build a PseudoLabelGenerator from Config and ClassSchema.

#         Args:
#             config:       Fully initialized Config object.
#             class_schema: ClassSchema from src.labels.schema.

#         Returns:
#             Configured PseudoLabelGenerator.
#         """
#         nodata = int(getattr(getattr(config, "labels", None), "nodata_value", 255))
#         return cls(
#             classification_engine = SpectralClassificationEngine.from_config(config),
#             conflict_resolver      = ConflictResolver.from_config(config),
#             morphology_processor   = MorphologyProcessor.from_config(config),
#             quality_assessment     = QualityAssessment.from_config(config, class_schema),
#             confidence_estimator   = ConfidenceEstimator.from_config(config),
#             nodata_value           = nodata,
#         )

#     def generate(
#         self,
#         patch_path:        Path,
#         patch_id:           str,
#         output_mask_path:   Path,
#     ) -> PseudoLabelResult:
#         """
#         Run the full pseudo-label generation pipeline for one patch.

#         Sequence:
#             1. Read spectral bands from patch GeoTIFF.
#             2. Apply RuleEngine to get per-class confidence maps.
#             3. Resolve conflicts -> ClassificationResult.
#             4. Apply morphological cleanup -> MorphologyResult.
#             5. Assess quality -> QualityResult.
#             6. Estimate confidence -> ConfidenceResult.
#             7. Write mask GeoTIFF.
#             8. Return PseudoLabelResult.

#         Args:
#             patch_path:        Path to the source patch GeoTIFF.
#             patch_id:           Unique patch identifier.
#             output_mask_path:   Where to write the generated mask GeoTIFF.

#         Returns:
#             Immutable PseudoLabelResult.

#         Raises:
#             OSError: Patch cannot be read, or mask cannot be written.
#         """
#         self._logger.debug("Generating pseudo-label for %s", patch_id)

#         # Step 1 & 2: Classify.
#         classification = self._engine.classify(patch_path)

#         # Step 3: Resolve conflicts.
#         resolved = self._resolver.resolve(classification)

#         # Step 4: Morphological cleanup.
#         morphed = self._morphology.process(resolved)

#         # Step 5: Quality assessment.
#         quality = self._quality.assess(morphed)

#         # Step 6: Confidence estimation.
#         confidence = self._confidence.estimate(resolved, morphed.class_map)

#         # Step 7: Write mask.
#         # Write nodata_value to unclassified/nodata pixels.
#         final_mask = morphed.class_map.copy()
#         nodata_src = resolved.nodata_mask | resolved.unclassified_mask
#         # After morphology, some previously-unclassified pixels may have
#         # received a class; recompute based on the morphed map.
#         final_mask[
#             (morphed.class_map == 0) & nodata_src
#         ] = self._nodata_value

#         self._write_mask(
#             mask_array=final_mask,
#             output_path=output_mask_path,
#             source_patch_path=patch_path,
#         )

#         # Collect spectral indices that contributed.
#         indices_used: set[str] = set()
#         for rr in classification.rule_results:
#             indices_used.update(rr.bands_used)

#         # Determine num_classes_present after morphological processing.
#         valid_classes = {
#             int(v) for v in np.unique(final_mask)
#             if int(v) != self._nodata_value
#         }

#         total_px = int(final_mask.size)
#         valid_px = int(np.sum(final_mask != self._nodata_value))
#         valid_ratio = valid_px / total_px if total_px > 0 else 0.0

#         all_issues = list(quality.issues)
#         if confidence.mask_confidence < self._confidence._cfg.min_mask_confidence:
#             all_issues.append(
#                 f"mask_confidence {confidence.mask_confidence:.3f} below "
#                 f"threshold {self._confidence._cfg.min_mask_confidence:.3f}"
#             )
#         is_acceptable = (
#             quality.is_acceptable
#             and confidence.mask_confidence
#             >= self._confidence._cfg.min_mask_confidence
#         )

#         result = PseudoLabelResult(
#             patch_id=patch_id,
#             mask_path=output_mask_path.resolve(),
#             mask_confidence=confidence.mask_confidence,
#             quality_score=quality.quality_score,
#             is_acceptable=is_acceptable,
#             num_classes_present=len(valid_classes),
#             valid_pixel_ratio=valid_ratio,
#             unclassified_ratio=quality.unclassified_ratio,
#             spectral_indices_used=tuple(sorted(indices_used)),
#             issues=tuple(all_issues),
#             crs=classification.rule_results[0].bands_used[0]
#             if False else self._read_crs(patch_path),
#         )

#         self._logger.debug(
#             "PseudoLabel %s: quality=%.2f, confidence=%.2f, "
#             "classes=%d, acceptable=%s",
#             patch_id, quality.quality_score,
#             confidence.mask_confidence,
#             len(valid_classes), is_acceptable,
#         )
#         return result

#     # ------------------------------------------------------------------
#     # Private helpers
#     # ------------------------------------------------------------------

#     def _write_mask(
#         self,
#         mask_array:        np.ndarray,
#         output_path:        Path,
#         source_patch_path:  Path,
#     ) -> None:
#         """
#         Write a uint8 mask as a single-band GeoTIFF.

#         Copies CRS and affine transform from the source patch.

#         Args:
#             mask_array:       (H, W) uint8 class map.
#             output_path:       Destination path.
#             source_patch_path: Source patch GeoTIFF for CRS/transform.

#         Raises:
#             OSError: rasterio is not installed or write fails.
#         """
#         try:
#             import rasterio
#         except ImportError as exc:
#             raise OSError("rasterio is not installed.") from exc

#         try:
#             with rasterio.open(source_patch_path) as src:
#                 crs        = src.crs
#                 transform  = src.transform
#                 h, w       = src.height, src.width

#             output_path.parent.mkdir(parents=True, exist_ok=True)
#             with rasterio.open(
#                 output_path, "w",
#                 driver="GTiff", dtype="uint8",
#                 width=w, height=h, count=1,
#                 crs=crs, transform=transform,
#                 nodata=self._nodata_value,
#                 compress="LZW",
#             ) as dst:
#                 dst.write(mask_array.astype(np.uint8), 1)
#         except Exception as exc:
#             raise OSError(
#                 f"Failed to write mask GeoTIFF: {output_path}: {exc}"
#             ) from exc

#     @staticmethod
#     def _read_crs(patch_path: Path) -> str:
#         """Read CRS string from a patch GeoTIFF."""
#         try:
#             import rasterio
#             with rasterio.open(patch_path) as ds:
#                 return ds.crs.to_string() if ds.crs else ""
#         except Exception:
#             return ""