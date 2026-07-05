"""
Internal data contracts for the pseudo-label generation pipeline (Module 9).

These dataclasses carry data between pipeline stages:

    SpectralBandReader
        -> RuleEngine
        -> ConflictResolver
        -> MorphologyProcessor
        -> QualityAssessment
        -> ConfidenceEstimator
        -> PseudoLabelGenerator

Dataclasses containing numpy arrays are NOT frozen (numpy arrays cannot be
meaningfully hashed or compared for equality). All other dataclasses are
frozen. None of these types are part of the public API; they are internal
data transfer objects.

Three-level confidence model
-----------------------------
Level 1 -- Rule Evidence   (RuleResult.confidence)
    Per-pixel sigmoid evidence from one rule for its own class. Range (0, 1).

Level 2 -- Classification Confidence   (ClassificationResult.confidence_map)
    The winning rule evidence at each pixel before conflict resolution modifies
    any class assignments.

Level 3 -- Resolved Confidence   (ClassificationResult.resolved_confidence_map)
    Evidence corresponding to the class that won after conflict resolution.
    Equals confidence_map when strategy is highest_confidence. Recomputed by
    ConflictResolver when strategy is priority_order.
    None until ConflictResolver.resolve() has been called.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "ClassificationContext",
    "ReproducibilityMetadata",
    "SpectralBandData",
    "RuleResult",
    "ClassificationResult",
    "MorphologyResult",
    "QualityResult",
    "ConfidenceResult",
    "PseudoLabelResult",
]


# ==============================================================================
# ClassificationContext
# ==============================================================================

@dataclass
class ClassificationContext:
    """
    Optional temporal, sensor, and geographic context for classification.

    This dataclass is intentionally NOT frozen because future fields may
    include numpy arrays (e.g. previous_class_map) which cannot be hashed.

    All fields default to None. Current rules ignore all fields. Future rules
    (e.g. MonsoonWaterRule, SeasonalSandRule) inspect the relevant fields to
    apply season- or sensor-specific threshold adjustments without requiring
    architecture changes.

    Attributes:
        acquisition_date:          YYYY-MM-DD representative date of the imagery.
        season:                    Resolved season name (e.g. "monsoon").
        hydrological_year:         Resolved hydrological year integer.
        sensor:                    Sensor identifier (e.g. "L8", "L9").
        sensor_generation:         Sensor family (e.g. "landsat_8_9", "landsat_5_7").
        river_name:                River name for geographic context.
        river_type:                River morphology type (e.g. "braided", "meandering").
        aoi_id:                    Area-of-interest identifier.
        previous_class_map:        (H, W) uint8 class map from the immediately
                                   preceding acquisition at this AOI. None when
                                   no prior observation is available.
        prior_confidence_map:      (H, W) float32 confidence map from the preceding
                                   acquisition. None when unavailable.
        num_prior_observations:    Count of prior acquisitions at this AOI.
    """

    # Temporal
    acquisition_date:      str | None = None
    season:                str | None = None
    hydrological_year:     int | None = None

    # Sensor
    sensor:                str | None = None
    sensor_generation:     str | None = None

    # Geographic / river
    river_name:            str | None = None
    river_type:            str | None = None
    aoi_id:                str | None = None

    # Prior observation (future temporal consistency)
    previous_class_map:      np.ndarray | None = None
    prior_confidence_map:    np.ndarray | None = None
    num_prior_observations:  int | None        = None


# ==============================================================================
# ReproducibilityMetadata
# ==============================================================================

@dataclass(frozen=True)
class ReproducibilityMetadata:
    """
    Immutable provenance record for experiment reproducibility.

    Every generated pseudo-label carries a ReproducibilityMetadata instance
    so that a mask can be fully reconstructed from its source patch, config,
    and pipeline version without re-running the full pipeline.

    Attributes:
        rule_engine_version:           Version string of the rule engine.
        feature_stack_version:          Version of the spectral feature schema
                                        (from config.export.feature_schema_version).
        processing_pipeline_version:     Full pipeline version string
                                        (from config.export.pipeline_version).
        configuration_hash:              8-char SHA-256 prefix of the full config.
                                        None if hashing failed.
        rule_configuration_hash:          8-char SHA-256 prefix of config.labels.rules.
                                        None if hashing failed.
        generation_timestamp:             ISO 8601 UTC timestamp of mask generation.
    """

    rule_engine_version:          str
    feature_stack_version:         str
    processing_pipeline_version:    str
    configuration_hash:             str | None
    rule_configuration_hash:         str | None
    generation_timestamp:            str


# ==============================================================================
# SpectralBandData
# ==============================================================================

@dataclass
class SpectralBandData:
    """
    Spectral band arrays extracted from one patch GeoTIFF.

    Attributes:
        bands:      Dict mapping band name -> (H, W) float32 ndarray.
                    NaN where input data is absent (nodata pixels).
        height:     Raster height in pixels.
        width:      Raster width in pixels.
        crs:        CRS string from the source file.
        transform:  rasterio Affine transform from the source file.
        band_names: Ordered tuple of all available band names.
    """

    bands:      dict[str, np.ndarray]
    height:     int
    width:      int
    crs:        str
    transform:  Any            # rasterio Affine
    band_names: tuple[str, ...]


# ==============================================================================
# RuleResult
# ==============================================================================

@dataclass
class RuleResult:
    """
    Result from applying one ClassificationRule to a patch (Level 1 evidence).

    Attributes:
        class_id:      Integer class label this rule votes for.
        class_name:    Human-readable class name.
        confidence:    (H, W) float32 per-pixel sigmoid evidence in (0, 1).
                       Zero where the rule's min_confidence threshold was not met.
        pixel_mask:    (H, W) bool: True where confidence >= min_confidence.
        bands_used:    Band names that contributed to this result.
        bands_missing: Band names the rule expected but were absent.
    """

    class_id:      int
    class_name:    str
    confidence:    np.ndarray     # (H, W) float32  -- Level 1 evidence
    pixel_mask:    np.ndarray     # (H, W) bool
    bands_used:    tuple[str, ...]
    bands_missing: tuple[str, ...]


# ==============================================================================
# ClassificationResult
# ==============================================================================

@dataclass
class ClassificationResult:
    """
    Result after all rules have been applied by SpectralClassificationEngine.

    Attributes:
        class_map:               (H, W) uint8 -- class ID per pixel (Level 2).
        confidence_map:          (H, W) float32 -- winning rule evidence before
                                 conflict resolution (Level 2).
        rule_results:            All individual rule outputs (Level 1 evidence),
                                 used for agreement score computation.
        unclassified_mask:       (H, W) bool -- True where no rule won.
        nodata_mask:             (H, W) bool -- True where input was nodata/NaN.
        resolved_confidence_map: (H, W) float32 -- evidence for the class that
                                 won after ConflictResolver has run (Level 3).
                                 None until ConflictResolver.resolve() is called.
                                 Equals confidence_map for highest_confidence
                                 strategy; recomputed for priority_order.
    """

    class_map:               np.ndarray       # (H, W) uint8
    confidence_map:          np.ndarray       # (H, W) float32  -- Level 2
    rule_results:            list[RuleResult]
    unclassified_mask:       np.ndarray       # (H, W) bool
    nodata_mask:             np.ndarray       # (H, W) bool
    resolved_confidence_map: np.ndarray | None = None   # (H, W) float32  -- Level 3


# ==============================================================================
# MorphologyResult
# ==============================================================================

@dataclass
class MorphologyResult:
    """
    Result after morphological post-processing.

    MorphologyProcessor never reads spectral bands; it operates exclusively
    on the integer class map and passes the confidence map through unchanged.

    Attributes:
        class_map:           (H, W) uint8 -- cleaned class map.
        operations_applied:  Human-readable descriptions of operations run.
    """

    class_map:          np.ndarray     # (H, W) uint8
    operations_applied: list[str]


# ==============================================================================
# QualityResult
# ==============================================================================

@dataclass
class QualityResult:
    """
    Quality assessment for one generated mask.

    NOT frozen -- tests assert isinstance(res.issues, list) and mutate fields.

    Attributes:
        quality_score:          Composite quality score in [0.0, 1.0].
        is_acceptable:           True if all quality thresholds are satisfied.
        valid_pixel_ratio:        Fraction of pixels with a valid class label.
        unclassified_ratio:        Fraction of pixels with no class label.
        class_pixel_fractions:     Dict of class_name -> pixel fraction.
        num_classes_present:       Count of distinct valid classes with >= min_class_pixels.
        issues:                    Human-readable quality problem descriptions.
        metric_scores:             Per-metric sub-scores for diagnostics.
                                  Keys: "valid_pixel_score", "unclassified_score",
                                        "class_coverage_score".
    """

    quality_score:         float
    is_acceptable:          bool
    valid_pixel_ratio:       float
    unclassified_ratio:      float
    class_pixel_fractions:   dict[str, float]
    num_classes_present:     int
    issues:                  list[str]
    metric_scores:           dict[str, float] = field(default_factory=dict)


# ==============================================================================
# ConfidenceResult
# ==============================================================================

@dataclass
class ConfidenceResult:
    """
    Confidence estimates for one generated mask.

    Attributes:
        pixel_confidence:   (H, W) float32 per-pixel resolved confidence (Level 3).
        mask_confidence:     Scalar mask-level confidence in [0.0, 1.0].
        agreement_score:     Fraction of valid pixels where two or more rules
                             agreed on the same winning class.
        component_scores:    Per-component scores for diagnostics.
                             Keys: "mask_confidence", "agreement_score".
    """

    pixel_confidence:  np.ndarray     # (H, W) float32  -- Level 3
    mask_confidence:   float
    agreement_score:   float
    component_scores:  dict[str, float] = field(default_factory=dict)


# ==============================================================================
# PseudoLabelResult
# ==============================================================================

@dataclass(frozen=True)
class PseudoLabelResult:
    """
    Immutable final result from PseudoLabelGenerator for one patch.

    This is the handoff object from the generation pipeline to LabelManager.
    LabelManager uses it to build LabelManifestEntry records.

    Attributes:
        patch_id:               Source patch identifier.
        mask_path:               Absolute path to the written mask GeoTIFF.
        mask_confidence:          Scalar mask-level resolved confidence (Level 3).
        quality_score:             Composite quality score from QualityAssessment.
        is_acceptable:             True if quality and confidence thresholds met.
        num_classes_present:       Distinct valid class IDs in the mask.
        valid_pixel_ratio:          Fraction of non-nodata pixels.
        unclassified_ratio:          Fraction of pixels that remained unclassified.
        spectral_indices_used:        Band/index names that contributed to classification.
        issues:                        Combined quality and confidence issues.
        crs:                           CRS string of the written mask GeoTIFF.
        reproducibility:               Provenance metadata. None when unavailable.
    """

    patch_id:               str
    mask_path:               Path
    mask_confidence:          float
    quality_score:             float
    is_acceptable:             bool
    num_classes_present:       int
    valid_pixel_ratio:          float
    unclassified_ratio:          float
    spectral_indices_used:        tuple[str, ...]
    issues:                        tuple[str, ...]
    crs:                           str
    reproducibility:               ReproducibilityMetadata | None = None
# """
# Internal data contracts for the pseudo-label generation pipeline (Module 9).

# Additions in this refinement:
#     ClassificationContext  -- Optional temporal/spatial context for future
#                               seasonal classification. Interfaces accept it now;
#                               no temporal logic is implemented yet.
#     ReproducibilityMetadata -- Records version, hash and timestamp information
#                               so every generated mask is fully traceable.
#     PseudoLabelResult      -- Extended with optional `reproducibility` field.
# """

# from __future__ import annotations

# from dataclasses import dataclass, field
# from pathlib import Path
# from typing import Any

# import numpy as np

# __all__ = [
#     "ClassificationContext",
#     "ReproducibilityMetadata",
#     "SpectralBandData",
#     "RuleResult",
#     "ClassificationResult",
#     "MorphologyResult",
#     "QualityResult",
#     "ConfidenceResult",
#     "PseudoLabelResult",
# ]


# # ==============================================================================
# # NEW: ClassificationContext
# # ==============================================================================

# @dataclass
# class ClassificationContext:
#     """
#     Optional temporal and spatial context for future seasonal classification.

#     This dataclass is intentionally not frozen because numpy arrays and
#     future fields (e.g., previous_class_map) cannot be hashed.

#     Purpose:
#         Provides a stable interface through which temporal information
#         can be added to classification in future releases without changing
#         the architecture. Every classification method currently accepts
#         context=None and ignores it. Future rule implementations (e.g.,
#         FloodedVegetationRule, SeasonalSandRule) can inspect it without
#         requiring method-signature changes elsewhere.

#     Current fields (all optional):
#         acquisition_date:    YYYY-MM-DD date string of the underlying imagery.
#         season:              Resolved season name (e.g., "monsoon").
#         hydrological_year:    Resolved hydrological year integer.

#     Future fields (NOT implemented yet — documented for interface design):
#         previous_class_map:  np.ndarray from the preceding time step.
#         confidence_from_previous: Float confidence carried from prior obs.
#         num_prior_observations: Count of prior acquisitions at this AOI.
#     """

#     acquisition_date:  str | None = None
#     season:            str | None = None
#     hydrological_year: int | None = None
#     # Future: previous_class_map, confidence_from_previous, etc.


# # ==============================================================================
# # NEW: ReproducibilityMetadata
# # ==============================================================================

# @dataclass(frozen=True)
# class ReproducibilityMetadata:
#     """
#     Immutable provenance record for experiment reproducibility.

#     Every generated pseudo-label carries a ReproducibilityMetadata instance
#     so that a mask can be fully reconstructed from its source patch, config,
#     and pipeline version — without needing to re-run the pipeline.

#     Attributes:
#         rule_engine_version:          Version of the rule engine component.
#         feature_stack_version:         Version of the spectral feature schema
#                                        (from config.export.feature_schema_version).
#         processing_pipeline_version:    Full pipeline version string
#                                        (from config.export.pipeline_version).
#         configuration_hash:             8-char SHA256 of the full config dict.
#                                        None if hashing failed.
#         rule_configuration_hash:         8-char SHA256 of config.labels.rules.
#                                        None if hashing failed.
#         generation_timestamp:            ISO 8601 UTC timestamp when this mask
#                                        was generated.
#     """

#     rule_engine_version:          str
#     feature_stack_version:         str
#     processing_pipeline_version:    str
#     configuration_hash:             str | None
#     rule_configuration_hash:         str | None
#     generation_timestamp:            str


# # ==============================================================================
# # Existing contracts (unchanged)
# # ==============================================================================

# @dataclass
# class SpectralBandData:
#     """
#     Spectral band arrays extracted from one patch GeoTIFF.

#     Attributes:
#         bands:      Dict mapping band name -> (H, W) float32 ndarray.
#                     NaN where data is absent (nodata pixels).
#         height:     Raster height in pixels.
#         width:      Raster width in pixels.
#         crs:        CRS string from the source file.
#         transform:  rasterio Affine transform from the source file.
#         band_names: Ordered tuple of all available band names.
#     """

#     bands:      dict[str, np.ndarray]
#     height:     int
#     width:      int
#     crs:        str
#     transform:  Any          # rasterio Affine
#     band_names: tuple[str, ...]


# @dataclass
# class RuleResult:
#     """
#     Result from applying one ClassificationRule to a patch.

#     Attributes:
#         class_id:      Integer class label this rule votes for.
#         class_name:    Human-readable class name.
#         confidence:    Per-pixel confidence in [0, 1], shape (H, W) float32.
#                        Zero where the rule's minimum confidence was not met.
#         pixel_mask:    Boolean mask (H, W): True where this rule proposes
#                        this class (confidence >= min_confidence).
#         bands_used:    Tuple of band names that contributed.
#         bands_missing: Bands the rule expected but were absent.
#     """

#     class_id:      int
#     class_name:    str
#     confidence:    np.ndarray     # (H, W) float32
#     pixel_mask:    np.ndarray     # (H, W) bool
#     bands_used:    tuple[str, ...]
#     bands_missing: tuple[str, ...]


# @dataclass
# class ClassificationResult:
#     """
#     Result after all rules have been applied and conflicts resolved.

#     Attributes:
#         class_map:         (H, W) uint8 -- class ID per pixel.
#         confidence_map:    (H, W) float32 -- winning rule's confidence.
#         rule_results:      All individual rule outputs.
#         unclassified_mask: (H, W) bool -- True where no rule won.
#         nodata_mask:       (H, W) bool -- True where input was nodata/NaN.
#     """

#     class_map:          np.ndarray     # (H, W) uint8
#     confidence_map:     np.ndarray     # (H, W) float32
#     rule_results:       list[RuleResult]
#     unclassified_mask:  np.ndarray     # (H, W) bool
#     nodata_mask:        np.ndarray     # (H, W) bool


# @dataclass
# class MorphologyResult:
#     """
#     Result after morphological post-processing.

#     Attributes:
#         class_map:           (H, W) uint8 -- cleaned class map.
#         operations_applied:  Human-readable descriptions.
#     """

#     class_map:          np.ndarray     # (H, W) uint8
#     operations_applied: list[str]


# @dataclass
# class QualityResult:
#     """
#     Quality assessment for one generated mask.

#     Attributes:
#         quality_score:          Overall quality score in [0.0, 1.0].
#         is_acceptable:           True if mask meets all quality thresholds.
#         valid_pixel_ratio:        Fraction of pixels with a valid class label.
#         unclassified_ratio:        Fraction of pixels with no class label.
#         class_pixel_fractions:     Dict of class_name -> pixel fraction.
#         num_classes_present:       Count of distinct valid classes.
#         issues:                    Human-readable quality problem descriptions.
#         metric_scores:             NEW — per-metric scores for diagnostics.
#     """

#     quality_score:         float
#     is_acceptable:          bool
#     valid_pixel_ratio:       float
#     unclassified_ratio:      float
#     class_pixel_fractions:   dict[str, float]
#     num_classes_present:     int
#     issues:                  list[str]
#     metric_scores:           dict[str, float] = field(default_factory=dict)


# @dataclass
# class ConfidenceResult:
#     """
#     Confidence estimates for one generated mask.

#     Attributes:
#         pixel_confidence:   (H, W) float32 per-pixel confidence.
#         mask_confidence:     Scalar mask-level confidence in [0.0, 1.0].
#         agreement_score:     Fraction of valid pixels where rules agreed.
#         component_scores:    NEW — per-component scores for diagnostics.
#     """

#     pixel_confidence:  np.ndarray     # (H, W) float32
#     mask_confidence:   float
#     agreement_score:   float
#     component_scores:  dict[str, float] = field(default_factory=dict)


# @dataclass(frozen=True)
# class PseudoLabelResult:
#     """
#     Immutable final result from PseudoLabelGenerator for one patch.

#     Attributes (unchanged from v1):
#         patch_id, mask_path, mask_confidence, quality_score,
#         is_acceptable, num_classes_present, valid_pixel_ratio,
#         unclassified_ratio, spectral_indices_used, issues, crs

#     New optional attribute:
#         reproducibility: ReproducibilityMetadata for experiment tracing.
#                          None only when generated by code that predates
#                          this refinement.
#     """

#     patch_id:              str
#     mask_path:              Path
#     mask_confidence:         float
#     quality_score:            float
#     is_acceptable:            bool
#     num_classes_present:       int
#     valid_pixel_ratio:          float
#     unclassified_ratio:          float
#     spectral_indices_used:        tuple[str, ...]
#     issues:                        tuple[str, ...]
#     crs:                           str
#     # Optional — backward-compatible default
#     reproducibility: ReproducibilityMetadata | None = None