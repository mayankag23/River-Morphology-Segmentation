"""
Public data contracts for the River Morphology Analytics Framework (Module 17).

Contract chain:
    InferenceResult  (Module 16)  ──> RiverMorphologyEngine.analyze() ──> RiverMorphologyResult

Package: src.morphology  (renamed from src.analytics for scientific clarity)

Design invariants
-----------------
- All public contracts are frozen dataclasses.
- No torch types anywhere in this module.
- Area values are in pixels by default; converted to m2 when pixel_area_m2 > 0.
- All floating-point fields default to 0.0 (never NaN).
- Class names come from InferenceResult.class_names — never hardcoded.
- Connected-region statistics are generic (not named "island" or "bar") so
  higher-level morphology metrics can be derived downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "AnalyticsConfig",
    "ConnectedRegionStats",
    "ClassRegionMetrics",
    "ClassMorphologyMetrics",
    "GeometryMetrics",
    "UncertaintyMetrics",
    "TemporalChange",
    "SeasonalSummary",
    "SpatialSummary",
    "SampleAnalysis",
    "RiverMorphologyResult",
]


# ==============================================================================
# AnalyticsConfig
# ==============================================================================

@dataclass(frozen=True)
class AnalyticsConfig:
    """
    Immutable analytics configuration.

    Attributes:
        pixel_area_m2:           Area of one pixel in square metres.
                                 0.0 = pixel-unit reporting only.
        pixel_width_m:           Pixel width in metres (GSD column direction).
                                 0.0 when not specified. Used for multi-resolution
                                 satellite imagery support.
        pixel_height_m:          Pixel height in metres (GSD row direction).
                                 0.0 when not specified.
        low_confidence_threshold: Pixels below this are flagged uncertain [0,1].
        min_component_area:      Minimum connected-region area (pixels) to
                                 include in region statistics.
        seasons:                 Dict season_name -> tuple of month numbers.
                                 Empty = use SamplePrediction.season directly.
        water_class:             Name of the water class.
        sand_class:              Name of the sand/exposed-sediment class.
        vegetation_class:        Name of the vegetation class.
        background_class:        Name of the background class.
        compute_geometry:        Enable connected-region geometry analysis.
        compute_temporal:        Enable temporal change computation.
        compute_seasonal:        Enable seasonal aggregation.
        compute_uncertainty:     Enable per-class confidence analysis.
        compute_shape_descriptors: When True, compute optional shape descriptors
                                 (perimeter, compactness, elongation, aspect ratio)
                                 for each connected region. Requires scipy.
                                 Slower than basic region analysis.
    """

    pixel_area_m2:             float             = 0.0
    pixel_width_m:             float             = 0.0
    pixel_height_m:            float             = 0.0
    low_confidence_threshold:  float             = 0.5
    min_component_area:        int               = 4
    seasons:                   dict[str, tuple]  = field(default_factory=dict)
    water_class:               str               = "water"
    sand_class:                str               = "sand"
    vegetation_class:          str               = "vegetation"
    background_class:          str               = "background"
    compute_geometry:          bool              = True
    compute_temporal:          bool              = True
    compute_seasonal:          bool              = True
    compute_uncertainty:       bool              = True
    compute_shape_descriptors: bool              = False

    @classmethod
    def from_config(cls, config: Any) -> AnalyticsConfig:
        """Build AnalyticsConfig from config.analytics (or config.morphology)."""
        # Accept both config.analytics and config.morphology for transition.
        acfg = getattr(config, "morphology", None) or getattr(config, "analytics", None)
        if acfg is None:
            return cls()
        raw_seasons = getattr(acfg, "seasons", {})
        seasons: dict[str, tuple] = {}
        if isinstance(raw_seasons, dict):
            for k, v in raw_seasons.items():
                seasons[str(k)] = tuple(int(m) for m in v)
        return cls(
            pixel_area_m2             = float(getattr(acfg, "pixel_area_m2",            0.0)),
            pixel_width_m             = float(getattr(acfg, "pixel_width_m",            0.0)),
            pixel_height_m            = float(getattr(acfg, "pixel_height_m",           0.0)),
            low_confidence_threshold  = float(getattr(acfg, "low_confidence_threshold", 0.5)),
            min_component_area        = int(getattr(acfg,   "min_component_area",       4)),
            seasons                   = seasons,
            water_class               = str(getattr(acfg,   "water_class",              "water")),
            sand_class                = str(getattr(acfg,   "sand_class",               "sand")),
            vegetation_class          = str(getattr(acfg,   "vegetation_class",         "vegetation")),
            background_class          = str(getattr(acfg,   "background_class",         "background")),
            compute_geometry          = bool(getattr(acfg,  "compute_geometry",         True)),
            compute_temporal          = bool(getattr(acfg,  "compute_temporal",         True)),
            compute_seasonal          = bool(getattr(acfg,  "compute_seasonal",         True)),
            compute_uncertainty       = bool(getattr(acfg,  "compute_uncertainty",      True)),
            compute_shape_descriptors = bool(getattr(acfg,  "compute_shape_descriptors", False)),
        )


# ==============================================================================
# ConnectedRegionStats
# ==============================================================================

@dataclass(frozen=True)
class ConnectedRegionStats:
    """
    Statistics for one connected region (connected component) of a single class.

    These are generic descriptors that apply equally to a water body, a sand
    region, or a vegetation patch. Higher-level morphology concepts (channel,
    bar, island, encroachment) are derived from these in downstream modules.

    Attributes:
        region_id:      Integer label of this region in the labeled array.
        area_px:        Area in pixels.
        area_m2:        Area in square metres. 0.0 when pixel_area_m2=0.
        bbox_row_min:   Bounding box top row index.
        bbox_row_max:   Bounding box bottom row index (exclusive).
        bbox_col_min:   Bounding box left column index.
        bbox_col_max:   Bounding box right column index (exclusive).
        bbox_height:    Bounding box height in pixels.
        bbox_width:     Bounding box width in pixels.
        aspect_ratio:   bbox_width / bbox_height. 0.0 when bbox_height == 0.
        mean_confidence: Mean model confidence for pixels in this region.
        perimeter_px:   Perimeter length in pixels (4-connectivity boundary).
                        0.0 when compute_shape_descriptors=False.
        compactness:    4*pi*area / perimeter^2. 1.0 for a circle, <1 for
                        elongated/complex shapes. 0.0 when perimeter == 0 or
                        shape descriptors disabled.
        elongation:     1 - (bbox_short_side / bbox_long_side). 0.0 for square,
                        approaches 1.0 for very elongated shapes. 0.0 when
                        shape descriptors disabled.
    """

    region_id:        int
    area_px:          int
    area_m2:          float
    bbox_row_min:     int
    bbox_row_max:     int
    bbox_col_min:     int
    bbox_col_max:     int
    bbox_height:      int
    bbox_width:       int
    aspect_ratio:     float
    mean_confidence:  float
    perimeter_px:     float = 0.0
    compactness:      float = 0.0
    elongation:       float = 0.0

    def as_dict(self) -> dict:
        return {
            "region_id":       self.region_id,
            "area_px":         self.area_px,
            "area_m2":         round(self.area_m2,         3),
            "bbox_row_min":    self.bbox_row_min,
            "bbox_row_max":    self.bbox_row_max,
            "bbox_col_min":    self.bbox_col_min,
            "bbox_col_max":    self.bbox_col_max,
            "bbox_height":     self.bbox_height,
            "bbox_width":      self.bbox_width,
            "aspect_ratio":    round(self.aspect_ratio,    6),
            "mean_confidence": round(self.mean_confidence, 6),
            "perimeter_px":    round(self.perimeter_px,    3),
            "compactness":     round(self.compactness,     6),
            "elongation":      round(self.elongation,      6),
        }


# ==============================================================================
# ClassRegionMetrics
# ==============================================================================

@dataclass(frozen=True)
class ClassRegionMetrics:
    """
    Aggregated connected-region statistics for one class in one sample.

    These generic statistics replace the domain-specific "island_count",
    "bar_count", etc. from the first draft. Higher-level concepts are derived
    in downstream analytical modules.

    Attributes:
        class_name:           Class name (e.g. "water", "sand").
        class_id:             Integer class label.
        region_count:         Number of connected regions >= min_component_area.
        largest_region_px:    Area of the largest connected region (pixels).
        largest_region_m2:    Area of the largest region in square metres.
        mean_region_size_px:  Mean area of all valid regions (pixels).
        std_region_size_px:   Standard deviation of region areas.
        fragmentation_index:  region_count / total_class_pixels if total > 0.
                              A higher value indicates more fragmented coverage.
                              0.0 when class has no pixels.
        estimated_width_px:   Mean row-width of the largest region. Useful as
                              a proxy for channel width (water) or bar width (sand).
                              0.0 when no regions present.
        regions:              Tuple of ConnectedRegionStats, sorted by area
                              descending (largest first). Includes only regions
                              >= min_component_area.
    """

    class_name:           str
    class_id:             int
    region_count:         int
    largest_region_px:    int
    largest_region_m2:    float
    mean_region_size_px:  float
    std_region_size_px:   float
    fragmentation_index:  float
    estimated_width_px:   float
    regions:              tuple[ConnectedRegionStats, ...]

    def as_dict(self) -> dict:
        return {
            "class_name":          self.class_name,
            "class_id":            self.class_id,
            "region_count":        self.region_count,
            "largest_region_px":   self.largest_region_px,
            "largest_region_m2":   round(self.largest_region_m2,    3),
            "mean_region_size_px": round(self.mean_region_size_px,  3),
            "std_region_size_px":  round(self.std_region_size_px,   3),
            "fragmentation_index": round(self.fragmentation_index,  6),
            "estimated_width_px":  round(self.estimated_width_px,   3),
            "regions":             [r.as_dict() for r in self.regions],
        }


# ==============================================================================
# ClassMorphologyMetrics
# ==============================================================================

@dataclass(frozen=True)
class ClassMorphologyMetrics:
    """
    Immutable morphology metrics for one segmentation class in one sample.

    Attributes:
        class_name:                Class name (e.g. "water", "sand").
        class_id:                  Integer class label.
        pixel_count:               Number of pixels classified as this class.
        area_fraction:             Fraction of total scene pixels [0, 1].
        total_fraction:            Alias for area_fraction (explicit denominator).
        area_m2:                   Area in square metres. 0.0 when pixel_area_m2=0.
        mean_confidence:           Mean model confidence for this class's pixels.
        low_conf_pixels:           Count of pixels below low_confidence_threshold.
        confidence_weighted_area:  Sum of confidence values over all class pixels.
                                   This is the confidence-weighted pixel count:
                                   sum(confidence[mask == class_id]).
                                   More reliable than raw pixel_count when model
                                   uncertainty is high. 0.0 for empty classes.
        confidence_weighted_area_m2: confidence_weighted_area * pixel_area_m2.
                                   0.0 when pixel_area_m2=0.
    """

    class_name:                 str
    class_id:                   int
    pixel_count:                int
    area_fraction:              float
    total_fraction:             float
    area_m2:                    float
    mean_confidence:            float
    low_conf_pixels:            int
    confidence_weighted_area:   float = 0.0
    confidence_weighted_area_m2: float = 0.0

    def as_dict(self) -> dict:
        return {
            "class_name":                  self.class_name,
            "class_id":                    self.class_id,
            "pixel_count":                 self.pixel_count,
            "area_fraction":               round(self.area_fraction,               6),
            "total_fraction":              round(self.total_fraction,              6),
            "area_m2":                     round(self.area_m2,                     3),
            "mean_confidence":             round(self.mean_confidence,             6),
            "low_conf_pixels":             self.low_conf_pixels,
            "confidence_weighted_area":    round(self.confidence_weighted_area,    3),
            "confidence_weighted_area_m2": round(self.confidence_weighted_area_m2, 3),
        }


# ==============================================================================
# GeometryMetrics
# ==============================================================================

@dataclass(frozen=True)
class GeometryMetrics:
    """
    Geometry metrics derived from connected-region analysis of the mask.

    The per-class region statistics (ClassRegionMetrics) replace the legacy
    island_count / bar_count / water_body_count fields. Those higher-level
    concepts are derivable from ClassRegionMetrics in downstream modules.

    Attributes:
        per_class_regions:       Dict class_name -> ClassRegionMetrics.
                                 Contains region statistics for every class.
        estimated_channel_width_px: Estimated wetted channel width in pixels
                                 (mean row-width of the largest water region).
                                 0.0 when no water regions present.
        pixel_width_m:           Pixel width in metres (from config). 0.0 if
                                 not configured.
        pixel_height_m:          Pixel height in metres (from config). 0.0 if
                                 not configured.
    """

    per_class_regions:           dict[str, ClassRegionMetrics]
    estimated_channel_width_px:  float
    pixel_width_m:               float
    pixel_height_m:              float

    def as_dict(self) -> dict:
        return {
            "per_class_regions":          {k: v.as_dict() for k, v in self.per_class_regions.items()},
            "estimated_channel_width_px": round(self.estimated_channel_width_px, 3),
            "pixel_width_m":              self.pixel_width_m,
            "pixel_height_m":             self.pixel_height_m,
        }


# ==============================================================================
# UncertaintyMetrics
# ==============================================================================

@dataclass(frozen=True)
class UncertaintyMetrics:
    """
    Uncertainty metrics summarising model confidence over one sample.

    Attributes:
        mean_confidence:      Mean confidence across all pixels.
        std_confidence:       Standard deviation of confidence.
        min_confidence:       Minimum confidence value.
        max_confidence:       Maximum confidence value.
        low_conf_pixel_count: Pixels below low_confidence_threshold.
        low_conf_fraction:    Fraction of total pixels with low confidence.
        per_class_confidence: Dict class_name -> mean confidence for that class.
    """

    mean_confidence:       float
    std_confidence:        float
    min_confidence:        float
    max_confidence:        float
    low_conf_pixel_count:  int
    low_conf_fraction:     float
    per_class_confidence:  dict[str, float]

    def as_dict(self) -> dict:
        return {
            "mean_confidence":      round(self.mean_confidence,  6),
            "std_confidence":       round(self.std_confidence,   6),
            "min_confidence":       round(self.min_confidence,   6),
            "max_confidence":       round(self.max_confidence,   6),
            "low_conf_pixel_count": self.low_conf_pixel_count,
            "low_conf_fraction":    round(self.low_conf_fraction, 6),
            "per_class_confidence": {
                k: round(v, 6) for k, v in self.per_class_confidence.items()
            },
        }


# ==============================================================================
# TemporalChange
# ==============================================================================

@dataclass(frozen=True)
class TemporalChange:
    """Change metrics between two consecutive acquisition dates for one class."""

    class_name:          str
    date_from:           str
    date_to:             str
    pixel_count_from:    int
    pixel_count_to:      int
    pixel_delta:         int
    area_fraction_from:  float
    area_fraction_to:    float
    fraction_delta:      float
    pct_change:          float

    def as_dict(self) -> dict:
        return {
            "class_name":         self.class_name,
            "date_from":          self.date_from,
            "date_to":            self.date_to,
            "pixel_count_from":   self.pixel_count_from,
            "pixel_count_to":     self.pixel_count_to,
            "pixel_delta":        self.pixel_delta,
            "area_fraction_from": round(self.area_fraction_from, 6),
            "area_fraction_to":   round(self.area_fraction_to,   6),
            "fraction_delta":     round(self.fraction_delta,      6),
            "pct_change":         round(self.pct_change,          4),
        }


# ==============================================================================
# SeasonalSummary
# ==============================================================================

@dataclass(frozen=True)
class SeasonalSummary:
    """Aggregated metrics for one season across multiple samples."""

    season:               str
    num_samples:          int
    mean_class_fractions: dict[str, float]
    std_class_fractions:  dict[str, float]
    sample_ids:           tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "season":               self.season,
            "num_samples":          self.num_samples,
            "mean_class_fractions": {k: round(v, 6) for k, v in self.mean_class_fractions.items()},
            "std_class_fractions":  {k: round(v, 6) for k, v in self.std_class_fractions.items()},
            "sample_ids":           list(self.sample_ids),
        }


# ==============================================================================
# SpatialSummary
# ==============================================================================

@dataclass(frozen=True)
class SpatialSummary:
    """Hierarchical spatial aggregation for one grouping key."""

    group_type:           str
    group_id:             str
    num_samples:          int
    total_pixels:         int
    mean_class_fractions: dict[str, float]
    sample_ids:           tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "group_type":           self.group_type,
            "group_id":             self.group_id,
            "num_samples":          self.num_samples,
            "total_pixels":         self.total_pixels,
            "mean_class_fractions": {k: round(v, 6) for k, v in self.mean_class_fractions.items()},
            "sample_ids":           list(self.sample_ids),
        }


# ==============================================================================
# SampleAnalysis
# ==============================================================================

@dataclass(frozen=True)
class SampleAnalysis:
    """Complete analytics for one SamplePrediction."""

    sample_id:          str
    acquisition_date:   str
    season:             str
    hydrological_year:  int
    sensor:             str
    river_name:         str
    reach_id:           str
    basin_id:           str
    aoi_id:             str
    total_pixels:       int
    class_metrics:      dict[str, ClassMorphologyMetrics]
    geometry:           GeometryMetrics | None
    uncertainty:        UncertaintyMetrics | None

    def as_dict(self) -> dict:
        return {
            "sample_id":         self.sample_id,
            "acquisition_date":  self.acquisition_date,
            "season":            self.season,
            "hydrological_year": self.hydrological_year,
            "sensor":            self.sensor,
            "river_name":        self.river_name,
            "reach_id":          self.reach_id,
            "basin_id":          self.basin_id,
            "aoi_id":            self.aoi_id,
            "total_pixels":      self.total_pixels,
            "class_metrics":     {k: v.as_dict() for k, v in self.class_metrics.items()},
            "geometry":          self.geometry.as_dict() if self.geometry else None,
            "uncertainty":       self.uncertainty.as_dict() if self.uncertainty else None,
        }


# ==============================================================================
# RiverMorphologyResult
# ==============================================================================

@dataclass(frozen=True)
class RiverMorphologyResult:
    """Immutable public output of RiverMorphologyEngine.analyze()."""

    sample_analyses:     tuple[SampleAnalysis, ...]
    temporal_changes:    tuple[TemporalChange, ...]
    seasonal_summaries:  dict[str, SeasonalSummary]
    spatial_summaries:   dict[str, SpatialSummary]
    class_names:         tuple[str, ...]
    num_samples:         int
    num_classes:         int
    total_pixels:        int
    mean_water_fraction: float
    mean_sand_fraction:  float
    mean_veg_fraction:   float
    mean_confidence:     float
    analytics_config:    AnalyticsConfig
    architecture:        str
    operations_log:      tuple[str, ...]
    analysis_time_s:     float

    def summary_lines(self) -> list[str]:
        return [
            f"  architecture:        {self.architecture}",
            f"  num_samples:         {self.num_samples}",
            f"  num_classes:         {self.num_classes}",
            f"  total_pixels:        {self.total_pixels:,}",
            f"  mean_water_fraction: {self.mean_water_fraction:.4f}",
            f"  mean_sand_fraction:  {self.mean_sand_fraction:.4f}",
            f"  mean_veg_fraction:   {self.mean_veg_fraction:.4f}",
            f"  mean_confidence:     {self.mean_confidence:.4f}",
            f"  temporal_changes:    {len(self.temporal_changes)}",
            f"  seasonal_summaries:  {len(self.seasonal_summaries)}",
            f"  spatial_summaries:   {len(self.spatial_summaries)}",
            f"  analysis_time_s:     {self.analysis_time_s:.2f}",
        ]

    def as_dict(self) -> dict:
        return {
            "num_samples":         self.num_samples,
            "num_classes":         self.num_classes,
            "total_pixels":        self.total_pixels,
            "class_names":         list(self.class_names),
            "architecture":        self.architecture,
            "mean_water_fraction": round(self.mean_water_fraction, 6),
            "mean_sand_fraction":  round(self.mean_sand_fraction,  6),
            "mean_veg_fraction":   round(self.mean_veg_fraction,   6),
            "mean_confidence":     round(self.mean_confidence,     6),
            "analysis_time_s":     round(self.analysis_time_s,     3),
            "operations_log":      list(self.operations_log),
            "sample_analyses":     [s.as_dict() for s in self.sample_analyses],
            "temporal_changes":    [t.as_dict() for t in self.temporal_changes],
            "seasonal_summaries":  {k: v.as_dict() for k, v in self.seasonal_summaries.items()},
            "spatial_summaries":   {k: v.as_dict() for k, v in self.spatial_summaries.items()},
        }
