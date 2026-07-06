"""
src/morphology -- River Morphology Analytics Framework (Module 17).

Renamed from src/analytics/ to src/morphology/ for scientific clarity and
to avoid ambiguity with future reporting or business-analytics modules.

Single public entry point:
    RiverMorphologyEngine.analyze(inference_result) -> RiverMorphologyResult

Usage
-----
    from src.morphology import RiverMorphologyEngine, AnalyticsConfig

    config = AnalyticsConfig(
        pixel_area_m2             = 900.0,    # 30m Landsat pixel = 900 m^2
        pixel_width_m             = 30.0,     # GSD column
        pixel_height_m            = 30.0,     # GSD row
        low_confidence_threshold  = 0.5,
        water_class               = "water",
        sand_class                = "sand",
        vegetation_class          = "vegetation",
        compute_geometry          = True,
        compute_shape_descriptors = True,     # perimeter, compactness, elongation
    )
    engine = RiverMorphologyEngine(config)
    result = engine.analyze(inference_result)

    # Per-sample generic region statistics (class-agnostic).
    for analysis in result.sample_analyses:
        geo = analysis.geometry
        if geo:
            water_regions = geo.per_class_regions["water"]
            print(f"  water region count: {water_regions.region_count}")
            for region in water_regions.regions:
                print(f"    region {region.region_id}: "
                      f"{region.area_px}px, "
                      f"compactness={region.compactness:.3f}, "
                      f"conf={region.mean_confidence:.3f}")

    print(result.mean_water_fraction)
    print(result.as_dict())   # JSON-serializable
"""

# Primary entry point
from src.morphology.engine import RiverMorphologyEngine

# Public contracts
from src.morphology.contracts import (
    AnalyticsConfig,
    ConnectedRegionStats,
    ClassRegionMetrics,
    ClassMorphologyMetrics,
    GeometryMetrics,
    UncertaintyMetrics,
    TemporalChange,
    SeasonalSummary,
    SpatialSummary,
    SampleAnalysis,
    RiverMorphologyResult,
)

# Analysis components (for advanced / testing use)
from src.morphology.statistics import MorphologyStatisticsComputer
from src.morphology.geometry import GeometryAnalyzer
from src.morphology.uncertainty import UncertaintyAnalyzer
from src.morphology.temporal import TemporalAnalyzer, SeasonalAggregator
from src.morphology.analyzer import MorphologyAnalyzer, SpatialAggregator
from src.morphology.validator import AnalyticsValidator, AnalyticsValidationResult
from src.morphology.factory import AnalyticsFactory

__all__ = [
    # Primary
    "RiverMorphologyEngine",
    # Contracts
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
    # Components
    "MorphologyStatisticsComputer",
    "GeometryAnalyzer",
    "UncertaintyAnalyzer",
    "TemporalAnalyzer",
    "SeasonalAggregator",
    "MorphologyAnalyzer",
    "SpatialAggregator",
    "AnalyticsValidator",
    "AnalyticsValidationResult",
    "AnalyticsFactory",
]
