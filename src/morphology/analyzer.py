"""
Per-sample morphology analyzer and spatial aggregation for Module 17
(src.morphology package).

MorphologyAnalyzer orchestrates the per-sample pipeline:
    SamplePrediction -> MorphologyStatisticsComputer -> ClassMorphologyMetrics
                     -> GeometryAnalyzer (with confidence map) -> GeometryMetrics
                     -> UncertaintyAnalyzer                    -> UncertaintyMetrics
                     -> SampleAnalysis

Refinement: confidence map is now forwarded to GeometryAnalyzer so that each
ConnectedRegionStats receives mean_confidence from the actual model output.

SpatialAggregator groups SampleAnalysis objects by aoi_id, reach_id,
river_name, and basin_id and computes SpatialSummary objects at each level.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np

from src.morphology.contracts import (
    AnalyticsConfig,
    SampleAnalysis,
    SpatialSummary,
)
from src.morphology.geometry import GeometryAnalyzer
from src.morphology.statistics import MorphologyStatisticsComputer
from src.morphology.uncertainty import UncertaintyAnalyzer

__all__ = ["MorphologyAnalyzer", "SpatialAggregator"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


class MorphologyAnalyzer:
    """
    Converts one SamplePrediction into a SampleAnalysis.

    Args:
        config:      AnalyticsConfig.
        class_names: Ordered class names from InferenceResult.
    """

    def __init__(
        self,
        config:      AnalyticsConfig,
        class_names: tuple[str, ...],
    ) -> None:
        self._config       = config
        self._class_names  = class_names
        self._stats_comp   = MorphologyStatisticsComputer(config, class_names)
        self._geo_analyzer = GeometryAnalyzer(config, class_names)
        self._unc_analyzer = UncertaintyAnalyzer(config, class_names)

    def analyze(self, prediction: Any) -> SampleAnalysis:
        """
        Analyze one SamplePrediction.

        Args:
            prediction: SamplePrediction from InferenceResult.predictions.

        Returns:
            Frozen SampleAnalysis.
        """
        mask       = np.asarray(prediction.predicted_mask, dtype=np.uint8)
        confidence = np.asarray(prediction.confidence,     dtype=np.float32)

        # Core class metrics (includes confidence-weighted area).
        class_metrics = self._stats_comp.compute(mask, confidence)

        # Geometry: confidence map is forwarded for per-region confidence.
        geometry = (
            self._geo_analyzer.compute(mask, confidence)
            if self._config.compute_geometry
            else None
        )

        # Uncertainty.
        uncertainty = (
            self._unc_analyzer.compute(confidence, mask)
            if self._config.compute_uncertainty
            else None
        )

        return SampleAnalysis(
            sample_id         = str(getattr(prediction, "sample_id",        "")),
            acquisition_date  = str(getattr(prediction, "acquisition_date",  "")),
            season            = str(getattr(prediction, "season",            "")),
            hydrological_year = int(getattr(prediction, "hydrological_year", 0)),
            sensor            = str(getattr(prediction, "sensor",            "")),
            river_name        = str(getattr(prediction, "river_name",        "")),
            reach_id          = str(getattr(prediction, "reach_id",          "")),
            basin_id          = str(getattr(prediction, "basin_id",          "")),
            aoi_id            = str(getattr(prediction, "aoi_id",            "")),
            total_pixels      = int(mask.size),
            class_metrics     = class_metrics,
            geometry          = geometry,
            uncertainty       = uncertainty,
        )


class SpatialAggregator:
    """
    Groups SampleAnalysis objects by spatial hierarchy and builds SpatialSummary.

    Hierarchy levels: aoi, reach, river, basin.

    Args:
        class_names: Ordered class names.
    """

    def __init__(self, class_names: tuple[str, ...]) -> None:
        self._class_names = class_names

    def compute(
        self,
        analyses: list[SampleAnalysis],
    ) -> dict[str, SpatialSummary]:
        """
        Build SpatialSummary for every spatial grouping.

        Returns:
            Dict "{group_type}:{group_id}" -> SpatialSummary.
        """
        result: dict[str, SpatialSummary] = {}

        for group_type, id_fn in [
            ("aoi",    lambda a: a.aoi_id),
            ("reach",  lambda a: a.reach_id),
            ("river",  lambda a: a.river_name),
            ("basin",  lambda a: a.basin_id),
        ]:
            groups: dict[str, list[SampleAnalysis]] = defaultdict(list)
            for analysis in analyses:
                gid = id_fn(analysis)
                if gid:
                    groups[gid].append(analysis)

            for gid, group in groups.items():
                key           = f"{group_type}:{gid}"
                result[key]   = self._build_summary(group_type, gid, group)

        return result

    def _build_summary(
        self,
        group_type: str,
        group_id:   str,
        analyses:   list[SampleAnalysis],
    ) -> SpatialSummary:
        total_pixels = sum(a.total_pixels for a in analyses)
        frac_lists: dict[str, list[float]] = {n: [] for n in self._class_names}
        for analysis in analyses:
            for class_name, cm in analysis.class_metrics.items():
                if class_name in frac_lists:
                    frac_lists[class_name].append(cm.area_fraction)

        mean_fracs = {
            name: float(np.mean(vals)) if vals else 0.0
            for name, vals in frac_lists.items()
        }

        return SpatialSummary(
            group_type           = group_type,
            group_id             = group_id,
            num_samples          = len(analyses),
            total_pixels         = total_pixels,
            mean_class_fractions = mean_fracs,
            sample_ids           = tuple(a.sample_id for a in analyses),
        )
