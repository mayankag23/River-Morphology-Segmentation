"""
RiverMorphologyEngine -- the single public interface for Module 17.

Usage
-----
    from src.morphology import RiverMorphologyEngine, AnalyticsConfig

    config = AnalyticsConfig(
        pixel_area_m2            = 900.0,   # 30m Landsat pixel
        low_confidence_threshold = 0.5,
        water_class              = "water",
        sand_class               = "sand",
        vegetation_class         = "vegetation",
    )
    engine = RiverMorphologyEngine(config)
    result = engine.analyze(inference_result)

    print(result.mean_water_fraction)
    for analysis in result.sample_analyses:
        print(analysis.sample_id, analysis.class_metrics["water"].pixel_count)

RiverMorphologyEngine orchestrates:
    AnalyticsValidator    -> pre-flight checks
    AnalyticsFactory      -> builds all analysis components
    MorphologyAnalyzer    -> per-sample SampleAnalysis
    TemporalAnalyzer      -> TemporalChange (consecutive date pairs)
    SeasonalAggregator    -> SeasonalSummary (grouped by season)
    SpatialAggregator     -> SpatialSummary (aoi / reach / river / basin)
    RiverMorphologyResult -> immutable public output
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from src.morphology.contracts import (
    AnalyticsConfig,
    RiverMorphologyResult,
    SampleAnalysis,
)
from src.morphology.factory import AnalyticsFactory
from src.morphology.validator import AnalyticsValidator

__all__ = ["RiverMorphologyEngine"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


class RiverMorphologyEngine:
    """
    Orchestrates a complete river morphology analytics run.

    Args:
        config: AnalyticsConfig or project Config object.
    """

    def __init__(self, config: Any) -> None:
        if isinstance(config, AnalyticsConfig):
            self._config = config
        else:
            self._config = AnalyticsConfig.from_config(config)
        self._validator = AnalyticsValidator()
        self._logger    = _LOGGER

    def analyze(
        self,
        inference_result: Any,
    ) -> RiverMorphologyResult:
        """
        Run a complete river morphology analysis.

        Args:
            inference_result: InferenceResult from Module 16.

        Returns:
            Frozen RiverMorphologyResult with all metrics and summaries.
        """
        ops: list[str] = []
        t0             = time.perf_counter()

        # Step 1: Pre-flight validation.
        validation = self._validator.validate(self._config, inference_result)
        for issue in validation.issues:
            self._logger.warning("RiverMorphologyEngine pre-flight: %s", issue)
        ops.append(f"validation: {len(validation.issues)} issue(s)")

        # Step 2: Build components.
        context     = AnalyticsFactory.build(self._config, inference_result)
        class_names = context["class_names"]
        ops.append(f"classes: {len(class_names)} ({', '.join(class_names)})")

        morph_analyzer  = context["morphology_analyzer"]
        temp_analyzer   = context["temporal_analyzer"]
        season_agg      = context["seasonal_aggregator"]
        spatial_agg     = context["spatial_aggregator"]

        # Step 3: Per-sample analysis.
        predictions = getattr(inference_result, "predictions", ())
        sample_analyses: list[SampleAnalysis] = []
        for pred in predictions:
            try:
                analysis = morph_analyzer.analyze(pred)
                sample_analyses.append(analysis)
            except Exception as exc:
                self._logger.warning(
                    "RiverMorphologyEngine: skipping sample '%s': %s",
                    getattr(pred, "sample_id", "unknown"), exc,
                )
        # Sort by acquisition_date for deterministic output.
        sample_analyses.sort(key=lambda a: a.acquisition_date)
        ops.append(f"per-sample: {len(sample_analyses)} analysed")

        # Step 4: Temporal change.
        temporal_changes = []
        if self._config.compute_temporal:
            temporal_changes = temp_analyzer.compute(sample_analyses)
        ops.append(f"temporal: {len(temporal_changes)} change records")

        # Step 5: Seasonal aggregation.
        seasonal_summaries: dict = {}
        if self._config.compute_seasonal:
            seasonal_summaries = season_agg.compute(sample_analyses)
        ops.append(f"seasonal: {len(seasonal_summaries)} season(s)")

        # Step 6: Spatial aggregation.
        spatial_summaries = spatial_agg.compute(sample_analyses)
        ops.append(f"spatial: {len(spatial_summaries)} group(s)")

        # Step 7: Dataset-level aggregate statistics.
        total_pixels   = sum(a.total_pixels for a in sample_analyses)
        mean_water     = self._mean_fraction(sample_analyses, self._config.water_class)
        mean_sand      = self._mean_fraction(sample_analyses, self._config.sand_class)
        mean_veg       = self._mean_fraction(sample_analyses, self._config.vegetation_class)
        mean_conf      = float(
            np.mean([
                a.uncertainty.mean_confidence
                for a in sample_analyses
                if a.uncertainty is not None
            ]) if any(a.uncertainty for a in sample_analyses) else
            getattr(inference_result, "mean_confidence", 0.0)
        )

        elapsed = time.perf_counter() - t0
        ops.append(f"total_time: {elapsed:.3f}s")

        result = RiverMorphologyResult(
            sample_analyses     = tuple(sample_analyses),
            temporal_changes    = tuple(temporal_changes),
            seasonal_summaries  = seasonal_summaries,
            spatial_summaries   = spatial_summaries,
            class_names         = class_names,
            num_samples         = len(sample_analyses),
            num_classes         = len(class_names),
            total_pixels        = total_pixels,
            mean_water_fraction = mean_water,
            mean_sand_fraction  = mean_sand,
            mean_veg_fraction   = mean_veg,
            mean_confidence     = mean_conf,
            analytics_config    = self._config,
            architecture        = str(getattr(inference_result, "architecture", "")),
            operations_log      = tuple(ops),
            analysis_time_s     = elapsed,
        )

        for line in result.summary_lines():
            self._logger.info(line)

        return result

    @staticmethod
    def _mean_fraction(
        analyses:   list[SampleAnalysis],
        class_name: str,
    ) -> float:
        """Compute mean area fraction for class_name across all samples."""
        fracs = [
            a.class_metrics[class_name].area_fraction
            for a in analyses
            if class_name in a.class_metrics
        ]
        return float(np.mean(fracs)) if fracs else 0.0
