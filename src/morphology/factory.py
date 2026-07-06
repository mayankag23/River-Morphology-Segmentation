"""
Analytics factory for Module 17.

AnalyticsFactory assembles MorphologyAnalyzer, TemporalAnalyzer,
SeasonalAggregator, and SpatialAggregator from AnalyticsConfig and
InferenceResult metadata. RiverMorphologyEngine calls this internally.
"""

from __future__ import annotations

import logging
from typing import Any

from src.morphology.analyzer import MorphologyAnalyzer, SpatialAggregator
from src.morphology.contracts import AnalyticsConfig
from src.morphology.temporal import SeasonalAggregator, TemporalAnalyzer

__all__ = ["AnalyticsFactory"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


class AnalyticsFactory:
    """Assembles all analytics components from config."""

    @classmethod
    def build(
        cls,
        config:           AnalyticsConfig,
        inference_result: Any,
    ) -> dict:
        """
        Build the complete analytics context.

        Args:
            config:           AnalyticsConfig.
            inference_result: InferenceResult from Module 16.

        Returns:
            Dict with keys: morphology_analyzer, temporal_analyzer,
            seasonal_aggregator, spatial_aggregator, class_names.
        """
        class_names = tuple(getattr(inference_result, "class_names", ()))
        if not class_names:
            class_names = ("background", "water", "sand", "vegetation")
            _LOGGER.warning(
                "AnalyticsFactory: class_names empty; defaulting to %s",
                class_names,
            )

        morphology_analyzer = MorphologyAnalyzer(config, class_names)
        temporal_analyzer   = TemporalAnalyzer(config, class_names)
        seasonal_aggregator = SeasonalAggregator(config, class_names)
        spatial_aggregator  = SpatialAggregator(class_names)

        _LOGGER.debug(
            "AnalyticsFactory: built context for %d classes: %s",
            len(class_names), class_names,
        )

        return {
            "morphology_analyzer": morphology_analyzer,
            "temporal_analyzer":   temporal_analyzer,
            "seasonal_aggregator": seasonal_aggregator,
            "spatial_aggregator":  spatial_aggregator,
            "class_names":         class_names,
        }
