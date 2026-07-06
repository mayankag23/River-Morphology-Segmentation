"""
Visualization factory for Module 18.

VisualizationFactory assembles ClassColorMap, MaskRenderer, OverlayRenderer,
TimelineRenderer, ComparisonRenderer, and FigureExporter from VisualizationConfig
and RiverMorphologyResult metadata.
"""

from __future__ import annotations

import logging
from typing import Any

from src.visualization.colormap import ClassColorMap
from src.visualization.comparison import ComparisonRenderer
from src.visualization.contracts import VisualizationConfig
from src.visualization.exporter import FigureExporter
from src.visualization.overlay import OverlayRenderer
from src.visualization.renderer import MaskRenderer
from src.visualization.timeline import TimelineRenderer

__all__ = ["VisualizationFactory"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


class VisualizationFactory:
    """Assembles all visualization components from config."""

    @classmethod
    def build(
        cls,
        config:            VisualizationConfig,
        morphology_result: Any,
    ) -> dict:
        """
        Build the complete visualization context.

        Args:
            config:            VisualizationConfig.
            morphology_result: RiverMorphologyResult from Module 17.

        Returns:
            Dict with keys: colormap, mask_renderer, overlay_renderer,
            timeline_renderer, comparison_renderer, exporter, class_names.
        """
        class_names = tuple(getattr(morphology_result, "class_names", ()))
        if not class_names:
            class_names = ("background", "water", "sand", "vegetation")
            _LOGGER.warning(
                "VisualizationFactory: class_names empty in result; "
                "defaulting to standard 4-class schema."
            )

        colormap           = ClassColorMap(config, class_names)
        mask_renderer      = MaskRenderer(config, colormap)
        overlay_renderer   = OverlayRenderer(config, colormap)
        timeline_renderer  = TimelineRenderer(config, colormap)
        comparison_renderer = ComparisonRenderer(config, colormap)
        exporter           = FigureExporter(config)

        _LOGGER.debug(
            "VisualizationFactory: built context for %d classes: %s",
            len(class_names), class_names,
        )

        return {
            "colormap":            colormap,
            "mask_renderer":       mask_renderer,
            "overlay_renderer":    overlay_renderer,
            "timeline_renderer":   timeline_renderer,
            "comparison_renderer": comparison_renderer,
            "exporter":            exporter,
            "class_names":         class_names,
        }