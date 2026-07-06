"""
src/visualization -- River Morphology Visualization Framework (Module 18).

Single public entry point:
    VisualizationEngine.visualize(morphology_result) -> VisualizationResult

Usage
-----
    from src.visualization import VisualizationEngine, VisualizationConfig

    config = VisualizationConfig(
        output_dir    = "figures",
        export_png    = True,
        dpi           = 150,
        alpha_overlay = 0.5,
        class_colors  = {
            "background": (0.2, 0.2, 0.2),
            "water":      (0.1, 0.5, 0.8),
            "sand":       (0.9, 0.8, 0.5),
            "vegetation": (0.2, 0.6, 0.2),
        },
    )
    engine = VisualizationEngine(config)
    result = engine.visualize(
        morphology_result,
        prediction_masks  = {"sample_001": mask_array},
        confidence_maps   = {"sample_001": conf_array},
    )
    for spec in result.figures:
        print(spec.figure_id, spec.figure_type, spec.export_paths)
"""

# Primary entry point
from src.visualization.engine import VisualizationEngine

# Public contracts
from src.visualization.contracts import (
    FigureSpec,
    VisualizationConfig,
    VisualizationResult,
)

# Color management
from src.visualization.colormap import ClassColorMap, ColorRegistry

# Renderers
from src.visualization.renderer import MaskRenderer
from src.visualization.overlay import OverlayRenderer
from src.visualization.timeline import TimelineRenderer
from src.visualization.comparison import ComparisonRenderer

# Export
from src.visualization.exporter import FigureExporter

# Validation
from src.visualization.validator import VisualizationValidator, VisualizationValidationResult

# Factory
from src.visualization.factory import VisualizationFactory

__all__ = [
    # Primary
    "VisualizationEngine",
    # Contracts
    "VisualizationConfig",
    "FigureSpec",
    "VisualizationResult",
    # Color
    "ClassColorMap",
    "ColorRegistry",
    # Renderers
    "MaskRenderer",
    "OverlayRenderer",
    "TimelineRenderer",
    "ComparisonRenderer",
    # Export
    "FigureExporter",
    # Validation
    "VisualizationValidator",
    "VisualizationValidationResult",
    # Factory
    "VisualizationFactory",
]