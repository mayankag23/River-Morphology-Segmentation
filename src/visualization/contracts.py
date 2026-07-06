"""
Public data contracts for the River Morphology Visualization Framework (Module 18).

Contract chain:
    RiverMorphologyResult (Module 17) ──> VisualizationEngine.visualize() ──> VisualizationResult

Design invariants
-----------------
- All public contracts are frozen dataclasses.
- No matplotlib types at module level (lazy import policy).
- All rendering parameters (DPI, figure size, colors, fonts) come exclusively
  from VisualizationConfig — nothing is hardcoded in any renderer.
- FigureSpec carries both the matplotlib Figure object (for Jupyter display)
  and the export path (for file-based workflows) so both use cases work.
- VisualizationResult is the single deliverable: fully auditable, JSON-summarizable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "VisualizationConfig",
    "FigureSpec",
    "VisualizationResult",
]


# ==============================================================================
# VisualizationConfig
# ==============================================================================

@dataclass(frozen=True)
class VisualizationConfig:
    """
    Immutable visualization configuration.

    Attributes:
        output_dir:          Directory for exported figures. "" = no export.
        export_png:          Export each figure as PNG.
        export_svg:          Export each figure as SVG.
        export_pdf:          Export each figure as PDF.
        dpi:                 Dots per inch for raster exports (PNG).
        figure_width:        Default figure width in inches.
        figure_height:       Default figure height in inches.
        font_size:           Base font size for axes labels and titles.
        title_font_size:     Font size for figure titles.
        colorbar:            Show colorbars on scalar maps.
        alpha_overlay:       Transparency of prediction overlay on imagery [0,1].
        alpha_confidence:    Transparency of confidence overlay [0,1].
        class_colors:        Dict class_name -> (R, G, B) tuple, values in [0,1].
                             Empty = use ColorRegistry defaults.
        background_color:    Matplotlib-compatible background color string.
        render_masks:        Render segmentation mask figures.
        render_overlays:     Render prediction overlay figures.
        render_confidence:   Render confidence map figures.
        render_per_class:    Render per-class coverage maps.
        render_timeline:     Render temporal timeline figures.
        render_seasonal:     Render seasonal comparison figures.
        render_comparison:   Render before/after comparison figures.
        max_samples:         Maximum samples to render. 0 = all.
        colormap_name:       Matplotlib colormap name for scalar fields
                             (confidence, uncertainty). Default "viridis".
    """

    output_dir:         str                     = ""
    export_png:         bool                    = True
    export_svg:         bool                    = False
    export_pdf:         bool                    = False
    dpi:                int                     = 150
    figure_width:       float                   = 10.0
    figure_height:      float                   = 6.0
    font_size:          int                     = 10
    title_font_size:    int                     = 12
    colorbar:           bool                    = True
    alpha_overlay:      float                   = 0.5
    alpha_confidence:   float                   = 0.6
    class_colors:       dict[str, tuple]        = field(default_factory=dict)
    background_color:   str                     = "white"
    render_masks:       bool                    = True
    render_overlays:    bool                    = True
    render_confidence:  bool                    = True
    render_per_class:   bool                    = True
    render_timeline:    bool                    = True
    render_seasonal:    bool                    = True
    render_comparison:  bool                    = True
    max_samples:        int                     = 0
    colormap_name:      str                     = "viridis"

    @classmethod
    def from_config(cls, config: Any) -> VisualizationConfig:
        """Build VisualizationConfig from config.visualization."""
        vcfg = getattr(config, "visualization", None)
        if vcfg is None:
            return cls()
        raw_colors = getattr(vcfg, "class_colors", {})
        class_colors: dict[str, tuple] = {}
        if isinstance(raw_colors, dict):
            for k, v in raw_colors.items():
                class_colors[str(k)] = tuple(float(c) for c in v)
        return cls(
            output_dir        = str(getattr(vcfg,  "output_dir",        "")),
            export_png        = bool(getattr(vcfg, "export_png",        True)),
            export_svg        = bool(getattr(vcfg, "export_svg",        False)),
            export_pdf        = bool(getattr(vcfg, "export_pdf",        False)),
            dpi               = int(getattr(vcfg,  "dpi",               150)),
            figure_width      = float(getattr(vcfg,"figure_width",      10.0)),
            figure_height     = float(getattr(vcfg,"figure_height",     6.0)),
            font_size         = int(getattr(vcfg,  "font_size",         10)),
            title_font_size   = int(getattr(vcfg,  "title_font_size",   12)),
            colorbar          = bool(getattr(vcfg, "colorbar",          True)),
            alpha_overlay     = float(getattr(vcfg,"alpha_overlay",     0.5)),
            alpha_confidence  = float(getattr(vcfg,"alpha_confidence",  0.6)),
            class_colors      = class_colors,
            background_color  = str(getattr(vcfg,  "background_color",  "white")),
            render_masks      = bool(getattr(vcfg, "render_masks",      True)),
            render_overlays   = bool(getattr(vcfg, "render_overlays",   True)),
            render_confidence = bool(getattr(vcfg, "render_confidence", True)),
            render_per_class  = bool(getattr(vcfg, "render_per_class",  True)),
            render_timeline   = bool(getattr(vcfg, "render_timeline",   True)),
            render_seasonal   = bool(getattr(vcfg, "render_seasonal",   True)),
            render_comparison = bool(getattr(vcfg, "render_comparison", True)),
            max_samples       = int(getattr(vcfg,  "max_samples",       0)),
            colormap_name     = str(getattr(vcfg,  "colormap_name",     "viridis")),
        )


# ==============================================================================
# FigureSpec
# ==============================================================================

@dataclass
class FigureSpec:
    """
    One generated figure with its metadata.

    NOT frozen — the figure object is mutable (matplotlib Figure).

    Attributes:
        figure_id:      Unique identifier for this figure (e.g. "mask_p001").
        figure_type:    Category: "mask", "overlay", "confidence", "per_class",
                        "timeline", "seasonal", "comparison", "uncertainty".
        title:          Human-readable figure title.
        sample_id:      Sample ID this figure relates to. "" for multi-sample.
        acquisition_date: YYYY-MM-DD for single-sample figures. "" for multi.
        figure:         The matplotlib Figure object. None when matplotlib is
                        unavailable or rendering was skipped.
        export_paths:   Absolute paths of all exported files for this figure.
        metadata:       Free-form dict for additional provenance.
    """

    figure_id:        str
    figure_type:      str
    title:            str
    sample_id:        str        = ""
    acquisition_date: str        = ""
    figure:           Any        = None     # matplotlib.figure.Figure at runtime
    export_paths:     list[str]  = field(default_factory=list)
    metadata:         dict       = field(default_factory=dict)

    def summary_dict(self) -> dict:
        """Return a JSON-serializable summary (no Figure object)."""
        return {
            "figure_id":        self.figure_id,
            "figure_type":      self.figure_type,
            "title":            self.title,
            "sample_id":        self.sample_id,
            "acquisition_date": self.acquisition_date,
            "has_figure":       self.figure is not None,
            "export_paths":     self.export_paths,
            "metadata":         self.metadata,
        }


# ==============================================================================
# VisualizationResult
# ==============================================================================

@dataclass(frozen=True)
class VisualizationResult:
    """
    Immutable public output of VisualizationEngine.visualize().

    Attributes:
        figures:              Tuple of all FigureSpec objects generated.
        num_figures:          Total number of figures generated.
        num_exported:         Total number of files written to disk.
        visualization_config: VisualizationConfig used for this run.
        architecture:         Model architecture from RiverMorphologyResult.
        num_samples:          Number of samples visualized.
        class_names:          Ordered class names from RiverMorphologyResult.
        output_dir:           Absolute output directory path. "" when no export.
        operations_log:       Ordered log of rendering steps.
        visualization_time_s: Wall-clock seconds for the full run.
    """

    figures:               tuple[FigureSpec, ...]
    num_figures:           int
    num_exported:          int
    visualization_config:  VisualizationConfig
    architecture:          str
    num_samples:           int
    class_names:           tuple[str, ...]
    output_dir:            str
    operations_log:        tuple[str, ...]
    visualization_time_s:  float

    def summary_lines(self) -> list[str]:
        """Return ASCII-formatted summary lines."""
        return [
            f"  architecture:         {self.architecture}",
            f"  num_samples:          {self.num_samples}",
            f"  num_figures:          {self.num_figures}",
            f"  num_exported:         {self.num_exported}",
            f"  output_dir:           {self.output_dir or '(none)'}",
            f"  visualization_time_s: {self.visualization_time_s:.2f}",
        ]

    def as_dict(self) -> dict:
        """Return a fully JSON-serializable summary dict."""
        return {
            "num_figures":          self.num_figures,
            "num_exported":         self.num_exported,
            "architecture":         self.architecture,
            "num_samples":          self.num_samples,
            "class_names":          list(self.class_names),
            "output_dir":           self.output_dir,
            "visualization_time_s": round(self.visualization_time_s, 3),
            "operations_log":       list(self.operations_log),
            "figures":              [f.summary_dict() for f in self.figures],
        }

    def figures_by_type(self, figure_type: str) -> list[FigureSpec]:
        """Return all figures of a specific type."""
        return [f for f in self.figures if f.figure_type == figure_type]

    def figures_for_sample(self, sample_id: str) -> list[FigureSpec]:
        """Return all figures related to a specific sample."""
        return [f for f in self.figures if f.sample_id == sample_id]