"""
Comparison visualization for Module 18.

ComparisonRenderer produces multi-panel figures that compare two samples
(before/after, seasonal, or yearly). All data comes from SampleAnalysis
objects already present in RiverMorphologyResult — no recomputation.

Figure types produced:
    "comparison"       Side-by-side class masks for two samples.
    "change_map"       Pixel-level difference map highlighting class changes.
    "class_diff"       Per-class stacked area differences (bar chart).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.visualization.colormap import ClassColorMap
from src.visualization.contracts import FigureSpec, VisualizationConfig
from src.visualization.renderer import _use_agg

__all__ = ["ComparisonRenderer"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


class ComparisonRenderer:
    """
    Renders before/after and change comparison figures.

    Args:
        config:   VisualizationConfig.
        colormap: ClassColorMap.
    """

    def __init__(
        self,
        config:   VisualizationConfig,
        colormap: ClassColorMap,
    ) -> None:
        self._config   = config
        self._colormap = colormap

    def render_side_by_side(
        self,
        analysis_a:   Any,
        analysis_b:   Any,
        mask_a:       np.ndarray,
        mask_b:       np.ndarray,
    ) -> FigureSpec:
        """
        Side-by-side class mask comparison for two samples.

        Args:
            analysis_a: Earlier SampleAnalysis.
            analysis_b: Later SampleAnalysis.
            mask_a:     (H, W) uint8 class-ID mask for sample A.
            mask_b:     (H, W) uint8 class-ID mask for sample B.

        Returns:
            FigureSpec with figure_type="comparison".
        """
        label_a = f"{analysis_a.sample_id}\n({analysis_a.acquisition_date})"
        label_b = f"{analysis_b.sample_id}\n({analysis_b.acquisition_date})"
        title   = f"Comparison: {analysis_a.acquisition_date} vs {analysis_b.acquisition_date}"
        fig_id  = f"comparison_{analysis_a.sample_id}_{analysis_b.sample_id}"

        fig = _make_figure(self._config, n_cols=2)
        if fig is None:
            return FigureSpec(figure_id=fig_id, figure_type="comparison", title=title)

        try:
            axes     = fig.axes
            rgba_a   = self._colormap.to_rgba_mask(mask_a)
            rgba_b   = self._colormap.to_rgba_mask(mask_b)
            axes[0].imshow(rgba_a, interpolation="nearest")
            axes[0].set_title(label_a, fontsize=self._config.font_size)
            axes[0].axis("off")
            axes[1].imshow(rgba_b, interpolation="nearest")
            axes[1].set_title(label_b, fontsize=self._config.font_size)
            axes[1].axis("off")
            handles = self._colormap.legend_handles()
            if handles:
                axes[1].legend(handles=handles, loc="lower right",
                               fontsize=self._config.font_size)
            fig.suptitle(title, fontsize=self._config.title_font_size)
            fig.tight_layout()
        except Exception as exc:
            _LOGGER.warning("ComparisonRenderer.render_side_by_side failed: %s", exc)

        return FigureSpec(
            figure_id  = fig_id,
            figure_type = "comparison",
            title       = title,
            figure      = fig,
        )

    def render_change_map(
        self,
        analysis_a: Any,
        analysis_b: Any,
        mask_a:     np.ndarray,
        mask_b:     np.ndarray,
    ) -> FigureSpec:
        """
        Pixel-level change map: highlights pixels that changed class between
        two samples using a distinctive color scheme.

        Args:
            analysis_a: SampleAnalysis for the earlier sample.
            analysis_b: SampleAnalysis for the later sample.
            mask_a:     (H, W) uint8 class-ID mask for sample A.
            mask_b:     (H, W) uint8 class-ID mask for sample B.

        Returns:
            FigureSpec with figure_type="change_map".
        """
        title  = (f"Change Map: {analysis_a.acquisition_date}"
                  f" → {analysis_b.acquisition_date}")
        fig_id = f"change_map_{analysis_a.sample_id}_{analysis_b.sample_id}"

        fig = _make_figure(self._config, n_cols=1)
        if fig is None:
            return FigureSpec(figure_id=fig_id, figure_type="change_map", title=title)

        try:
            # Build RGBA change map.
            changed  = (mask_a != mask_b)
            H, W     = mask_a.shape
            rgba     = np.zeros((H, W, 4), dtype=np.float32)
            rgba[:, :, 3] = 1.0

            # Unchanged pixels: show as class A color at reduced intensity.
            rgba_a = self._colormap.to_rgba_mask(mask_a)
            rgba[~changed] = rgba_a[~changed] * 0.4

            # Changed pixels: bright red overlay to highlight changes.
            rgba[changed, 0] = 0.9
            rgba[changed, 1] = 0.1
            rgba[changed, 2] = 0.1

            ax = fig.axes[0]
            ax.imshow(rgba, interpolation="nearest")
            ax.set_title(title, fontsize=self._config.title_font_size)
            ax.axis("off")

            try:
                from matplotlib.patches import Patch
                handles = [
                    Patch(facecolor=(0.9, 0.1, 0.1), label="Changed"),
                    Patch(facecolor=(0.4, 0.4, 0.4), label="Unchanged"),
                ]
                ax.legend(handles=handles, loc="lower right",
                          fontsize=self._config.font_size)
            except ImportError:
                pass

            fig.tight_layout()
        except Exception as exc:
            _LOGGER.warning("ComparisonRenderer.render_change_map failed: %s", exc)

        return FigureSpec(
            figure_id  = fig_id,
            figure_type = "change_map",
            title       = title,
            figure      = fig,
        )

    def render_class_diff(
        self,
        analysis_a:  Any,
        analysis_b:  Any,
        class_names: tuple[str, ...],
    ) -> FigureSpec:
        """
        Bar chart showing per-class area fraction differences (B - A).

        Reads from ClassMorphologyMetrics — no pixel data required.

        Args:
            analysis_a:  Earlier SampleAnalysis.
            analysis_b:  Later SampleAnalysis.
            class_names: Ordered class names.

        Returns:
            FigureSpec with figure_type="class_diff".
        """
        title  = (f"Class Difference: {analysis_a.acquisition_date}"
                  f" → {analysis_b.acquisition_date}")
        fig_id = f"class_diff_{analysis_a.sample_id}_{analysis_b.sample_id}"

        fig = _make_figure(self._config, n_cols=1)
        if fig is None:
            return FigureSpec(figure_id=fig_id, figure_type="class_diff", title=title)

        try:
            diffs  = []
            colors = []
            for cls_name in class_names:
                frac_a = (analysis_a.class_metrics.get(cls_name).area_fraction
                          if analysis_a.class_metrics.get(cls_name) else 0.0)
                frac_b = (analysis_b.class_metrics.get(cls_name).area_fraction
                          if analysis_b.class_metrics.get(cls_name) else 0.0)
                diffs.append(frac_b - frac_a)
                colors.append(self._colormap.get(cls_name))

            x  = np.arange(len(class_names))
            ax = fig.axes[0]
            ax.bar(x, diffs, color=colors, width=0.6)
            ax.set_xticks(x)
            ax.set_xticklabels(list(class_names), fontsize=self._config.font_size)
            ax.set_ylabel("Area Fraction Delta (B - A)", fontsize=self._config.font_size)
            ax.set_title(title, fontsize=self._config.title_font_size)
            ax.axhline(0, color="black", linewidth=0.8)
            fig.tight_layout()
        except Exception as exc:
            _LOGGER.warning("ComparisonRenderer.render_class_diff failed: %s", exc)

        return FigureSpec(
            figure_id  = fig_id,
            figure_type = "class_diff",
            title       = title,
            figure      = fig,
        )


# ==============================================================================
# Helpers
# ==============================================================================

def _make_figure(config: VisualizationConfig, n_cols: int = 1) -> Any | None:
    if not _use_agg():
        return None
    try:
        import matplotlib.pyplot as plt
        fig, _ = plt.subplots(
            1, n_cols,
            figsize=(config.figure_width * n_cols, config.figure_height),
            facecolor=config.background_color,
        )
        return fig
    except Exception as exc:
        _LOGGER.warning("comparison._make_figure failed: %s", exc)
        return None