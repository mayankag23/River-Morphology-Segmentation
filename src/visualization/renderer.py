"""
Core segmentation mask renderer for Module 18.

MaskRenderer produces three figure types from a SampleAnalysis:
    "mask"        Class-colored segmentation mask (one color per class).
    "per_class"   Grid of binary coverage maps, one subplot per class.
    "confidence"  Scalar confidence map using a sequential colormap.
    "uncertainty" Low-confidence pixel overlay (pixels below threshold).

Design rules
------------
- matplotlib is imported inside each method (lazy import; never at module level).
- matplotlib.use('Agg') is called before any Figure is created to ensure
  headless rendering in server/pipeline environments.
- No computation is performed here — all data comes from SampleAnalysis.
- Figure size, DPI, font size, colormap all come from VisualizationConfig.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.visualization.colormap import ClassColorMap
from src.visualization.contracts import FigureSpec, VisualizationConfig

__all__ = ["MaskRenderer"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


def _use_agg() -> bool:
    """Switch matplotlib to Agg backend. Returns True on success."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        return True
    except ImportError:
        return False


class MaskRenderer:
    """
    Renders segmentation masks and scalar maps from SampleAnalysis objects.

    Args:
        config:    VisualizationConfig.
        colormap:  ClassColorMap for the current class schema.
    """

    def __init__(
        self,
        config:   VisualizationConfig,
        colormap: ClassColorMap,
    ) -> None:
        self._config   = config
        self._colormap = colormap

    def render_mask(
        self,
        sample_analysis: Any,
        predicted_mask:  np.ndarray,
    ) -> FigureSpec:
        """
        Render a class-colored segmentation mask.

        Args:
            sample_analysis: SampleAnalysis from RiverMorphologyResult.
            predicted_mask:  (H, W) uint8 class-ID array.

        Returns:
            FigureSpec with figure_type="mask".
        """
        sid   = sample_analysis.sample_id
        date  = sample_analysis.acquisition_date
        title = f"Segmentation Mask — {sid} ({date})"

        fig = self._make_figure(title)
        if fig is None:
            return FigureSpec(
                figure_id="mask_" + sid, figure_type="mask",
                title=title, sample_id=sid, acquisition_date=date,
            )

        try:
            import matplotlib.pyplot as plt
            rgba = self._colormap.to_rgba_mask(predicted_mask)
            ax   = fig.axes[0]
            ax.imshow(rgba, interpolation="nearest")
            ax.set_title(title, fontsize=self._config.title_font_size)
            ax.axis("off")
            handles = self._colormap.legend_handles()
            if handles:
                ax.legend(handles=handles, loc="lower right",
                          fontsize=self._config.font_size)
            fig.tight_layout()
        except Exception as exc:
            _LOGGER.warning("MaskRenderer.render_mask failed: %s", exc)

        return FigureSpec(
            figure_id       = f"mask_{sid}",
            figure_type     = "mask",
            title           = title,
            sample_id       = sid,
            acquisition_date = date,
            figure          = fig,
        )

    def render_per_class(
        self,
        sample_analysis: Any,
        predicted_mask:  np.ndarray,
        class_names:     tuple[str, ...],
    ) -> FigureSpec:
        """
        Render a grid of per-class binary coverage maps.

        Args:
            sample_analysis: SampleAnalysis.
            predicted_mask:  (H, W) uint8 class-ID array.
            class_names:     Ordered class names.

        Returns:
            FigureSpec with figure_type="per_class".
        """
        sid   = sample_analysis.sample_id
        date  = sample_analysis.acquisition_date
        title = f"Per-Class Coverage — {sid} ({date})"
        n_cls = len(class_names)

        figsize = (self._config.figure_width * max(1, n_cls / 2),
                   self._config.figure_height)
        fig = self._make_figure(title, figsize=figsize, n_axes=n_cls)
        if fig is None:
            return FigureSpec(
                figure_id=f"per_class_{sid}", figure_type="per_class",
                title=title, sample_id=sid, acquisition_date=date,
            )

        try:
            axes = fig.axes
            for idx, (ax, class_name) in enumerate(zip(axes, class_names)):
                binary = (predicted_mask == idx).astype(np.float32)
                r, g, b = self._colormap.get(class_name)
                cmap    = _single_color_cmap(r, g, b)
                ax.imshow(binary, cmap=cmap, vmin=0, vmax=1,
                          interpolation="nearest")
                cm_val = sample_analysis.class_metrics.get(class_name)
                frac   = cm_val.area_fraction if cm_val else 0.0
                ax.set_title(f"{class_name}\n{frac:.1%}",
                             fontsize=self._config.font_size)
                ax.axis("off")
            fig.suptitle(title, fontsize=self._config.title_font_size)
            fig.tight_layout()
        except Exception as exc:
            _LOGGER.warning("MaskRenderer.render_per_class failed: %s", exc)

        return FigureSpec(
            figure_id       = f"per_class_{sid}",
            figure_type     = "per_class",
            title           = title,
            sample_id       = sid,
            acquisition_date = date,
            figure          = fig,
        )

    def render_confidence(
        self,
        sample_analysis: Any,
        confidence:      np.ndarray,
    ) -> FigureSpec:
        """
        Render a scalar confidence map using the configured colormap.

        Args:
            sample_analysis: SampleAnalysis.
            confidence:      (H, W) float32 confidence array in [0, 1].

        Returns:
            FigureSpec with figure_type="confidence".
        """
        sid   = sample_analysis.sample_id
        date  = sample_analysis.acquisition_date
        title = f"Prediction Confidence — {sid} ({date})"

        fig = self._make_figure(title)
        if fig is None:
            return FigureSpec(
                figure_id=f"confidence_{sid}", figure_type="confidence",
                title=title, sample_id=sid, acquisition_date=date,
            )

        try:
            import matplotlib.pyplot as plt
            ax  = fig.axes[0]
            im  = ax.imshow(confidence, cmap=self._config.colormap_name,
                            vmin=0.0, vmax=1.0, interpolation="nearest")
            ax.set_title(title, fontsize=self._config.title_font_size)
            ax.axis("off")
            if self._config.colorbar:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                             label="Confidence")
            fig.tight_layout()
        except Exception as exc:
            _LOGGER.warning("MaskRenderer.render_confidence failed: %s", exc)

        return FigureSpec(
            figure_id       = f"confidence_{sid}",
            figure_type     = "confidence",
            title           = title,
            sample_id       = sid,
            acquisition_date = date,
            figure          = fig,
        )

    def render_uncertainty(
        self,
        sample_analysis: Any,
        confidence:      np.ndarray,
        threshold:       float,
    ) -> FigureSpec:
        """
        Render a binary map highlighting low-confidence pixels.

        Args:
            sample_analysis: SampleAnalysis.
            confidence:      (H, W) float32 confidence array.
            threshold:       Confidence threshold below which pixels are uncertain.

        Returns:
            FigureSpec with figure_type="uncertainty".
        """
        sid   = sample_analysis.sample_id
        date  = sample_analysis.acquisition_date
        title = f"Uncertainty Map — {sid} ({date})"

        fig = self._make_figure(title)
        if fig is None:
            return FigureSpec(
                figure_id=f"uncertainty_{sid}", figure_type="uncertainty",
                title=title, sample_id=sid, acquisition_date=date,
            )

        try:
            low_conf = (confidence < threshold).astype(np.float32)
            ax       = fig.axes[0]
            ax.imshow(low_conf, cmap="Reds", vmin=0, vmax=1,
                      interpolation="nearest")
            ax.set_title(title, fontsize=self._config.title_font_size)
            ax.axis("off")
            if self._config.colorbar:
                import matplotlib
                from matplotlib.colors import ListedColormap
                from matplotlib.patches import Patch
                handles = [
                    Patch(facecolor=(0.9, 0.9, 0.9), label="High confidence"),
                    Patch(facecolor=(0.8, 0.1, 0.1), label="Low confidence"),
                ]
                ax.legend(handles=handles, loc="lower right",
                          fontsize=self._config.font_size)
            fig.tight_layout()
        except Exception as exc:
            _LOGGER.warning("MaskRenderer.render_uncertainty failed: %s", exc)

        return FigureSpec(
            figure_id       = f"uncertainty_{sid}",
            figure_type     = "uncertainty",
            title           = title,
            sample_id       = sid,
            acquisition_date = date,
            figure          = fig,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_figure(
        self,
        title:   str = "",
        figsize: tuple | None = None,
        n_axes:  int = 1,
    ) -> Any | None:
        """Create and return a matplotlib Figure. Returns None on ImportError."""
        if not _use_agg():
            return None
        try:
            import matplotlib.pyplot as plt
            if figsize is None:
                figsize = (self._config.figure_width, self._config.figure_height)
            if n_axes == 1:
                fig, _ = plt.subplots(1, 1, figsize=figsize,
                                       facecolor=self._config.background_color)
            else:
                ncols = min(n_axes, 4)
                nrows = (n_axes + ncols - 1) // ncols
                fig, _ = plt.subplots(nrows, ncols, figsize=figsize,
                                       facecolor=self._config.background_color)
                # Flatten axes if subplots returns a 2-D array.
                all_axes = fig.axes
                # Hide extra subplots.
                for ax in all_axes[n_axes:]:
                    ax.set_visible(False)
            plt.rcParams.update({
                "font.size":       self._config.font_size,
                "axes.titlesize":  self._config.title_font_size,
            })
            return fig
        except Exception as exc:
            _LOGGER.warning("MaskRenderer._make_figure failed: %s", exc)
            return None


def _single_color_cmap(r: float, g: float, b: float) -> Any:
    """Create a white-to-class-color matplotlib colormap."""
    try:
        from matplotlib.colors import LinearSegmentedColormap
        return LinearSegmentedColormap.from_list(
            "cls_cmap", [(1.0, 1.0, 1.0), (r, g, b)], N=256
        )
    except ImportError:
        return "viridis"