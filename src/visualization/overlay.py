"""
Overlay renderer for Module 18.

OverlayRenderer composites predicted class masks and confidence maps on top
of a reference image (e.g. a greyscale or RGB patch from the dataset).

When no source image is available, the mask is rendered stand-alone with a
neutral background so the pipeline never crashes.

Overlay rules
-------------
- Overlay alpha comes from VisualizationConfig.alpha_overlay (class colors).
- Confidence overlay alpha comes from VisualizationConfig.alpha_confidence.
- All blending is done in float32 RGBA space via numpy; no PIL dependency.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.visualization.colormap import ClassColorMap
from src.visualization.contracts import FigureSpec, VisualizationConfig
from src.visualization.renderer import _use_agg

__all__ = ["OverlayRenderer"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


class OverlayRenderer:
    """
    Composites prediction overlays on source imagery.

    Args:
        config:   VisualizationConfig.
        colormap: ClassColorMap for the current class schema.
    """

    def __init__(
        self,
        config:   VisualizationConfig,
        colormap: ClassColorMap,
    ) -> None:
        self._config   = config
        self._colormap = colormap

    def render_overlay(
        self,
        sample_analysis: Any,
        predicted_mask:  np.ndarray,
        source_image:    np.ndarray | None = None,
    ) -> FigureSpec:
        """
        Render prediction overlay on source imagery or a neutral background.

        Args:
            sample_analysis: SampleAnalysis.
            predicted_mask:  (H, W) uint8 class-ID array.
            source_image:    Optional (H, W) or (H, W, C) image array.
                             When None, a neutral grey background is used.

        Returns:
            FigureSpec with figure_type="overlay".
        """
        sid   = sample_analysis.sample_id
        date  = sample_analysis.acquisition_date
        title = f"Prediction Overlay — {sid} ({date})"

        fig = _make_figure(self._config, n_axes=1)
        if fig is None:
            return FigureSpec(
                figure_id=f"overlay_{sid}", figure_type="overlay",
                title=title, sample_id=sid, acquisition_date=date,
            )

        try:
            H, W = predicted_mask.shape
            # Build background.
            if source_image is not None:
                bg = _to_rgb(source_image)
            else:
                bg = np.full((H, W, 3), 0.5, dtype=np.float32)

            # Class-colored overlay.
            rgba_mask = self._colormap.to_rgba_mask(predicted_mask)
            alpha     = self._config.alpha_overlay
            composite = bg * (1.0 - alpha) + rgba_mask[:, :, :3] * alpha
            composite = np.clip(composite, 0.0, 1.0)

            ax = fig.axes[0]
            ax.imshow(composite, interpolation="nearest")
            ax.set_title(title, fontsize=self._config.title_font_size)
            ax.axis("off")
            handles = self._colormap.legend_handles()
            if handles:
                ax.legend(handles=handles, loc="lower right",
                          fontsize=self._config.font_size)
            fig.tight_layout()
        except Exception as exc:
            _LOGGER.warning("OverlayRenderer.render_overlay failed: %s", exc)

        return FigureSpec(
            figure_id       = f"overlay_{sid}",
            figure_type     = "overlay",
            title           = title,
            sample_id       = sid,
            acquisition_date = date,
            figure          = fig,
        )

    def render_confidence_overlay(
        self,
        sample_analysis: Any,
        predicted_mask:  np.ndarray,
        confidence:      np.ndarray,
        threshold:       float,
    ) -> FigureSpec:
        """
        Render low-confidence pixels highlighted on top of the class mask.

        Args:
            sample_analysis: SampleAnalysis.
            predicted_mask:  (H, W) uint8 class-ID array.
            confidence:      (H, W) float32 confidence array.
            threshold:       Low-confidence threshold.

        Returns:
            FigureSpec with figure_type="confidence_overlay".
        """
        sid   = sample_analysis.sample_id
        date  = sample_analysis.acquisition_date
        title = f"Confidence Overlay — {sid} ({date})"

        fig = _make_figure(self._config, n_axes=1)
        if fig is None:
            return FigureSpec(
                figure_id=f"confidence_overlay_{sid}",
                figure_type="confidence_overlay",
                title=title, sample_id=sid, acquisition_date=date,
            )

        try:
            # Base: class-colored mask.
            base = self._colormap.to_rgba_mask(predicted_mask)[:, :, :3]
            # Red channel highlight for uncertain pixels.
            uncertain  = confidence < threshold
            alpha      = self._config.alpha_confidence
            overlay    = base.copy()
            overlay[uncertain] = overlay[uncertain] * (1 - alpha) + \
                                 np.array([0.9, 0.1, 0.1]) * alpha

            ax = fig.axes[0]
            ax.imshow(np.clip(overlay, 0, 1), interpolation="nearest")
            ax.set_title(title, fontsize=self._config.title_font_size)
            ax.axis("off")
            fig.tight_layout()
        except Exception as exc:
            _LOGGER.warning("OverlayRenderer.render_confidence_overlay failed: %s", exc)

        return FigureSpec(
            figure_id       = f"confidence_overlay_{sid}",
            figure_type     = "confidence_overlay",
            title           = title,
            sample_id       = sid,
            acquisition_date = date,
            figure          = fig,
        )


# ==============================================================================
# Private helpers
# ==============================================================================

def _make_figure(config: VisualizationConfig, n_axes: int = 1) -> Any | None:
    """Create a matplotlib Figure. Returns None when matplotlib unavailable."""
    if not _use_agg():
        return None
    try:
        import matplotlib.pyplot as plt
        figsize = (config.figure_width, config.figure_height)
        fig, _  = plt.subplots(1, n_axes, figsize=figsize,
                                facecolor=config.background_color)
        return fig
    except Exception as exc:
        _LOGGER.warning("OverlayRenderer._make_figure failed: %s", exc)
        return None


def _to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert (H,W), (H,W,1), or (H,W,C) image to (H,W,3) float32 in [0,1]."""
    img = np.asarray(image, dtype=np.float32)
    # Normalize to [0,1] if needed.
    if img.max() > 1.0:
        img = img / 255.0
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = np.concatenate([img, img, img], axis=-1)
    elif img.ndim == 3 and img.shape[2] > 3:
        img = img[:, :, :3]
    return np.clip(img, 0.0, 1.0)