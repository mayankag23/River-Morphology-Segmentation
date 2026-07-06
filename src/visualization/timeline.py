"""
Temporal and seasonal visualization for Module 18.

TimelineRenderer produces charts from pre-computed analytics in
RiverMorphologyResult. It reads TemporalChange, SeasonalSummary, and
SampleAnalysis objects — it never recomputes any metric.

Figure types produced:
    "timeline"    Stacked bar chart of class area fractions over time
                  (one bar per acquisition date, coloured by class).
    "seasonal"    Grouped bar chart of mean class fractions by season.
    "change"      Bar chart of pixel_delta values per class per date pair.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.visualization.colormap import ClassColorMap
from src.visualization.contracts import FigureSpec, VisualizationConfig
from src.visualization.renderer import _use_agg

__all__ = ["TimelineRenderer"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


class TimelineRenderer:
    """
    Renders temporal and seasonal figures from pre-computed analytics.

    Args:
        config:    VisualizationConfig.
        colormap:  ClassColorMap.
    """

    def __init__(
        self,
        config:   VisualizationConfig,
        colormap: ClassColorMap,
    ) -> None:
        self._config   = config
        self._colormap = colormap

    def render_timeline(
        self,
        morphology_result: Any,
    ) -> FigureSpec:
        """
        Stacked bar chart of class area fractions ordered by acquisition date.

        Args:
            morphology_result: RiverMorphologyResult from Module 17.

        Returns:
            FigureSpec with figure_type="timeline".
        """
        title  = "Class Area Timeline"
        fig_id = "timeline_all"

        fig = _make_figure(self._config)
        if fig is None:
            return FigureSpec(figure_id=fig_id, figure_type="timeline", title=title)

        try:
            import matplotlib.pyplot as plt
            analyses    = morphology_result.sample_analyses
            class_names = morphology_result.class_names
            dates       = [a.acquisition_date for a in analyses]
            n_dates     = len(dates)

            if n_dates == 0:
                _set_no_data(fig, title, self._config)
                return FigureSpec(figure_id=fig_id, figure_type="timeline",
                                  title=title, figure=fig)

            x      = np.arange(n_dates)
            bottom = np.zeros(n_dates)
            ax     = fig.axes[0]

            for cls_name in class_names:
                fracs = np.array([
                    a.class_metrics.get(cls_name).area_fraction
                    if a.class_metrics.get(cls_name) else 0.0
                    for a in analyses
                ])
                color = self._colormap.get(cls_name)
                ax.bar(x, fracs, bottom=bottom, label=cls_name,
                       color=color, width=0.8)
                bottom += fracs

            ax.set_xticks(x)
            ax.set_xticklabels(dates, rotation=45, ha="right",
                               fontsize=self._config.font_size)
            ax.set_ylabel("Area Fraction", fontsize=self._config.font_size)
            ax.set_title(title, fontsize=self._config.title_font_size)
            ax.legend(loc="upper right", fontsize=self._config.font_size)
            ax.set_ylim(0, 1)
            fig.tight_layout()
        except Exception as exc:
            _LOGGER.warning("TimelineRenderer.render_timeline failed: %s", exc)

        return FigureSpec(
            figure_id  = fig_id,
            figure_type = "timeline",
            title       = title,
            figure      = fig,
        )

    def render_seasonal(
        self,
        morphology_result: Any,
    ) -> FigureSpec:
        """
        Grouped bar chart of mean class fractions per season.

        Args:
            morphology_result: RiverMorphologyResult.

        Returns:
            FigureSpec with figure_type="seasonal".
        """
        title  = "Seasonal Class Fractions"
        fig_id = "seasonal_all"

        fig = _make_figure(self._config)
        if fig is None:
            return FigureSpec(figure_id=fig_id, figure_type="seasonal", title=title)

        try:
            summaries   = morphology_result.seasonal_summaries
            class_names = morphology_result.class_names

            if not summaries:
                _set_no_data(fig, title, self._config)
                return FigureSpec(figure_id=fig_id, figure_type="seasonal",
                                  title=title, figure=fig)

            seasons = sorted(summaries.keys())
            n_cls   = len(class_names)
            n_seas  = len(seasons)
            x       = np.arange(n_seas)
            width   = 0.8 / max(1, n_cls)
            ax      = fig.axes[0]

            for cls_idx, cls_name in enumerate(class_names):
                means = [summaries[s].mean_class_fractions.get(cls_name, 0.0)
                         for s in seasons]
                stds  = [summaries[s].std_class_fractions.get(cls_name, 0.0)
                         for s in seasons]
                offset = (cls_idx - n_cls / 2 + 0.5) * width
                color  = self._colormap.get(cls_name)
                ax.bar(x + offset, means, width=width, label=cls_name,
                       color=color, yerr=stds, capsize=3, error_kw={"linewidth": 1})

            ax.set_xticks(x)
            ax.set_xticklabels(seasons, fontsize=self._config.font_size)
            ax.set_ylabel("Mean Area Fraction", fontsize=self._config.font_size)
            ax.set_title(title, fontsize=self._config.title_font_size)
            ax.legend(loc="upper right", fontsize=self._config.font_size)
            fig.tight_layout()
        except Exception as exc:
            _LOGGER.warning("TimelineRenderer.render_seasonal failed: %s", exc)

        return FigureSpec(
            figure_id  = fig_id,
            figure_type = "seasonal",
            title       = title,
            figure      = fig,
        )

    def render_change_chart(
        self,
        morphology_result: Any,
    ) -> FigureSpec:
        """
        Horizontal bar chart showing pixel_delta per class per temporal pair.

        Reads pre-computed TemporalChange objects — no recomputation.

        Returns:
            FigureSpec with figure_type="change_chart".
        """
        title  = "Temporal Change (pixel delta)"
        fig_id = "change_chart_all"

        fig = _make_figure(self._config)
        if fig is None:
            return FigureSpec(figure_id=fig_id, figure_type="change_chart", title=title)

        try:
            changes     = morphology_result.temporal_changes
            class_names = morphology_result.class_names

            if not changes:
                _set_no_data(fig, title, self._config)
                return FigureSpec(figure_id=fig_id, figure_type="change_chart",
                                  title=title, figure=fig)

            ax    = fig.axes[0]
            # One bar group per class, grouped by date pair.
            pairs = sorted({(c.date_from, c.date_to) for c in changes})
            n_pairs = len(pairs)
            n_cls   = len(class_names)
            x       = np.arange(n_pairs)
            width   = 0.8 / max(1, n_cls)

            for cls_idx, cls_name in enumerate(class_names):
                deltas = []
                for df, dt in pairs:
                    matching = [c.pixel_delta for c in changes
                                if c.class_name == cls_name
                                and c.date_from == df and c.date_to == dt]
                    deltas.append(matching[0] if matching else 0)
                offset = (cls_idx - n_cls / 2 + 0.5) * width
                color  = self._colormap.get(cls_name)
                ax.bar(x + offset, deltas, width=width,
                       label=cls_name, color=color)

            labels = [f"{df}\n→{dt}" for df, dt in pairs]
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=self._config.font_size)
            ax.set_ylabel("Pixel Delta", fontsize=self._config.font_size)
            ax.set_title(title, fontsize=self._config.title_font_size)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.legend(loc="upper right", fontsize=self._config.font_size)
            fig.tight_layout()
        except Exception as exc:
            _LOGGER.warning("TimelineRenderer.render_change_chart failed: %s", exc)

        return FigureSpec(
            figure_id  = fig_id,
            figure_type = "change_chart",
            title       = title,
            figure      = fig,
        )


# ==============================================================================
# Helpers
# ==============================================================================

def _make_figure(config: VisualizationConfig) -> Any | None:
    if not _use_agg():
        return None
    try:
        import matplotlib.pyplot as plt
        fig, _ = plt.subplots(
            1, 1,
            figsize=(config.figure_width, config.figure_height),
            facecolor=config.background_color,
        )
        return fig
    except Exception as exc:
        _LOGGER.warning("timeline._make_figure failed: %s", exc)
        return None


def _set_no_data(fig: Any, title: str, config: VisualizationConfig) -> None:
    """Place a 'No data' annotation when the result has no samples."""
    try:
        ax = fig.axes[0]
        ax.text(0.5, 0.5, "No data available", ha="center", va="center",
                fontsize=config.font_size, transform=ax.transAxes)
        ax.set_title(title, fontsize=config.title_font_size)
        ax.axis("off")
    except Exception:
        pass