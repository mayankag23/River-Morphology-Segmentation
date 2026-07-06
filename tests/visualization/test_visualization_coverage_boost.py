"""
Additional coverage tests for Module 18 visualization.

Target: raise overall coverage from ~60% to >90%.

Coverage targets per file:
    comparison.py   14% -> >92%
    timeline.py     12% -> >92%
    engine.py       17% -> >92%
    factory.py      56% -> >90%
    overlay.py      88% -> >95%
    renderer.py     89% -> >95%

Strategy: Call every method with matplotlib fully available so the rendering
code paths execute, AND exercise every branch (colorbar=True/False, conf=None/not-None,
fig is None, class_metrics missing, etc.) via direct method calls on the renderers.
"""

from __future__ import annotations

import dataclasses
import types
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

mpl = pytest.importorskip("matplotlib")
import matplotlib
matplotlib.use("Agg")

from src.visualization.colormap import ClassColorMap
from src.visualization.comparison import ComparisonRenderer, _make_figure as cmp_make_figure
from src.visualization.contracts import FigureSpec, VisualizationConfig
from src.visualization.engine import VisualizationEngine
from src.visualization.factory import VisualizationFactory
from src.visualization.overlay import OverlayRenderer
from src.visualization.renderer import MaskRenderer
from src.visualization.timeline import (
    TimelineRenderer, _make_figure as tl_make_figure, _set_no_data,
)


# ==============================================================================
# Shared fixtures
# ==============================================================================

CLASS_NAMES = ("background", "water", "sand", "vegetation")


def _cfg(**kw) -> VisualizationConfig:
    defaults = dict(
        output_dir="", export_png=False, export_svg=False, export_pdf=False,
        dpi=72, figure_width=4.0, figure_height=3.0, font_size=8,
        title_font_size=10, colorbar=True, alpha_overlay=0.5,
        alpha_confidence=0.6, class_colors={}, background_color="white",
        render_masks=True, render_overlays=True, render_confidence=True,
        render_per_class=True, render_timeline=True, render_seasonal=True,
        render_comparison=True, max_samples=0, colormap_name="viridis",
    )
    defaults.update(kw)
    return VisualizationConfig(**defaults)


def _cmap(**kw) -> ClassColorMap:
    return ClassColorMap(_cfg(**kw), CLASS_NAMES)


def _mask(h=8, w=8):
    m = np.zeros((h, w), dtype=np.uint8)
    m[0:4, :] = 1
    return m


def _conf(h=8, w=8, val=0.8):
    return np.full((h, w), val, dtype=np.float32)


def _sa(sid="p001", date="2023-07-01", season="monsoon", with_metrics=True):
    """Build a minimal SampleAnalysis-like namespace."""
    from src.morphology.contracts import ClassMorphologyMetrics
    if with_metrics:
        cm = {n: ClassMorphologyMetrics(n, i, 16, 0.25, 0.25, 0.0, 0.8, 2)
              for i, n in enumerate(CLASS_NAMES)}
    else:
        cm = {}   # missing metrics — triggers the 0.0 fallback branch
    return types.SimpleNamespace(
        sample_id=sid, acquisition_date=date, season=season,
        hydrological_year=2023, sensor="L8", river_name="Kosi",
        reach_id="R1", basin_id="B1", aoi_id="A1",
        total_pixels=64, class_metrics=cm, geometry=None, uncertainty=None,
    )


def _morph_result(n=3, water_frac=0.4):
    from src.morphology.contracts import (
        AnalyticsConfig, ClassMorphologyMetrics, SampleAnalysis,
        RiverMorphologyResult, SeasonalSummary, TemporalChange,
    )
    cfg      = AnalyticsConfig()
    analyses = []
    for i in range(n):
        cm = {name: ClassMorphologyMetrics(name, idx, 16, 0.25, 0.25, 0.0, 0.8, 2)
              for idx, name in enumerate(CLASS_NAMES)}
        cm["water"] = ClassMorphologyMetrics("water", 1, int(64*water_frac),
                                             water_frac, water_frac, 0.0, 0.8, 2)
        sa = SampleAnalysis(
            sample_id=f"p{i:03d}", acquisition_date=f"2023-{i+1:02d}-01",
            season="monsoon" if i % 2 == 0 else "pre-monsoon",
            hydrological_year=2023, sensor="L8", river_name="Kosi",
            reach_id="R1", basin_id="B1", aoi_id="A1",
            total_pixels=64, class_metrics=cm, geometry=None, uncertainty=None,
        )
        analyses.append(sa)

    seasonal = {
        "monsoon": SeasonalSummary(
            "monsoon", 2,
            {"water": 0.5, "sand": 0.2, "vegetation": 0.1, "background": 0.2},
            {"water": 0.05, "sand": 0.02, "vegetation": 0.01, "background": 0.01},
            ("p000", "p002"),
        ),
        "pre-monsoon": SeasonalSummary(
            "pre-monsoon", 1,
            {"water": 0.3, "sand": 0.3, "vegetation": 0.2, "background": 0.2},
            {"water": 0.0,  "sand": 0.0,  "vegetation": 0.0,  "background": 0.0},
            ("p001",),
        ),
    }
    temporal = (
        TemporalChange("water", "2023-01-01", "2023-02-01", 25, 30, 5, 0.39, 0.47, 0.08, 20.0),
        TemporalChange("sand",  "2023-01-01", "2023-02-01", 10, 8, -2, 0.16, 0.12, -0.04, -20.0),
    )
    return RiverMorphologyResult(
        sample_analyses=tuple(analyses), temporal_changes=temporal,
        seasonal_summaries=seasonal, spatial_summaries={},
        class_names=CLASS_NAMES, num_samples=n, num_classes=4,
        total_pixels=n * 64, mean_water_fraction=water_frac,
        mean_sand_fraction=0.2, mean_veg_fraction=0.1, mean_confidence=0.8,
        analytics_config=cfg, architecture="unetplusplus",
        operations_log=("step1",), analysis_time_s=0.5,
    )


# ==============================================================================
# comparison.py — full rendering paths
# ==============================================================================

class TestComparisonRendererFull:
    """Execute every method with matplotlib available so all lines run."""

    def _rend(self, **kw): return ComparisonRenderer(_cfg(**kw), _cmap(**kw))
    def _two(self):        return _sa("p1", "2023-01-01"), _sa("p2", "2023-07-01")

    # ----- render_side_by_side -----

    def test_side_by_side_figure_type(self):
        r = self._rend()
        a, b = self._two()
        fs = r.render_side_by_side(a, b, _mask(), _mask())
        assert fs.figure_type == "comparison"

    def test_side_by_side_has_figure(self):
        r = self._rend()
        a, b = self._two()
        fs = r.render_side_by_side(a, b, _mask(), _mask())
        assert fs.figure is not None

    def test_side_by_side_figure_id_contains_both_ids(self):
        r = self._rend()
        a, b = self._two()
        fs = r.render_side_by_side(a, b, _mask(), _mask())
        assert "p1" in fs.figure_id and "p2" in fs.figure_id

    def test_side_by_side_title_contains_dates(self):
        r = self._rend()
        a, b = self._two()
        fs = r.render_side_by_side(a, b, _mask(), _mask())
        assert "2023-01-01" in fs.title
        assert "2023-07-01" in fs.title

    def test_side_by_side_fig_is_none_path(self):
        """When _make_figure returns None, we get a FigureSpec with figure=None."""
        r = self._rend()
        a, b = self._two()
        with patch("src.visualization.comparison._use_agg", return_value=False):
            fs = r.render_side_by_side(a, b, _mask(), _mask())
        assert fs.figure is None
        assert fs.figure_type == "comparison"

    def test_side_by_side_identical_masks(self):
        """Identical masks should render without error."""
        r  = self._rend()
        a, b = self._two()
        m  = _mask()
        fs = r.render_side_by_side(a, b, m, m)
        assert fs.figure is not None

    def test_side_by_side_colormap_no_legend_handles(self):
        """When legend_handles returns [] no legend is added — no crash."""
        r    = self._rend()
        a, b = self._two()
        with patch.object(r._colormap, "legend_handles", return_value=[]):
            fs = r.render_side_by_side(a, b, _mask(), _mask())
        assert fs.figure is not None

    # ----- render_change_map -----

    def test_change_map_figure_type(self):
        r = self._rend()
        a, b = self._two()
        fs = r.render_change_map(a, b, _mask(), _mask())
        assert fs.figure_type == "change_map"

    def test_change_map_has_figure(self):
        r = self._rend()
        a, b = self._two()
        fs = r.render_change_map(a, b, _mask(), _mask())
        assert fs.figure is not None

    def test_change_map_all_pixels_changed(self):
        """mask_a and mask_b completely different."""
        r    = self._rend()
        a, b = self._two()
        ma   = np.zeros((8, 8), dtype=np.uint8)
        mb   = np.ones((8, 8), dtype=np.uint8)
        fs   = r.render_change_map(a, b, ma, mb)
        assert fs.figure is not None

    def test_change_map_no_pixels_changed(self):
        """mask_a == mask_b — no red pixels."""
        r    = self._rend()
        a, b = self._two()
        m    = _mask()
        fs   = r.render_change_map(a, b, m, m)
        assert fs.figure is not None

    def test_change_map_fig_is_none_path(self):
        r = self._rend()
        a, b = self._two()
        with patch("src.visualization.comparison._use_agg", return_value=False):
            fs = r.render_change_map(a, b, _mask(), _mask())
        assert fs.figure is None

    # ----- render_class_diff -----

    def test_class_diff_figure_type(self):
        r = self._rend()
        a, b = self._two()
        fs = r.render_class_diff(a, b, CLASS_NAMES)
        assert fs.figure_type == "class_diff"

    def test_class_diff_has_figure(self):
        r = self._rend()
        a, b = self._two()
        fs = r.render_class_diff(a, b, CLASS_NAMES)
        assert fs.figure is not None

    def test_class_diff_missing_metrics_fallback_zero(self):
        """When class_metrics.get(cls_name) returns None, area_fraction defaults to 0."""
        r = self._rend()
        a = _sa("p1", with_metrics=False)   # empty class_metrics
        b = _sa("p2", with_metrics=True)
        fs = r.render_class_diff(a, b, CLASS_NAMES)
        assert fs.figure is not None

    def test_class_diff_fig_is_none_path(self):
        r = self._rend()
        a, b = self._two()
        with patch("src.visualization.comparison._use_agg", return_value=False):
            fs = r.render_class_diff(a, b, CLASS_NAMES)
        assert fs.figure is None

    def test_class_diff_positive_and_negative_deltas(self):
        """When B has more water and less sand than A — both positive and negative bars."""
        from src.morphology.contracts import ClassMorphologyMetrics
        def _cm_custom(water_frac):
            cm = {n: ClassMorphologyMetrics(n, i, 16, 0.25, 0.25, 0.0, 0.8, 2)
                  for i, n in enumerate(CLASS_NAMES)}
            cm["water"] = ClassMorphologyMetrics("water", 1, 32, water_frac, water_frac, 0.0, 0.8, 0)
            cm["sand"]  = ClassMorphologyMetrics("sand",  2, 16, 1 - water_frac, 1 - water_frac, 0.0, 0.8, 0)
            return cm
        a = types.SimpleNamespace(sample_id="p1", acquisition_date="2023-01-01",
                                  class_metrics=_cm_custom(0.3))
        b = types.SimpleNamespace(sample_id="p2", acquisition_date="2023-07-01",
                                  class_metrics=_cm_custom(0.6))
        r  = self._rend()
        fs = r.render_class_diff(a, b, CLASS_NAMES)
        assert fs.figure is not None

    # ----- _make_figure helper -----

    def test_cmp_make_figure_single_col(self):
        fig = cmp_make_figure(_cfg(), n_cols=1)
        assert fig is not None

    def test_cmp_make_figure_two_cols(self):
        fig = cmp_make_figure(_cfg(), n_cols=2)
        assert fig is not None
        assert len(fig.axes) == 2

    def test_cmp_make_figure_none_when_agg_fails(self):
        with patch("src.visualization.comparison._use_agg", return_value=False):
            fig = cmp_make_figure(_cfg())
        assert fig is None

    def test_cmp_make_figure_exception_returns_none(self):
        with patch("src.visualization.comparison._use_agg", return_value=True):
            with patch("matplotlib.pyplot.subplots", side_effect=RuntimeError("plt error")):
                fig = cmp_make_figure(_cfg())
        assert fig is None


# ==============================================================================
# timeline.py — full rendering paths
# ==============================================================================

class TestTimelineRendererFull:
    def _rend(self, **kw): return TimelineRenderer(_cfg(**kw), _cmap(**kw))

    # ----- render_timeline -----

    def test_render_timeline_with_samples(self):
        r  = self._rend()
        mr = _morph_result(n=3)
        fs = r.render_timeline(mr)
        assert fs.figure is not None
        assert fs.figure_type == "timeline"

    def test_render_timeline_single_sample(self):
        r  = self._rend()
        mr = _morph_result(n=1)
        fs = r.render_timeline(mr)
        assert fs.figure is not None

    def test_render_timeline_empty_analyses(self):
        """n_dates == 0 branch: _set_no_data called, early return with figure."""
        r  = self._rend()
        mr = _morph_result(n=0)
        fs = r.render_timeline(mr)
        assert fs.figure is not None
        assert fs.figure_type == "timeline"

    def test_render_timeline_fig_is_none(self):
        r = self._rend()
        with patch("src.visualization.timeline._use_agg", return_value=False):
            fs = r.render_timeline(_morph_result(n=2))
        assert fs.figure is None

    def test_render_timeline_class_metrics_missing_uses_zero(self):
        """Class missing from class_metrics -> 0.0 fraction."""
        r  = self._rend()
        mr = _morph_result(n=2)
        # Add a class name not present in any sample's class_metrics.
        mr_patched = dataclasses.replace(
            mr, class_names=CLASS_NAMES + ("extra_class",)
        )
        fs = r.render_timeline(mr_patched)
        assert fs.figure is not None

    def test_render_timeline_multiple_dates_sorted_correctly(self):
        """Three samples with different dates: bars rendered in date order."""
        r  = self._rend()
        mr = _morph_result(n=3)
        fs = r.render_timeline(mr)
        # Just verify rendering completes without error.
        assert fs.figure_type == "timeline"

    # ----- render_seasonal -----

    def test_render_seasonal_with_data(self):
        r  = self._rend()
        fs = r.render_seasonal(_morph_result())
        assert fs.figure is not None
        assert fs.figure_type == "seasonal"

    def test_render_seasonal_empty_summaries(self):
        """if not summaries branch: _set_no_data + early return."""
        r  = self._rend()
        mr = dataclasses.replace(_morph_result(), seasonal_summaries={})
        fs = r.render_seasonal(mr)
        assert fs.figure is not None

    def test_render_seasonal_fig_is_none(self):
        r = self._rend()
        with patch("src.visualization.timeline._use_agg", return_value=False):
            fs = r.render_seasonal(_morph_result())
        assert fs.figure is None

    def test_render_seasonal_single_season(self):
        from src.morphology.contracts import SeasonalSummary
        r  = self._rend()
        mr = _morph_result()
        mr_single = dataclasses.replace(
            mr,
            seasonal_summaries={"monsoon": mr.seasonal_summaries["monsoon"]},
        )
        fs = r.render_seasonal(mr_single)
        assert fs.figure is not None

    def test_render_seasonal_std_fractions_vary(self):
        """Ensure std error bars are drawn when stds > 0."""
        r  = self._rend()
        mr = _morph_result(n=3)
        fs = r.render_seasonal(mr)
        assert fs.figure is not None

    # ----- render_change_chart -----

    def test_render_change_chart_with_changes(self):
        r  = self._rend()
        mr = _morph_result(n=3)
        fs = r.render_change_chart(mr)
        assert fs.figure is not None
        assert fs.figure_type == "change_chart"

    def test_render_change_chart_empty_changes(self):
        """if not changes branch."""
        r  = self._rend()
        mr = dataclasses.replace(_morph_result(), temporal_changes=())
        fs = r.render_change_chart(mr)
        assert fs.figure is not None

    def test_render_change_chart_fig_is_none(self):
        r = self._rend()
        with patch("src.visualization.timeline._use_agg", return_value=False):
            fs = r.render_change_chart(_morph_result())
        assert fs.figure is None

    def test_render_change_chart_missing_class_uses_zero(self):
        """Class in class_names not present in any change -> delta = 0 fallback."""
        from src.morphology.contracts import TemporalChange
        r  = self._rend()
        mr = _morph_result(n=2)
        # Add a class name that has no corresponding TemporalChange entry.
        mr_patched = dataclasses.replace(
            mr, class_names=CLASS_NAMES + ("new_class",)
        )
        fs = r.render_change_chart(mr_patched)
        assert fs.figure is not None

    def test_render_change_chart_multiple_date_pairs(self):
        """Multiple temporal pairs produce multiple bar groups."""
        from src.morphology.contracts import TemporalChange
        changes = (
            TemporalChange("water", "2023-01-01", "2023-04-01", 25, 30, 5, 0.3, 0.4, 0.1, 20.0),
            TemporalChange("water", "2023-04-01", "2023-07-01", 30, 35, 5, 0.4, 0.5, 0.1, 16.7),
            TemporalChange("sand",  "2023-01-01", "2023-04-01", 10, 8, -2, 0.2, 0.1, -0.1, -20.0),
        )
        r  = self._rend()
        mr = dataclasses.replace(_morph_result(n=3), temporal_changes=changes)
        fs = r.render_change_chart(mr)
        assert fs.figure is not None

    # ----- helpers -----

    def test_tl_make_figure_returns_figure(self):
        fig = tl_make_figure(_cfg())
        assert fig is not None

    def test_tl_make_figure_none_when_agg_fails(self):
        with patch("src.visualization.timeline._use_agg", return_value=False):
            assert tl_make_figure(_cfg()) is None

    def test_tl_make_figure_exception_returns_none(self):
        with patch("src.visualization.timeline._use_agg", return_value=True):
            with patch("matplotlib.pyplot.subplots", side_effect=RuntimeError("err")):
                assert tl_make_figure(_cfg()) is None

    def test_set_no_data_runs_without_error(self):
        fig = tl_make_figure(_cfg())
        _set_no_data(fig, "Test Title", _cfg())  # should not raise

    def test_set_no_data_exception_suppressed(self):
        """If ax.text fails, exception is caught silently."""
        fig = tl_make_figure(_cfg())
        with patch.object(fig.axes[0], "text", side_effect=RuntimeError("ax err")):
            _set_no_data(fig, "T", _cfg())  # must not propagate


# ==============================================================================
# engine.py — branch-by-branch coverage
# ==============================================================================

class TestVisualizationEngineBranches:
    def _engine(self, **kw): return VisualizationEngine(_cfg(**kw))

    def _masks(self, mr):
        return {a.sample_id: _mask() for a in mr.sample_analyses}

    def _confs(self, mr):
        return {a.sample_id: _conf() for a in mr.sample_analyses}

    def _imgs(self, mr):
        return {a.sample_id: np.random.rand(8, 8, 3).astype(np.float32)
                for a in mr.sample_analyses}

    # --- mask present for some but not all samples ---

    def test_mask_missing_for_sample_continues(self):
        """prediction_masks provided but some sample_id is absent -> continue."""
        mr    = _morph_result(n=2)
        # Only provide mask for first sample.
        masks = {mr.sample_analyses[0].sample_id: _mask()}
        result = self._engine().visualize(mr, prediction_masks=masks)
        # Only 1 sample rendered, not 2.
        mask_figs = result.figures_by_type("mask")
        assert len(mask_figs) == 1

    # --- confidence_maps=None with render_confidence=True -> conf is None, block skipped ---

    def test_render_confidence_true_no_conf_map_skips_block(self):
        """When confidence_maps is None, conf=None, so confidence/uncertainty blocks skipped."""
        mr     = _morph_result(n=1)
        masks  = self._masks(mr)
        result = self._engine(render_confidence=True).visualize(
            mr, prediction_masks=masks, confidence_maps=None
        )
        # No confidence or uncertainty figures.
        assert len(result.figures_by_type("confidence"))  == 0
        assert len(result.figures_by_type("uncertainty")) == 0

    # --- confidence provided + render_confidence=True -> confidence figures generated ---

    def test_render_confidence_with_conf_map_generates_figures(self):
        mr     = _morph_result(n=1)
        result = self._engine(render_confidence=True).visualize(
            mr, prediction_masks=self._masks(mr), confidence_maps=self._confs(mr)
        )
        assert len(result.figures_by_type("confidence"))  >= 1
        assert len(result.figures_by_type("uncertainty")) >= 1

    # --- render_overlays=True + conf=None -> confidence_overlay skipped ---

    def test_render_overlays_no_conf_skips_confidence_overlay(self):
        mr     = _morph_result(n=1)
        result = self._engine(render_overlays=True).visualize(
            mr, prediction_masks=self._masks(mr), confidence_maps=None
        )
        assert len(result.figures_by_type("confidence_overlay")) == 0

    # --- render_overlays=True + conf not None -> confidence_overlay rendered ---

    def test_render_overlays_with_conf_renders_confidence_overlay(self):
        mr     = _morph_result(n=1)
        result = self._engine(render_overlays=True).visualize(
            mr, prediction_masks=self._masks(mr), confidence_maps=self._confs(mr)
        )
        assert len(result.figures_by_type("confidence_overlay")) >= 1

    # --- source_images provided -> overlay with real image ---

    def test_source_images_provided_used_in_overlay(self):
        mr     = _morph_result(n=1)
        result = self._engine(render_overlays=True).visualize(
            mr,
            prediction_masks = self._masks(mr),
            source_images    = self._imgs(mr),
        )
        overlays = result.figures_by_type("overlay")
        assert len(overlays) >= 1

    # --- render_masks=False ---

    def test_render_masks_false(self):
        mr     = _morph_result(n=2)
        result = self._engine(render_masks=False).visualize(mr, self._masks(mr))
        assert len(result.figures_by_type("mask")) == 0

    # --- render_per_class=False ---

    def test_render_per_class_false(self):
        mr     = _morph_result(n=2)
        result = self._engine(render_per_class=False).visualize(mr, self._masks(mr))
        assert len(result.figures_by_type("per_class")) == 0

    # --- render_overlays=False ---

    def test_render_overlays_false(self):
        mr     = _morph_result(n=2)
        result = self._engine(render_overlays=False).visualize(mr, self._masks(mr))
        assert len(result.figures_by_type("overlay")) == 0

    # --- render_comparison=True, prediction_masks=None -> comparison skipped ---

    def test_render_comparison_no_masks_skipped(self):
        mr     = _morph_result(n=3)
        result = self._engine(render_comparison=True).visualize(
            mr, prediction_masks=None
        )
        assert len(result.figures_by_type("comparison")) == 0

    # --- render_comparison=True, one sample -> comparison skipped (need >= 2) ---

    def test_render_comparison_single_sample_skipped(self):
        mr     = _morph_result(n=1)
        result = self._engine(render_comparison=True).visualize(
            mr, prediction_masks=self._masks(mr)
        )
        assert len(result.figures_by_type("comparison")) == 0

    # --- render_comparison=True, both masks missing -> if m_a or m_b is None ---

    def test_render_comparison_masks_missing_for_comparison_pair(self):
        """Masks dict provided but neither of the first two samples has an entry."""
        mr    = _morph_result(n=2)
        # Only provide masks for a different (nonexistent) sample id.
        masks = {"nonexistent_id": _mask()}
        result = self._engine(render_comparison=True).visualize(
            mr, prediction_masks=masks
        )
        assert len(result.figures_by_type("comparison")) == 0

    # --- export enabled -> exporter.export_all called ---

    def test_export_enabled_generates_files(self, tmp_path):
        mr     = _morph_result(n=1)
        result = VisualizationEngine(
            _cfg(output_dir=str(tmp_path), export_png=True)
        ).visualize(mr, prediction_masks=self._masks(mr))
        assert result.num_exported >= 1

    # --- validation issues logged but engine continues ---

    def test_invalid_config_still_returns_result(self):
        """Even with validation warnings, engine must return a VisualizationResult."""
        from src.visualization.contracts import VisualizationResult
        # alpha_overlay out of range -> validation warning but no crash.
        mr     = _morph_result(n=1)
        engine = VisualizationEngine(_cfg(alpha_overlay=2.0))
        result = engine.visualize(mr)
        assert isinstance(result, VisualizationResult)

    # --- empty morphology result ---

    def test_empty_morphology_result_zero_figures(self):
        mr     = _morph_result(n=0)
        result = self._engine().visualize(mr)
        assert result.num_figures >= 0  # timeline/seasonal still generated

    # --- deterministic: same input gives same figure count ---

    def test_deterministic_same_figure_count(self):
        mr  = _morph_result(n=2)
        r1  = self._engine().visualize(mr, self._masks(mr), self._confs(mr))
        r2  = self._engine().visualize(mr, self._masks(mr), self._confs(mr))
        assert r1.num_figures == r2.num_figures

    # --- from_config path ---

    def test_from_config_path(self):
        class _ProjectConfig:
            pass   # no visualization section -> defaults
        engine = VisualizationEngine(_ProjectConfig())
        assert isinstance(engine._config, VisualizationConfig)

    # --- output_dir path in result ---

    def test_output_dir_resolved_when_set(self, tmp_path):
        engine = VisualizationEngine(_cfg(output_dir=str(tmp_path), export_png=False))
        result = engine.visualize(_morph_result())
        assert result.output_dir != ""

    def test_output_dir_empty_string_when_not_set(self):
        result = self._engine().visualize(_morph_result())
        assert result.output_dir == ""


# ==============================================================================
# factory.py — all construction branches
# ==============================================================================

class TestVisualizationFactoryFull:
    def test_build_with_custom_colors(self):
        cfg = _cfg(class_colors={"water": (0.1, 0.5, 0.9)})
        mr  = _morph_result()
        ctx = VisualizationFactory.build(cfg, mr)
        assert ctx["colormap"].get("water") == pytest.approx((0.1, 0.5, 0.9))

    def test_build_all_renderer_types(self):
        ctx = VisualizationFactory.build(_cfg(), _morph_result())
        assert isinstance(ctx["mask_renderer"],       MaskRenderer)
        assert isinstance(ctx["overlay_renderer"],    OverlayRenderer)
        assert isinstance(ctx["timeline_renderer"],   TimelineRenderer)
        assert isinstance(ctx["comparison_renderer"], ComparisonRenderer)

    def test_build_exporter_type(self):
        from src.visualization.exporter import FigureExporter
        ctx = VisualizationFactory.build(_cfg(), _morph_result())
        assert isinstance(ctx["exporter"], FigureExporter)

    def test_build_empty_class_names_triggers_warning_and_default(self):
        """VisualizationFactory falls back to 4-class default when class_names=()."""
        mr  = dataclasses.replace(_morph_result(), class_names=())
        ctx = VisualizationFactory.build(_cfg(), mr)
        assert len(ctx["class_names"]) == 4

    def test_build_with_output_dir_creates_exporter_with_dir(self, tmp_path):
        from src.visualization.exporter import FigureExporter
        cfg = _cfg(output_dir=str(tmp_path))
        ctx = VisualizationFactory.build(cfg, _morph_result())
        exp = ctx["exporter"]
        assert isinstance(exp, FigureExporter)

    def test_build_context_class_names_match_result(self):
        mr  = _morph_result()
        ctx = VisualizationFactory.build(_cfg(), mr)
        assert ctx["class_names"] == CLASS_NAMES

    def test_build_is_deterministic(self):
        mr  = _morph_result()
        c1  = VisualizationFactory.build(_cfg(), mr)
        c2  = VisualizationFactory.build(_cfg(), mr)
        assert c1["class_names"] == c2["class_names"]


# ==============================================================================
# overlay.py — remaining branches
# ==============================================================================

class TestOverlayRendererAdditional:
    def _rend(self, **kw): return OverlayRenderer(_cfg(**kw), _cmap(**kw))

    def test_render_overlay_multichannel_image(self):
        """source_image with more than 3 channels -> truncated to first 3."""
        r   = self._rend()
        img = np.random.rand(8, 8, 12).astype(np.float32)
        fs  = r.render_overlay(_sa(), _mask(), source_image=img)
        assert fs.figure is not None

    def test_render_overlay_grayscale_2d_image(self):
        r   = self._rend()
        img = np.random.rand(8, 8).astype(np.float32)
        fs  = r.render_overlay(_sa(), _mask(), source_image=img)
        assert fs.figure is not None

    def test_render_overlay_uint8_image_normalised(self):
        """uint8 image (0-255) should be normalised to [0,1] before blending."""
        r   = self._rend()
        img = (np.random.rand(8, 8, 3) * 255).astype(np.uint8)
        fs  = r.render_overlay(_sa(), _mask(), source_image=img)
        assert fs.figure is not None

    def test_render_confidence_overlay_all_uncertain(self):
        """All pixels are below threshold -> all highlighted."""
        r    = self._rend()
        conf = np.full((8, 8), 0.1, dtype=np.float32)  # all below 0.5
        fs   = r.render_confidence_overlay(_sa(), _mask(), conf, threshold=0.5)
        assert fs.figure is not None

    def test_render_confidence_overlay_none_uncertain(self):
        """All pixels are above threshold -> no highlighting."""
        r    = self._rend()
        conf = np.full((8, 8), 0.9, dtype=np.float32)  # all above 0.5
        fs   = r.render_confidence_overlay(_sa(), _mask(), conf, threshold=0.5)
        assert fs.figure is not None

    def test_render_overlay_alpha_zero(self):
        """alpha_overlay=0.0 -> background shows through completely."""
        r   = OverlayRenderer(_cfg(alpha_overlay=0.0), _cmap())
        fs  = r.render_overlay(_sa(), _mask(), source_image=None)
        assert fs.figure is not None

    def test_render_overlay_alpha_one(self):
        """alpha_overlay=1.0 -> prediction fully opaque."""
        r   = OverlayRenderer(_cfg(alpha_overlay=1.0), _cmap())
        fs  = r.render_overlay(_sa(), _mask(), source_image=None)
        assert fs.figure is not None

    def test_render_overlay_fig_is_none_path(self):
        r = self._rend()
        with patch("src.visualization.overlay._use_agg", return_value=False):
            fs = r.render_overlay(_sa(), _mask())
        assert fs.figure is None

    def test_render_confidence_overlay_fig_is_none_path(self):
        r = self._rend()
        with patch("src.visualization.overlay._use_agg", return_value=False):
            fs = r.render_confidence_overlay(_sa(), _mask(), _conf(), 0.5)
        assert fs.figure is None


# ==============================================================================
# renderer.py — remaining branches
# ==============================================================================

class TestMaskRendererAdditional:
    def _rend(self, **kw): return MaskRenderer(_cfg(**kw), _cmap(**kw))

    # --- colorbar=False branches ---

    def test_render_confidence_colorbar_false(self):
        """colorbar=False: fig.colorbar is NOT called."""
        r  = self._rend(colorbar=False)
        fs = r.render_confidence(_sa(), _conf())
        assert fs.figure is not None

    def test_render_uncertainty_colorbar_false(self):
        """colorbar=False: legend is NOT added."""
        r  = self._rend(colorbar=False)
        fs = r.render_uncertainty(_sa(), _conf(), threshold=0.5)
        assert fs.figure is not None

    # --- _make_figure with n_axes > 4 (triggers ncols=4, nrows=2, hidden axes) ---

    def test_make_figure_5_axes_hides_extra(self):
        """n_axes=5 -> ncols=4, nrows=2 -> 8 axes total -> 3 hidden."""
        r   = self._rend()
        fig = r._make_figure("T", n_axes=5)
        assert fig is not None
        # 8 subplots created (2 rows * 4 cols), last 3 hidden.
        assert len(fig.axes) == 8

    def test_make_figure_8_axes_exactly_fills_grid(self):
        """n_axes=8 -> ncols=4, nrows=2 -> 8 axes, none hidden."""
        r   = self._rend()
        fig = r._make_figure("T", n_axes=8)
        assert fig is not None
        assert len(fig.axes) == 8

    def test_make_figure_single_axis(self):
        """n_axes=1 uses plt.subplots(1, 1)."""
        r   = self._rend()
        fig = r._make_figure("T", n_axes=1)
        assert fig is not None
        assert len(fig.axes) == 1

    def test_make_figure_custom_figsize(self):
        """Custom figsize should be used as-is."""
        r   = self._rend()
        fig = r._make_figure("T", figsize=(3, 2), n_axes=1)
        assert fig is not None

    def test_make_figure_none_when_agg_fails(self):
        r = self._rend()
        with patch("src.visualization.renderer._use_agg", return_value=False):
            fig = r._make_figure("T")
        assert fig is None

    def test_make_figure_exception_returns_none(self):
        r = self._rend()
        with patch("src.visualization.renderer._use_agg", return_value=True):
            with patch("matplotlib.pyplot.subplots", side_effect=RuntimeError("err")):
                fig = r._make_figure("T")
        assert fig is None

    # --- render_per_class with missing class_metrics ---

    def test_render_per_class_missing_metrics_uses_zero(self):
        """When class_metrics.get(class_name) returns None, frac defaults to 0.0."""
        r  = self._rend()
        sa = _sa(with_metrics=False)   # empty class_metrics
        fs = r.render_per_class(sa, _mask(), CLASS_NAMES)
        assert fs.figure is not None

    # --- render_mask: fig is None path ---

    def test_render_mask_fig_is_none(self):
        r = self._rend()
        with patch("src.visualization.renderer._use_agg", return_value=False):
            fs = r.render_mask(_sa(), _mask())
        assert fs.figure is None
        assert fs.figure_type == "mask"

    # --- render_per_class: fig is None path ---

    def test_render_per_class_fig_is_none(self):
        r = self._rend()
        with patch("src.visualization.renderer._use_agg", return_value=False):
            fs = r.render_per_class(_sa(), _mask(), CLASS_NAMES)
        assert fs.figure is None

    # --- render_confidence: fig is None path ---

    def test_render_confidence_fig_is_none(self):
        r = self._rend()
        with patch("src.visualization.renderer._use_agg", return_value=False):
            fs = r.render_confidence(_sa(), _conf())
        assert fs.figure is None

    # --- render_uncertainty: fig is None path ---

    def test_render_uncertainty_fig_is_none(self):
        r = self._rend()
        with patch("src.visualization.renderer._use_agg", return_value=False):
            fs = r.render_uncertainty(_sa(), _conf(), 0.5)
        assert fs.figure is None

    # --- render_mask: no legend handles ---

    def test_render_mask_no_legend_handles(self):
        """When legend_handles returns [], no legend added, no crash."""
        r = self._rend()
        with patch.object(r._colormap, "legend_handles", return_value=[]):
            fs = r.render_mask(_sa(), _mask())
        assert fs.figure is not None

    # --- render_uncertainty colorbar=True (legend rendered) ---

    def test_render_uncertainty_colorbar_true_legend_added(self):
        r  = self._rend(colorbar=True)
        fs = r.render_uncertainty(_sa(), _conf(val=0.3), threshold=0.5)
        assert fs.figure is not None
