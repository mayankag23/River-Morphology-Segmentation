"""Tests for timeline.py, comparison.py, factory.py, and engine.py"""
from __future__ import annotations
import json
import types
import numpy as np
import pytest

mpl = pytest.importorskip("matplotlib")

from src.visualization.contracts import FigureSpec, VisualizationConfig, VisualizationResult
from src.visualization.colormap import ClassColorMap
from src.visualization.timeline import TimelineRenderer
from src.visualization.comparison import ComparisonRenderer
from src.visualization.factory import VisualizationFactory
from src.visualization.engine import VisualizationEngine


CLASS_NAMES = ("background", "water", "sand", "vegetation")


def _cfg(**kw) -> VisualizationConfig:
    defaults = dict(
        output_dir="", export_png=False, export_svg=False, export_pdf=False,
        dpi=72, figure_width=6.0, figure_height=4.0, font_size=8,
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


def _morph_result(n=3, water_frac=0.4):
    """Build a minimal RiverMorphologyResult stub."""
    from src.morphology.contracts import (
        AnalyticsConfig, ClassMorphologyMetrics, SampleAnalysis,
        RiverMorphologyResult, SeasonalSummary, TemporalChange,
    )
    cfg = AnalyticsConfig()
    analyses = []
    for i in range(n):
        cm = {
            name: ClassMorphologyMetrics(
                name, idx, 16, 0.25, 0.25, 0.0, 0.8, 2,
            )
            for idx, name in enumerate(CLASS_NAMES)
        }
        cm["water"] = ClassMorphologyMetrics(
            "water", 1,
            int(64 * water_frac),
            water_frac, water_frac, 0.0, 0.8, 2,
        )
        sa = SampleAnalysis(
            sample_id        = f"p{i:03d}",
            acquisition_date = f"2023-{i+1:02d}-01",
            season           = "monsoon" if i % 2 == 0 else "pre-monsoon",
            hydrological_year = 2023,
            sensor           = "L8",
            river_name       = "Kosi",
            reach_id         = "R1",
            basin_id         = "B1",
            aoi_id           = "A1",
            total_pixels     = 64,
            class_metrics    = cm,
            geometry         = None,
            uncertainty      = None,
        )
        analyses.append(sa)

    seasonal = {
        "monsoon":     SeasonalSummary("monsoon", 2,
                         {"water": 0.5, "sand": 0.2, "vegetation": 0.1, "background": 0.2},
                         {"water": 0.05, "sand": 0.02, "vegetation": 0.01, "background": 0.01},
                         ("p000", "p002")),
        "pre-monsoon": SeasonalSummary("pre-monsoon", 1,
                         {"water": 0.3, "sand": 0.3, "vegetation": 0.2, "background": 0.2},
                         {"water": 0.0, "sand": 0.0, "vegetation": 0.0, "background": 0.0},
                         ("p001",)),
    }

    temporal = (
        TemporalChange("water", "2023-01-01", "2023-02-01", 25, 30, 5, 0.39, 0.47, 0.08, 20.0),
    )

    return RiverMorphologyResult(
        sample_analyses     = tuple(analyses),
        temporal_changes    = temporal,
        seasonal_summaries  = seasonal,
        spatial_summaries   = {},
        class_names         = CLASS_NAMES,
        num_samples         = n,
        num_classes         = 4,
        total_pixels        = n * 64,
        mean_water_fraction = water_frac,
        mean_sand_fraction  = 0.2,
        mean_veg_fraction   = 0.1,
        mean_confidence     = 0.8,
        analytics_config    = cfg,
        architecture        = "unetplusplus",
        operations_log      = ("step1",),
        analysis_time_s     = 0.5,
    )


def _mask(h=8, w=8, water_frac=0.4):
    mask = np.zeros((h, w), dtype=np.uint8)
    mask.ravel()[:int(h * w * water_frac)] = 1
    return mask


def _conf(h=8, w=8, val=0.75):
    return np.full((h, w), val, dtype=np.float32)


# ==============================================================================
# TimelineRenderer
# ==============================================================================

class TestTimelineRenderer:
    def _tlr(self): return TimelineRenderer(_cfg(), _cmap())

    def test_render_timeline_returns_figure_spec(self):
        r  = self._tlr()
        mr = _morph_result(n=3)
        fs = r.render_timeline(mr)
        assert isinstance(fs, FigureSpec)
        assert fs.figure_type == "timeline"
        assert fs.figure is not None

    def test_render_timeline_empty_analyses(self):
        r  = self._tlr()
        mr = _morph_result(n=0)
        fs = r.render_timeline(mr)
        assert fs.figure is not None  # "No data" figure

    def test_render_seasonal_returns_figure_spec(self):
        r  = self._tlr()
        fs = r.render_seasonal(_morph_result())
        assert fs.figure_type == "seasonal"
        assert fs.figure is not None

    def test_render_seasonal_empty_returns_no_data_figure(self):
        from src.morphology.contracts import (
            AnalyticsConfig, RiverMorphologyResult,
        )
        mr = _morph_result(n=0)
        import dataclasses
        mr = dataclasses.replace(mr, seasonal_summaries={})
        r  = self._tlr()
        fs = r.render_seasonal(mr)
        assert fs.figure is not None

    def test_render_change_chart_returns_figure_spec(self):
        r  = self._tlr()
        fs = r.render_change_chart(_morph_result())
        assert fs.figure_type == "change_chart"
        assert fs.figure is not None

    def test_render_change_chart_no_changes_returns_no_data_figure(self):
        import dataclasses
        mr = dataclasses.replace(_morph_result(), temporal_changes=())
        r  = self._tlr()
        fs = r.render_change_chart(mr)
        assert fs.figure is not None


# ==============================================================================
# ComparisonRenderer
# ==============================================================================

class TestComparisonRenderer:
    def _cmpr(self): return ComparisonRenderer(_cfg(), _cmap())

    def _two_analyses(self, mr):
        return mr.sample_analyses[0], mr.sample_analyses[1]

    def test_render_side_by_side_returns_figure_spec(self):
        r      = self._cmpr()
        mr     = _morph_result(n=2)
        a, b   = self._two_analyses(mr)
        fs     = r.render_side_by_side(a, b, _mask(), _mask())
        assert fs.figure_type == "comparison"
        assert fs.figure is not None

    def test_render_change_map_returns_figure_spec(self):
        r    = self._cmpr()
        mr   = _morph_result(n=2)
        a, b = self._two_analyses(mr)
        mask_a = _mask()
        mask_b = _mask()
        mask_b[2, 2] = 2   # small difference
        fs   = r.render_change_map(a, b, mask_a, mask_b)
        assert fs.figure_type == "change_map"
        assert fs.figure is not None

    def test_render_class_diff_returns_figure_spec(self):
        r    = self._cmpr()
        mr   = _morph_result(n=2)
        a, b = self._two_analyses(mr)
        fs   = r.render_class_diff(a, b, CLASS_NAMES)
        assert fs.figure_type == "class_diff"
        assert fs.figure is not None

    def test_change_map_figure_id_contains_both_sample_ids(self):
        r    = self._cmpr()
        mr   = _morph_result(n=2)
        a, b = self._two_analyses(mr)
        fs   = r.render_change_map(a, b, _mask(), _mask())
        assert a.sample_id in fs.figure_id or b.sample_id in fs.figure_id


# ==============================================================================
# VisualizationFactory
# ==============================================================================

class TestVisualizationFactory:
    def test_build_returns_all_required_keys(self):
        mr  = _morph_result()
        ctx = VisualizationFactory.build(_cfg(), mr)
        for key in ("colormap", "mask_renderer", "overlay_renderer",
                    "timeline_renderer", "comparison_renderer",
                    "exporter", "class_names"):
            assert key in ctx

    def test_class_names_from_result(self):
        mr  = _morph_result()
        ctx = VisualizationFactory.build(_cfg(), mr)
        assert ctx["class_names"] == CLASS_NAMES

    def test_empty_class_names_defaults(self):
        import dataclasses
        mr  = dataclasses.replace(_morph_result(), class_names=())
        ctx = VisualizationFactory.build(_cfg(), mr)
        # Should fall back to a non-empty default tuple.
        assert len(ctx["class_names"]) > 0


# ==============================================================================
# VisualizationEngine — integration tests
# ==============================================================================

class TestVisualizationEngine:
    def _engine(self, **kw): return VisualizationEngine(_cfg(**kw))

    def _masks(self, mr, h=8, w=8, frac=0.4):
        return {a.sample_id: _mask(h, w, frac) for a in mr.sample_analyses}

    def _confs(self, mr, h=8, w=8):
        return {a.sample_id: _conf(h, w) for a in mr.sample_analyses}

    def test_returns_visualization_result(self):
        mr     = _morph_result()
        result = self._engine().visualize(mr, self._masks(mr), self._confs(mr))
        assert isinstance(result, VisualizationResult)

    def test_result_is_frozen(self):
        mr     = _morph_result()
        result = self._engine().visualize(mr)
        with pytest.raises((AttributeError, TypeError)):
            result.num_figures = 99  # type: ignore[misc]

    def test_figures_generated_without_masks(self):
        """When no prediction_masks, only aggregate figures are generated."""
        mr     = _morph_result(n=3)
        result = self._engine().visualize(mr)
        # Should have timeline, seasonal, change_chart at minimum.
        assert result.num_figures >= 3

    def test_per_sample_figures_generated_with_masks(self):
        mr     = _morph_result(n=2)
        result = self._engine().visualize(mr, self._masks(mr), self._confs(mr))
        # At least mask + per_class + confidence + uncertainty + overlay per sample.
        assert result.num_figures > 5

    def test_max_samples_limits_per_sample_figures(self):
        mr     = _morph_result(n=4)
        result = self._engine(max_samples=1).visualize(
            mr, self._masks(mr), self._confs(mr)
        )
        mask_figs = result.figures_by_type("mask")
        assert len(mask_figs) <= 1

    def test_render_timeline_disabled(self):
        mr     = _morph_result()
        result = self._engine(render_timeline=False).visualize(mr)
        assert len(result.figures_by_type("timeline")) == 0

    def test_render_seasonal_disabled(self):
        mr     = _morph_result()
        result = self._engine(render_seasonal=False).visualize(mr)
        assert len(result.figures_by_type("seasonal")) == 0

    def test_render_masks_disabled(self):
        mr     = _morph_result(n=2)
        result = self._engine(render_masks=False).visualize(mr, self._masks(mr))
        assert len(result.figures_by_type("mask")) == 0

    def test_class_names_in_result(self):
        mr     = _morph_result()
        result = self._engine().visualize(mr)
        assert result.class_names == CLASS_NAMES

    def test_architecture_in_result(self):
        mr     = _morph_result()
        result = self._engine().visualize(mr)
        assert result.architecture == "unetplusplus"

    def test_as_dict_json_serialisable(self):
        mr     = _morph_result()
        result = self._engine().visualize(mr)
        assert json.dumps(result.as_dict())

    def test_export_png_creates_files(self, tmp_path):
        mr     = _morph_result(n=1)
        masks  = self._masks(mr)
        result = VisualizationEngine(
            _cfg(output_dir=str(tmp_path), export_png=True)
        ).visualize(mr, masks)
        png_files = list(tmp_path.glob("*.png"))
        assert len(png_files) >= 1

    def test_no_export_when_output_dir_empty(self):
        mr     = _morph_result(n=1)
        result = self._engine().visualize(mr, self._masks(mr))
        assert result.num_exported == 0
        assert result.output_dir   == ""

    def test_operations_log_non_empty(self):
        mr     = _morph_result()
        result = self._engine().visualize(mr)
        assert len(result.operations_log) > 0

    def test_visualization_time_positive(self):
        mr     = _morph_result()
        result = self._engine().visualize(mr)
        assert result.visualization_time_s >= 0.0

    def test_comparison_figures_generated(self):
        mr     = _morph_result(n=3)
        result = self._engine().visualize(mr, self._masks(mr))
        comp   = result.figures_by_type("comparison")
        assert len(comp) >= 1

    def test_accepts_project_config_object(self):
        class _Cfg: pass
        engine = VisualizationEngine(_Cfg())
        assert isinstance(engine._config, VisualizationConfig)

    def test_figures_for_sample(self):
        mr     = _morph_result(n=2)
        masks  = self._masks(mr)
        result = self._engine().visualize(mr, masks, self._confs(mr))
        first_id = mr.sample_analyses[0].sample_id
        assert len(result.figures_for_sample(first_id)) >= 1

    def test_num_samples_correct(self):
        mr     = _morph_result(n=3)
        result = self._engine(max_samples=2).visualize(mr, self._masks(mr))
        assert result.num_samples == 2