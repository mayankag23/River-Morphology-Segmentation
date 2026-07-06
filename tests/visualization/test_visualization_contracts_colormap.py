"""Tests for contracts.py and colormap.py"""
from __future__ import annotations
import json
import numpy as np
import pytest
from src.visualization.contracts import FigureSpec, VisualizationConfig, VisualizationResult
from src.visualization.colormap import ClassColorMap, ColorRegistry, DEFAULT_COLORS


CLASS_NAMES = ("background", "water", "sand", "vegetation")


# ==============================================================================
# VisualizationConfig
# ==============================================================================

class TestVisualizationConfig:
    def test_frozen(self):
        cfg = VisualizationConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.dpi = 300  # type: ignore[misc]

    def test_defaults(self):
        cfg = VisualizationConfig()
        assert cfg.dpi             == 150
        assert cfg.alpha_overlay   == pytest.approx(0.5)
        assert cfg.export_png      is True
        assert cfg.colormap_name   == "viridis"
        assert cfg.max_samples     == 0
        assert cfg.render_masks    is True

    def test_from_config_reads_values(self):
        class _V:
            output_dir="out"; export_png=True; export_svg=False; export_pdf=False
            dpi=200; figure_width=8.0; figure_height=5.0; font_size=9
            title_font_size=11; colorbar=False; alpha_overlay=0.4
            alpha_confidence=0.7; class_colors={}; background_color="white"
            render_masks=True; render_overlays=False; render_confidence=True
            render_per_class=True; render_timeline=True; render_seasonal=True
            render_comparison=False; max_samples=5; colormap_name="plasma"
        class _Cfg:
            visualization = _V()
        cfg = VisualizationConfig.from_config(_Cfg())
        assert cfg.dpi           == 200
        assert cfg.alpha_overlay == pytest.approx(0.4)
        assert cfg.max_samples   == 5
        assert cfg.colormap_name == "plasma"

    def test_from_config_class_colors(self):
        class _V:
            output_dir=""; export_png=True; export_svg=False; export_pdf=False
            dpi=150; figure_width=10.0; figure_height=6.0; font_size=10
            title_font_size=12; colorbar=True; alpha_overlay=0.5
            alpha_confidence=0.6; class_colors={"water": (0.1, 0.5, 0.9)}
            background_color="white"; render_masks=True; render_overlays=True
            render_confidence=True; render_per_class=True; render_timeline=True
            render_seasonal=True; render_comparison=True; max_samples=0
            colormap_name="viridis"
        class _Cfg:
            visualization = _V()
        cfg = VisualizationConfig.from_config(_Cfg())
        assert "water" in cfg.class_colors
        assert cfg.class_colors["water"][0] == pytest.approx(0.1)

    def test_from_config_no_section_returns_defaults(self):
        class _Cfg: pass
        assert VisualizationConfig.from_config(_Cfg()) == VisualizationConfig()


# ==============================================================================
# FigureSpec
# ==============================================================================

class TestFigureSpec:
    def test_mutable(self):
        spec = FigureSpec("f1", "mask", "Test Figure")
        spec.export_paths.append("/tmp/f1.png")
        assert len(spec.export_paths) == 1

    def test_summary_dict_no_figure(self):
        spec = FigureSpec("f1", "mask", "Test", figure=None)
        d    = spec.summary_dict()
        assert d["has_figure"] is False
        assert json.dumps(d)

    def test_summary_dict_with_figure_object(self):
        spec = FigureSpec("f1", "mask", "Test", figure=object())
        d    = spec.summary_dict()
        assert d["has_figure"] is True

    def test_defaults(self):
        spec = FigureSpec("f1", "mask", "Test")
        assert spec.sample_id        == ""
        assert spec.acquisition_date == ""
        assert spec.export_paths     == []


# ==============================================================================
# VisualizationResult
# ==============================================================================

class TestVisualizationResult:
    def _make(self):
        specs = (FigureSpec("f1", "mask", "M1"), FigureSpec("f2", "timeline", "TL"))
        return VisualizationResult(
            figures=specs, num_figures=2, num_exported=0,
            visualization_config=VisualizationConfig(),
            architecture="unetplusplus", num_samples=3,
            class_names=CLASS_NAMES, output_dir="",
            operations_log=("step1",), visualization_time_s=1.2,
        )

    def test_frozen(self):
        r = self._make()
        with pytest.raises((AttributeError, TypeError)):
            r.num_figures = 99  # type: ignore[misc]

    def test_summary_lines_ascii(self):
        for l in self._make().summary_lines():
            assert all(ord(c) < 128 for c in l)

    def test_as_dict_json_serialisable(self):
        assert json.dumps(self._make().as_dict())

    def test_figures_by_type(self):
        r      = self._make()
        masks  = r.figures_by_type("mask")
        assert len(masks) == 1 and masks[0].figure_id == "f1"

    def test_figures_for_sample(self):
        specs = (FigureSpec("m_p1", "mask", "M", sample_id="p1"),
                 FigureSpec("m_p2", "mask", "M", sample_id="p2"))
        r = VisualizationResult(
            figures=specs, num_figures=2, num_exported=0,
            visualization_config=VisualizationConfig(),
            architecture="unetplusplus", num_samples=2,
            class_names=CLASS_NAMES, output_dir="",
            operations_log=(), visualization_time_s=0.5,
        )
        assert len(r.figures_for_sample("p1")) == 1
        assert len(r.figures_for_sample("p2")) == 1
        assert len(r.figures_for_sample("p3")) == 0


# ==============================================================================
# ClassColorMap
# ==============================================================================

class TestClassColorMap:
    def _cmap(self, colors: dict | None = None) -> ClassColorMap:
        cfg = VisualizationConfig(class_colors=colors or {})
        return ClassColorMap(cfg, CLASS_NAMES)

    def teardown_method(self):
        ColorRegistry.clear()

    def test_default_colors_assigned(self):
        cmap = self._cmap()
        # Water should be the default steel blue
        assert cmap.get("water") == DEFAULT_COLORS["water"]

    def test_config_color_overrides_default(self):
        cmap = self._cmap({"water": (0.0, 0.0, 1.0)})
        assert cmap.get("water") == pytest.approx((0.0, 0.0, 1.0))

    def test_registry_override(self):
        ColorRegistry.register("water", (0.5, 0.5, 0.5))
        cmap = self._cmap()   # no config override for water
        assert cmap.get("water") == pytest.approx((0.5, 0.5, 0.5))

    def test_config_beats_registry(self):
        ColorRegistry.register("water", (0.5, 0.5, 0.5))
        cmap = self._cmap({"water": (0.1, 0.2, 0.3)})
        # config.class_colors is highest priority
        assert cmap.get("water") == pytest.approx((0.1, 0.2, 0.3))

    def test_unknown_class_returns_cycle_color(self):
        cfg  = VisualizationConfig()
        cmap = ClassColorMap(cfg, CLASS_NAMES + ("new_class",))
        color = cmap.get("new_class")
        assert len(color) == 3
        assert all(0.0 <= c <= 1.0 for c in color)

    def test_unknown_class_returns_grey_fallback(self):
        cmap  = self._cmap()
        color = cmap.get("not_in_schema")
        assert color == (0.5, 0.5, 0.5)

    def test_as_list_length(self):
        cmap = self._cmap()
        assert len(cmap.as_list()) == len(CLASS_NAMES)

    def test_to_rgba_mask_shape(self):
        cmap = self._cmap()
        mask = np.zeros((8, 8), dtype=np.uint8)
        rgba = cmap.to_rgba_mask(mask)
        assert rgba.shape == (8, 8, 4)
        assert rgba.dtype == np.float32

    def test_to_rgba_mask_alpha_one(self):
        cmap = self._cmap()
        mask = np.zeros((4, 4), dtype=np.uint8)
        rgba = cmap.to_rgba_mask(mask)
        assert (rgba[:, :, 3] == 1.0).all()

    def test_to_rgba_mask_water_color_correct(self):
        cmap = self._cmap()
        mask = np.ones((4, 4), dtype=np.uint8)   # class 1 = water
        rgba = cmap.to_rgba_mask(mask)
        r, g, b = DEFAULT_COLORS["water"]
        assert rgba[0, 0, 0] == pytest.approx(r)
        assert rgba[0, 0, 1] == pytest.approx(g)

    def test_legend_handles(self):
        mpl = pytest.importorskip("matplotlib")
        cmap    = self._cmap()
        handles = cmap.legend_handles()
        assert len(handles) == len(CLASS_NAMES)


# ==============================================================================
# ColorRegistry
# ==============================================================================

class TestColorRegistry:
    def teardown_method(self):
        ColorRegistry.clear()

    def test_register_and_get(self):
        ColorRegistry.register("my_class", (0.3, 0.4, 0.5))
        assert ColorRegistry.get("my_class") == (0.3, 0.4, 0.5)

    def test_get_unregistered_returns_none(self):
        assert ColorRegistry.get("nonexistent") is None

    def test_clear_removes_all(self):
        ColorRegistry.register("a", (0.1, 0.2, 0.3))
        ColorRegistry.clear()
        assert ColorRegistry.get("a") is None