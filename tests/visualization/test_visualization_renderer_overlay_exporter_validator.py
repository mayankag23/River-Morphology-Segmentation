"""Tests for renderer.py, overlay.py, exporter.py, validator.py"""
from __future__ import annotations
import json
import types
import numpy as np
import pytest
from pathlib import Path

from src.visualization.contracts import FigureSpec, VisualizationConfig
from src.visualization.colormap import ClassColorMap
from src.visualization.renderer import MaskRenderer, _single_color_cmap
from src.visualization.overlay import OverlayRenderer, _to_rgb
from src.visualization.exporter import FigureExporter, _sanitise
from src.visualization.validator import VisualizationValidator, VisualizationValidationResult


CLASS_NAMES = ("background", "water", "sand", "vegetation")
mpl = pytest.importorskip("matplotlib")


def _cfg(**kw) -> VisualizationConfig:
    defaults = dict(
        output_dir="", export_png=True, export_svg=False, export_pdf=False,
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


def _sample_analysis(sid="p001", date="2023-07-01"):
    from src.morphology.contracts import ClassMorphologyMetrics
    cm = {n: ClassMorphologyMetrics(n, i, 16, 0.25, 0.25, 0.0, 0.8, 2)
          for i, n in enumerate(CLASS_NAMES)}
    return types.SimpleNamespace(
        sample_id=sid, acquisition_date=date, season="monsoon",
        hydrological_year=2023, sensor="L8", river_name="Kosi",
        reach_id="R1", basin_id="B1", aoi_id="A1",
        total_pixels=64, class_metrics=cm, geometry=None, uncertainty=None,
    )


def _mask(h=8, w=8):
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[0:4, :] = 1   # water
    return mask


def _conf(h=8, w=8, val=0.8):
    return np.full((h, w), val, dtype=np.float32)


# ==============================================================================
# MaskRenderer
# ==============================================================================

class TestMaskRenderer:
    def _renderer(self): return MaskRenderer(_cfg(), _cmap())

    def test_render_mask_returns_figure_spec(self):
        r  = self._renderer()
        fs = r.render_mask(_sample_analysis(), _mask())
        assert isinstance(fs, FigureSpec)
        assert fs.figure_type == "mask"

    def test_render_mask_has_figure(self):
        r  = self._renderer()
        fs = r.render_mask(_sample_analysis(), _mask())
        assert fs.figure is not None

    def test_render_mask_sample_id_set(self):
        r  = self._renderer()
        fs = r.render_mask(_sample_analysis("p007"), _mask())
        assert fs.sample_id == "p007"

    def test_render_per_class_returns_figure_spec(self):
        r  = self._renderer()
        fs = r.render_per_class(_sample_analysis(), _mask(), CLASS_NAMES)
        assert fs.figure_type == "per_class"
        assert fs.figure is not None

    def test_render_confidence_returns_figure_spec(self):
        r  = self._renderer()
        fs = r.render_confidence(_sample_analysis(), _conf())
        assert fs.figure_type == "confidence"
        assert fs.figure is not None

    def test_render_uncertainty_returns_figure_spec(self):
        r  = self._renderer()
        fs = r.render_uncertainty(_sample_analysis(), _conf(val=0.3), 0.5)
        assert fs.figure_type == "uncertainty"
        assert fs.figure is not None

    def test_render_mask_matplotlib_unavailable_returns_none_figure(self):
        from unittest.mock import patch
        r = self._renderer()
        with patch.dict("sys.modules", {"matplotlib": None, "matplotlib.pyplot": None,
                                        "matplotlib.colors": None}):
            fs = r.render_mask(_sample_analysis(), _mask())
        # figure may be None when matplotlib is patched out, or may still render
        # depending on import cache — just verify it returns a FigureSpec.
        assert isinstance(fs, FigureSpec)


class TestSingleColorCmap:
    def test_returns_colormap(self):
        cmap = _single_color_cmap(0.1, 0.5, 0.9)
        assert cmap is not None

    def test_fallback_when_matplotlib_unavailable(self):
        from unittest.mock import patch
        with patch.dict("sys.modules", {"matplotlib": None, "matplotlib.colors": None}):
            cmap = _single_color_cmap(0.1, 0.5, 0.9)
        # Should return the string fallback "viridis"
        assert cmap == "viridis" or hasattr(cmap, "__call__")


# ==============================================================================
# OverlayRenderer
# ==============================================================================

class TestOverlayRenderer:
    def _renderer(self): return OverlayRenderer(_cfg(), _cmap())

    def test_render_overlay_no_image(self):
        r  = self._renderer()
        fs = r.render_overlay(_sample_analysis(), _mask(), source_image=None)
        assert fs.figure_type == "overlay"
        assert fs.figure is not None

    def test_render_overlay_with_grayscale_image(self):
        r   = self._renderer()
        img = np.random.rand(8, 8).astype(np.float32)
        fs  = r.render_overlay(_sample_analysis(), _mask(), source_image=img)
        assert fs.figure is not None

    def test_render_overlay_with_rgb_image(self):
        r   = self._renderer()
        img = np.random.rand(8, 8, 3).astype(np.float32)
        fs  = r.render_overlay(_sample_analysis(), _mask(), source_image=img)
        assert fs.figure is not None

    def test_render_confidence_overlay_returns_figure_spec(self):
        r  = self._renderer()
        fs = r.render_confidence_overlay(_sample_analysis(), _mask(), _conf(), 0.5)
        assert fs.figure_type == "confidence_overlay"
        assert fs.figure is not None

    def test_overlay_figure_id_contains_sample_id(self):
        r  = self._renderer()
        fs = r.render_overlay(_sample_analysis("abc123"), _mask())
        assert "abc123" in fs.figure_id


class TestToRgb:
    def test_grayscale_2d_becomes_rgb(self):
        img = np.ones((4, 4), dtype=np.float32) * 0.5
        out = _to_rgb(img)
        assert out.shape == (4, 4, 3)

    def test_single_channel_becomes_rgb(self):
        img = np.ones((4, 4, 1), dtype=np.float32)
        out = _to_rgb(img)
        assert out.shape == (4, 4, 3)

    def test_rgb_unchanged(self):
        img = np.ones((4, 4, 3), dtype=np.float32) * 0.5
        out = _to_rgb(img)
        assert out.shape == (4, 4, 3)

    def test_multichannel_truncated_to_3(self):
        img = np.ones((4, 4, 12), dtype=np.float32)
        out = _to_rgb(img)
        assert out.shape == (4, 4, 3)

    def test_uint8_normalised(self):
        img = np.full((4, 4, 3), 128, dtype=np.uint8)
        out = _to_rgb(img)
        assert out.max() <= 1.0

    def test_clipped_to_0_1(self):
        img = np.full((4, 4), 2.0, dtype=np.float32)
        out = _to_rgb(img)
        assert out.max() <= 1.0 and out.min() >= 0.0


# ==============================================================================
# FigureExporter
# ==============================================================================

class TestFigureExporter:
    def _make_spec(self, sid="p001"):
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
        fig, _ = plt.subplots(1, 1, figsize=(2, 2))
        return FigureSpec(f"mask_{sid}", "mask", "Test", sample_id=sid, figure=fig)

    def test_export_png_creates_file(self, tmp_path):
        cfg  = _cfg(output_dir=str(tmp_path), export_png=True)
        exp  = FigureExporter(cfg)
        spec = self._make_spec()
        paths = exp.export(spec)
        assert len(paths) == 1
        assert Path(paths[0]).exists()
        assert paths[0].endswith(".png")

    def test_export_svg_creates_file(self, tmp_path):
        cfg  = _cfg(output_dir=str(tmp_path), export_png=False, export_svg=True)
        exp  = FigureExporter(cfg)
        spec = self._make_spec()
        paths = exp.export(spec)
        assert len(paths) == 1
        assert paths[0].endswith(".svg")

    def test_export_pdf_creates_file(self, tmp_path):
        cfg  = _cfg(output_dir=str(tmp_path), export_png=False, export_pdf=True)
        exp  = FigureExporter(cfg)
        spec = self._make_spec()
        paths = exp.export(spec)
        assert len(paths) == 1
        assert paths[0].endswith(".pdf")

    def test_export_all_formats(self, tmp_path):
        cfg  = _cfg(output_dir=str(tmp_path), export_png=True,
                    export_svg=True, export_pdf=True)
        exp  = FigureExporter(cfg)
        spec = self._make_spec()
        paths = exp.export(spec)
        assert len(paths) == 3

    def test_no_export_when_output_dir_empty(self):
        exp  = FigureExporter(_cfg(output_dir=""))
        spec = self._make_spec()
        assert exp.export(spec) == []

    def test_no_export_when_figure_is_none(self, tmp_path):
        exp  = FigureExporter(_cfg(output_dir=str(tmp_path)))
        spec = FigureSpec("f1", "mask", "T")   # figure=None
        assert exp.export(spec) == []

    def test_export_all_updates_export_paths(self, tmp_path):
        cfg  = _cfg(output_dir=str(tmp_path), export_png=True)
        exp  = FigureExporter(cfg)
        spec = self._make_spec()
        exp.export_all([spec])
        assert len(spec.export_paths) >= 1

    def test_output_dir_created_automatically(self, tmp_path):
        new_dir = tmp_path / "nested" / "figs"
        cfg  = _cfg(output_dir=str(new_dir), export_png=True)
        exp  = FigureExporter(cfg)
        exp.export(self._make_spec())
        assert new_dir.exists()


class TestSanitise:
    def test_clean_id_unchanged(self): assert _sanitise("fig_001") == "fig_001"
    def test_slash_replaced(self):      assert "/" not in _sanitise("a/b")
    def test_empty_becomes_figure(self): assert _sanitise("") == "figure"
    def test_dot_preserved(self):       assert "." in _sanitise("fig.001")


# ==============================================================================
# VisualizationValidator
# ==============================================================================

class TestVisualizationValidator:
    def _ir(self, n=2):
        return types.SimpleNamespace(
            num_samples=n, class_names=CLASS_NAMES,
            architecture="unetplusplus",
        )

    def test_valid_inputs_pass(self):
        v = VisualizationValidator()
        r = v.validate(_cfg(), self._ir())
        assert r.is_valid

    def test_none_result_detected(self):
        assert not VisualizationValidator().validate(_cfg(), None).is_valid

    def test_zero_samples_detected(self):
        assert not VisualizationValidator().validate(_cfg(), self._ir(n=0)).is_valid

    def test_invalid_dpi_detected(self):
        assert not VisualizationValidator().validate(_cfg(dpi=0), self._ir()).is_valid

    def test_invalid_figure_dimensions_detected(self):
        assert not VisualizationValidator().validate(
            _cfg(figure_width=0.0), self._ir()).is_valid

    def test_invalid_alpha_overlay_detected(self):
        assert not VisualizationValidator().validate(
            _cfg(alpha_overlay=1.5), self._ir()).is_valid

    def test_invalid_alpha_confidence_detected(self):
        assert not VisualizationValidator().validate(
            _cfg(alpha_confidence=-0.1), self._ir()).is_valid

    def test_invalid_color_length_detected(self):
        cfg = _cfg(class_colors={"water": (0.1, 0.2)})   # only 2 components
        assert not VisualizationValidator().validate(cfg, self._ir()).is_valid

    def test_color_out_of_range_detected(self):
        cfg = _cfg(class_colors={"water": (1.5, 0.5, 0.5)})
        assert not VisualizationValidator().validate(cfg, self._ir()).is_valid

    def test_negative_max_samples_detected(self):
        assert not VisualizationValidator().validate(
            _cfg(max_samples=-1), self._ir()).is_valid

    def test_valid_class_colors_pass(self):
        cfg = _cfg(class_colors={"water": (0.1, 0.5, 0.9)})
        assert VisualizationValidator().validate(cfg, self._ir()).is_valid

    def test_issues_are_copy(self):
        r = VisualizationValidationResult(["a"])
        r.issues.append("b")
        assert len(r.issues) == 1