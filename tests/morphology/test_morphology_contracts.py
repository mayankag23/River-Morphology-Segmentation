"""Tests for src/morphology/contracts.py"""
from __future__ import annotations
import json
import pytest
from src.morphology.contracts import (
    AnalyticsConfig, ClassMorphologyMetrics, ClassRegionMetrics,
    ConnectedRegionStats, GeometryMetrics, RiverMorphologyResult,
    SampleAnalysis, SeasonalSummary, SpatialSummary, TemporalChange,
    UncertaintyMetrics,
)


class TestAnalyticsConfig:
    def test_frozen(self):
        cfg = AnalyticsConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.pixel_area_m2 = 100.0  # type: ignore[misc]

    def test_defaults(self):
        cfg = AnalyticsConfig()
        assert cfg.pixel_area_m2  == 0.0
        assert cfg.pixel_width_m  == 0.0
        assert cfg.pixel_height_m == 0.0
        assert cfg.water_class    == "water"
        assert cfg.compute_shape_descriptors is False

    def test_from_config_reads_morphology_section(self):
        class _A:
            pixel_area_m2=900.0; pixel_width_m=30.0; pixel_height_m=30.0
            low_confidence_threshold=0.4; min_component_area=8; seasons={}
            water_class="water"; sand_class="sand"; vegetation_class="vegetation"
            background_class="background"; compute_geometry=False
            compute_temporal=True; compute_seasonal=True; compute_uncertainty=True
            compute_shape_descriptors=True
        class _Cfg:
            morphology = _A()
        cfg = AnalyticsConfig.from_config(_Cfg())
        assert cfg.pixel_area_m2 == pytest.approx(900.0)
        assert cfg.pixel_width_m == pytest.approx(30.0)
        assert cfg.compute_shape_descriptors is True

    def test_from_config_accepts_analytics_section_for_backward_compat(self):
        class _A:
            pixel_area_m2=100.0; pixel_width_m=0.0; pixel_height_m=0.0
            low_confidence_threshold=0.5; min_component_area=4; seasons={}
            water_class="water"; sand_class="sand"; vegetation_class="vegetation"
            background_class="background"; compute_geometry=True
            compute_temporal=True; compute_seasonal=True; compute_uncertainty=True
            compute_shape_descriptors=False
        class _Cfg:
            analytics = _A()
        cfg = AnalyticsConfig.from_config(_Cfg())
        assert cfg.pixel_area_m2 == pytest.approx(100.0)

    def test_from_config_no_section_returns_defaults(self):
        class _Cfg: pass
        assert AnalyticsConfig.from_config(_Cfg()) == AnalyticsConfig()


class TestConnectedRegionStats:
    def _make(self, **kw):
        defaults = dict(
            region_id=1, area_px=100, area_m2=90000.0,
            bbox_row_min=5, bbox_row_max=15, bbox_col_min=10, bbox_col_max=20,
            bbox_height=10, bbox_width=10, aspect_ratio=1.0, mean_confidence=0.8,
        )
        defaults.update(kw)
        return ConnectedRegionStats(**defaults)

    def test_frozen(self):
        r = self._make()
        with pytest.raises((AttributeError, TypeError)):
            r.area_px = 999  # type: ignore[misc]

    def test_as_dict_json_serialisable(self):
        assert json.dumps(self._make().as_dict())

    def test_optional_shape_descriptors_default_zero(self):
        r = self._make()
        assert r.perimeter_px  == 0.0
        assert r.compactness   == 0.0
        assert r.elongation    == 0.0

    def test_shape_descriptors_settable(self):
        r = ConnectedRegionStats(
            region_id=1, area_px=50, area_m2=0.0,
            bbox_row_min=0, bbox_row_max=10, bbox_col_min=0, bbox_col_max=5,
            bbox_height=10, bbox_width=5, aspect_ratio=0.5, mean_confidence=0.7,
            perimeter_px=30.0, compactness=0.7, elongation=0.5,
        )
        assert r.perimeter_px  == 30.0
        assert r.compactness   == pytest.approx(0.7)
        assert r.elongation    == pytest.approx(0.5)


class TestClassRegionMetrics:
    def test_frozen(self):
        crm = ClassRegionMetrics("water", 1, 3, 200, 18000.0, 80.0, 20.0, 0.015, 12.5, ())
        with pytest.raises((AttributeError, TypeError)):
            crm.region_count = 99  # type: ignore[misc]

    def test_as_dict_json_serialisable(self):
        crm = ClassRegionMetrics("sand", 2, 2, 100, 0.0, 60.0, 10.0, 0.02, 8.0, ())
        assert json.dumps(crm.as_dict())

    def test_regions_tuple(self):
        crm = ClassRegionMetrics("water", 1, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, ())
        assert isinstance(crm.regions, tuple)


class TestClassMorphologyMetrics:
    def test_frozen(self):
        cm = ClassMorphologyMetrics("water", 1, 100, 0.5, 0.5, 90000.0, 0.8, 10,
                                    confidence_weighted_area=82.5,
                                    confidence_weighted_area_m2=74250.0)
        with pytest.raises((AttributeError, TypeError)):
            cm.pixel_count = 999  # type: ignore[misc]

    def test_confidence_weighted_area_in_as_dict(self):
        cm = ClassMorphologyMetrics("water", 1, 100, 0.5, 0.5, 0.0, 0.8, 5,
                                    confidence_weighted_area=80.0,
                                    confidence_weighted_area_m2=72000.0)
        d = cm.as_dict()
        assert "confidence_weighted_area"    in d
        assert "confidence_weighted_area_m2" in d
        assert d["confidence_weighted_area"]    == pytest.approx(80.0)
        assert d["confidence_weighted_area_m2"] == pytest.approx(72000.0)

    def test_default_confidence_weighted_area_zero(self):
        cm = ClassMorphologyMetrics("bg", 0, 0, 0.0, 0.0, 0.0, 0.0, 0)
        assert cm.confidence_weighted_area    == 0.0
        assert cm.confidence_weighted_area_m2 == 0.0

    def test_as_dict_json_serialisable(self):
        cm = ClassMorphologyMetrics("sand", 2, 50, 0.25, 0.25, 0.0, 0.7, 3,
                                    confidence_weighted_area=35.0)
        assert json.dumps(cm.as_dict())


class TestGeometryMetrics:
    def _make(self):
        crm = ClassRegionMetrics("water", 1, 1, 200, 0.0, 200.0, 0.0, 0.005, 20.0, ())
        return GeometryMetrics(
            per_class_regions={"water": crm},
            estimated_channel_width_px=20.0,
            pixel_width_m=30.0,
            pixel_height_m=30.0,
        )

    def test_frozen(self):
        gm = self._make()
        with pytest.raises((AttributeError, TypeError)):
            gm.pixel_width_m = 10.0  # type: ignore[misc]

    def test_as_dict_json_serialisable(self):
        assert json.dumps(self._make().as_dict())

    def test_pixel_resolution_in_dict(self):
        d = self._make().as_dict()
        assert d["pixel_width_m"]  == 30.0
        assert d["pixel_height_m"] == 30.0


class TestRiverMorphologyResult:
    def _make(self):
        cfg = AnalyticsConfig()
        sa  = SampleAnalysis("p1","2023-07-01","monsoon",2023,"L8","Kosi","R1","B1","A1",
                              1024, {"water": ClassMorphologyMetrics(
                                  "water",1,512,0.5,0.5,0.0,0.8,10)},
                              None, None)
        return RiverMorphologyResult(
            sample_analyses=(sa,), temporal_changes=(), seasonal_summaries={},
            spatial_summaries={}, class_names=("background","water","sand","vegetation"),
            num_samples=1, num_classes=4, total_pixels=1024,
            mean_water_fraction=0.5, mean_sand_fraction=0.1, mean_veg_fraction=0.2,
            mean_confidence=0.8, analytics_config=cfg, architecture="unetplusplus",
            operations_log=("step1",), analysis_time_s=0.5,
        )

    def test_frozen(self):
        r = self._make()
        with pytest.raises((AttributeError, TypeError)):
            r.num_samples = 99  # type: ignore[misc]

    def test_summary_lines_ascii(self):
        lines = self._make().summary_lines()
        assert all(ord(c) < 128 for l in lines for c in l)

    def test_as_dict_json_serialisable(self):
        assert json.dumps(self._make().as_dict())
