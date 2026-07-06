"""Tests for temporal.py, validator.py, analyzer.py, and engine.py"""
from __future__ import annotations
import json
import types
import numpy as np
import pytest

from src.morphology.contracts import (
    AnalyticsConfig, ClassMorphologyMetrics, SampleAnalysis,
)
from src.morphology.temporal import SeasonalAggregator, TemporalAnalyzer
from src.morphology.validator import AnalyticsValidator, AnalyticsValidationResult
from src.morphology.analyzer import MorphologyAnalyzer, SpatialAggregator
from src.morphology.engine import RiverMorphologyEngine


CLASS_NAMES = ("background", "water", "sand", "vegetation")


def _cfg(**kw) -> AnalyticsConfig:
    defaults = dict(
        pixel_area_m2=0.0, pixel_width_m=0.0, pixel_height_m=0.0,
        low_confidence_threshold=0.5, min_component_area=1, seasons={},
        water_class="water", sand_class="sand",
        vegetation_class="vegetation", background_class="background",
        compute_geometry=False, compute_temporal=True,
        compute_seasonal=True, compute_uncertainty=True,
        compute_shape_descriptors=False,
    )
    defaults.update(kw)
    return AnalyticsConfig(**defaults)


def _cm(class_name="water", class_id=1, pixel_count=100, area_frac=0.5):
    return ClassMorphologyMetrics(class_name, class_id, pixel_count,
                                  area_frac, area_frac, 0.0, 0.8, 5)


def _sample_analysis(
    sample_id="p1", date="2023-07-01", season="monsoon", year=2023,
    river="Kosi", reach="R1", basin="B1", aoi="A1",
    water_px=100, total_px=200,
) -> SampleAnalysis:
    cm = {
        "background": _cm("background", 0, total_px - water_px,
                          (total_px - water_px) / total_px),
        "water":      _cm("water", 1, water_px, water_px / total_px),
        "sand":       _cm("sand",  2, 0, 0.0),
        "vegetation": _cm("vegetation", 3, 0, 0.0),
    }
    return SampleAnalysis(
        sample_id=sample_id, acquisition_date=date, season=season,
        hydrological_year=year, sensor="L8", river_name=river,
        reach_id=reach, basin_id=basin, aoi_id=aoi,
        total_pixels=total_px, class_metrics=cm, geometry=None, uncertainty=None,
    )


# ==============================================================================
# TemporalAnalyzer
# ==============================================================================

class TestTemporalAnalyzer:
    def _ta(self, **kw): return TemporalAnalyzer(_cfg(**kw), CLASS_NAMES)

    def test_single_sample_no_changes(self):
        assert self._ta().compute([_sample_analysis()]) == []

    def test_two_dates_produces_changes(self):
        ta = self._ta()
        changes = ta.compute([
            _sample_analysis("p1", "2023-01-01", water_px=100),
            _sample_analysis("p2", "2023-07-01", water_px=150),
        ])
        water_ch = [c for c in changes if c.class_name == "water"]
        assert water_ch[0].pixel_delta == 50

    def test_pct_change_zero_from_count(self):
        ta = self._ta()
        changes = ta.compute([
            _sample_analysis("p1", "2023-01-01", water_px=0),
            _sample_analysis("p2", "2023-07-01", water_px=50),
        ])
        water_ch = next(c for c in changes if c.class_name == "water")
        assert water_ch.pct_change == pytest.approx(0.0)

    def test_analyses_sorted_before_diffing(self):
        ta = self._ta()
        changes = ta.compute([
            _sample_analysis("p1", "2023-07-01", water_px=150),
            _sample_analysis("p2", "2023-01-01", water_px=100),
        ])
        water_ch = next(c for c in changes if c.class_name == "water")
        assert water_ch.date_from == "2023-01-01"

    def test_same_date_averages(self):
        ta = self._ta()
        changes = ta.compute([
            _sample_analysis("p1", "2023-01-01", water_px=100),
            _sample_analysis("p2", "2023-01-01", water_px=200),
            _sample_analysis("p3", "2023-07-01", water_px=180),
        ])
        water_ch = next(c for c in changes if c.class_name == "water")
        assert water_ch.pixel_count_from == 150  # avg of 100 and 200


# ==============================================================================
# SeasonalAggregator
# ==============================================================================

class TestSeasonalAggregator:
    def _sa(self, **kw): return SeasonalAggregator(_cfg(**kw), CLASS_NAMES)

    def test_empty_returns_empty(self):
        assert self._sa().compute([]) == {}

    def test_two_seasons(self):
        sa = self._sa()
        r  = sa.compute([
            _sample_analysis(season="monsoon"),
            _sample_analysis(season="pre-monsoon"),
        ])
        assert "monsoon" in r and "pre-monsoon" in r

    def test_empty_season_grouped_unknown(self):
        sa = self._sa()
        r  = sa.compute([_sample_analysis(season="")])
        assert "unknown" in r

    def test_month_to_season_mapping(self):
        seasons = {"monsoon": (6, 7, 8, 9)}
        sa = SeasonalAggregator(_cfg(seasons=seasons), CLASS_NAMES)
        a  = _sample_analysis(date="2023-07-01", season="wrong", year=2023)
        r  = sa.compute([a])
        assert "monsoon" in r


# ==============================================================================
# AnalyticsValidator
# ==============================================================================

class TestAnalyticsValidator:
    def _ir(self, n=2):
        mask = np.zeros((8, 8), dtype=np.uint8)
        conf = np.ones((8, 8), dtype=np.float32)
        pred = types.SimpleNamespace(sample_id="p1", predicted_mask=mask, confidence=conf)
        return types.SimpleNamespace(
            num_samples=n, class_names=CLASS_NAMES,
            predictions=[pred] * n, mean_confidence=0.8,
        )

    def test_valid_inputs_pass(self):
        assert AnalyticsValidator().validate(_cfg(), self._ir()).is_valid

    def test_none_inference_result_detected(self):
        assert not AnalyticsValidator().validate(_cfg(), None).is_valid

    def test_zero_samples_detected(self):
        assert not AnalyticsValidator().validate(_cfg(), self._ir(n=0)).is_valid

    def test_negative_pixel_area_detected(self):
        assert not AnalyticsValidator().validate(_cfg(pixel_area_m2=-1.0), self._ir()).is_valid

    def test_invalid_threshold_detected(self):
        assert not AnalyticsValidator().validate(_cfg(low_confidence_threshold=1.5), self._ir()).is_valid

    def test_nan_confidence_detected(self):
        mask = np.zeros((4, 4), dtype=np.uint8)
        conf = np.full((4, 4), np.nan, dtype=np.float32)
        pred = types.SimpleNamespace(sample_id="p1", predicted_mask=mask, confidence=conf)
        ir   = types.SimpleNamespace(num_samples=1, class_names=CLASS_NAMES,
                                     predictions=[pred], mean_confidence=0.8)
        assert not AnalyticsValidator().validate(_cfg(), ir).is_valid

    def test_issues_are_copy(self):
        r = AnalyticsValidationResult(["a"])
        r.issues.append("b")
        assert len(r.issues) == 1


# ==============================================================================
# MorphologyAnalyzer
# ==============================================================================

class TestMorphologyAnalyzer:
    def _pred(self, water_frac=0.5, h=8, w=8, river="Kosi"):
        mask = np.zeros((h, w), dtype=np.uint8)
        mask.ravel()[:int(h * w * water_frac)] = 1
        conf = np.ones((h, w), dtype=np.float32) * 0.8
        return types.SimpleNamespace(
            sample_id="p1", predicted_mask=mask, confidence=conf,
            acquisition_date="2023-07-01", season="monsoon",
            hydrological_year=2023, sensor="L8", river_name=river,
            reach_id="R1", basin_id="B1", aoi_id="A1",
        )

    def test_returns_sample_analysis(self):
        ma = MorphologyAnalyzer(_cfg(), CLASS_NAMES)
        r  = ma.analyze(self._pred())
        assert isinstance(r, SampleAnalysis)

    def test_class_metrics_all_classes(self):
        ma = MorphologyAnalyzer(_cfg(), CLASS_NAMES)
        r  = ma.analyze(self._pred())
        assert set(r.class_metrics.keys()) == set(CLASS_NAMES)

    def test_confidence_weighted_area_in_class_metrics(self):
        ma   = MorphologyAnalyzer(_cfg(), CLASS_NAMES)
        pred = self._pred(water_frac=0.5, h=4, w=4)
        pred.confidence = np.full((4, 4), 0.6, dtype=np.float32)
        r    = ma.analyze(pred)
        expected_cwa = 8 * 0.6   # 8 water pixels * 0.6 confidence
        assert r.class_metrics["water"].confidence_weighted_area == pytest.approx(expected_cwa)

    def test_geometry_none_when_disabled(self):
        ma = MorphologyAnalyzer(_cfg(compute_geometry=False), CLASS_NAMES)
        assert ma.analyze(self._pred()).geometry is None

    def test_uncertainty_none_when_disabled(self):
        ma = MorphologyAnalyzer(_cfg(compute_uncertainty=False), CLASS_NAMES)
        assert ma.analyze(self._pred()).uncertainty is None

    def test_metadata_preserved(self):
        ma = MorphologyAnalyzer(_cfg(), CLASS_NAMES)
        r  = ma.analyze(self._pred(river="Brahmaputra"))
        assert r.river_name == "Brahmaputra"

    def test_total_pixels_correct(self):
        ma = MorphologyAnalyzer(_cfg(), CLASS_NAMES)
        assert ma.analyze(self._pred(h=8, w=8)).total_pixels == 64


# ==============================================================================
# SpatialAggregator
# ==============================================================================

class TestSpatialAggregator:
    def test_groups_by_aoi(self):
        sa = SpatialAggregator(CLASS_NAMES)
        r  = sa.compute([
            _sample_analysis("p1", aoi="AOI1"),
            _sample_analysis("p2", aoi="AOI1"),
            _sample_analysis("p3", aoi="AOI2"),
        ])
        assert r["aoi:AOI1"].num_samples == 2
        assert "aoi:AOI2" in r

    def test_empty_id_not_grouped(self):
        sa = SpatialAggregator(CLASS_NAMES)
        r  = sa.compute([_sample_analysis(river="", aoi="")])
        assert not any(k.startswith("river:") or k.startswith("aoi:") for k in r)

    def test_total_pixels_summed(self):
        sa = SpatialAggregator(CLASS_NAMES)
        r  = sa.compute([
            _sample_analysis("p1", aoi="A1", total_px=100),
            _sample_analysis("p2", aoi="A1", total_px=200),
        ])
        assert r["aoi:A1"].total_pixels == 300


# ==============================================================================
# RiverMorphologyEngine integration
# ==============================================================================

class TestRiverMorphologyEngine:
    def _ir(self, n=3, water_frac=0.4):
        preds = []
        for i in range(n):
            mask = np.zeros((8, 8), dtype=np.uint8)
            mask.ravel()[:int(64 * water_frac)] = 1
            conf = np.ones((8, 8), dtype=np.float32) * 0.75
            preds.append(types.SimpleNamespace(
                sample_id        = f"p{i:03d}",
                predicted_mask   = mask,
                confidence       = conf,
                probabilities    = np.ones((4, 8, 8), dtype=np.float32) * 0.25,
                acquisition_date = f"2023-{i+1:02d}-01",
                season           = "monsoon" if i % 2 == 0 else "pre-monsoon",
                hydrological_year = 2023,
                sensor           = "L8",
                river_name       = "Kosi",
                reach_id         = "R1",
                basin_id         = "B1",
                aoi_id           = "A1",
            ))
        return types.SimpleNamespace(
            predictions=preds, num_samples=n,
            class_names=CLASS_NAMES, architecture="unetplusplus",
            mean_confidence=0.75,
        )

    def test_returns_river_morphology_result(self):
        from src.morphology.contracts import RiverMorphologyResult
        engine = RiverMorphologyEngine(_cfg())
        assert isinstance(engine.analyze(self._ir()), RiverMorphologyResult)

    def test_result_is_frozen(self):
        engine = RiverMorphologyEngine(_cfg())
        result = engine.analyze(self._ir())
        with pytest.raises((AttributeError, TypeError)):
            result.num_samples = 999  # type: ignore[misc]

    def test_correct_num_samples(self):
        engine = RiverMorphologyEngine(_cfg())
        assert engine.analyze(self._ir(n=4)).num_samples == 4

    def test_sample_analyses_sorted_by_date(self):
        engine = RiverMorphologyEngine(_cfg())
        dates  = [a.acquisition_date for a in engine.analyze(self._ir(n=3)).sample_analyses]
        assert dates == sorted(dates)

    def test_mean_water_fraction_correct(self):
        engine = RiverMorphologyEngine(_cfg())
        result = engine.analyze(self._ir(water_frac=0.5))
        assert result.mean_water_fraction == pytest.approx(0.5, abs=0.02)

    def test_temporal_changes_produced(self):
        assert len(RiverMorphologyEngine(_cfg()).analyze(self._ir(n=3)).temporal_changes) > 0

    def test_temporal_changes_empty_when_disabled(self):
        assert len(RiverMorphologyEngine(_cfg(compute_temporal=False)).analyze(self._ir()).temporal_changes) == 0

    def test_seasonal_summaries_produced(self):
        assert len(RiverMorphologyEngine(_cfg()).analyze(self._ir(n=3)).seasonal_summaries) > 0

    def test_spatial_summaries_produced(self):
        assert len(RiverMorphologyEngine(_cfg()).analyze(self._ir()).spatial_summaries) > 0

    def test_class_names_match(self):
        assert RiverMorphologyEngine(_cfg()).analyze(self._ir()).class_names == CLASS_NAMES

    def test_as_dict_json_serialisable(self):
        result = RiverMorphologyEngine(_cfg()).analyze(self._ir())
        assert json.dumps(result.as_dict())

    def test_empty_predictions_zero_metrics(self):
        ir = types.SimpleNamespace(predictions=[], num_samples=0,
                                   class_names=CLASS_NAMES, architecture="unetplusplus",
                                   mean_confidence=0.0)
        result = RiverMorphologyEngine(_cfg()).analyze(ir)
        assert result.num_samples == 0
        assert result.mean_water_fraction == 0.0

    def test_accepts_project_config_object(self):
        class _Cfg: pass
        engine = RiverMorphologyEngine(_Cfg())
        assert isinstance(engine._config, AnalyticsConfig)

    def test_pixel_resolution_in_analytics_config(self):
        engine = RiverMorphologyEngine(_cfg(pixel_width_m=30.0, pixel_height_m=30.0))
        assert engine._config.pixel_width_m  == pytest.approx(30.0)
        assert engine._config.pixel_height_m == pytest.approx(30.0)

    def test_engine_with_geometry_enabled(self):
        """Full integration with geometry enabled (requires scipy)."""
        pytest.importorskip("scipy")
        engine = RiverMorphologyEngine(_cfg(compute_geometry=True, min_component_area=1))
        result = engine.analyze(self._ir(n=2))
        for sa in result.sample_analyses:
            assert sa.geometry is not None
            assert "water" in sa.geometry.per_class_regions

    def test_confidence_weighted_area_propagated_to_result(self):
        engine = RiverMorphologyEngine(_cfg())
        result = engine.analyze(self._ir(n=1, water_frac=0.5))
        water_cm = result.sample_analyses[0].class_metrics["water"]
        assert water_cm.confidence_weighted_area > 0.0

    def test_shape_descriptors_populated_when_enabled(self):
        pytest.importorskip("scipy")
        engine = RiverMorphologyEngine(
            _cfg(compute_geometry=True, compute_shape_descriptors=True, min_component_area=1)
        )
        result = engine.analyze(self._ir(n=1, water_frac=0.5))
        geo    = result.sample_analyses[0].geometry
        assert geo is not None
        water_regions = geo.per_class_regions["water"]
        if water_regions.region_count > 0:
            r = water_regions.regions[0]
            assert r.perimeter_px > 0.0

    def test_total_pixels_correct(self):
        engine = RiverMorphologyEngine(_cfg())
        result = engine.analyze(self._ir(n=2))
        assert result.total_pixels == 2 * 64
