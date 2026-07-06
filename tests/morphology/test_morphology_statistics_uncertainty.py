"""Tests for statistics.py and uncertainty.py"""
from __future__ import annotations
import numpy as np
import pytest
from src.morphology.contracts import AnalyticsConfig
from src.morphology.statistics import MorphologyStatisticsComputer
from src.morphology.uncertainty import UncertaintyAnalyzer


CLASS_NAMES = ("background", "water", "sand", "vegetation")


def _cfg(**kw) -> AnalyticsConfig:
    defaults = dict(
        pixel_area_m2=0.0, pixel_width_m=0.0, pixel_height_m=0.0,
        low_confidence_threshold=0.5, min_component_area=1, seasons={},
        water_class="water", sand_class="sand",
        vegetation_class="vegetation", background_class="background",
        compute_geometry=True, compute_temporal=True,
        compute_seasonal=True, compute_uncertainty=True,
        compute_shape_descriptors=False,
    )
    defaults.update(kw)
    return AnalyticsConfig(**defaults)


class TestMorphologyStatisticsComputer:
    def _comp(self, **kw):
        return MorphologyStatisticsComputer(_cfg(**kw), CLASS_NAMES)

    def test_returns_all_class_names(self):
        mask = np.zeros((8, 8), dtype=np.uint8)
        conf = np.ones((8, 8), dtype=np.float32)
        r    = self._comp().compute(mask, conf)
        assert set(r.keys()) == set(CLASS_NAMES)

    def test_pixel_count_correct(self):
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[0:4, :] = 1   # 32 water pixels
        conf = np.ones((8, 8), dtype=np.float32)
        r    = self._comp().compute(mask, conf)
        assert r["water"].pixel_count == 32

    def test_total_fraction_sums_to_one(self):
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[0:4, :] = 1
        conf = np.ones((8, 8), dtype=np.float32)
        r    = self._comp().compute(mask, conf)
        total = sum(m.total_fraction for m in r.values())
        assert abs(total - 1.0) < 1e-5

    def test_area_m2_correct(self):
        mask = np.zeros((4, 4), dtype=np.uint8)   # 16 pixels all background
        conf = np.ones((4, 4), dtype=np.float32)
        r    = self._comp(pixel_area_m2=100.0).compute(mask, conf)
        assert r["background"].area_m2 == pytest.approx(16 * 100.0)

    def test_confidence_weighted_area_correct(self):
        """sum(confidence[mask==class]) for water pixels."""
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[0, :] = 1   # 4 water pixels
        conf = np.full((4, 4), 0.6, dtype=np.float32)
        r    = self._comp().compute(mask, conf)
        assert r["water"].confidence_weighted_area == pytest.approx(4 * 0.6)

    def test_confidence_weighted_area_m2_correct(self):
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[0, :] = 1   # 4 water pixels
        conf = np.full((4, 4), 0.5, dtype=np.float32)
        r    = self._comp(pixel_area_m2=100.0).compute(mask, conf)
        assert r["water"].confidence_weighted_area_m2 == pytest.approx(4 * 0.5 * 100.0)

    def test_confidence_weighted_area_zero_for_empty_class(self):
        mask = np.zeros((4, 4), dtype=np.uint8)   # no sand
        conf = np.ones((4, 4), dtype=np.float32)
        r    = self._comp().compute(mask, conf)
        assert r["sand"].confidence_weighted_area == 0.0

    def test_confidence_weighted_area_less_or_equal_pixel_count_when_conf_le_1(self):
        """conf_weighted_area <= pixel_count when all confidence <= 1."""
        mask = np.ones((8, 8), dtype=np.uint8)   # all water
        conf = np.random.rand(8, 8).astype(np.float32)  # conf in [0,1]
        r    = self._comp().compute(mask, conf)
        assert r["water"].confidence_weighted_area <= r["water"].pixel_count + 1e-5

    def test_low_conf_pixels_counted(self):
        mask = np.ones((4, 4), dtype=np.uint8)   # all water
        conf = np.full((4, 4), 0.3, dtype=np.float32)  # all below 0.5
        r    = self._comp(low_confidence_threshold=0.5).compute(mask, conf)
        assert r["water"].low_conf_pixels == 16

    def test_empty_mask_returns_empty(self):
        mask = np.zeros((0, 0), dtype=np.uint8)
        conf = np.zeros((0, 0), dtype=np.float32)
        r    = self._comp().compute(mask, conf)
        assert r == {}

    def test_dataset_mean_fractions_empty_list(self):
        comp  = MorphologyStatisticsComputer(_cfg(), CLASS_NAMES)
        means = comp.dataset_mean_fractions([])
        assert all(v == 0.0 for v in means.values())


class TestUncertaintyAnalyzer:
    def _ua(self, **kw):
        return UncertaintyAnalyzer(_cfg(**kw), CLASS_NAMES)

    def test_mean_confidence_correct(self):
        conf = np.array([[0.8, 0.6], [0.4, 0.9]], dtype=np.float32)
        mask = np.zeros((2, 2), dtype=np.uint8)
        r    = self._ua().compute(conf, mask)
        assert r.mean_confidence == pytest.approx(float(conf.mean()))

    def test_low_conf_fraction_all_below_threshold(self):
        conf = np.full((4, 4), 0.3, dtype=np.float32)
        mask = np.zeros((4, 4), dtype=np.uint8)
        r    = self._ua(low_confidence_threshold=0.5).compute(conf, mask)
        assert r.low_conf_fraction == pytest.approx(1.0)

    def test_per_class_confidence(self):
        conf = np.zeros((4, 4), dtype=np.float32)
        conf[0:2, :] = 0.9
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[0:2, :] = 1   # class 1 = water
        r    = self._ua().compute(conf, mask)
        assert r.per_class_confidence["water"] == pytest.approx(0.9)

    def test_empty_class_zero_confidence(self):
        conf = np.ones((4, 4), dtype=np.float32)
        mask = np.zeros((4, 4), dtype=np.uint8)   # no sand
        r    = self._ua().compute(conf, mask)
        assert r.per_class_confidence["sand"] == 0.0

    def test_empty_mask_returns_zero_uncertainty(self):
        conf = np.zeros((0, 0), dtype=np.float32)
        mask = np.zeros((0, 0), dtype=np.uint8)
        r    = self._ua().compute(conf, mask)
        assert r.mean_confidence == 0.0
