"""Tests for src/morphology/geometry.py — extended GeometryAnalyzer"""
from __future__ import annotations
import numpy as np
import pytest
import math
from src.morphology.contracts import AnalyticsConfig
from src.morphology.geometry import (
    GeometryAnalyzer, _component_sizes, _estimate_width,
    _compute_compactness, _compute_elongation, _compute_perimeter,
)


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


def _ga(scipy_required=True, **kw) -> GeometryAnalyzer:
    if scipy_required:
        pytest.importorskip("scipy")
    return GeometryAnalyzer(_cfg(**kw), CLASS_NAMES)


# ==============================================================================
# GeometryAnalyzer — per-class region counts
# ==============================================================================

class TestGeometryAnalyzerRegionCounts:
    def test_single_water_body_one_region(self):
        ga   = _ga()
        mask = np.ones((8, 8), dtype=np.uint8)   # class 1 = water
        r    = ga.compute(mask)
        assert r.per_class_regions["water"].region_count == 1

    def test_two_disconnected_water_regions(self):
        ga   = _ga(min_component_area=1)
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[0:3, 0:3] = 1
        mask[5:8, 5:8] = 1
        r = ga.compute(mask)
        assert r.per_class_regions["water"].region_count == 2

    def test_no_water_zero_regions(self):
        ga   = _ga()
        mask = np.zeros((8, 8), dtype=np.uint8)
        r    = ga.compute(mask)
        assert r.per_class_regions["water"].region_count == 0
        assert r.per_class_regions["water"].largest_region_px == 0

    def test_sand_region_counted(self):
        ga   = _ga(min_component_area=1)
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:5, 2:5] = 2   # sand (class 2)
        r    = ga.compute(mask)
        assert r.per_class_regions["sand"].region_count >= 1

    def test_vegetation_region_counted(self):
        ga   = _ga(min_component_area=1)
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[1:3, 1:3] = 3   # vegetation (class 3)
        r    = ga.compute(mask)
        assert r.per_class_regions["vegetation"].region_count >= 1

    def test_background_class_not_analysed(self):
        ga   = _ga()
        mask = np.zeros((8, 8), dtype=np.uint8)   # all background
        r    = ga.compute(mask)
        assert r.per_class_regions["background"].region_count == 0

    def test_min_component_area_filters_small_regions(self):
        ga   = _ga(min_component_area=10)
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[0, 0] = 1   # 1-pixel water region (below min_area=10)
        r    = ga.compute(mask)
        assert r.per_class_regions["water"].region_count == 0

    def test_all_class_names_in_per_class_regions(self):
        ga   = _ga()
        mask = np.zeros((8, 8), dtype=np.uint8)
        r    = ga.compute(mask)
        for name in CLASS_NAMES:
            assert name in r.per_class_regions


class TestGeometryAnalyzerRegionProperties:
    def test_largest_region_correct(self):
        ga   = _ga(min_component_area=1)
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[0:3, 0:3] = 1   # 9 pixels
        mask[5:6, 5:6] = 1   # 1 pixel
        r    = ga.compute(mask)
        assert r.per_class_regions["water"].largest_region_px == 9

    def test_mean_region_size(self):
        ga   = _ga(min_component_area=1)
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[0:2, 0:2] = 1   # 4 pixels
        mask[4:6, 4:6] = 1   # 4 pixels
        r    = ga.compute(mask)
        assert r.per_class_regions["water"].mean_region_size_px == pytest.approx(4.0)

    def test_fragmentation_index_single_region(self):
        """region_count / total_class_pixels = 1/n for a single region."""
        ga   = _ga(min_component_area=1)
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[:, :] = 1   # 16 water pixels, 1 region
        r    = ga.compute(mask)
        frag = r.per_class_regions["water"].fragmentation_index
        assert frag == pytest.approx(1.0 / 16.0)

    def test_fragmentation_index_more_regions_higher(self):
        """More regions per pixel -> higher fragmentation."""
        ga   = _ga(min_component_area=1)
        mask1 = np.zeros((8, 8), dtype=np.uint8)
        mask1[3:5, 3:5] = 1   # 1 compact region
        mask2 = np.zeros((8, 8), dtype=np.uint8)
        for i in range(4):
            mask2[i*2, i*2] = 1   # 4 isolated pixels
        r1 = ga.compute(mask1)
        r2 = ga.compute(mask2)
        assert r2.per_class_regions["water"].fragmentation_index > \
               r1.per_class_regions["water"].fragmentation_index

    def test_regions_sorted_by_area_descending(self):
        ga   = _ga(min_component_area=1)
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[0:4, 0:4] = 1   # 16 pixels
        mask[6:8, 6:8] = 1   # 4 pixels
        r    = ga.compute(mask)
        regions = r.per_class_regions["water"].regions
        assert len(regions) == 2
        assert regions[0].area_px >= regions[1].area_px

    def test_estimated_width_nonzero_when_water_present(self):
        ga   = _ga(min_component_area=1)
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[3:6, :] = 1   # 3-row water band
        r    = ga.compute(mask)
        assert r.per_class_regions["water"].estimated_width_px > 0.0

    def test_channel_width_matches_water_estimated_width(self):
        ga   = _ga(min_component_area=1)
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[3:6, :] = 1
        r    = ga.compute(mask)
        assert r.estimated_channel_width_px == \
               r.per_class_regions["water"].estimated_width_px

    def test_largest_region_m2_correct(self):
        ga   = _ga(min_component_area=1, pixel_area_m2=100.0)
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[0:4, 0:4] = 1   # 16 pixels
        r    = ga.compute(mask)
        assert r.per_class_regions["water"].largest_region_m2 == pytest.approx(16 * 100.0)


class TestGeometryAnalyzerConfidenceWeighting:
    def test_region_mean_confidence_correct(self):
        ga   = _ga(min_component_area=1)
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[0:2, 0:2] = 1   # 4 water pixels in top-left
        conf = np.zeros((4, 4), dtype=np.float32)
        conf[0:2, 0:2] = 0.9
        r    = ga.compute(mask, conf)
        regions = r.per_class_regions["water"].regions
        assert len(regions) == 1
        assert regions[0].mean_confidence == pytest.approx(0.9)

    def test_region_mean_confidence_zero_when_no_confidence_passed(self):
        ga   = _ga(min_component_area=1)
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[0:2, 0:2] = 1
        r    = ga.compute(mask, confidence=None)
        regions = r.per_class_regions["water"].regions
        assert regions[0].mean_confidence == pytest.approx(0.0)


class TestGeometryAnalyzerPixelResolution:
    def test_pixel_resolution_in_result(self):
        ga   = _ga(pixel_width_m=30.0, pixel_height_m=30.0)
        mask = np.zeros((8, 8), dtype=np.uint8)
        r    = ga.compute(mask)
        assert r.pixel_width_m  == pytest.approx(30.0)
        assert r.pixel_height_m == pytest.approx(30.0)

    def test_zero_pixel_resolution_when_not_configured(self):
        ga   = _ga()
        mask = np.zeros((8, 8), dtype=np.uint8)
        r    = ga.compute(mask)
        assert r.pixel_width_m  == 0.0
        assert r.pixel_height_m == 0.0


class TestShapeDescriptors:
    def test_shape_descriptors_zero_when_disabled(self):
        ga   = _ga(compute_shape_descriptors=False, min_component_area=1)
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:6, 2:6] = 1
        r    = ga.compute(mask)
        region = r.per_class_regions["water"].regions[0]
        assert region.perimeter_px  == 0.0
        assert region.compactness   == 0.0
        assert region.elongation    == 0.0

    def test_shape_descriptors_populated_when_enabled(self):
        ga   = _ga(compute_shape_descriptors=True, min_component_area=1)
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:6, 2:6] = 1   # 4x4 square
        r    = ga.compute(mask)
        region = r.per_class_regions["water"].regions[0]
        assert region.perimeter_px > 0.0
        assert region.compactness  > 0.0

    def test_square_region_compactness_near_max(self):
        """A nearly square region should have compactness close to pi/4 ~ 0.785."""
        ga   = _ga(compute_shape_descriptors=True, min_component_area=1)
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[5:15, 5:15] = 1   # 10x10 square
        r    = ga.compute(mask)
        region = r.per_class_regions["water"].regions[0]
        # Compactness of a square is pi/4 ~ 0.785
        assert region.compactness == pytest.approx(math.pi / 4, abs=0.05)

    def test_elongated_region_elongation_high(self):
        """A 1x8 strip should have elongation close to 1.0."""
        ga   = _ga(compute_shape_descriptors=True, min_component_area=1)
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[4, 1:9] = 1   # 1-row, 8-column strip
        r    = ga.compute(mask)
        region = r.per_class_regions["water"].regions[0]
        assert region.elongation > 0.7

    def test_square_region_elongation_zero(self):
        """A square bounding box should give elongation = 0."""
        ga   = _ga(compute_shape_descriptors=True, min_component_area=1)
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:6, 2:6] = 1   # 4x4 square
        r    = ga.compute(mask)
        region = r.per_class_regions["water"].regions[0]
        assert region.elongation == pytest.approx(0.0, abs=1e-6)

    def test_aspect_ratio_correct(self):
        ga   = _ga(min_component_area=1)
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[0:2, 0:6] = 1   # 2 rows, 6 cols -> aspect = 6/2 = 3.0
        r    = ga.compute(mask)
        region = r.per_class_regions["water"].regions[0]
        assert region.aspect_ratio == pytest.approx(3.0)


class TestGeometryAnalyzerFallback:
    def test_scipy_unavailable_returns_empty_geometry(self):
        from unittest.mock import patch
        ga   = _ga(scipy_required=False)
        mask = np.ones((8, 8), dtype=np.uint8)
        with patch.dict("sys.modules", {"scipy": None, "scipy.ndimage": None}):
            result = ga.compute(mask)
        for name in CLASS_NAMES:
            assert result.per_class_regions[name].region_count == 0


# ==============================================================================
# Shape descriptor helper functions
# ==============================================================================

class TestShapeHelpers:
    def test_compute_compactness_zero_perimeter(self):
        assert _compute_compactness(100, 0.0) == 0.0

    def test_compute_compactness_positive(self):
        # Area=100, perimeter=40 -> 4*pi*100/1600 ~ 0.785
        c = _compute_compactness(100, 40.0)
        assert c == pytest.approx(4 * math.pi * 100 / (40.0 ** 2), rel=1e-5)

    def test_compute_elongation_square(self):
        assert _compute_elongation(5, 5) == pytest.approx(0.0)

    def test_compute_elongation_strip(self):
        assert _compute_elongation(1, 8) == pytest.approx(7/8)

    def test_compute_elongation_both_zero(self):
        assert _compute_elongation(0, 0) == 0.0

    def test_compute_perimeter_requires_scipy(self):
        scipy = pytest.importorskip("scipy")
        mask  = np.zeros((6, 6), dtype=bool)
        mask[1:5, 1:5] = True   # 4x4 square
        p = _compute_perimeter(mask)
        # 4x4 square perimeter by erosion boundary: 12 boundary pixels
        assert p == pytest.approx(16.0)


# ==============================================================================
# Legacy helper compatibility
# ==============================================================================

class TestLegacyHelpers:
    def test_component_sizes_empty(self):
        labeled = np.zeros((4, 4), dtype=np.int32)
        assert _component_sizes(labeled, 0) == []

    def test_component_sizes_single(self):
        labeled = np.array([[1, 1], [1, 0]], dtype=np.int32)
        assert _component_sizes(labeled, 1) == [3]

    def test_estimate_width_empty(self):
        labeled = np.zeros((4, 4), dtype=np.int32)
        assert _estimate_width(labeled, [], 1) == 0.0
