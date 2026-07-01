"""
Unit tests for src/patches/tiler.py.

Pure Python, no I/O.

Run:
    pytest tests/patches/test_patch_tiler.py -v
    pytest tests/patches/test_patch_tiler.py -v \
        --cov=src/patches/tiler --cov-report=term-missing
"""

from __future__ import annotations

import pytest

from src.core.exceptions import InvalidValueError
from src.patches.tiler import PatchTiler, PatchWindow


# ==============================================================================
# PatchWindow tests
# ==============================================================================

class TestPatchWindow:
    """Tests for the frozen PatchWindow dataclass."""

    def test_frozen(self) -> None:
        w = PatchWindow(row_index=0, col_index=0, row_off=0, col_off=0, height=8, width=8)
        with pytest.raises((AttributeError, TypeError)):
            w.row_off = 5  # type: ignore[misc]

    def test_fields(self) -> None:
        w = PatchWindow(row_index=1, col_index=2, row_off=8, col_off=16, height=8, width=8)
        assert w.row_index == 1
        assert w.col_index == 2
        assert w.row_off   == 8
        assert w.col_off   == 16
        assert w.height    == 8
        assert w.width     == 8


# ==============================================================================
# PatchTiler construction tests
# ==============================================================================

class TestPatchTilerConstruction:
    """Tests for PatchTiler.__init__() validation."""

    def test_valid_construction(self) -> None:
        tiler = PatchTiler(patch_size=8, stride=8)
        assert tiler.patch_size == 8
        assert tiler.stride     == 8

    def test_zero_patch_size_raises(self) -> None:
        with pytest.raises(InvalidValueError, match="patch_size"):
            PatchTiler(patch_size=0, stride=8)

    def test_negative_patch_size_raises(self) -> None:
        with pytest.raises(InvalidValueError, match="patch_size"):
            PatchTiler(patch_size=-4, stride=8)

    def test_zero_stride_raises(self) -> None:
        with pytest.raises(InvalidValueError, match="stride"):
            PatchTiler(patch_size=8, stride=0)

    def test_negative_stride_raises(self) -> None:
        with pytest.raises(InvalidValueError, match="stride"):
            PatchTiler(patch_size=8, stride=-1)

    def test_non_int_patch_size_raises(self) -> None:
        with pytest.raises(InvalidValueError, match="patch_size"):
            PatchTiler(patch_size=8.5, stride=8)  # type: ignore[arg-type]


# ==============================================================================
# Non-overlapping tiling tests
# ==============================================================================

class TestComputeWindowsNonOverlapping:
    """Tests for compute_windows() with stride == patch_size."""

    def test_exact_division_produces_correct_count(self) -> None:
        tiler   = PatchTiler(patch_size=8, stride=8)
        windows = tiler.compute_windows(raster_width=16, raster_height=16)
        assert len(windows) == 4

    def test_row_major_order(self) -> None:
        tiler   = PatchTiler(patch_size=8, stride=8)
        windows = tiler.compute_windows(16, 16)
        expected = [(0, 0), (0, 1), (1, 0), (1, 1)]
        actual   = [(w.row_index, w.col_index) for w in windows]
        assert actual == expected

    def test_pixel_offsets_correct(self) -> None:
        tiler   = PatchTiler(patch_size=8, stride=8)
        windows = tiler.compute_windows(16, 16)
        offsets = {(w.row_off, w.col_off) for w in windows}
        assert offsets == {(0, 0), (0, 8), (8, 0), (8, 8)}

    def test_all_windows_full_patch_size(self) -> None:
        tiler   = PatchTiler(patch_size=8, stride=8)
        windows = tiler.compute_windows(16, 16)
        for w in windows:
            assert w.height == 8
            assert w.width  == 8


# ==============================================================================
# Overlapping tiling tests
# ==============================================================================

class TestComputeWindowsOverlap:
    """Tests for compute_windows() with stride < patch_size."""

    def test_overlapping_stride_produces_more_windows(self) -> None:
        tiler   = PatchTiler(patch_size=8, stride=4)
        windows = tiler.compute_windows(16, 16)
        # offsets [0, 4, 8] per axis -> 3x3 = 9
        assert len(windows) == 9

    def test_heavy_overlap_stride_one(self) -> None:
        tiler   = PatchTiler(patch_size=8, stride=1)
        windows = tiler.compute_windows(16, 16)
        # offsets [0..8] per axis (9 values) -> 81 windows
        assert len(windows) == 81

    def test_overlapping_windows_have_correct_offsets(self) -> None:
        tiler   = PatchTiler(patch_size=8, stride=4)
        windows = tiler.compute_windows(16, 16)
        offsets = sorted({w.row_off for w in windows})
        assert offsets == [0, 4, 8]


# ==============================================================================
# Gap tiling tests
# ==============================================================================

class TestComputeWindowsGaps:
    """Tests for compute_windows() with stride > patch_size."""

    def test_stride_larger_than_patch_size_produces_gaps(self) -> None:
        tiler   = PatchTiler(patch_size=4, stride=8)
        windows = tiler.compute_windows(16, 16)
        # offsets along each axis: range(0, 16-4+1, 8) = [0, 8] -> 2x2 = 4
        assert len(windows) == 4

    def test_gap_offsets_correct(self) -> None:
        tiler   = PatchTiler(patch_size=4, stride=8)
        windows = tiler.compute_windows(16, 16)
        offsets = sorted({w.col_off for w in windows})
        assert offsets == [0, 8]


# ==============================================================================
# Edge handling tests
# ==============================================================================

class TestComputeWindowsEdgeHandling:
    """Tests for incomplete edge window dropping."""

    def test_incomplete_edge_dropped(self) -> None:
        # 17x17 raster, patch_size=8, stride=8: offsets = range(0, 10, 8) = [0, 8].
        # A 1px strip at the far edge is uncovered and dropped (not a partial window).
        tiler   = PatchTiler(patch_size=8, stride=8)
        windows = tiler.compute_windows(17, 17)
        assert len(windows) == 4
        for w in windows:
            assert w.row_off + w.height <= 17
            assert w.col_off + w.width  <= 17

    def test_all_windows_uniform_size_at_edges(self) -> None:
        tiler   = PatchTiler(patch_size=8, stride=8)
        windows = tiler.compute_windows(20, 13)
        sizes   = {(w.height, w.width) for w in windows}
        assert sizes == {(8, 8)}

    def test_raster_smaller_than_patch_size_returns_empty(self) -> None:
        tiler   = PatchTiler(patch_size=32, stride=32)
        windows = tiler.compute_windows(16, 16)
        assert windows == ()

    def test_raster_exactly_patch_size_returns_one_window(self) -> None:
        tiler   = PatchTiler(patch_size=16, stride=16)
        windows = tiler.compute_windows(16, 16)
        assert len(windows) == 1
        assert windows[0].row_off == 0
        assert windows[0].col_off == 0

    def test_rectangular_raster(self) -> None:
        """Non-square rasters must tile each axis independently."""
        tiler   = PatchTiler(patch_size=8, stride=8)
        windows = tiler.compute_windows(raster_width=24, raster_height=8)
        # width: offsets [0, 8, 16] (3); height: offsets [0] (1) -> 3 windows
        assert len(windows) == 3


# ==============================================================================
# Validation tests
# ==============================================================================

class TestComputeWindowsValidation:
    """Tests for compute_windows() input validation."""

    def test_negative_width_raises(self) -> None:
        tiler = PatchTiler(patch_size=8, stride=8)
        with pytest.raises(InvalidValueError, match="raster dimensions"):
            tiler.compute_windows(-1, 16)

    def test_zero_height_raises(self) -> None:
        tiler = PatchTiler(patch_size=8, stride=8)
        with pytest.raises(InvalidValueError, match="raster dimensions"):
            tiler.compute_windows(16, 0)