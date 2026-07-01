"""
Unit tests for src/patches/reader.py.

Uses real rasterio I/O via GeoTiffWriter (Module 7) to create test fixtures.
No mocking of rasterio's internal API for the happy path.

Run:
    pytest tests/patches/test_patch_reader.py -v
    pytest tests/patches/test_patch_reader.py -v \
        --cov=src/patches/reader --cov-report=term-missing
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from affine import Affine

from src.export.downloader import AffineTransform, AoiBounds, DownloadResult
from src.export.geotiff import GeoTiffProfile, GeoTiffWriter
from src.patches.reader import PatchReader
from src.patches.tiler import PatchWindow

_CRS    = "EPSG:4326"
_BANDS  = ("Blue", "Green", "Red")
_AFFINE = Affine(0.001, 0.0, 87.0, 0.0, -0.001, 26.5)


def _write_test_image(tmp_path: Path, width: int = 12, height: int = 12) -> Path:
    data = np.arange(len(_BANDS) * height * width, dtype=np.float32).reshape(
        len(_BANDS), height, width
    )
    dr = DownloadResult(
        data=data, crs=_CRS, transform=AffineTransform.from_affine(_AFFINE),
        band_names=_BANDS, width=width, height=height,
        aoi_bounds=AoiBounds(87.0, 26.0, 87.5, 26.5), num_tiles=1,
    )
    writer = GeoTiffWriter(GeoTiffProfile(tiled=False, overviews=False))
    return writer.write(dr, tmp_path / "src.tif").path


@pytest.fixture
def image_path(tmp_path: Path) -> Path:
    return _write_test_image(tmp_path)


# ==============================================================================
# Construction tests
# ==============================================================================

class TestPatchReaderConstruction:
    """Tests for PatchReader.__init__(), properties, and lifecycle."""

    def test_opens_dataset(self, image_path: Path) -> None:
        with PatchReader(image_path) as reader:
            assert reader.width    == 12
            assert reader.height   == 12
            assert reader.num_bands == 3

    def test_missing_file_raises_oserror(self, tmp_path: Path) -> None:
        with pytest.raises(OSError, match="not found"):
            PatchReader(tmp_path / "missing.tif")

    def test_corrupt_file_raises_oserror(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.tif"
        bad.write_bytes(b"not a real tiff")
        with pytest.raises(OSError):
            PatchReader(bad)

    def test_context_manager_closes(self, image_path: Path) -> None:
        reader = PatchReader(image_path)
        reader.close()
        assert reader._dataset is None

    def test_close_idempotent(self, image_path: Path) -> None:
        reader = PatchReader(image_path)
        reader.close()
        reader.close()  # must not raise

    def test_with_statement_closes_on_exit(self, image_path: Path) -> None:
        with PatchReader(image_path) as reader:
            pass
        assert reader._dataset is None


# ==============================================================================
# read_window tests
# ==============================================================================

class TestPatchReaderReadWindow:
    """Tests for PatchReader.read_window()."""

    def test_returns_correct_shape(self, image_path: Path) -> None:
        with PatchReader(image_path) as reader:
            window = PatchWindow(0, 0, 0, 0, 4, 4)
            data, _ = reader.read_window(window)
        assert data.shape == (3, 4, 4)

    def test_returns_float32(self, image_path: Path) -> None:
        with PatchReader(image_path) as reader:
            window = PatchWindow(0, 0, 0, 0, 4, 4)
            data, _ = reader.read_window(window)
        assert data.dtype == np.float32

    def test_pixel_values_correct(self, image_path: Path) -> None:
        with PatchReader(image_path) as reader:
            window = PatchWindow(0, 0, 0, 0, 4, 4)
            data, _ = reader.read_window(window)
        expected = np.arange(12 * 12, dtype=np.float32).reshape(12, 12)[:4, :4]
        np.testing.assert_allclose(data[0], expected)

    def test_offset_pixel_values_correct(self, image_path: Path) -> None:
        with PatchReader(image_path) as reader:
            window = PatchWindow(row_index=1, col_index=1, row_off=4, col_off=4, height=4, width=4)
            data, _ = reader.read_window(window)
        expected = np.arange(12 * 12, dtype=np.float32).reshape(12, 12)[4:8, 4:8]
        np.testing.assert_allclose(data[0], expected)

    def test_transform_origin_shifted(self, image_path: Path) -> None:
        with PatchReader(image_path) as reader:
            window = PatchWindow(row_index=1, col_index=1, row_off=4, col_off=4, height=4, width=4)
            _, transform = reader.read_window(window)
        assert transform.c == pytest.approx(87.0  + 4 * 0.001)
        assert transform.f == pytest.approx(26.5  + 4 * (-0.001))

    def test_pixel_size_preserved_in_transform(self, image_path: Path) -> None:
        with PatchReader(image_path) as reader:
            window = PatchWindow(0, 0, 0, 0, 4, 4)
            _, transform = reader.read_window(window)
        assert transform.a == pytest.approx(0.001)
        assert transform.e == pytest.approx(-0.001)

    def test_returns_affine_transform_type(self, image_path: Path) -> None:
        with PatchReader(image_path) as reader:
            window = PatchWindow(0, 0, 0, 0, 4, 4)
            _, transform = reader.read_window(window)
        assert isinstance(transform, AffineTransform)

    def test_multiple_reads_independent(self, image_path: Path) -> None:
        with PatchReader(image_path) as reader:
            w1 = PatchWindow(0, 0, 0, 0, 4, 4)
            w2 = PatchWindow(1, 1, 4, 4, 4, 4)
            data1, t1 = reader.read_window(w1)
            data2, t2 = reader.read_window(w2)
        assert not np.allclose(data1, data2)
        assert t1.c != t2.c