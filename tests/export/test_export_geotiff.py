"""
Unit tests for src/export/geotiff.py.

Uses real rasterio I/O with small (8x8) numpy arrays written to tmp_path.
No EE, no mocking of rasterio.

Run:
    pytest tests/export/test_export_geotiff.py -v
    pytest tests/export/test_export_geotiff.py -v \
        --cov=src/export/geotiff --cov-report=term-missing
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine

from src.core.exceptions import InvalidValueError
from src.export.downloader import AffineTransform, AoiBounds, DownloadResult
from src.export.geotiff import (
    GeoTiffProfile,
    GeoTiffValidationResult,
    GeoTiffValidator,
    GeoTiffWriteResult,
    GeoTiffWriter,
)


# ==============================================================================
# Fixtures
# ==============================================================================

_CRS       = "EPSG:4326"
_AFFINE    = Affine(0.001, 0.0, 87.0, 0.0, -0.001, 26.5)
_TRANSFORM = AffineTransform.from_affine(_AFFINE)
_AOI       = AoiBounds(87.0, 26.0, 87.5, 26.5)
_BANDS     = ("Blue", "Green", "Red", "NIR", "MNDWI")


def _make_download_result(n_bands: int = 5) -> DownloadResult:
    data = np.stack([
        np.full((8, 8), float(i), dtype=np.float32) for i in range(n_bands)
    ])
    return DownloadResult(
        data=data,
        crs=_CRS,
        transform=_TRANSFORM,
        band_names=_BANDS[:n_bands],
        width=8, height=8,
        aoi_bounds=_AOI,
        num_tiles=1,
    )


@pytest.fixture
def writer() -> GeoTiffWriter:
    return GeoTiffWriter(
        GeoTiffProfile(compress="LZW", tiled=False, overviews=False)
    )


@pytest.fixture
def validator() -> GeoTiffValidator:
    return GeoTiffValidator()


@pytest.fixture
def written(tmp_path: Path, writer: GeoTiffWriter) -> GeoTiffWriteResult:
    return writer.write(_make_download_result(), tmp_path / "test.tif")


# ==============================================================================
# GeoTiffProfile tests
# ==============================================================================

class TestGeoTiffProfile:
    def test_defaults(self) -> None:
        p = GeoTiffProfile()
        assert p.compress  == "LZW"
        assert p.tiled     is True
        assert p.tile_size == 256
        assert p.dtype     == "float32"
        assert p.overviews is True

    def test_frozen_prevents_mutation(self) -> None:
        p = GeoTiffProfile()
        with pytest.raises((AttributeError, TypeError)):
            p.compress = "NONE"  # type: ignore[misc]

    def test_custom_values(self) -> None:
        p = GeoTiffProfile(compress="DEFLATE", tiled=False, overviews=False)
        assert p.compress  == "DEFLATE"
        assert p.tiled     is False
        assert p.overviews is False


# ==============================================================================
# GeoTiffWriteResult tests
# ==============================================================================

class TestGeoTiffWriteResult:
    def test_frozen(self, written: GeoTiffWriteResult) -> None:
        with pytest.raises((AttributeError, TypeError)):
            written.num_bands = 0  # type: ignore[misc]

    def test_path_is_absolute(self, written: GeoTiffWriteResult) -> None:
        assert written.path.is_absolute()

    def test_num_bands_correct(self, written: GeoTiffWriteResult) -> None:
        assert written.num_bands == 5

    def test_band_names_tuple(self, written: GeoTiffWriteResult) -> None:
        assert written.band_names == _BANDS

    def test_crs_correct(self, written: GeoTiffWriteResult) -> None:
        assert written.crs == _CRS


# ==============================================================================
# GeoTiffWriter tests
# ==============================================================================

class TestGeoTiffWriter:
    def test_creates_file(self, written: GeoTiffWriteResult) -> None:
        assert written.path.exists()

    def test_file_non_empty(self, written: GeoTiffWriteResult) -> None:
        assert written.path.stat().st_size > 0

    def test_file_size_bytes_in_result(self, written: GeoTiffWriteResult) -> None:
        assert written.file_size_bytes == written.path.stat().st_size

    def test_correct_band_count(self, written: GeoTiffWriteResult) -> None:
        with rasterio.open(written.path) as ds:
            assert ds.count == 5

    def test_correct_crs(self, written: GeoTiffWriteResult) -> None:
        with rasterio.open(written.path) as ds:
            assert ds.crs.to_string() == _CRS

    def test_band_descriptions_set(self, written: GeoTiffWriteResult) -> None:
        with rasterio.open(written.path) as ds:
            assert list(ds.descriptions) == list(_BANDS)

    def test_band_order_preserved(
        self, tmp_path: Path, writer: GeoTiffWriter
    ) -> None:
        ordered_bands = ("E", "D", "C", "B", "A")
        data = np.zeros((5, 4, 4), dtype=np.float32)
        dr   = DownloadResult(
            data=data, crs=_CRS, transform=_TRANSFORM,
            band_names=ordered_bands, width=4, height=4,
            aoi_bounds=_AOI, num_tiles=1,
        )
        result = writer.write(dr, tmp_path / "order.tif")
        with rasterio.open(result.path) as ds:
            assert list(ds.descriptions) == list(ordered_bands)

    def test_pixel_values_preserved(
        self, tmp_path: Path, writer: GeoTiffWriter
    ) -> None:
        data = np.array([[[1.5, 2.5], [3.5, 4.5]]], dtype=np.float32)
        dr   = DownloadResult(
            data=data, crs=_CRS, transform=_TRANSFORM,
            band_names=("Band1",), width=2, height=2,
            aoi_bounds=_AOI, num_tiles=1,
        )
        result = writer.write(dr, tmp_path / "vals.tif")
        with rasterio.open(result.path) as ds:
            np.testing.assert_allclose(ds.read(1), data[0], rtol=1e-5)

    def test_transform_preserved(self, written: GeoTiffWriteResult) -> None:
        with rasterio.open(written.path) as ds:
            assert ds.transform.a == pytest.approx(_AFFINE.a)
            assert ds.transform.c == pytest.approx(_AFFINE.c)
            assert ds.transform.f == pytest.approx(_AFFINE.f)

    def test_dtype_float32(self, written: GeoTiffWriteResult) -> None:
        with rasterio.open(written.path) as ds:
            assert ds.dtypes[0] == "float32"

    def test_raises_for_non_3d_array(
        self, writer: GeoTiffWriter, tmp_path: Path
    ) -> None:
        bad_dr = DownloadResult(
            data=np.zeros((8, 8)),  # 2-D
            crs=_CRS, transform=_TRANSFORM,
            band_names=("B",), width=8, height=8,
            aoi_bounds=_AOI, num_tiles=1,
        )
        with pytest.raises(InvalidValueError, match="3-D"):
            writer.write(bad_dr, tmp_path / "bad.tif")

    def test_raises_for_band_name_count_mismatch(
        self, writer: GeoTiffWriter, tmp_path: Path
    ) -> None:
        dr = DownloadResult(
            data=np.zeros((3, 8, 8), dtype=np.float32),
            crs=_CRS, transform=_TRANSFORM,
            band_names=("A", "B"),  # 2 names for 3 bands
            width=8, height=8,
            aoi_bounds=_AOI, num_tiles=1,
        )
        with pytest.raises(InvalidValueError, match="band_names"):
            writer.write(dr, tmp_path / "bad.tif")


# ==============================================================================
# GeoTiffValidator tests
# ==============================================================================

class TestGeoTiffValidator:
    def test_valid_file_passes(
        self, validator: GeoTiffValidator, written: GeoTiffWriteResult
    ) -> None:
        result = validator.validate(written)
        assert result.is_valid is True
        assert len(result.issues) == 0

    def test_missing_file_fails(
        self, validator: GeoTiffValidator, tmp_path: Path
    ) -> None:
        fake_write = GeoTiffWriteResult(
            path=tmp_path / "nonexistent.tif",
            num_bands=5, width=8, height=8,
            crs=_CRS, transform=_TRANSFORM,
            band_names=_BANDS, file_size_bytes=0,
            compress="LZW",
        )
        result = validator.validate(fake_write)
        assert result.is_valid is False
        assert any("does not exist" in i for i in result.issues)

    def test_empty_file_fails(
        self, validator: GeoTiffValidator, tmp_path: Path, written: GeoTiffWriteResult
    ) -> None:
        empty = tmp_path / "empty.tif"
        empty.write_bytes(b"")
        fake = GeoTiffWriteResult(
            path=empty, num_bands=5, width=8, height=8,
            crs=_CRS, transform=_TRANSFORM,
            band_names=_BANDS, file_size_bytes=0, compress="LZW",
        )
        result = validator.validate(fake)
        assert result.is_valid is False

    def test_wrong_band_count_detected(
        self, validator: GeoTiffValidator, written: GeoTiffWriteResult
    ) -> None:
        wrong = GeoTiffWriteResult(
            path=written.path, num_bands=99, width=written.width,
            height=written.height, crs=written.crs, transform=written.transform,
            band_names=tuple(f"B{i}" for i in range(99)),
            file_size_bytes=written.file_size_bytes, compress="LZW",
        )
        result = validator.validate(wrong)
        assert result.is_valid is False
        assert any("Band count" in i for i in result.issues)

    def test_wrong_crs_detected(
        self, validator: GeoTiffValidator, written: GeoTiffWriteResult
    ) -> None:
        wrong_crs = GeoTiffWriteResult(
            path=written.path, num_bands=written.num_bands,
            width=written.width, height=written.height,
            crs="EPSG:32644",    # file has 4326
            transform=written.transform,
            band_names=written.band_names,
            file_size_bytes=written.file_size_bytes, compress="LZW",
        )
        result = validator.validate(wrong_crs, expected_crs="EPSG:32644")
        assert result.is_valid is False
        assert any("CRS" in i for i in result.issues)

    def test_summary_is_ascii(
        self, validator: GeoTiffValidator, written: GeoTiffWriteResult
    ) -> None:
        result  = validator.validate(written)
        summary = result.summary()
        assert all(ord(c) < 128 for c in summary)

    def test_result_is_frozen(
        self, validator: GeoTiffValidator, written: GeoTiffWriteResult
    ) -> None:
        result = validator.validate(written)
        with pytest.raises((AttributeError, TypeError)):
            result.is_valid = False  # type: ignore[misc]

    def test_band_names_in_result(
        self, validator: GeoTiffValidator, written: GeoTiffWriteResult
    ) -> None:
        result = validator.validate(written)
        assert result.band_names == _BANDS

    def test_no_expected_args_always_passes(
        self, validator: GeoTiffValidator, written: GeoTiffWriteResult
    ) -> None:
        result = validator.validate(written)
        assert result.is_valid is True