"""
Unit tests for src/labels/validator.py.

Run:
    pytest tests/labels/test_label_validator.py -v \
        --cov=src/labels/validator --cov-report=term-missing
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.crs import CRS

from src.core.exceptions import InvalidValueError
from src.export.downloader import AffineTransform, AoiBounds, DownloadResult
from src.export.geotiff import GeoTiffProfile, GeoTiffWriter
from src.labels.schema import ClassDefinition, ClassSchema
from src.labels.validator import LabelValidator

_CRS = "EPSG:4326"
_AFFINE = Affine(0.001, 0.0, 87.0, 0.0, -0.001, 26.5)
_SIZE = 8


def _schema() -> ClassSchema:
    return ClassSchema(classes=(
        ClassDefinition(0, "background", (128, 128, 128)),
        ClassDefinition(1, "water", (0, 119, 190)),
        ClassDefinition(2, "sand", (255, 200, 87)),
        ClassDefinition(3, "vegetation", (34, 139, 34)),
    ))


def _write_patch(tmp_path: Path, name: str = "patch.tif") -> Path:
    data = np.random.rand(4, _SIZE, _SIZE).astype(np.float32)
    dr = DownloadResult(
        data=data, crs=_CRS, transform=AffineTransform.from_affine(_AFFINE),
        band_names=("Blue", "Green", "Red", "NIR"), width=_SIZE, height=_SIZE,
        aoi_bounds=AoiBounds(87.0, 26.0, 87.5, 26.5), num_tiles=1,
    )
    writer = GeoTiffWriter(GeoTiffProfile(tiled=False, overviews=False))
    return writer.write(dr, tmp_path / name).path


def _write_mask(
    tmp_path: Path, name: str, values: np.ndarray,
    crs: str = _CRS, transform: Affine = _AFFINE,
) -> Path:
    path = tmp_path / name
    profile = {
        "driver": "GTiff", "dtype": "uint8", "width": values.shape[1],
        "height": values.shape[0], "count": 1, "crs": CRS.from_string(crs),
        "transform": transform,
    }
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(values.astype("uint8"), 1)
    return path


@pytest.fixture
def schema() -> ClassSchema:
    return _schema()


@pytest.fixture
def validator(schema: ClassSchema) -> LabelValidator:
    return LabelValidator(schema, nodata_value=255, max_nodata_ratio=0.5)


@pytest.fixture
def patch_path(tmp_path: Path) -> Path:
    return _write_patch(tmp_path)


class TestLabelValidatorConstruction:
    def test_valid_construction(self, schema: ClassSchema) -> None:
        v = LabelValidator(schema, nodata_value=255, max_nodata_ratio=0.5)
        assert v is not None

    def test_nodata_collides_with_class_raises(self, schema: ClassSchema) -> None:
        with pytest.raises(InvalidValueError, match="collide"):
            LabelValidator(schema, nodata_value=0, max_nodata_ratio=0.5)


class TestValidateHappyPath:
    def test_valid_multi_class_mask(self, validator, tmp_path: Path, patch_path: Path) -> None:
        values = np.zeros((_SIZE, _SIZE), dtype=np.uint8)
        values[:4, :] = 1
        values[4:, :4] = 2
        mask_path = _write_mask(tmp_path, "mask.tif", values)
        result = validator.validate(patch_path, mask_path)
        assert result.is_valid is True
        assert result.num_classes_present == 3
        assert result.mask_exists is True


class TestValidateFailures:
    def test_none_mask_path(self, validator, patch_path: Path) -> None:
        result = validator.validate(patch_path, None)
        assert result.is_valid is False
        assert result.mask_exists is False

    def test_missing_mask_file(self, validator, tmp_path: Path, patch_path: Path) -> None:
        result = validator.validate(patch_path, tmp_path / "nonexistent.tif")
        assert result.is_valid is False
        assert any("missing" in i for i in result.issues)

    def test_crs_mismatch(self, validator, tmp_path: Path, patch_path: Path) -> None:
        values = np.ones((_SIZE, _SIZE), dtype=np.uint8)
        mask_path = _write_mask(tmp_path, "mask.tif", values, crs="EPSG:32644")
        result = validator.validate(patch_path, mask_path)
        assert result.is_valid is False
        assert result.crs_match is False

    def test_dimension_mismatch(self, validator, tmp_path: Path, patch_path: Path) -> None:
        values = np.ones((_SIZE * 2, _SIZE * 2), dtype=np.uint8)
        mask_path = _write_mask(tmp_path, "mask.tif", values)
        result = validator.validate(patch_path, mask_path)
        assert result.dimension_match is False

    def test_transform_mismatch(self, validator, tmp_path: Path, patch_path: Path) -> None:
        values = np.ones((_SIZE, _SIZE), dtype=np.uint8)
        shifted = Affine(0.001, 0.0, 90.0, 0.0, -0.001, 26.5)
        mask_path = _write_mask(tmp_path, "mask.tif", values, transform=shifted)
        result = validator.validate(patch_path, mask_path)
        assert result.transform_match is False

    def test_invalid_class_id(self, validator, tmp_path: Path, patch_path: Path) -> None:
        values = np.full((_SIZE, _SIZE), 99, dtype=np.uint8)
        mask_path = _write_mask(tmp_path, "mask.tif", values)
        result = validator.validate(patch_path, mask_path)
        assert result.is_valid is False
        assert any("invalid class" in i for i in result.issues)

    def test_empty_mask_all_nodata(self, validator, tmp_path: Path, patch_path: Path) -> None:
        values = np.full((_SIZE, _SIZE), 255, dtype=np.uint8)
        mask_path = _write_mask(tmp_path, "mask.tif", values)
        result = validator.validate(patch_path, mask_path)
        assert result.valid_pixel_ratio == pytest.approx(0.0)

    def test_exceeds_max_nodata_ratio(self, schema, tmp_path: Path, patch_path: Path) -> None:
        strict = LabelValidator(schema, nodata_value=255, max_nodata_ratio=0.1)
        values = np.zeros((_SIZE, _SIZE), dtype=np.uint8)
        values[:6, :] = 255
        mask_path = _write_mask(tmp_path, "mask.tif", values)
        result = strict.validate(patch_path, mask_path)
        assert result.is_valid is False


class TestSingleClassBehaviour:
    def test_allowed_by_default(self, validator, tmp_path: Path, patch_path: Path) -> None:
        values = np.zeros((_SIZE, _SIZE), dtype=np.uint8)
        mask_path = _write_mask(tmp_path, "mask.tif", values)
        result = validator.validate(patch_path, mask_path)
        assert result.is_single_class is True
        assert result.is_valid is True

    def test_rejected_when_configured(self, schema, tmp_path: Path, patch_path: Path) -> None:
        strict = LabelValidator(
            schema, nodata_value=255, max_nodata_ratio=0.5,
            reject_single_class_masks=True,
        )
        values = np.zeros((_SIZE, _SIZE), dtype=np.uint8)
        mask_path = _write_mask(tmp_path, "mask.tif", values)
        result = strict.validate(patch_path, mask_path)
        assert result.is_valid is False