"""
Unit tests for src/training/dataset.py.

Existing tests are unchanged. New tests for the six read-only properties
and summary() have been appended.

Run:
    pytest tests/training/test_dataset.py -v \
        --cov=src/training/dataset --cov-report=term-missing
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.crs import CRS

from src.export.downloader import AffineTransform, AoiBounds, DownloadResult
from src.export.geotiff import GeoTiffProfile, GeoTiffWriter
from src.training.dataset import RiverMorphologyDataset, SampleMetadata

pytest.importorskip("torch", reason="torch is required for training tests")

_CRS      = "EPSG:4326"
_AFFINE   = Affine(0.001, 0.0, 87.0, 0.0, -0.001, 26.5)
_SIZE     = 8
_BANDS    = ("Blue", "Green", "Red", "NIR")
_N_BANDS  = len(_BANDS)
_CLASSES  = ("background", "water", "sand", "vegetation")


def _write_patch(path: Path) -> Path:
    data = np.random.rand(_N_BANDS, _SIZE, _SIZE).astype(np.float32)
    dr = DownloadResult(
        data=data, crs=_CRS, transform=AffineTransform.from_affine(_AFFINE),
        band_names=_BANDS, width=_SIZE, height=_SIZE,
        aoi_bounds=AoiBounds(87.0, 26.0, 87.5, 26.5), num_tiles=1,
    )
    writer = GeoTiffWriter(GeoTiffProfile(tiled=False, overviews=False))
    return writer.write(dr, path).path


def _write_mask(path: Path, values: np.ndarray) -> Path:
    profile = {
        "driver": "GTiff", "dtype": "uint8", "width": _SIZE, "height": _SIZE,
        "count": 1, "crs": CRS.from_string(_CRS), "transform": _AFFINE,
    }
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(values.astype("uint8"), 1)
    return path


def _make_entry(patch_path: Path, mask_path: Path, patch_id: str = "p1") -> object:
    from unittest.mock import MagicMock
    e = MagicMock()
    e.sample_id               = patch_id
    e.patch_id                = patch_id
    e.scene_id                = "scene001"
    e.patch_path              = str(patch_path)
    e.mask_path               = str(mask_path)
    e.crs                     = _CRS
    e.width                   = _SIZE
    e.height                  = _SIZE
    e.num_bands               = _N_BANDS
    e.row_index               = 0
    e.col_index               = 0
    e.acquisition_date        = "2023-07-15"
    e.year                    = 2023
    e.month                   = 7
    e.season                  = "monsoon"
    e.hydrological_year       = 2023
    e.sensor                  = "L8"
    e.aoi_id                  = "aoi_1"
    e.river_name               = ""
    e.reach_id                 = ""
    e.basin_id                 = ""
    e.label_valid_pixel_ratio  = 0.95
    return e


@pytest.fixture
def patch_path(tmp_path: Path) -> Path:
    return _write_patch(tmp_path / "patch.tif")


@pytest.fixture
def mask_path(tmp_path: Path) -> Path:
    values = np.zeros((_SIZE, _SIZE), dtype=np.uint8)
    values[:4, :] = 1
    values[4:, :4] = 2
    return _write_mask(tmp_path / "mask.tif", values)


@pytest.fixture
def entry(patch_path: Path, mask_path: Path):
    return _make_entry(patch_path, mask_path)


@pytest.fixture
def dataset(entry):
    return RiverMorphologyDataset([entry], split="train")


# ==============================================================================
# Existing tests (unchanged)
# ==============================================================================

class TestRiverMorphologyDataset:
    def test_len(self, dataset: RiverMorphologyDataset) -> None:
        assert len(dataset) == 1

    def test_getitem_returns_three_tuple(self, dataset) -> None:
        result = dataset[0]
        assert len(result) == 3

    def test_image_tensor_shape(self, dataset) -> None:
        import torch
        img, _, _ = dataset[0]
        assert isinstance(img, torch.Tensor)
        assert img.shape == (_N_BANDS, _SIZE, _SIZE)

    def test_image_tensor_dtype_float32(self, dataset) -> None:
        import torch
        img, _, _ = dataset[0]
        assert img.dtype == torch.float32

    def test_mask_tensor_shape(self, dataset) -> None:
        import torch
        _, mask, _ = dataset[0]
        assert isinstance(mask, torch.Tensor)
        assert mask.shape == (_SIZE, _SIZE)

    def test_mask_tensor_dtype_long(self, dataset) -> None:
        import torch
        _, mask, _ = dataset[0]
        assert mask.dtype == torch.int64

    def test_metadata_type(self, dataset) -> None:
        _, _, meta = dataset[0]
        assert isinstance(meta, SampleMetadata)

    def test_metadata_sample_id(self, dataset) -> None:
        _, _, meta = dataset[0]
        assert meta.sample_id == "p1"

    def test_metadata_split(self, dataset) -> None:
        _, _, meta = dataset[0]
        assert meta.split == "train"

    def test_metadata_season(self, dataset) -> None:
        _, _, meta = dataset[0]
        assert meta.season == "monsoon"

    def test_nodata_replaced_with_ignore_index(self, tmp_path: Path) -> None:
        patch = _write_patch(tmp_path / "p2.tif")
        values = np.full((_SIZE, _SIZE), 255, dtype=np.uint8)
        mask   = _write_mask(tmp_path / "m2.tif", values)
        e      = _make_entry(patch, mask, "p2")
        ds     = RiverMorphologyDataset(
            [e], nodata_value=255, ignore_index=255, split="train"
        )
        import torch
        _, mask_tensor, _ = ds[0]
        assert int(mask_tensor.unique()[0]) == 255

    def test_num_bands_property(self, dataset) -> None:
        assert dataset.num_bands == _N_BANDS

    def test_split_property(self, dataset) -> None:
        assert dataset.split == "train"

    def test_num_samples_property(self, dataset) -> None:
        assert dataset.num_samples == 1

    def test_multiple_entries(self, tmp_path: Path) -> None:
        entries = []
        for i in range(3):
            p = _write_patch(tmp_path / f"p{i}.tif")
            m = _write_mask(
                tmp_path / f"m{i}.tif",
                np.zeros((_SIZE, _SIZE), dtype=np.uint8),
            )
            entries.append(_make_entry(p, m, f"p{i}"))
        ds = RiverMorphologyDataset(entries, split="train")
        assert len(ds) == 3
        for i in range(3):
            img, mask, meta = ds[i]
            assert meta.sample_id == f"p{i}"

    def test_missing_patch_raises_oserror(self, tmp_path: Path) -> None:
        values = np.zeros((_SIZE, _SIZE), dtype=np.uint8)
        mask   = _write_mask(tmp_path / "m.tif", values)
        e      = _make_entry(tmp_path / "nonexistent.tif", mask)
        ds     = RiverMorphologyDataset([e], split="train")
        with pytest.raises(OSError):
            ds[0]

    def test_normalization_applied_when_stats_given(self, tmp_path: Path) -> None:
        from src.training.normalizer import NormalizationStats
        patch = _write_patch(tmp_path / "pn.tif")
        values = np.zeros((_SIZE, _SIZE), dtype=np.uint8)
        mask   = _write_mask(tmp_path / "mn.tif", values)
        e      = _make_entry(patch, mask)
        stats  = NormalizationStats(
            strategy="per_band_mean_std", num_bands=_N_BANDS,
            band_names=_BANDS, mean=tuple(0.0 for _ in _BANDS),
            std=tuple(1.0 for _ in _BANDS), percentile_min=2, percentile_max=98,
        )
        ds = RiverMorphologyDataset([e], norm_stats=stats, split="train")
        img, _, _ = ds[0]
        assert img.shape[0] == _N_BANDS


# ==============================================================================
# Transform interface tests
# ==============================================================================

class TestRiverMorphologyDatasetTransform:
    """Verify RiverMorphologyDataset uses Transform interface correctly."""

    def test_default_transform_is_identity(self, entry) -> None:
        from src.training.transforms import IdentityTransform
        ds = RiverMorphologyDataset([entry], split="train")
        assert isinstance(ds._transform, IdentityTransform)

    def test_none_transform_installs_identity(self, entry) -> None:
        from src.training.transforms import IdentityTransform
        ds = RiverMorphologyDataset([entry], transform=None, split="train")
        assert isinstance(ds._transform, IdentityTransform)

    def test_identity_transform_accepted(self, entry) -> None:
        from src.training.transforms import IdentityTransform
        ds = RiverMorphologyDataset(
            [entry], transform=IdentityTransform(), split="train"
        )
        img, mask, _ = ds[0]
        import torch
        assert img.shape   == (_N_BANDS, _SIZE, _SIZE)
        assert mask.shape  == (_SIZE, _SIZE)
        assert img.dtype   == torch.float32
        assert mask.dtype  == torch.int64

    def test_custom_transform_called(
        self, entry, patch_path: Path, mask_path: Path
    ) -> None:
        """A custom Transform subclass must be called inside __getitem__."""
        from src.training.transforms import Transform

        called: list[bool] = []

        class RecordingTransform(Transform):
            def __call__(self, image, mask):
                called.append(True)
                return image, mask

        ds = RiverMorphologyDataset(
            [entry], transform=RecordingTransform(), split="train"
        )
        ds[0]
        assert called == [True]

    def test_transform_receives_chw_image(self, entry) -> None:
        """Transform must receive (C, H, W) not (H, W, C)."""
        from src.training.transforms import Transform

        received_shape: list[tuple] = []

        class ShapeCapture(Transform):
            def __call__(self, image, mask):
                received_shape.append(image.shape)
                return image, mask

        ds = RiverMorphologyDataset([entry], transform=ShapeCapture(), split="train")
        ds[0]
        assert len(received_shape) == 1
        assert received_shape[0] == (_N_BANDS, _SIZE, _SIZE)


# ==============================================================================
# New read-only property tests
# ==============================================================================

class TestRiverMorphologyDatasetProperties:
    """Tests for the six new read-only Dataset properties."""

    def test_band_names_set(self, entry) -> None:
        ds = RiverMorphologyDataset([entry], band_names=_BANDS, split="train")
        assert ds.band_names == _BANDS

    def test_band_names_default_empty(self, entry) -> None:
        ds = RiverMorphologyDataset([entry], split="train")
        assert ds.band_names == ()

    def test_class_names_set(self, entry) -> None:
        ds = RiverMorphologyDataset([entry], class_names=_CLASSES, split="train")
        assert ds.class_names == _CLASSES

    def test_class_names_default_empty(self, entry) -> None:
        ds = RiverMorphologyDataset([entry], split="train")
        assert ds.class_names == ()

    def test_num_classes_set(self, entry) -> None:
        ds = RiverMorphologyDataset([entry], num_classes=4, split="train")
        assert ds.num_classes == 4

    def test_num_classes_default_zero(self) -> None:
        ds = RiverMorphologyDataset([], split="train")
        assert ds.num_classes == 0

    def test_patch_size_from_entry(self, entry) -> None:
        ds = RiverMorphologyDataset([entry], split="train")
        assert ds.patch_size == _SIZE

    def test_patch_size_empty_dataset(self) -> None:
        ds = RiverMorphologyDataset([], split="train")
        assert ds.patch_size == 0

    def test_ignore_index_stored(self, entry) -> None:
        ds = RiverMorphologyDataset([entry], ignore_index=42, split="train")
        assert ds.ignore_index == 42

    def test_ignore_index_default(self, entry) -> None:
        ds = RiverMorphologyDataset([entry], split="train")
        assert ds.ignore_index == 255

    def test_split_property_validation(self, entry) -> None:
        for split_name in ("train", "validation", "test"):
            ds = RiverMorphologyDataset([entry], split=split_name)
            assert ds.split == split_name

    # ------------------------------------------------------------------
    # summary() tests
    # ------------------------------------------------------------------

    def test_summary_returns_string(self, entry) -> None:
        ds      = RiverMorphologyDataset([entry], split="train")
        summary = ds.summary()
        assert isinstance(summary, str)

    def test_summary_ascii_only(self, entry) -> None:
        ds      = RiverMorphologyDataset(
            [entry], band_names=_BANDS, class_names=_CLASSES,
            num_classes=4, split="validation",
        )
        summary = ds.summary()
        assert all(ord(c) < 128 for c in summary)

    def test_summary_contains_split_name(self, entry) -> None:
        ds      = RiverMorphologyDataset([entry], split="validation")
        summary = ds.summary()
        assert "validation" in summary

    def test_summary_contains_sample_count(self, entry) -> None:
        ds      = RiverMorphologyDataset([entry], split="train")
        summary = ds.summary()
        assert "1" in summary

    def test_summary_contains_transform_type(self, entry) -> None:
        from src.training.transforms import IdentityTransform
        ds      = RiverMorphologyDataset(
            [entry], transform=IdentityTransform(), split="train"
        )
        summary = ds.summary()
        assert "IdentityTransform" in summary

    def test_summary_reflects_band_names(self, entry) -> None:
        ds      = RiverMorphologyDataset(
            [entry], band_names=("Blue", "NIR"), split="train"
        )
        summary = ds.summary()
        assert "Blue" in summary

    def test_summary_reflects_num_classes(self, entry) -> None:
        ds      = RiverMorphologyDataset([entry], num_classes=4, split="train")
        summary = ds.summary()
        assert "4" in summary

    def test_properties_consistent_with_getitem(
        self, entry, patch_path: Path, mask_path: Path
    ) -> None:
        """The properties must be consistent with the actual data returned."""
        import torch
        ds      = RiverMorphologyDataset(
            [entry], band_names=_BANDS, class_names=_CLASSES,
            num_classes=len(_CLASSES), split="train",
        )
        img, mask, meta = ds[0]
        assert img.shape[0]  == ds.num_bands
        assert len(ds.band_names)  == ds.num_bands
        assert len(ds.class_names) == ds.num_classes