"""
Unit tests for src/labels/manager.py.

Uses real rasterio I/O via GeoTiffWriter (Module 7) for patch fixtures and
direct rasterio writes for mask fixtures. Real PatchDatasetResult,
PatchManifest, PatchManifestEntry, SceneMetadata, and FilesystemLabelSource
instances are constructed (no mocking needed -- all are frozen dataclasses
or simple, real components).

Run:
    pytest tests/labels/test_label_manager.py -v \
        --cov=src/labels/manager --cov-report=term-missing
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.crs import CRS

from src.core.exceptions import InvalidValueError
from src.export.downloader import AffineTransform, AoiBounds, DownloadResult
from src.export.geotiff import GeoTiffProfile, GeoTiffWriter
from src.export.metadata import METADATA_SCHEMA_VERSION, SceneMetadata
from src.labels.manager import LabelDatasetResult, LabelManager
from src.labels.source import FilesystemLabelSource
from src.patches.generator import PatchDatasetResult
from src.patches.manifest import PatchManifest, PatchManifestEntry
from tests.conftest import make_valid_config, write_config

_CRS = "EPSG:4326"
_AFFINE = Affine(0.001, 0.0, 87.0, 0.0, -0.001, 26.5)
_SIZE = 8
_BANDS = ("Blue", "Green", "Red", "NIR")

_SEASONS = {
    "pre_monsoon":  [3, 4, 5], "monsoon": [6, 7, 8, 9],
    "post_monsoon": [10, 11], "winter": [12, 1, 2],
}


# def _config(tmp_path: Path, labels_overrides: dict | None = None):
#     from src.core.config import Config
#     data = make_valid_config()
#     data["classes"] = {
#         "num_classes": 4,
#         "labels": {"background": 0, "water": 1, "sand": 2, "vegetation": 3},
#         "names": ["background", "water", "sand", "vegetation"],
#         "colors": {
#             "background": [128, 128, 128], "water": [0, 119, 190],
#             "sand": [255, 200, 87], "vegetation": [34, 139, 34],
#         },
#     }
#     data["temporal"] = {"seasons": _SEASONS, "hydrological_year_start_month": 6}
#     data["labels"] = {
#         "nodata_value": 255, "mask_filename_pattern": "{patch_id}_mask.tif",
#         "default_label_version": "1.0.0", "default_annotator": "auto_generated",
#         "default_confidence": 1.0, "default_confidence_source": "automatic",
#         "min_distinct_classes": 1, "reject_single_class_masks": False,
#         "max_nodata_ratio": 0.5, "output_formats": ["csv", "json"],
#         "ratios": {"water_sand_ratio": ["water", "sand"]},
#         "bare_sediment_numerator": ["sand"],
#         "bare_sediment_denominator": ["water", "sand", "vegetation"],
#     }
#     if labels_overrides:
#         data["labels"].update(labels_overrides)
#     return Config(config_path=write_config(tmp_path, data))
def _config(tmp_path: Path, labels_overrides: dict | None = None):
    from src.core.config import Config

    data = make_valid_config()

    data["classes"] = {
        "num_classes": 4,
        "labels": {
            "background": 0,
            "water": 1,
            "sand": 2,
            "vegetation": 3,
        },
        "names": [
            "background",
            "water",
            "sand",
            "vegetation",
        ],
        "colors": {
            "background": [128, 128, 128],
            "water": [0, 119, 190],
            "sand": [255, 200, 87],
            "vegetation": [34, 139, 34],
        },
    }

    # NEW
    num_classes = data["classes"]["num_classes"]
    data["model"]["num_classes"] = num_classes

    if "loss" in data:
        data["loss"]["num_classes"] = num_classes

    data["temporal"] = {
        "seasons": _SEASONS,
        "hydrological_year_start_month": 6,
    }

    data["labels"] = {
        "nodata_value": 255,
        "mask_filename_pattern": "{patch_id}_mask.tif",
        "default_label_version": "1.0.0",
        "default_annotator": "auto_generated",
        "default_confidence": 1.0,
        "default_confidence_source": "automatic",
        "min_distinct_classes": 1,
        "reject_single_class_masks": False,
        "max_nodata_ratio": 0.5,
        "output_formats": ["csv", "json"],
        "ratios": {"water_sand_ratio": ["water", "sand"]},
        "bare_sediment_numerator": ["sand"],
        "bare_sediment_denominator": [
            "water",
            "sand",
            "vegetation",
        ],
    }

    if labels_overrides:
        data["labels"].update(labels_overrides)

    return Config(config_path=write_config(tmp_path, data))


def _write_patch(directory: Path, patch_id: str) -> Path:
    data = np.random.rand(len(_BANDS), _SIZE, _SIZE).astype(np.float32)
    dr = DownloadResult(
        data=data, crs=_CRS, transform=AffineTransform.from_affine(_AFFINE),
        band_names=_BANDS, width=_SIZE, height=_SIZE,
        aoi_bounds=AoiBounds(87.0, 26.0, 87.5, 26.5), num_tiles=1,
    )
    writer = GeoTiffWriter(GeoTiffProfile(tiled=False, overviews=False))
    return writer.write(dr, directory / f"{patch_id}.tif").path


def _write_mask(directory: Path, patch_id: str, values: np.ndarray, crs: str = _CRS) -> Path:
    path = directory / f"{patch_id}_mask.tif"
    profile = {
        "driver": "GTiff", "dtype": "uint8", "width": values.shape[1],
        "height": values.shape[0], "count": 1, "crs": CRS.from_string(crs),
        "transform": _AFFINE,
    }
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(values.astype("uint8"), 1)
    return path


def _scene_metadata(start="2023-07-01", end="2023-07-31") -> SceneMetadata:
    return SceneMetadata(
        schema_version=METADATA_SCHEMA_VERSION, scene_id="scene001",
        export_timestamp=datetime.now(timezone.utc).isoformat(),
        aoi=AoiBounds(87.0, 26.0, 87.5, 26.5), crs=_CRS,
        transform=AffineTransform.from_affine(_AFFINE),
        width=64, height=64, resolution_meters=30.0, num_bands=4,
        band_names=_BANDS, composite_bands=_BANDS,
        spectral_indices=("NDWI", "MNDWI"), composite_method="median",
        sensors=("L8", "L9"), start_date=start, end_date=end,
        cloud_cover_limit=20.0, operations_applied=("scaling", "harmonization"),
        num_tiles=1,
    )


def _patch_dataset_result(patches_dir: Path, n: int = 4) -> tuple[PatchDatasetResult, list[str]]:
    entries: list[PatchManifestEntry] = []
    patch_ids: list[str] = []
    for i in range(n):
        row, col = divmod(i, 2)
        patch_id = f"scene001_r{row:03d}_c{col:03d}"
        patch_ids.append(patch_id)
        patch_path = _write_patch(patches_dir, patch_id)
        entries.append(PatchManifestEntry(
            patch_id=patch_id, scene_id="scene001",
            source_image_path=str(patches_dir / "image.tif"),
            patch_path=str(patch_path), row_index=row, col_index=col,
            row_off=row * _SIZE, col_off=col * _SIZE, height=_SIZE, width=_SIZE,
            num_bands=len(_BANDS), crs=_CRS, valid_pixel_ratio=1.0,
            file_size_bytes=patch_path.stat().st_size,
            created_at=datetime.now(timezone.utc).isoformat(),
        ))
    manifest = PatchManifest(entries=tuple(entries), csv_path=None, json_path=None)
    result = PatchDatasetResult(
        scene_id="scene001", output_dir=patches_dir, scene_patches_dir=patches_dir,
        manifest=manifest, patches_generated=n, patches_skipped=0, total_windows=n,
        patch_size=_SIZE, stride=_SIZE, operations_log=("tiling: ok",),
    )
    return result, patch_ids


@pytest.fixture
def config(tmp_path: Path):
    return _config(tmp_path)


@pytest.fixture
def manager(config) -> LabelManager:
    return LabelManager(config)


@pytest.fixture
def scene_metadata() -> SceneMetadata:
    return _scene_metadata()


@pytest.fixture
def patches_dir(tmp_path: Path) -> Path:
    d = tmp_path / "patches_src"
    d.mkdir()
    return d


@pytest.fixture
def patch_result_and_ids(patches_dir: Path):
    return _patch_dataset_result(patches_dir)


@pytest.fixture
def labels_source_dir(tmp_path: Path, patch_result_and_ids) -> Path:
    _, patch_ids = patch_result_and_ids
    d = tmp_path / "labels_src"
    d.mkdir()
    for patch_id in patch_ids:
        values = np.zeros((_SIZE, _SIZE), dtype=np.uint8)
        values[:4, :] = 1
        values[4:, :4] = 2
        _write_mask(d, patch_id, values)
    return d


class TestLabelManagerConstruction:
    def test_reads_config_values(self, manager: LabelManager) -> None:
        assert manager._nodata_value == 255
        assert manager._class_schema.num_classes == 4

    def test_no_filesystem_side_effects(self, config, tmp_path: Path) -> None:
        LabelManager(config)
        assert not (tmp_path / "labels_out").exists()


class TestLabelManagerGenerateWithDir:
    """Tests using the convenience labels_source_dir argument."""

    def test_returns_label_dataset_result(
        self, manager, patch_result_and_ids, scene_metadata, labels_source_dir, tmp_path
    ) -> None:
        result, _ = patch_result_and_ids
        out = manager.generate(
            result, scene_metadata, tmp_path / "labels_out",
            aoi_id="aoi_1", labels_source_dir=labels_source_dir,
        )
        assert isinstance(out, LabelDatasetResult)
        assert out.source_type == "filesystem"

    def test_all_labels_valid(
        self, manager, patch_result_and_ids, scene_metadata, labels_source_dir, tmp_path
    ) -> None:
        result, _ = patch_result_and_ids
        out = manager.generate(
            result, scene_metadata, tmp_path / "labels_out",
            aoi_id="aoi_1", labels_source_dir=labels_source_dir,
        )
        assert out.labels_valid == 4 and out.labels_rejected == 0 and out.labels_missing == 0

    def test_masks_copied(
        self, manager, patch_result_and_ids, scene_metadata, labels_source_dir, tmp_path
    ) -> None:
        result, patch_ids = patch_result_and_ids
        out = manager.generate(
            result, scene_metadata, tmp_path / "labels_out",
            aoi_id="aoi_1", labels_source_dir=labels_source_dir,
        )
        for patch_id in patch_ids:
            assert (out.scene_labels_dir / f"{patch_id}_mask.tif").exists()

    def test_temporal_metadata_attached(
        self, manager, patch_result_and_ids, scene_metadata, labels_source_dir, tmp_path
    ) -> None:
        result, _ = patch_result_and_ids
        out = manager.generate(
            result, scene_metadata, tmp_path / "labels_out", aoi_id="aoi_1",
            labels_source_dir=labels_source_dir, river_name="Kosi",
            reach_id="reach-1", basin_id="basin-1",
        )
        for entry in out.manifest.entries:
            assert entry.season == "monsoon"
            assert entry.hydrological_year == 2023
            assert entry.river_name == "Kosi"
            assert entry.reach_id == "reach-1"
            assert entry.basin_id == "basin-1"
            assert entry.source_type == "filesystem"

    def test_statistics_populated_with_ratios(
        self, manager, patch_result_and_ids, scene_metadata, labels_source_dir, tmp_path
    ) -> None:
        result, _ = patch_result_and_ids
        out = manager.generate(
            result, scene_metadata, tmp_path / "labels_out",
            aoi_id="aoi_1", labels_source_dir=labels_source_dir,
        )
        assert out.statistics.valid_labels == 4
        ratio = next(r for r in out.statistics.class_ratios if r.name == "water_sand_ratio")
        assert ratio.value is not None

    def test_summary_lines_ascii(
        self, manager, patch_result_and_ids, scene_metadata, labels_source_dir, tmp_path
    ) -> None:
        result, _ = patch_result_and_ids
        out = manager.generate(
            result, scene_metadata, tmp_path / "labels_out",
            aoi_id="aoi_1", labels_source_dir=labels_source_dir,
        )
        for line in out.summary_lines():
            assert all(ord(c) < 128 for c in line)


class TestLabelManagerGenerateWithExplicitSource:
    """Tests using an explicit LabelSource instance (the abstraction contract)."""

    def test_explicit_label_source_used(
        self, manager, patch_result_and_ids, scene_metadata, labels_source_dir, tmp_path
    ) -> None:
        result, _ = patch_result_and_ids
        source = FilesystemLabelSource(labels_source_dir, "{patch_id}_mask.tif")
        out = manager.generate(
            result, scene_metadata, tmp_path / "labels_out",
            aoi_id="aoi_1", label_source=source,
        )
        assert out.source_type == "filesystem"
        assert out.labels_valid == 4

    def test_label_source_takes_precedence_over_dir(
        self, manager, patch_result_and_ids, scene_metadata, labels_source_dir, tmp_path
    ) -> None:
        """A provided label_source must be used even if labels_source_dir is also passed."""
        result, _ = patch_result_and_ids
        empty_dir = tmp_path / "empty_labels"
        empty_dir.mkdir()
        source = FilesystemLabelSource(labels_source_dir, "{patch_id}_mask.tif")
        out = manager.generate(
            result, scene_metadata, tmp_path / "labels_out", aoi_id="aoi_1",
            label_source=source, labels_source_dir=empty_dir,
        )
        assert out.labels_valid == 4  # uses labels_source_dir contents, not empty_dir


class TestLabelManagerMissingAndInvalid:
    def test_missing_mask_for_one_patch(
        self, manager, patch_result_and_ids, scene_metadata, tmp_path
    ) -> None:
        result, patch_ids = patch_result_and_ids
        labels_dir = tmp_path / "partial_labels"
        labels_dir.mkdir()
        for patch_id in patch_ids[:3]:
            _write_mask(labels_dir, patch_id, np.full((_SIZE, _SIZE), 1, dtype=np.uint8))
        out = manager.generate(
            result, scene_metadata, tmp_path / "labels_out",
            aoi_id="aoi_1", labels_source_dir=labels_dir,
        )
        assert out.labels_missing == 1
        assert out.labels_rejected == 1
        assert out.labels_valid == 3

    def test_invalid_crs_rejected(
        self, manager, patch_result_and_ids, scene_metadata, tmp_path
    ) -> None:
        result, patch_ids = patch_result_and_ids
        labels_dir = tmp_path / "bad_labels"
        labels_dir.mkdir()
        for i, patch_id in enumerate(patch_ids):
            crs = "EPSG:32644" if i == 0 else _CRS
            _write_mask(labels_dir, patch_id, np.full((_SIZE, _SIZE), 1, dtype=np.uint8), crs=crs)
        out = manager.generate(
            result, scene_metadata, tmp_path / "labels_out",
            aoi_id="aoi_1", labels_source_dir=labels_dir,
        )
        assert out.labels_rejected == 1
        assert out.labels_valid == 3


class TestLabelManagerDuplicates:
    def test_append_run_skips_existing(
        self, manager, patch_result_and_ids, scene_metadata, labels_source_dir, tmp_path
    ) -> None:
        result, _ = patch_result_and_ids
        out_dir = tmp_path / "labels_out"
        manager.generate(
            result, scene_metadata, out_dir, aoi_id="aoi_1",
            labels_source_dir=labels_source_dir,
        )
        out2 = manager.generate(
            result, scene_metadata, out_dir, aoi_id="aoi_1",
            labels_source_dir=labels_source_dir, append_to_manifest=True,
        )
        assert out2.labels_duplicate == 4
        assert out2.manifest.entry_count == 4

    def test_no_append_does_not_detect_cross_run_duplicates(
        self, manager, patch_result_and_ids, scene_metadata, labels_source_dir, tmp_path
    ) -> None:
        result, _ = patch_result_and_ids
        out_dir = tmp_path / "labels_out"
        manager.generate(
            result, scene_metadata, out_dir, aoi_id="aoi_1",
            labels_source_dir=labels_source_dir,
        )
        out2 = manager.generate(
            result, scene_metadata, out_dir, aoi_id="aoi_1",
            labels_source_dir=labels_source_dir, append_to_manifest=False,
        )
        assert out2.labels_duplicate == 0


class TestLabelManagerValidation:
    def test_empty_patch_manifest_raises(self, manager, scene_metadata, tmp_path) -> None:
        empty_manifest = PatchManifest(entries=(), csv_path=None, json_path=None)
        empty_result = PatchDatasetResult(
            scene_id="empty_scene", output_dir=tmp_path, scene_patches_dir=tmp_path,
            manifest=empty_manifest, patches_generated=0, patches_skipped=0,
            total_windows=0, patch_size=8, stride=8, operations_log=(),
        )
        with pytest.raises(InvalidValueError, match="at least one patch entry"):
            manager.generate(
                empty_result, scene_metadata, tmp_path / "out", aoi_id="aoi_1",
                labels_source_dir=tmp_path,
            )

    def test_no_source_provided_raises(
        self, manager, patch_result_and_ids, scene_metadata, tmp_path
    ) -> None:
        result, _ = patch_result_and_ids
        with pytest.raises(InvalidValueError, match="label_source"):
            manager.generate(result, scene_metadata, tmp_path / "out", aoi_id="aoi_1")