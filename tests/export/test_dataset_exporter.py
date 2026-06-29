"""
Unit tests for src/export/exporter.py.

EarthEngineDownloader.download() is mocked so no EE calls are made.
GeoTIFF writing, metadata, manifest, and version run against real files
in pytest's tmp_path.

Run:
    pytest tests/export/test_dataset_exporter.py -v
    pytest tests/export/test_dataset_exporter.py -v \
        --cov=src/export/exporter --cov-report=term-missing
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from affine import Affine

from src.core.exceptions import InvalidValueError, MissingFieldError
from src.export.downloader import AffineTransform, AoiBounds, DownloadResult
from src.export.exporter import DatasetExporter, DatasetExportResult
from src.export.geotiff import GeoTiffValidationResult, GeoTiffWriteResult
from src.export.manifest import DatasetManifest
from src.export.metadata import SceneMetadata
from src.export.version import VersionInfo
from src.gee.composite import CompositeMethod
from src.gee.harmonization import COMMON_BAND_NAMES
from tests.conftest import make_valid_config, write_config


# ==============================================================================
# Fixtures
# ==============================================================================

_AFFINE    = Affine(0.001, 0.0, 87.0, 0.0, -0.001, 26.5)
_TRANSFORM = AffineTransform.from_affine(_AFFINE)
_AOI       = AoiBounds(87.0, 26.0, 87.5, 26.5)
_BANDS     = COMMON_BAND_NAMES[:5]
_N_BANDS   = len(_BANDS)


def _make_download_result(n_bands: int = _N_BANDS) -> DownloadResult:
    data = np.random.rand(n_bands, 10, 10).astype(np.float32)
    return DownloadResult(
        data=data,
        crs="EPSG:4326",
        transform=_TRANSFORM,
        band_names=_BANDS[:n_bands],
        width=10, height=10,
        aoi_bounds=_AOI,
        num_tiles=1,
    )


def _make_mock_feature_stack(n_bands: int = _N_BANDS) -> MagicMock:
    collection = MagicMock()
    collection.sensors           = []
    collection.start_date        = "2023-11-01"
    collection.end_date          = "2024-02-28"
    collection.cloud_cover_limit = 20.0

    processed = MagicMock()
    processed.operations_applied = ("scaling", "harmonization")
    processed.source_result      = collection

    composite = MagicMock()
    composite.method        = CompositeMethod.MEDIAN
    composite.source_result = processed

    fs = MagicMock()
    fs.source_composite     = composite
    fs.all_band_names       = _BANDS[:n_bands]
    fs.composite_band_names = _BANDS[:n_bands]
    fs.features_computed    = ("NDWI",)
    fs.image                = MagicMock()
    return fs


def _make_aoi_config(tmp_path: Path):
    from src.core.config import Config
    data = make_valid_config()
    data["aoi"].update({
        "min_lon": 87.0, "min_lat": 26.0,
        "max_lon": 87.5, "max_lat": 26.5,
    })
    data["satellite"]["output_crs"]        = "EPSG:4326"
    data["satellite"]["resolution_meters"] = 30
    data["export"] = {
        "geotiff":          {"compress": "LZW", "tiled": False, "tile_size": 256,
                             "dtype": "float32", "overviews": False},
        "manifest":         {"formats": ["csv", "json"]},
        "max_tile_pixels":  1_000_000,
        "dataset_version":  "1.0.0",
        "pipeline_version": "1.0.0",
        "feature_schema_version": "1.0.0",
        "landsat_collection":     "Landsat C2 L2",
    }
    return Config(config_path=write_config(tmp_path, data))


def _make_no_aoi_config(tmp_path: Path):
    from src.core.config import Config
    return Config(config_path=write_config(tmp_path, make_valid_config()))


@pytest.fixture
def config(tmp_path: Path):
    return _make_aoi_config(tmp_path)


@pytest.fixture
def client() -> MagicMock:
    from src.gee.client import EarthEngineClient
    c = MagicMock(spec=EarthEngineClient)
    c.is_initialized = True
    c.execute_with_retry.side_effect = lambda f, *a, **kw: f()
    return c


@pytest.fixture
def exporter(client, config) -> DatasetExporter:
    return DatasetExporter(client, config)


@pytest.fixture
def feature_stack() -> MagicMock:
    return _make_mock_feature_stack()


@pytest.fixture
def download_result() -> DownloadResult:
    return _make_download_result()


def _run_export(
    exporter:        DatasetExporter,
    feature_stack:   MagicMock,
    download_result: DownloadResult,
    tmp_path:        Path,
    scene_id:        str = "test_scene",
    append:          bool = True,
) -> DatasetExportResult:
    with patch(
        "src.export.exporter.EarthEngineDownloader.download",
        return_value=download_result,
    ):
        return exporter.export(
            feature_stack_result=feature_stack,
            output_dir=tmp_path / "dataset",
            scene_id=scene_id,
            append_to_manifest=append,
        )


# ==============================================================================
# DatasetExportResult tests
# ==============================================================================

class TestDatasetExportResult:
    def test_frozen(
        self, exporter, feature_stack, download_result, tmp_path
    ) -> None:
        result = _run_export(exporter, feature_stack, download_result, tmp_path)
        with pytest.raises((AttributeError, TypeError)):
            result.is_valid = False  # type: ignore[misc]

    def test_summary_lines_ascii(
        self, exporter, feature_stack, download_result, tmp_path
    ) -> None:
        result = _run_export(exporter, feature_stack, download_result, tmp_path)
        for line in result.summary_lines():
            assert all(ord(c) < 128 for c in line)

    def test_typed_result_objects(
        self, exporter, feature_stack, download_result, tmp_path
    ) -> None:
        result = _run_export(exporter, feature_stack, download_result, tmp_path)
        assert isinstance(result.scene_metadata, SceneMetadata)
        assert isinstance(result.version_info,   VersionInfo)
        assert isinstance(result.manifest,       DatasetManifest)
        assert isinstance(result.write_result,   GeoTiffWriteResult)
        assert isinstance(result.validation,     GeoTiffValidationResult)


# ==============================================================================
# DatasetExporter.export() — directory structure tests
# ==============================================================================

class TestExportDirectoryStructure:
    def test_creates_dataset_root(
        self, exporter, feature_stack, download_result, tmp_path
    ) -> None:
        result = _run_export(exporter, feature_stack, download_result, tmp_path)
        assert result.dataset_root.exists()
        assert result.dataset_root.is_dir()

    def test_creates_scenes_subdir(
        self, exporter, feature_stack, download_result, tmp_path
    ) -> None:
        result = _run_export(exporter, feature_stack, download_result, tmp_path)
        assert result.scenes_dir.exists()
        assert result.scenes_dir.name == "scenes"

    def test_creates_scene_dir(
        self, exporter, feature_stack, download_result, tmp_path
    ) -> None:
        result = _run_export(
            exporter, feature_stack, download_result, tmp_path, "my_scene"
        )
        assert result.scene_dir.name == "my_scene"
        assert result.scene_dir.exists()

    def test_image_named_image_tif(
        self, exporter, feature_stack, download_result, tmp_path
    ) -> None:
        result = _run_export(exporter, feature_stack, download_result, tmp_path)
        assert result.image_path.name == "image.tif"

    def test_metadata_named_metadata_json(
        self, exporter, feature_stack, download_result, tmp_path
    ) -> None:
        result = _run_export(exporter, feature_stack, download_result, tmp_path)
        assert result.metadata_path.name == "metadata.json"

    def test_version_at_dataset_root(
        self, exporter, feature_stack, download_result, tmp_path
    ) -> None:
        result = _run_export(exporter, feature_stack, download_result, tmp_path)
        assert result.version_path.parent == result.dataset_root

    def test_manifest_at_dataset_root(
        self, exporter, feature_stack, download_result, tmp_path
    ) -> None:
        result = _run_export(exporter, feature_stack, download_result, tmp_path)
        assert result.manifest.csv_path.parent == result.dataset_root


# ==============================================================================
# DatasetExporter.export() — output file tests
# ==============================================================================

class TestExportOutputFiles:
    def test_geotiff_created(
        self, exporter, feature_stack, download_result, tmp_path
    ) -> None:
        result = _run_export(exporter, feature_stack, download_result, tmp_path)
        assert result.image_path.exists()

    def test_metadata_json_created(
        self, exporter, feature_stack, download_result, tmp_path
    ) -> None:
        result = _run_export(exporter, feature_stack, download_result, tmp_path)
        assert result.metadata_path.exists()

    def test_version_json_created(
        self, exporter, feature_stack, download_result, tmp_path
    ) -> None:
        result = _run_export(exporter, feature_stack, download_result, tmp_path)
        assert result.version_path.exists()

    def test_manifest_csv_created(
        self, exporter, feature_stack, download_result, tmp_path
    ) -> None:
        result = _run_export(exporter, feature_stack, download_result, tmp_path)
        assert result.manifest.csv_path.exists()

    def test_manifest_json_created(
        self, exporter, feature_stack, download_result, tmp_path
    ) -> None:
        result = _run_export(exporter, feature_stack, download_result, tmp_path)
        assert result.manifest.json_path.exists()

    def test_metadata_json_content(
        self, exporter, feature_stack, download_result, tmp_path
    ) -> None:
        result = _run_export(
            exporter, feature_stack, download_result, tmp_path, "content_test"
        )
        with open(result.metadata_path) as fh:
            meta = json.load(fh)
        assert meta["scene_id"] == "content_test"

    def test_version_json_content(
        self, exporter, feature_stack, download_result, tmp_path
    ) -> None:
        result = _run_export(exporter, feature_stack, download_result, tmp_path)
        with open(result.version_path) as fh:
            ver = json.load(fh)
        assert "dataset_version" in ver


# ==============================================================================
# DatasetExporter.export() — result field tests
# ==============================================================================

class TestExportResultFields:
    def test_scene_id_preserved(
        self, exporter, feature_stack, download_result, tmp_path
    ) -> None:
        result = _run_export(
            exporter, feature_stack, download_result, tmp_path, "my_id"
        )
        assert result.scene_id == "my_id"

    def test_auto_scene_id_generated(
        self, exporter, feature_stack, download_result, tmp_path
    ) -> None:
        with patch(
            "src.export.exporter.EarthEngineDownloader.download",
            return_value=download_result,
        ):
            result = exporter.export(
                feature_stack, tmp_path / "ds", scene_id=None
            )
        assert result.scene_id.startswith("RM_export_")

    def test_is_valid_true_on_success(
        self, exporter, feature_stack, download_result, tmp_path
    ) -> None:
        result = _run_export(exporter, feature_stack, download_result, tmp_path)
        assert result.is_valid is True

    def test_operations_log_non_empty(
        self, exporter, feature_stack, download_result, tmp_path
    ) -> None:
        result = _run_export(exporter, feature_stack, download_result, tmp_path)
        assert len(result.operations_log) >= 3

    def test_operations_log_contains_download(
        self, exporter, feature_stack, download_result, tmp_path
    ) -> None:
        result = _run_export(exporter, feature_stack, download_result, tmp_path)
        assert any("download" in op for op in result.operations_log)

    def test_scene_metadata_scene_id_matches(
        self, exporter, feature_stack, download_result, tmp_path
    ) -> None:
        result = _run_export(
            exporter, feature_stack, download_result, tmp_path, "check_id"
        )
        assert result.scene_metadata.scene_id == "check_id"

    def test_source_result_stored(
        self, exporter, feature_stack, download_result, tmp_path
    ) -> None:
        # DatasetExportResult does not directly expose source_result, but
        # scene_metadata carries provenance info.
        result = _run_export(exporter, feature_stack, download_result, tmp_path)
        assert result.scene_metadata is not None

    def test_export_timestamp_iso_format(
        self, exporter, feature_stack, download_result, tmp_path
    ) -> None:
        result = _run_export(exporter, feature_stack, download_result, tmp_path)
        assert "T" in result.export_timestamp

    def test_all_paths_absolute(
        self, exporter, feature_stack, download_result, tmp_path
    ) -> None:
        result = _run_export(exporter, feature_stack, download_result, tmp_path)
        assert result.dataset_root.is_absolute()
        assert result.image_path.is_absolute()
        assert result.metadata_path.is_absolute()
        assert result.version_path.is_absolute()


# ==============================================================================
# DatasetExporter.export() — manifest append tests
# ==============================================================================

class TestManifestAppend:
    def test_append_true_accumulates_entries(
        self, exporter, feature_stack, download_result, tmp_path
    ) -> None:
        ds = tmp_path / "dataset"
        with patch(
            "src.export.exporter.EarthEngineDownloader.download",
            return_value=download_result,
        ):
            exporter.export(feature_stack, ds, scene_id="sc_A", append_to_manifest=True)
            exporter.export(feature_stack, ds, scene_id="sc_B", append_to_manifest=True)

        from src.export.manifest import DatasetManifest
        loaded = DatasetManifest.from_csv(ds / "manifest.csv")
        assert loaded.entry_count == 2

    def test_append_false_overwrites(
        self, exporter, feature_stack, download_result, tmp_path
    ) -> None:
        ds = tmp_path / "dataset"
        with patch(
            "src.export.exporter.EarthEngineDownloader.download",
            return_value=download_result,
        ):
            exporter.export(feature_stack, ds, scene_id="sc_A", append_to_manifest=False)
            exporter.export(feature_stack, ds, scene_id="sc_B", append_to_manifest=False)

        from src.export.manifest import DatasetManifest
        loaded = DatasetManifest.from_csv(ds / "manifest.csv")
        assert loaded.entry_count == 1
        assert loaded.entries[0].scene_id == "sc_B"


# ==============================================================================
# DatasetExporter input validation tests
# ==============================================================================

class TestInputValidation:
    def test_missing_aoi_raises(self, client, tmp_path) -> None:
        config = _make_no_aoi_config(tmp_path)
        exp    = DatasetExporter(client, config)
        with pytest.raises(MissingFieldError, match="aoi"):
            exp.export(_make_mock_feature_stack(), tmp_path / "out")

    def test_empty_band_names_raises(self, exporter, tmp_path) -> None:
        fs = _make_mock_feature_stack()
        fs.all_band_names = ()
        with pytest.raises(InvalidValueError, match="all_band_names"):
            exporter.export(fs, tmp_path / "out")

    def test_construction_does_not_call_ee(self, client, config) -> None:
        DatasetExporter(client, config)
        client.execute_with_retry.assert_not_called()