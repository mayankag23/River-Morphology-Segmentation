"""
Unit tests for src/export/metadata.py.

No rasterio, no EE. Uses synthetic DownloadResult and mocked FeatureStackResult.

Run:
    pytest tests/export/test_export_metadata.py -v
    pytest tests/export/test_export_metadata.py -v \
        --cov=src/export/metadata --cov-report=term-missing
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from affine import Affine

from src.export.downloader import AffineTransform, AoiBounds, DownloadResult
from src.export.metadata import METADATA_SCHEMA_VERSION, MetadataWriter, SceneMetadata
from src.gee.composite import CompositeMethod
from src.gee.harmonization import COMMON_BAND_NAMES
from tests.conftest import make_valid_config, write_config


# ==============================================================================
# Fixtures
# ==============================================================================

_AFFINE    = Affine(0.001, 0.0, 87.0, 0.0, -0.001, 26.5)
_TRANSFORM = AffineTransform.from_affine(_AFFINE)
_AOI       = AoiBounds(87.0, 26.0, 87.5, 26.5)


@pytest.fixture
def download_result() -> DownloadResult:
    return DownloadResult(
        data=np.zeros((5, 10, 10), dtype=np.float32),
        crs="EPSG:4326",
        transform=_TRANSFORM,
        band_names=("Blue", "Green", "Red", "NIR", "NDWI"),
        width=10, height=10,
        aoi_bounds=_AOI,
        num_tiles=1,
    )


def _mock_feature_stack() -> MagicMock:
    collection = MagicMock()
    collection.sensors           = []
    collection.start_date        = "2023-11-01"
    collection.end_date          = "2024-02-28"
    collection.cloud_cover_limit = 20.0

    processed = MagicMock()
    processed.operations_applied = ("scaling", "qa_masking", "harmonization")
    processed.source_result      = collection

    composite = MagicMock()
    composite.method        = CompositeMethod.MEDIAN
    composite.source_result = processed

    fs = MagicMock()
    fs.source_composite     = composite
    fs.all_band_names       = COMMON_BAND_NAMES + ("NDWI", "BSI")
    fs.composite_band_names = COMMON_BAND_NAMES
    fs.features_computed    = ("NDWI", "BSI")
    return fs


@pytest.fixture
def feature_stack() -> MagicMock:
    return _mock_feature_stack()


@pytest.fixture
def config(tmp_path: Path):
    from src.core.config import Config
    data = make_valid_config()
    data["aoi"].update({"min_lon": 87.0, "min_lat": 26.0,
                        "max_lon": 87.5, "max_lat": 26.5})
    return Config(config_path=write_config(tmp_path, data))


@pytest.fixture
def writer(config) -> MetadataWriter:
    return MetadataWriter(config)


@pytest.fixture
def metadata(writer, feature_stack, download_result) -> SceneMetadata:
    return writer.generate("test_scene_001", feature_stack, download_result)


# ==============================================================================
# SceneMetadata tests
# ==============================================================================

class TestSceneMetadata:
    def test_frozen(self, metadata: SceneMetadata) -> None:
        with pytest.raises((AttributeError, TypeError)):
            metadata.scene_id = "other"  # type: ignore[misc]

    def test_to_dict_json_serializable(self, metadata: SceneMetadata) -> None:
        raw = json.dumps(metadata.to_dict(), ensure_ascii=True)
        assert len(raw) > 0

    def test_to_dict_ascii_only(self, metadata: SceneMetadata) -> None:
        raw = json.dumps(metadata.to_dict())
        assert all(ord(c) < 128 for c in raw)

    def test_schema_version(self, metadata: SceneMetadata) -> None:
        assert metadata.schema_version == METADATA_SCHEMA_VERSION

    def test_aoi_is_aoi_bounds(self, metadata: SceneMetadata) -> None:
        assert isinstance(metadata.aoi, AoiBounds)
        assert metadata.aoi.min_lon == pytest.approx(87.0)

    def test_transform_is_affine_transform(self, metadata: SceneMetadata) -> None:
        assert isinstance(metadata.transform, AffineTransform)


# ==============================================================================
# MetadataWriter.generate() tests
# ==============================================================================

class TestMetadataWriterGenerate:
    def test_scene_id_set(self, metadata: SceneMetadata) -> None:
        assert metadata.scene_id == "test_scene_001"

    def test_crs_set(self, metadata: SceneMetadata) -> None:
        assert metadata.crs == "EPSG:4326"

    def test_width_height(self, metadata: SceneMetadata) -> None:
        assert metadata.width  == 10
        assert metadata.height == 10

    def test_band_names_contains_blue(self, metadata: SceneMetadata) -> None:
        assert "Blue" in metadata.band_names

    def test_spectral_indices(self, metadata: SceneMetadata) -> None:
        assert "NDWI" in metadata.spectral_indices
        assert "BSI"  in metadata.spectral_indices

    def test_composite_method_median(self, metadata: SceneMetadata) -> None:
        assert metadata.composite_method == "median"

    def test_start_end_dates(self, metadata: SceneMetadata) -> None:
        assert metadata.start_date == "2023-11-01"
        assert metadata.end_date   == "2024-02-28"

    def test_operations_applied(self, metadata: SceneMetadata) -> None:
        assert "scaling"       in metadata.operations_applied
        assert "harmonization" in metadata.operations_applied

    def test_export_timestamp_iso(self, metadata: SceneMetadata) -> None:
        assert "T" in metadata.export_timestamp

    def test_num_tiles(self, metadata: SceneMetadata) -> None:
        assert metadata.num_tiles == 1

    def test_none_chain_uses_defaults(
        self, writer: MetadataWriter, download_result: DownloadResult
    ) -> None:
        fs = MagicMock()
        fs.source_composite     = None
        fs.all_band_names       = ("Blue",)
        fs.composite_band_names = ("Blue",)
        fs.features_computed    = ()
        meta = writer.generate("fallback", fs, download_result)
        assert meta.scene_id == "fallback"
        assert isinstance(meta.sensors, tuple)


# ==============================================================================
# MetadataWriter.save() / load() tests
# ==============================================================================

class TestMetadataWriterSaveLoad:
    def test_save_creates_file(
        self, writer: MetadataWriter, metadata: SceneMetadata, tmp_path: Path
    ) -> None:
        path = writer.save(metadata, tmp_path / "meta.json")
        assert path.exists()

    def test_save_returns_absolute_path(
        self, writer: MetadataWriter, metadata: SceneMetadata, tmp_path: Path
    ) -> None:
        path = writer.save(metadata, tmp_path / "meta.json")
        assert path.is_absolute()

    def test_save_valid_json(
        self, writer: MetadataWriter, metadata: SceneMetadata, tmp_path: Path
    ) -> None:
        path = writer.save(metadata, tmp_path / "meta.json")
        with open(path) as fh:
            loaded = json.load(fh)
        assert loaded["scene_id"] == "test_scene_001"

    def test_load_roundtrip(
        self, writer: MetadataWriter, metadata: SceneMetadata, tmp_path: Path
    ) -> None:
        path   = writer.save(metadata, tmp_path / "meta.json")
        loaded = MetadataWriter.load(path)
        assert loaded.scene_id         == metadata.scene_id
        assert loaded.crs              == metadata.crs
        assert loaded.band_names       == metadata.band_names
        assert loaded.spectral_indices == metadata.spectral_indices
        assert loaded.aoi              == metadata.aoi
        assert loaded.transform        == metadata.transform

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            MetadataWriter.load(tmp_path / "nonexistent.json")

    def test_load_invalid_json_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("not_json{{", encoding="utf-8")
        with pytest.raises(ValueError):
            MetadataWriter.load(bad)