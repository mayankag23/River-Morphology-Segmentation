"""
Unit tests for src/export/version.py.

Pure Python -- no rasterio, no EE. subprocess is mocked.

Run:
    pytest tests/export/test_dataset_version.py -v
    pytest tests/export/test_dataset_version.py -v \
        --cov=src/export/version --cov-report=term-missing
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.export.version import (
    VERSION_FILE_NAME,
    DatasetVersionManager,
    VersionInfo,
)
from tests.conftest import make_valid_config, write_config


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def config(tmp_path: Path):
    from src.core.config import Config
    data = make_valid_config()
    data["export"] = {
        "geotiff":          {"compress": "LZW", "tiled": True, "tile_size": 256,
                             "dtype": "float32", "overviews": True},
        "manifest":         {"formats": ["csv", "json"]},
        "max_tile_pixels":  1_000_000,
        "dataset_version":  "2.0.0",
        "pipeline_version": "1.5.0",
        "feature_schema_version": "1.1.0",
        "landsat_collection":     "Landsat C2 L2",
    }
    return Config(config_path=write_config(tmp_path, data))


@pytest.fixture
def manager(config) -> DatasetVersionManager:
    return DatasetVersionManager(config)


@pytest.fixture
def version_info(manager: DatasetVersionManager) -> VersionInfo:
    return manager.generate()


# ==============================================================================
# VersionInfo tests
# ==============================================================================

class TestVersionInfo:
    def test_frozen(self, version_info: VersionInfo) -> None:
        with pytest.raises((AttributeError, TypeError)):
            version_info.dataset_version = "other"  # type: ignore[misc]

    def test_to_dict_json_serializable(self, version_info: VersionInfo) -> None:
        raw = json.dumps(version_info.to_dict(), ensure_ascii=True)
        assert len(raw) > 0

    def test_to_dict_ascii_only(self, version_info: VersionInfo) -> None:
        raw = json.dumps(version_info.to_dict())
        assert all(ord(c) < 128 for c in raw)

    def test_git_commit_is_str_or_none(self, version_info: VersionInfo) -> None:
        assert version_info.git_commit is None or isinstance(version_info.git_commit, str)

    def test_config_hash_is_str_or_none(self, version_info: VersionInfo) -> None:
        assert version_info.config_hash is None or isinstance(version_info.config_hash, str)

    def test_config_hash_length(self, version_info: VersionInfo) -> None:
        if version_info.config_hash is not None:
            assert len(version_info.config_hash) == 8


# ==============================================================================
# DatasetVersionManager.generate() tests
# ==============================================================================

class TestDatasetVersionManagerGenerate:
    def test_reads_dataset_version_from_config(
        self, manager: DatasetVersionManager
    ) -> None:
        info = manager.generate()
        assert info.dataset_version == "2.0.0"

    def test_reads_pipeline_version_from_config(
        self, manager: DatasetVersionManager
    ) -> None:
        info = manager.generate()
        assert info.pipeline_version == "1.5.0"

    def test_reads_feature_schema_version(
        self, manager: DatasetVersionManager
    ) -> None:
        info = manager.generate()
        assert info.feature_schema_version == "1.1.0"

    def test_reads_landsat_collection(
        self, manager: DatasetVersionManager
    ) -> None:
        info = manager.generate()
        assert info.landsat_collection == "Landsat C2 L2"

    def test_created_at_iso_format(
        self, manager: DatasetVersionManager
    ) -> None:
        info = manager.generate()
        assert "T" in info.created_at

    def test_uses_defaults_when_no_export_config(self, tmp_path: Path) -> None:
        from src.core.config import Config
        data = make_valid_config()
        cfg  = Config(config_path=write_config(tmp_path, data))
        mgr  = DatasetVersionManager(cfg)
        info = mgr.generate()
        assert info.dataset_version  == "1.0.0"
        assert info.pipeline_version == "1.0.0"

    def test_git_commit_captured_when_available(
        self, manager: DatasetVersionManager
    ) -> None:
        mock_result       = MagicMock()
        mock_result.stdout = "abc1234\n"
        with patch("subprocess.run", return_value=mock_result):
            info = manager.generate()
        assert info.git_commit == "abc1234"

    def test_git_commit_none_on_subprocess_failure(
        self, manager: DatasetVersionManager
    ) -> None:
        with patch("subprocess.run", side_effect=OSError("no git")):
            info = manager.generate()
        assert info.git_commit is None

    def test_config_hash_is_eight_chars(
        self, manager: DatasetVersionManager
    ) -> None:
        info = manager.generate()
        if info.config_hash:
            assert len(info.config_hash) == 8


# ==============================================================================
# DatasetVersionManager.save() / load() tests
# ==============================================================================

class TestDatasetVersionManagerSaveLoad:
    def test_save_creates_version_json(
        self, manager: DatasetVersionManager, tmp_path: Path
    ) -> None:
        info = manager.generate()
        path = manager.save(info, tmp_path)
        assert path.exists()
        assert path.name == VERSION_FILE_NAME

    def test_save_returns_absolute_path(
        self, manager: DatasetVersionManager, tmp_path: Path
    ) -> None:
        info = manager.generate()
        path = manager.save(info, tmp_path)
        assert path.is_absolute()

    def test_save_produces_valid_json(
        self, manager: DatasetVersionManager, tmp_path: Path
    ) -> None:
        info = manager.generate()
        path = manager.save(info, tmp_path)
        with open(path) as fh:
            data = json.load(fh)
        assert "dataset_version" in data

    def test_load_roundtrip(
        self, manager: DatasetVersionManager, tmp_path: Path
    ) -> None:
        info   = manager.generate()
        path   = manager.save(info, tmp_path)
        loaded = DatasetVersionManager.load(tmp_path)
        assert loaded.dataset_version        == info.dataset_version
        assert loaded.pipeline_version       == info.pipeline_version
        assert loaded.feature_schema_version == info.feature_schema_version

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            DatasetVersionManager.load(tmp_path / "empty_dir")

    def test_save_ascii_only(
        self, manager: DatasetVersionManager, tmp_path: Path
    ) -> None:
        info    = manager.generate()
        path    = manager.save(info, tmp_path)
        content = path.read_text(encoding="utf-8")
        assert all(ord(c) < 128 for c in content)