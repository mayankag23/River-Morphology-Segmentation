"""
Unit tests for src/dataset/version.py.

Run:
    pytest tests/dataset/test_dataset_version.py -v \
        --cov=src/dataset/version --cov-report=term-missing
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.dataset.version import DatasetVersionInfo, DatasetVersionManager
from tests.conftest import make_valid_config, write_config


def _config(tmp_path: Path, version: str = "2.0.0"):
    from src.core.config import Config
    data = make_valid_config()
    data["dataset"] = {
        "split": {
            "strategy": "temporal", "train_ratio": 0.7, "val_ratio": 0.15,
            "test_ratio": 0.15, "random_seed": 42,
        },
        "quality": {}, "output_formats": ["csv"],
        "dataset_version": version, "min_total_samples": 1,
    }
    return Config(config_path=write_config(tmp_path, data))


@pytest.fixture
def manager(tmp_path: Path) -> DatasetVersionManager:
    return DatasetVersionManager(_config(tmp_path))


@pytest.fixture
def version_info(manager: DatasetVersionManager) -> DatasetVersionInfo:
    return manager.generate(
        total_samples=100, train_samples=70, validation_samples=15,
        test_samples=15, excluded_samples=0, split_strategy="temporal",
        source_scenes=5,
    )


class TestDatasetVersionInfo:
    def test_frozen(self, version_info: DatasetVersionInfo) -> None:
        with pytest.raises((AttributeError, TypeError)):
            version_info.dataset_version = "other"  # type: ignore[misc]

    def test_to_dict_json_serializable(self, version_info: DatasetVersionInfo) -> None:
        raw = json.dumps(version_info.to_dict(), ensure_ascii=True)
        assert len(raw) > 0

    def test_to_dict_ascii_only(self, version_info: DatasetVersionInfo) -> None:
        raw = json.dumps(version_info.to_dict())
        assert all(ord(c) < 128 for c in raw)


class TestDatasetVersionManager:
    def test_reads_version_from_config(self, manager: DatasetVersionManager) -> None:
        info = manager.generate(
            total_samples=10, train_samples=7, validation_samples=2,
            test_samples=1, excluded_samples=0, split_strategy="temporal",
            source_scenes=2,
        )
        assert info.dataset_version == "2.0.0"

    def test_counts_stored_correctly(self, version_info: DatasetVersionInfo) -> None:
        assert version_info.total_samples == 100
        assert version_info.train_samples == 70
        assert version_info.validation_samples == 15
        assert version_info.test_samples == 15
        assert version_info.source_scenes == 5

    def test_split_strategy_stored(self, version_info: DatasetVersionInfo) -> None:
        assert version_info.split_strategy == "temporal"

    def test_ratios_stored(self, version_info: DatasetVersionInfo) -> None:
        assert version_info.train_ratio == pytest.approx(0.70)
        assert version_info.validation_ratio == pytest.approx(0.15)

    def test_created_at_iso_format(self, version_info: DatasetVersionInfo) -> None:
        assert "T" in version_info.assembly_timestamp

    def test_git_commit_on_failure_is_none(
        self, manager: DatasetVersionManager
    ) -> None:
        with patch("subprocess.run", side_effect=OSError("no git")):
            info = manager.generate(
                total_samples=10, train_samples=7, validation_samples=2,
                test_samples=1, excluded_samples=0, split_strategy="random",
                source_scenes=1,
            )
        assert info.git_commit is None

    def test_git_commit_captured_when_available(
        self, manager: DatasetVersionManager
    ) -> None:
        mock_result = MagicMock()
        mock_result.stdout = "abc1234\n"
        with patch("subprocess.run", return_value=mock_result):
            info = manager.generate(
                total_samples=10, train_samples=7, validation_samples=2,
                test_samples=1, excluded_samples=0, split_strategy="temporal",
                source_scenes=1,
            )
        assert info.git_commit == "abc1234"

    def test_save_creates_version_json(
        self, manager: DatasetVersionManager, version_info: DatasetVersionInfo, tmp_path: Path
    ) -> None:
        path = manager.save(version_info, tmp_path)
        assert path.exists()
        assert path.name == "version.json"

    def test_load_roundtrip(
        self, manager: DatasetVersionManager, version_info: DatasetVersionInfo, tmp_path: Path
    ) -> None:
        manager.save(version_info, tmp_path)
        loaded = DatasetVersionManager.load(tmp_path)
        assert loaded.dataset_version        == version_info.dataset_version
        assert loaded.total_samples          == version_info.total_samples
        assert loaded.split_strategy         == version_info.split_strategy

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            DatasetVersionManager.load(tmp_path / "empty_dir")