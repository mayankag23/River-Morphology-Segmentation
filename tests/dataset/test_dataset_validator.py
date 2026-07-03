"""
Unit tests for src/dataset/validator.py.

Run:
    pytest tests/dataset/test_dataset_validator.py -v \
        --cov=src/dataset/validator --cov-report=term-missing
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.dataset.manifest import DatasetSample
from src.dataset.validator import DatasetValidator
from tests.conftest import make_valid_config, write_config


def _sample(
    patch_id: str = "sc_r000_c000",
    crs: str = "EPSG:4326",
    ratio: float = 0.95,
) -> DatasetSample:
    return DatasetSample(
        sample_id=patch_id, patch_id=patch_id, scene_id="scene001",
        patch_path=f"/data/{patch_id}.tif", mask_path=f"/data/{patch_id}_mask.tif",
        crs=crs, width=256, height=256, num_bands=4, row_index=0, col_index=0,
        patch_valid_pixel_ratio=ratio, label_valid_pixel_ratio=ratio,
        num_classes_present=3, acquisition_date="2023-07-15",
        year=2023, month=7, season="monsoon", hydrological_year=2023,
        sensor="L8", river_name="", reach_id="", basin_id="",
        aoi_id="aoi_1", label_version="1.0.0", annotator="x",
        confidence=1.0, confidence_source="automatic",
    )


def _config(tmp_path: Path, min_total: int = 1, min_ratio: float = 0.5):
    from src.core.config import Config
    data = make_valid_config()
    data["dataset"] = {
        "split": {"strategy": "temporal", "train_ratio": 0.7, "val_ratio": 0.15,
                  "test_ratio": 0.15, "random_seed": 42},
        "quality": {"min_valid_pixel_ratio": min_ratio, "min_samples_per_split": 1},
        "output_formats": ["csv"], "dataset_version": "1.0.0",
        "min_total_samples": min_total,
    }
    return Config(config_path=write_config(tmp_path, data))


class TestDatasetValidator:
    def test_valid_samples_pass(self, tmp_path: Path) -> None:
        v = DatasetValidator(_config(tmp_path))
        result = v.validate([_sample("a"), _sample("b"), _sample("c")], check_files=False)
        assert result.is_valid is True
        assert result.valid_samples == 3

    def test_duplicate_sample_ids_detected(self, tmp_path: Path) -> None:
        v = DatasetValidator(_config(tmp_path))
        result = v.validate([_sample("dup"), _sample("dup")], check_files=False)
        assert result.is_valid is False
        assert "dup" in result.duplicate_sample_ids

    def test_inconsistent_crs_detected(self, tmp_path: Path) -> None:
        v = DatasetValidator(_config(tmp_path))
        samples = [_sample("a", crs="EPSG:4326"), _sample("b", crs="EPSG:32644")]
        result = v.validate(samples, check_files=False)
        assert result.crs_is_consistent is False
        assert result.is_valid is False

    def test_below_min_pixel_ratio_detected(self, tmp_path: Path) -> None:
        v = DatasetValidator(_config(tmp_path, min_ratio=0.9))
        result = v.validate([_sample("a", ratio=0.3)], check_files=False)
        assert result.below_min_pixel_ratio_count == 1

    def test_insufficient_samples_detected(self, tmp_path: Path) -> None:
        v = DatasetValidator(_config(tmp_path, min_total=10))
        result = v.validate([_sample("a")], check_files=False)
        assert result.min_total_samples_met is False
        assert result.is_valid is False

    def test_missing_patch_file_detected(self, tmp_path: Path) -> None:
        v = DatasetValidator(_config(tmp_path))
        result = v.validate([_sample("nonexistent")], check_files=True)
        assert "nonexistent" in result.missing_patch_files
        assert result.is_valid is False

    def test_result_is_frozen(self, tmp_path: Path) -> None:
        v = DatasetValidator(_config(tmp_path))
        result = v.validate([_sample("a")], check_files=False)
        with pytest.raises((AttributeError, TypeError)):
            result.is_valid = False  # type: ignore[misc]

    def test_crs_values_found(self, tmp_path: Path) -> None:
        v = DatasetValidator(_config(tmp_path))
        result = v.validate([_sample("a"), _sample("b")], check_files=False)
        assert "EPSG:4326" in result.crs_values_found