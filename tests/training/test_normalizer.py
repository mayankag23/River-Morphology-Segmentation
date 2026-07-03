"""
Unit tests for src/training/normalizer.py.

Run:
    pytest tests/training/test_normalizer.py -v \
        --cov=src/training/normalizer --cov-report=term-missing
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.core.exceptions import InvalidValueError
from src.training.normalizer import DatasetNormalizer, NormalizationStats, NormalizationStrategy
from tests.conftest import make_valid_config, write_config


def _config(tmp_path: Path, strategy: str = "per_band_mean_std"):
    from src.core.config import Config
    data = make_valid_config()
    data["training"] = {
        "normalization": {
            "strategy": strategy,
            "percentile_min": 2,
            "percentile_max": 98,
        }
    }
    data["labels"] = {"nodata_value": 255}
    return Config(config_path=write_config(tmp_path, data))


def _mock_entries(tmp_path: Path, n: int = 3) -> list:
    """Write n real GeoTIFF patches and return mock entries pointing to them."""
    import rasterio
    from affine import Affine
    from rasterio.crs import CRS
    entries = []
    for i in range(n):
        path = tmp_path / f"patch_{i}.tif"
        data = np.random.rand(4, 8, 8).astype(np.float32) + i
        profile = {
            "driver": "GTiff", "dtype": "float32", "width": 8, "height": 8,
            "count": 4, "crs": CRS.from_string("EPSG:4326"),
            "transform": Affine(0.001, 0, 87, 0, -0.001, 26.5),
        }
        with rasterio.open(path, "w", **profile) as ds:
            ds.write(data)
        e = MagicMock()
        e.patch_path = str(path)
        entries.append(e)
    return entries


class TestNormalizationStats:
    def test_frozen(self) -> None:
        stats = NormalizationStats(
            strategy="per_band_mean_std", num_bands=4,
            band_names=("B", "G", "R", "N"), mean=(0.0,) * 4, std=(1.0,) * 4,
            percentile_min=2, percentile_max=98,
        )
        with pytest.raises((AttributeError, TypeError)):
            stats.num_bands = 5  # type: ignore[misc]

    def test_normalize_shape_preserved(self) -> None:
        stats = NormalizationStats(
            strategy="per_band_mean_std", num_bands=4,
            band_names=("B", "G", "R", "N"),
            mean=(0.1, 0.2, 0.3, 0.4), std=(0.1, 0.1, 0.1, 0.1),
            percentile_min=2, percentile_max=98,
        )
        data   = np.random.rand(4, 8, 8).astype(np.float32)
        result = stats.normalize(data)
        assert result.shape == (4, 8, 8)
        assert result.dtype == np.float32

    def test_normalize_changes_values(self) -> None:
        stats = NormalizationStats(
            strategy="per_band_mean_std", num_bands=1,
            band_names=("B",), mean=(0.5,), std=(0.2,),
            percentile_min=2, percentile_max=98,
        )
        data   = np.full((1, 4, 4), 0.5, dtype=np.float32)
        result = stats.normalize(data)
        np.testing.assert_allclose(result, 0.0, atol=1e-5)

    def test_denormalize_roundtrip(self) -> None:
        stats = NormalizationStats(
            strategy="per_band_mean_std", num_bands=2,
            band_names=("B", "G"), mean=(0.3, 0.4), std=(0.1, 0.2),
            percentile_min=2, percentile_max=98,
        )
        data   = np.random.rand(2, 4, 4).astype(np.float32)
        normed = stats.normalize(data)
        back   = stats.denormalize(normed)
        np.testing.assert_allclose(back, data, atol=1e-5)

    def test_normalize_wrong_band_count_raises(self) -> None:
        stats = NormalizationStats(
            strategy="per_band_mean_std", num_bands=4,
            band_names=("B", "G", "R", "N"), mean=(0.0,) * 4, std=(1.0,) * 4,
            percentile_min=2, percentile_max=98,
        )
        data = np.zeros((3, 8, 8), dtype=np.float32)
        with pytest.raises(InvalidValueError, match="num_bands"):
            stats.normalize(data)

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        stats = NormalizationStats(
            strategy="per_band_mean_std", num_bands=4,
            band_names=("B", "G", "R", "N"),
            mean=(0.1, 0.2, 0.3, 0.4), std=(0.5, 0.6, 0.7, 0.8),
            percentile_min=2, percentile_max=98,
        )
        path   = stats.save(tmp_path / "norm.json")
        loaded = NormalizationStats.load(path)
        assert loaded.mean       == stats.mean
        assert loaded.std        == stats.std
        assert loaded.band_names == stats.band_names

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            NormalizationStats.load(tmp_path / "nonexistent.json")

    def test_none_strategy_normalize_identity(self) -> None:
        stats = NormalizationStats(
            strategy="none", num_bands=3,
            band_names=("A", "B", "C"), mean=(0.0,) * 3, std=(1.0,) * 3,
            percentile_min=2, percentile_max=98,
        )
        data   = np.random.rand(3, 4, 4).astype(np.float32)
        result = stats.normalize(data)
        np.testing.assert_array_equal(result, data)


class TestNormalizationStrategy:
    def test_from_string_valid(self) -> None:
        assert NormalizationStrategy.from_string("per_band_mean_std") == NormalizationStrategy.PER_BAND_MEAN_STD
        assert NormalizationStrategy.from_string("min_max") == NormalizationStrategy.MIN_MAX
        assert NormalizationStrategy.from_string("NONE") == NormalizationStrategy.NONE

    def test_from_string_invalid_raises(self) -> None:
        with pytest.raises(InvalidValueError, match="strategy"):
            NormalizationStrategy.from_string("z_score")


class TestDatasetNormalizer:
    def test_compute_mean_std(self, tmp_path: Path) -> None:
        cfg     = _config(tmp_path, strategy="per_band_mean_std")
        norm    = DatasetNormalizer(cfg)
        entries = _mock_entries(tmp_path, n=3)
        stats   = norm.compute(entries, ("B", "G", "R", "N"))
        assert stats.num_bands == 4
        assert stats.strategy == "per_band_mean_std"
        assert len(stats.mean) == 4
        assert all(s > 0.0 for s in stats.std)

    def test_compute_min_max(self, tmp_path: Path) -> None:
        cfg  = _config(tmp_path, strategy="min_max")
        norm = DatasetNormalizer(cfg)
        entries = _mock_entries(tmp_path, n=2)
        stats   = norm.compute(entries, ("B", "G", "R", "N"))
        assert stats.strategy == "min_max"
        assert all(s > 0.0 for s in stats.std)

    def test_compute_none_strategy(self, tmp_path: Path) -> None:
        cfg  = _config(tmp_path, strategy="none")
        norm = DatasetNormalizer(cfg)
        stats = norm.compute([], ("B", "G", "R", "N"))
        assert stats.strategy == "none"
        assert all(m == 0.0 for m in stats.mean)
        assert all(s == 1.0 for s in stats.std)

    def test_save_to_dir(self, tmp_path: Path) -> None:
        cfg   = _config(tmp_path, strategy="none")
        norm  = DatasetNormalizer(cfg)
        stats = norm.compute([], ("B", "G", "R", "N"))
        path  = norm.save_to_dir(stats, tmp_path)
        assert path.exists()
        with open(path) as fh:
            data = json.load(fh)
        assert "mean" in data

    def test_compute_handles_missing_file_gracefully(self, tmp_path: Path) -> None:
        cfg     = _config(tmp_path, strategy="per_band_mean_std")
        norm    = DatasetNormalizer(cfg)
        entries = _mock_entries(tmp_path, n=2)
        bad     = MagicMock()
        bad.patch_path = str(tmp_path / "nonexistent.tif")
        stats   = norm.compute(entries + [bad], ("B", "G", "R", "N"))
        # Should complete without raising; bad file is skipped.
        assert stats.num_bands == 4