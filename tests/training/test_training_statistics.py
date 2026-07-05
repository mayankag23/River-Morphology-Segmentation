"""
Tests for src/training/statistics.py

Run:
    pytest tests/training/test_training_statistics.py -v \
        --cov=src/training/statistics --cov-report=term-missing
"""

from __future__ import annotations

import numpy as np
import pytest

from src.training.statistics import DatasetStatisticsAccumulator


class TestDatasetStatisticsAccumulator:
    def test_invalid_num_bands_raises(self) -> None:
        with pytest.raises(Exception):
            DatasetStatisticsAccumulator(num_bands=0)

    def test_finalize_before_update_raises(self) -> None:
        acc = DatasetStatisticsAccumulator(num_bands=3)
        with pytest.raises(RuntimeError, match="before any data"):
            acc.finalize()

    def test_single_sample_mean(self) -> None:
        acc = DatasetStatisticsAccumulator(num_bands=2)
        image = np.array([[[1.0, 2.0], [3.0, 4.0]],
                          [[2.0, 4.0], [6.0, 8.0]]], dtype=np.float64)
        acc.update(image)
        stats = acc.finalize(num_samples=1)
        expected_mean_b0 = (1.0 + 2.0 + 3.0 + 4.0) / 4
        assert abs(stats.mean[0] - expected_mean_b0) < 1e-6

    def test_known_mean_and_std(self) -> None:
        """A constant image has mean == value and std == 1.0 (replaced)."""
        acc = DatasetStatisticsAccumulator(num_bands=1)
        for _ in range(10):
            acc.update(np.ones((1, 4, 4), dtype=np.float64) * 3.0)
        stats = acc.finalize(num_samples=10)
        assert abs(stats.mean[0] - 3.0) < 1e-6
        # All pixels are identical -> variance = 0 -> std replaced with 1.0.
        assert stats.std[0] == pytest.approx(1.0)

    def test_std_positive_for_varying_data(self) -> None:
        acc = DatasetStatisticsAccumulator(num_bands=2)
        np.random.seed(0)
        for _ in range(20):
            acc.update(np.random.rand(2, 8, 8).astype(np.float64))
        stats = acc.finalize(num_samples=20)
        for s in stats.std:
            assert s > 0.0

    def test_band_names_preserved(self) -> None:
        acc = DatasetStatisticsAccumulator(num_bands=3)
        acc.update(np.ones((3, 4, 4)))
        stats = acc.finalize(band_names=("Blue", "Green", "Red"), num_samples=1)
        assert stats.band_names == ("Blue", "Green", "Red")

    def test_source_is_computed(self) -> None:
        acc = DatasetStatisticsAccumulator(num_bands=1)
        acc.update(np.ones((1, 4, 4)))
        stats = acc.finalize(num_samples=1)
        assert stats.source == "computed"

    def test_min_max_populated(self) -> None:
        acc = DatasetStatisticsAccumulator(num_bands=1)
        img = np.array([[[0.0, 1.0], [0.5, 0.25]]])
        acc.update(img)
        stats = acc.finalize(num_samples=1)
        assert abs(stats.min_values[0] - 0.0) < 1e-6
        assert abs(stats.max_values[0] - 1.0) < 1e-6

    def test_reset_clears_accumulation(self) -> None:
        acc = DatasetStatisticsAccumulator(num_bands=1)
        acc.update(np.ones((1, 4, 4)) * 5.0)
        acc.reset()
        with pytest.raises(RuntimeError):
            acc.finalize()

    def test_nan_pixels_ignored(self) -> None:
        """NaN pixels must not contaminate the mean/std computation."""
        acc = DatasetStatisticsAccumulator(num_bands=1)
        img = np.array([[[1.0, np.nan], [1.0, 1.0]]])
        acc.update(img)
        stats = acc.finalize(num_samples=1)
        assert abs(stats.mean[0] - 1.0) < 1e-6

    def test_samples_seen_property(self) -> None:
        acc = DatasetStatisticsAccumulator(num_bands=2)
        assert acc.samples_seen == 0
        acc.update(np.ones((2, 4, 4)))
        assert acc.samples_seen > 0

    def test_multi_sample_mean_matches_numpy(self) -> None:
        """Welford mean must match numpy reference over 50 samples."""
        np.random.seed(42)
        data = [np.random.rand(3, 8, 8).astype(np.float64) for _ in range(50)]
        acc = DatasetStatisticsAccumulator(num_bands=3)
        for d in data:
            acc.update(d)
        stats = acc.finalize(num_samples=50)
        all_data = np.concatenate([d.reshape(3, -1) for d in data], axis=1)
        np_means = all_data.mean(axis=1)
        for i in range(3):
            assert abs(stats.mean[i] - np_means[i]) < 1e-4
