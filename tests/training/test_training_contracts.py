"""
Tests for src/training/contracts.py

Run:
    pytest tests/training/test_training_contracts.py -v \
        --cov=src/training/contracts --cov-report=term-missing
"""

from __future__ import annotations

import numpy as np
import pytest

from src.training.contracts import (
    NormalizationStatistics,
    TransformMetadata,
    TransformPipelineResult,
    TransformSample,
)


class TestNormalizationStatistics:
    def _make(self, n: int = 3, source: str = "computed") -> NormalizationStatistics:
        return NormalizationStatistics(
            band_names=tuple(f"B{i}" for i in range(n)),
            mean=tuple(float(i) * 0.1 for i in range(n)),
            std=tuple(float(i) * 0.05 + 0.01 for i in range(n)),
            num_samples=100,
            source=source,
        )

    def test_frozen(self) -> None:
        stats = self._make()
        with pytest.raises((AttributeError, TypeError)):
            stats.mean = (0.0,)  # type: ignore[misc]

    def test_num_bands(self) -> None:
        assert self._make(5).num_bands == 5

    def test_as_numpy_shapes(self) -> None:
        stats = self._make(4)
        mean, std = stats.as_numpy()
        assert mean.shape == (4,)
        assert std.shape  == (4,)
        assert mean.dtype == np.float32
        assert std.dtype  == np.float32

    def test_source_field(self) -> None:
        assert self._make(source="supplied").source == "supplied"

    def test_min_max_optional(self) -> None:
        stats = self._make()
        assert stats.min_values == ()
        assert stats.max_values == ()


class TestTransformSample:
    def _make(self) -> TransformSample:
        return TransformSample(
            image    = np.zeros((4, 8, 8), dtype=np.float32),
            mask     = np.zeros((8, 8), dtype=np.uint8),
            sample_id = "patch_001",
            split     = "train",
        )

    def test_mutable(self) -> None:
        sample = self._make()
        sample.image = np.ones((4, 8, 8), dtype=np.float32)
        assert sample.image.mean() == 1.0

    def test_default_metadata_fields(self) -> None:
        sample = self._make()
        assert sample.acquisition_date == ""
        assert sample.season          == ""
        assert sample.hydrological_year == 0
        assert sample.sensor          == ""
        assert sample.river_name      == ""
        assert sample.reach_id        == ""
        assert sample.basin_id        == ""
        assert sample.aoi_id          == ""

    def test_metadata_dict_is_dict(self) -> None:
        sample = self._make()
        assert isinstance(sample.metadata, dict)


class TestTransformPipelineResult:
    def _dummy_stats(self) -> NormalizationStatistics:
        return NormalizationStatistics(
            band_names=("B0", "B1"), mean=(0.5, 0.3), std=(0.1, 0.08),
            num_samples=50, source="computed",
        )

    def _make(self, valid: bool = True) -> TransformPipelineResult:
        meta = TransformMetadata(
            pipeline_version="1.0.0", random_seed=42,
            normalization_source="computed",
            normalization_stats=self._dummy_stats(),
            train_augmentations=("normalization", "horizontal_flip"),
            val_augmentations=("normalization",),
            test_augmentations=("normalization",),
            created_at="2024-01-01T00:00:00+00:00",
            config_hash="abcd1234",
        )
        return TransformPipelineResult(
            train_dataset=object(), validation_dataset=object(),
            test_dataset=object(),
            normalization_stats=self._dummy_stats(),
            metadata=meta,
            num_train_samples=80, num_val_samples=10, num_test_samples=10,
            num_bands=2, num_classes=4,
            patch_size=(64, 64), is_valid=valid,
            validation_issues=(), operations_log=("step1", "step2"),
        )

    def test_frozen(self) -> None:
        r = self._make()
        with pytest.raises((AttributeError, TypeError)):
            r.num_bands = 10  # type: ignore[misc]

    def test_summary_lines_contains_valid(self) -> None:
        lines = self._make(valid=True).summary_lines()
        assert any("OK" in l for l in lines)

    def test_summary_lines_contains_fail(self) -> None:
        lines = self._make(valid=False).summary_lines()
        assert any("FAIL" in l for l in lines)
