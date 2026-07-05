"""
Tests for src/training/pipeline.py

These tests use a stub TorchDatasetResult (no real GeoTIFF or torch DataLoader
required), following the project's policy of mocking external dependencies.

Run:
    pytest tests/training/test_training_pipeline.py -v \
        --cov=src/training/pipeline --cov-report=term-missing
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.training.contracts import NormalizationStatistics, TransformPipelineResult
from src.training.pipeline import AugmentedDataset, TransformPipeline, _to_numpy_float32


# ==============================================================================
# Stubs
# ==============================================================================

class _FakeDataset:
    """Minimal Dataset stub: returns (image, mask) numpy arrays."""

    def __init__(self, n: int, c: int = 4, h: int = 8, w: int = 8) -> None:
        self._n = n
        self._c = c
        self._h = h
        self._w = w

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, index: int):
        image = np.ones((self._c, self._h, self._w), dtype=np.float32) * 0.5
        mask  = np.zeros((self._h, self._w), dtype=np.uint8)
        meta  = {
            "sample_id":         f"p_{index:04d}",
            "acquisition_date":  "2023-07-15",
            "season":            "monsoon",
            "hydrological_year": 2023,
            "sensor":            "L8",
            "river_name":        "Kosi",
            "reach_id":          "R1",
            "basin_id":          "B1",
            "aoi_id":            "A1",
        }
        return image, mask, meta


class _FakeTorchDatasetResult:
    def __init__(self, n_train: int = 20, n_val: int = 5, n_test: int = 5,
                 c: int = 4) -> None:
        self.train_dataset      = _FakeDataset(n_train, c=c)
        self.validation_dataset  = _FakeDataset(n_val, c=c)
        self.test_dataset        = _FakeDataset(n_test, c=c)


class _MinimalConfig:
    class labels:
        class classes:
            # Four classes: background, water, sand, vegetation
            pass

    class training:
        random_seed      = 42
        nodata_value     = 255
        pipeline_version = "1.0.0"
        validate_metadata = False

        class normalization:
            source = "supplied"
            mean   = [0.5, 0.5, 0.5, 0.5]
            std    = [0.1, 0.1, 0.1, 0.1]
            band_names = ["B0", "B1", "B2", "B3"]

        class augmentation:
            class horizontal_flip:
                enabled = False
            class vertical_flip:
                enabled = False
            class rotate_90:
                enabled = False
            class brightness:
                enabled = False
            class contrast:
                enabled = False
            class gaussian_noise:
                enabled = False
            class random_crop:
                enabled = False
            class random_scale:
                enabled = False


# Patch ClassSchema.from_config to return a stub.
class _StubClassDef:
    def __init__(self, cid: int, name: str) -> None:
        self.class_id = cid
        self.name = name


class _StubSchema:
    def __init__(self) -> None:
        self.classes = [
            _StubClassDef(0, "background"),
            _StubClassDef(1, "water"),
            _StubClassDef(2, "sand"),
            _StubClassDef(3, "vegetation"),
        ]
        self.num_classes = 4

    def from_config(self, _config):
        return self

    @property
    def class_ids(self):
        return [d.class_id for d in self.classes]


# ==============================================================================
# AugmentedDataset tests
# ==============================================================================

class TestAugmentedDataset:
    def _make(self, n: int = 10) -> AugmentedDataset:
        from src.training.registry import TransformRegistry
        from src.training.transform import IdentityTransform, ComposedTransform
        pipeline = ComposedTransform([IdentityTransform()])
        return AugmentedDataset(_FakeDataset(n), pipeline, "train")

    def test_len(self) -> None:
        ds = self._make(7)
        assert len(ds) == 7

    def test_getitem_returns_two_elements(self) -> None:
        ds = self._make()
        item = ds[0]
        assert len(item) == 2

    def test_split_property(self) -> None:
        ds = self._make()
        assert ds.split == "train"

    def test_base_dataset_property(self) -> None:
        base = _FakeDataset(5)
        from src.training.transform import ComposedTransform, IdentityTransform
        ds = AugmentedDataset(base, ComposedTransform([IdentityTransform()]), "validation")
        assert ds.base_dataset is base


# ==============================================================================
# TransformPipeline tests (using supplied statistics to avoid I/O)
# ==============================================================================

class TestTransformPipeline:
    def _make_pipeline(self) -> TransformPipeline:
        import unittest.mock as mock
        # Patch ClassSchema.from_config to avoid Config loading.
        stub_schema = _StubSchema()
        with mock.patch(
            "src.training.pipeline.TransformPipeline.__init__",
            lambda self, config: _pipeline_init_stub(self, config, stub_schema),
        ):
            pipe = TransformPipeline.__new__(TransformPipeline)
            _pipeline_init_stub(pipe, _MinimalConfig, stub_schema)
        return pipe

    def _build_result(self) -> TransformPipelineResult:
        pipe = self._make_pipeline()
        tdr  = _FakeTorchDatasetResult(n_train=10, n_val=3, n_test=3, c=4)
        # Provide external stats to skip heavy computation.
        external = NormalizationStatistics(
            band_names=("B0", "B1", "B2", "B3"),
            mean=(0.5, 0.5, 0.5, 0.5),
            std=(0.1, 0.1, 0.1, 0.1),
            num_samples=10, source="supplied",
        )
        return pipe.build(tdr, external_stats=external)

    def test_result_is_transform_pipeline_result(self) -> None:
        result = self._build_result()
        assert isinstance(result, TransformPipelineResult)

    def test_result_is_frozen(self) -> None:
        result = self._build_result()
        with pytest.raises((AttributeError, TypeError)):
            result.num_bands = 99  # type: ignore[misc]

    def test_sample_counts(self) -> None:
        result = self._build_result()
        assert result.num_train_samples == 10
        assert result.num_val_samples   == 3
        assert result.num_test_samples  == 3

    def test_num_bands(self) -> None:
        result = self._build_result()
        assert result.num_bands == 4

    def test_num_classes(self) -> None:
        result = self._build_result()
        assert result.num_classes == 4

    def test_normalization_source(self) -> None:
        result = self._build_result()
        assert result.normalization_stats.source == "supplied"

    def test_metadata_seed(self) -> None:
        result = self._build_result()
        assert result.metadata.random_seed == 42

    def test_summary_lines_non_empty(self) -> None:
        result = self._build_result()
        lines = result.summary_lines()
        assert len(lines) > 0
        assert all(isinstance(l, str) for l in lines)

    def test_operations_log_non_empty(self) -> None:
        result = self._build_result()
        assert len(result.operations_log) > 0

    def test_train_augmentations_present_in_metadata(self) -> None:
        result = self._build_result()
        assert isinstance(result.metadata.train_augmentations, tuple)


# ==============================================================================
# Numpy conversion helpers
# ==============================================================================

class TestNumpyConversionHelpers:
    def test_to_numpy_from_numpy(self) -> None:
        arr = np.ones((3, 4, 4), dtype=np.float64)
        out = _to_numpy_float32(arr)
        assert out.dtype == np.float32

    def test_to_numpy_from_list(self) -> None:
        lst = [[[1.0, 2.0], [3.0, 4.0]]]   # 1x2x2
        out = _to_numpy_float32(lst)
        assert out.dtype == np.float32
        assert out.shape == (1, 2, 2)


# ==============================================================================
# Private stub for __init__
# ==============================================================================

import logging

def _pipeline_init_stub(
    self: Any, config: Any, stub_schema: Any
) -> None:
    """Lightweight replacement for TransformPipeline.__init__ in tests."""
    self._config = config
    self._logger = logging.getLogger("test_pipeline")
    self._seed   = 42
    self._norm_source       = "supplied"
    self._class_schema      = stub_schema
    self._valid_class_ids   = {0, 1, 2, 3}
    self._nodata_value      = 255
    self._num_classes       = 4
    self._pipeline_version  = "1.0.0"
    from src.training.validator import TransformValidator
    self._validator = TransformValidator(
        valid_class_ids={0, 1, 2, 3},
        check_metadata=False,
        nodata_class_id=255,
    )
