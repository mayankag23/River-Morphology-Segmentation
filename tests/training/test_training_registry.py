"""
Tests for src/training/registry.py

Run:
    pytest tests/training/test_training_registry.py -v \
        --cov=src/training/registry --cov-report=term-missing
"""

from __future__ import annotations

import numpy as np
import pytest

from src.training.augmentation import HorizontalFlipTransform
from src.training.contracts import NormalizationStatistics, TransformSample
from src.training.normalization import NormalizationTransform
from src.training.registry import TransformRegistry
from src.training.transform import ComposedTransform, IdentityTransform


def _stats(n: int = 3) -> NormalizationStatistics:
    return NormalizationStatistics(
        band_names=tuple(f"B{i}" for i in range(n)),
        mean=tuple(0.0 for _ in range(n)),
        std=tuple(1.0 for _ in range(n)),
        num_samples=10,
        source="supplied",
    )


class _MinimalConfig:
    class training:
        class augmentation:
            class horizontal_flip:
                enabled = True
                probability = 0.5
            class vertical_flip:
                enabled = False
                probability = 0.5
            class rotate_90:
                enabled = False
                probability = 0.5
            class brightness:
                enabled = False
                probability = 0.3
                max_delta = 0.05
            class contrast:
                enabled = False
                probability = 0.3
                contrast_range = 0.1
            class gaussian_noise:
                enabled = False
                probability = 0.3
                std = 0.02
            class random_crop:
                enabled = False
                probability = 0.5
                crop_height = 8
                crop_width = 8
            class random_scale:
                enabled = False
                probability = 0.3
                min_scale = 0.75
                max_scale = 1.25


class TestTransformRegistry:
    def setup_method(self) -> None:
        self._saved = dict(TransformRegistry._registered)

    def teardown_method(self) -> None:
        TransformRegistry._registered.clear()
        TransformRegistry._registered.update(self._saved)

    def test_all_builtins_registered(self) -> None:
        names = TransformRegistry.registered_names()
        expected = {
            "horizontal_flip", "vertical_flip", "rotate_90",
            "brightness", "contrast", "gaussian_noise",
            "random_crop", "random_scale",
        }
        assert expected.issubset(set(names))

    def test_register_decorator(self) -> None:
        from src.training.transform import SegmentationTransform

        @TransformRegistry.register
        class _Temp(SegmentationTransform):
            _NAME = "_temp_test_xyz"

            @property
            def name(self) -> str:
                return self._NAME

            def apply(self, sample):
                return sample

        assert "_temp_test_xyz" in TransformRegistry.registered_names()

    def test_register_external(self) -> None:
        from src.training.transform import SegmentationTransform

        class _Ext(SegmentationTransform):
            _NAME = "_ext_test_xyz"

            @property
            def name(self) -> str:
                return self._NAME

            def apply(self, sample):
                return sample

        TransformRegistry.register_external(_Ext)
        assert "_ext_test_xyz" in TransformRegistry.registered_names()

    def test_clear_removes_all(self) -> None:
        TransformRegistry.clear()
        assert TransformRegistry.registered_names() == ()

    def test_create_pipeline_returns_composed(self) -> None:
        pipeline = TransformRegistry.create_pipeline(
            _MinimalConfig, normalization_stats=_stats(3)
        )
        assert isinstance(pipeline, ComposedTransform)

    def test_create_pipeline_normalization_first(self) -> None:
        pipeline = TransformRegistry.create_pipeline(
            _MinimalConfig, normalization_stats=_stats(3)
        )
        names = pipeline.transform_names
        assert names[0] == "normalization"

    def test_create_pipeline_includes_enabled_transforms(self) -> None:
        pipeline = TransformRegistry.create_pipeline(
            _MinimalConfig, normalization_stats=_stats(3)
        )
        assert "horizontal_flip" in pipeline.transform_names

    def test_create_pipeline_excludes_disabled_transforms(self) -> None:
        pipeline = TransformRegistry.create_pipeline(
            _MinimalConfig, normalization_stats=_stats(3)
        )
        assert "vertical_flip" not in pipeline.transform_names

    def test_create_pipeline_augmentation_only_skips_norm(self) -> None:
        pipeline = TransformRegistry.create_pipeline(
            _MinimalConfig,
            normalization_stats=_stats(3),
            augmentation_only=True,
        )
        assert "normalization" not in pipeline.transform_names

    def test_create_pipeline_no_enabled_returns_identity(self) -> None:
        class _NoAug:
            class training:
                class augmentation:
                    pass

        pipeline = TransformRegistry.create_pipeline(_NoAug)
        assert "identity" in pipeline.transform_names

    def test_create_inference_pipeline_norm_only(self) -> None:
        pipeline = TransformRegistry.create_inference_pipeline(_stats(3))
        assert "normalization" in pipeline.transform_names
        assert len(pipeline.transform_names) == 1

    def test_pipeline_applies_to_sample(self) -> None:
        np.random.seed(0)
        pipeline = TransformRegistry.create_pipeline(
            _MinimalConfig, normalization_stats=_stats(3)
        )
        s = TransformSample(
            image=np.ones((3, 8, 8), dtype=np.float32),
            mask=np.zeros((8, 8), dtype=np.uint8),
            sample_id="p1",
            split="train",
        )
        result = pipeline.apply(s)
        assert result.image.shape == (3, 8, 8)
        assert result.mask.shape  == (8, 8)
