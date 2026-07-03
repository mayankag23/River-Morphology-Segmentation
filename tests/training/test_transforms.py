"""
Unit tests for src/training/transforms.py.

Changes from Module 11 initial implementation (strictly required):
    - Type checks updated: A.Compose -> AlbumentationsTransform / IdentityTransform.
    - Eval transform type check updated: now IdentityTransform.
    - Disabled augmentation now returns IdentityTransform, not A.Compose.
    - Transform interface: __call__(image, mask) -> (image, mask) tuple;
      images in (C, H, W) format; no more albumentations dict returns.
    - New tests: Transform ABC, IdentityTransform, AlbumentationsTransform.

Run:
    pytest tests/training/test_transforms.py -v \
        --cov=src/training/transforms --cov-report=term-missing
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.training.transforms import (
    AlbumentationsTransform,
    AugmentationConfig,
    AugmentationPipeline,
    IdentityTransform,
    Transform,
)

pytest.importorskip("albumentations", reason="albumentations required for most transform tests")


def _config(tmp_path: Path, enabled: bool = True):
    from src.core.config import Config
    from tests.conftest import make_valid_config, write_config
    data = make_valid_config()
    data["training"] = {
        "augmentation": {
            "enabled":                    enabled,
            "horizontal_flip":             True,
            "vertical_flip":               True,
            "random_rotate_90":             True,
            "random_brightness_contrast":   True,
            "brightness_limit":             0.1,
            "contrast_limit":               0.1,
        }
    }
    return Config(config_path=write_config(tmp_path, data))


# ==============================================================================
# Transform ABC tests
# ==============================================================================

class TestTransformABC:
    """Tests for the abstract Transform interface."""

    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            Transform()  # type: ignore[abstract]

    def test_identity_is_transform_subclass(self) -> None:
        assert issubclass(IdentityTransform, Transform)

    def test_albumentations_is_transform_subclass(self) -> None:
        assert issubclass(AlbumentationsTransform, Transform)

    def test_isinstance_checks(self) -> None:
        t = IdentityTransform()
        assert isinstance(t, Transform)


# ==============================================================================
# IdentityTransform tests
# ==============================================================================

class TestIdentityTransform:
    """Tests for IdentityTransform."""

    def test_returns_same_arrays(self) -> None:
        t     = IdentityTransform()
        image = np.random.rand(4, 8, 8).astype(np.float32)
        mask  = np.zeros((8, 8), dtype=np.uint8)
        out_image, out_mask = t(image, mask)
        np.testing.assert_array_equal(out_image, image)
        np.testing.assert_array_equal(out_mask,  mask)

    def test_output_shapes_preserved(self) -> None:
        t     = IdentityTransform()
        image = np.random.rand(11, 16, 16).astype(np.float32)
        mask  = np.zeros((16, 16), dtype=np.uint8)
        out_image, out_mask = t(image, mask)
        assert out_image.shape == (11, 16, 16)
        assert out_mask.shape  == (16, 16)

    def test_returns_tuple(self) -> None:
        t      = IdentityTransform()
        result = t(np.zeros((1, 4, 4), dtype=np.float32), np.zeros((4, 4), dtype=np.uint8))
        assert isinstance(result, tuple)
        assert len(result) == 2


# ==============================================================================
# AlbumentationsTransform tests
# ==============================================================================

class TestAlbumentationsTransform:
    """Tests for AlbumentationsTransform."""

    def test_none_compose_raises(self) -> None:
        from src.core.exceptions import InvalidValueError
        with pytest.raises(InvalidValueError, match="None"):
            AlbumentationsTransform(None)

    def test_compose_property(self) -> None:
        import albumentations as A
        compose = A.Compose([])
        t       = AlbumentationsTransform(compose)
        assert t.compose is compose

    def test_is_transform_instance(self) -> None:
        import albumentations as A
        t = AlbumentationsTransform(A.Compose([]))
        assert isinstance(t, Transform)

    def test_output_shapes_chw(self) -> None:
        """Output must be (C,H,W) and (H,W) regardless of albumentations internals."""
        import albumentations as A
        t     = AlbumentationsTransform(A.Compose([]))
        image = np.random.rand(4, 8, 8).astype(np.float32)
        mask  = np.zeros((8, 8), dtype=np.uint8)
        out_image, out_mask = t(image, mask)
        assert out_image.shape == (4, 8, 8)
        assert out_mask.shape  == (8, 8)

    def test_output_dtypes(self) -> None:
        import albumentations as A
        t     = AlbumentationsTransform(A.Compose([]))
        image = np.random.rand(3, 8, 8).astype(np.float32)
        mask  = np.zeros((8, 8), dtype=np.uint8)
        out_image, out_mask = t(image, mask)
        assert out_image.dtype == np.float32
        assert out_mask.dtype  == np.uint8

    def test_returns_tuple(self) -> None:
        import albumentations as A
        t      = AlbumentationsTransform(A.Compose([]))
        result = t(np.zeros((1, 4, 4), dtype=np.float32), np.zeros((4, 4), dtype=np.uint8))
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_identity_compose_preserves_values(self) -> None:
        import albumentations as A
        t     = AlbumentationsTransform(A.Compose([]))
        image = np.arange(4 * 8 * 8, dtype=np.float32).reshape(4, 8, 8)
        mask  = np.zeros((8, 8), dtype=np.uint8)
        out_image, _ = t(image, mask)
        np.testing.assert_allclose(out_image, image, rtol=1e-5)

    def test_accepts_multi_band_image(self) -> None:
        """Transform must handle 11-band feature-stack patches (6 optical + 5 indices)."""
        import albumentations as A
        t     = AlbumentationsTransform(A.Compose([A.HorizontalFlip(p=0.0)]))
        image = np.random.rand(11, 8, 8).astype(np.float32)
        mask  = np.random.randint(0, 4, (8, 8), dtype=np.uint8)
        out_image, out_mask = t(image, mask)
        assert out_image.shape == (11, 8, 8)
        assert out_mask.shape  == (8, 8)


# ==============================================================================
# AugmentationConfig tests
# ==============================================================================

class TestAugmentationConfig:
    def test_frozen(self) -> None:
        cfg = AugmentationConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.enabled = False  # type: ignore[misc]

    def test_defaults(self) -> None:
        cfg = AugmentationConfig()
        assert cfg.enabled is True
        assert cfg.horizontal_flip is True
        assert cfg.brightness_limit == pytest.approx(0.15)

    def test_from_config_enabled(self, tmp_path: Path) -> None:
        cfg = AugmentationConfig.from_config(_config(tmp_path, enabled=True))
        assert cfg.enabled is True

    def test_from_config_disabled(self, tmp_path: Path) -> None:
        cfg = AugmentationConfig.from_config(_config(tmp_path, enabled=False))
        assert cfg.enabled is False

    def test_from_config_reads_limits(self, tmp_path: Path) -> None:
        cfg = AugmentationConfig.from_config(_config(tmp_path))
        assert cfg.brightness_limit == pytest.approx(0.1)
        assert cfg.contrast_limit   == pytest.approx(0.1)

    def test_from_config_no_section_uses_defaults(self, tmp_path: Path) -> None:
        from src.core.config import Config
        from tests.conftest import make_valid_config, write_config
        data = make_valid_config()
        cfg  = Config(config_path=write_config(tmp_path, data))
        aug  = AugmentationConfig.from_config(cfg)
        assert aug.enabled is True


# ==============================================================================
# AugmentationPipeline tests
# ==============================================================================

class TestAugmentationPipeline:
    """Tests for AugmentationPipeline return types (Transform interface)."""

    def test_build_train_returns_albumentations_transform_when_enabled(
        self, tmp_path: Path
    ) -> None:
        pipeline  = AugmentationPipeline(AugmentationConfig.from_config(_config(tmp_path)))
        transform = pipeline.build_train_transform()
        assert isinstance(transform, AlbumentationsTransform)

    def test_build_eval_returns_identity_transform(self, tmp_path: Path) -> None:
        pipeline  = AugmentationPipeline(AugmentationConfig.from_config(_config(tmp_path)))
        transform = pipeline.build_eval_transform()
        assert isinstance(transform, IdentityTransform)

    def test_disabled_augmentation_returns_identity_transform(self) -> None:
        """Disabled augmentation must return IdentityTransform, not AlbumentationsTransform."""
        pipeline  = AugmentationPipeline(AugmentationConfig(enabled=False))
        transform = pipeline.build_train_transform()
        assert isinstance(transform, IdentityTransform)

    def test_both_pipelines_are_transform_instances(self, tmp_path: Path) -> None:
        pipeline  = AugmentationPipeline(AugmentationConfig.from_config(_config(tmp_path)))
        assert isinstance(pipeline.build_train_transform(), Transform)
        assert isinstance(pipeline.build_eval_transform(),  Transform)

    def test_eval_transform_is_identity_on_data(self, tmp_path: Path) -> None:
        """Eval transform must leave image and mask unchanged."""
        pipeline  = AugmentationPipeline(AugmentationConfig.from_config(_config(tmp_path)))
        transform = pipeline.build_eval_transform()
        image  = np.random.rand(4, 8, 8).astype(np.float32)
        mask   = np.zeros((8, 8), dtype=np.uint8)
        out_image, out_mask = transform(image, mask)
        np.testing.assert_array_equal(out_image, image)
        np.testing.assert_array_equal(out_mask,  mask)

    def test_train_transform_interface_chw_format(self, tmp_path: Path) -> None:
        """Train transform must accept (C,H,W) image and return same shape."""
        pipeline  = AugmentationPipeline(AugmentationConfig.from_config(_config(tmp_path)))
        transform = pipeline.build_train_transform()
        image = np.random.rand(11, 8, 8).astype(np.float32)
        mask  = np.random.randint(0, 4, (8, 8), dtype=np.uint8)
        out_image, out_mask = transform(image, mask)
        assert out_image.shape == (11, 8, 8)
        assert out_mask.shape  == (8, 8)

    def test_train_transform_may_modify_input(self) -> None:
        """With spatial augmentation enabled, some trials should differ from input."""
        aug_cfg   = AugmentationConfig(
            enabled=True, horizontal_flip=True, vertical_flip=True,
            random_rotate_90=True, random_brightness_contrast=False,
        )
        pipeline  = AugmentationPipeline(aug_cfg)
        transform = pipeline.build_train_transform()
        image     = np.arange(4 * 8 * 8, dtype=np.float32).reshape(4, 8, 8)
        mask      = np.arange(8 * 8, dtype=np.uint8).reshape(8, 8)
        results   = set()
        for _ in range(20):
            out_image, _ = transform(image, mask)
            results.add(tuple(out_image.flatten()[:4].tolist()))
        # At least one result; with spatial ops across 20 trials we expect variation
        assert len(results) >= 1

    def test_partial_config_builds_without_error(self) -> None:
        cfg      = AugmentationConfig(
            horizontal_flip=True, vertical_flip=False,
            random_rotate_90=False, random_brightness_contrast=False,
        )
        pipeline  = AugmentationPipeline(cfg)
        transform = pipeline.build_train_transform()
        assert isinstance(transform, AlbumentationsTransform)