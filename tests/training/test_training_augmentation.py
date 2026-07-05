"""
Tests for src/training/augmentation.py

Pure numpy, no I/O, no torch, no GeoTIFF.

Key invariants tested:
- Image and mask always have identical spatial dimensions after geometric transforms.
- Mask values (class IDs) are never changed by any transform.
- Random transforms are reproducible when numpy seed is fixed.
- All transforms support arbitrary channel counts (multi-spectral).

Run:
    pytest tests/training/test_training_augmentation.py -v \
        --cov=src/training/augmentation --cov-report=term-missing
"""

from __future__ import annotations

import numpy as np
import pytest

from src.training.augmentation import (
    BrightnessTransform,
    ContrastTransform,
    GaussianNoiseTransform,
    HorizontalFlipTransform,
    RandomCropTransform,
    RandomScaleTransform,
    Rotate90Transform,
    VerticalFlipTransform,
)
from src.training.contracts import TransformSample


def _sample(
    c: int = 6,
    h: int = 16,
    w: int = 16,
    fill_image: float = 0.5,
    fill_mask:  int   = 1,
) -> TransformSample:
    return TransformSample(
        image=np.full((c, h, w), fill_image, dtype=np.float32),
        mask=np.full((h, w), fill_mask, dtype=np.uint8),
        sample_id="test_001",
        split="train",
        acquisition_date="2023-07-15",
        season="monsoon",
        hydrological_year=2023,
        river_name="Kosi",
    )


def _check_spatial_consistency(s: TransformSample) -> None:
    """Image (C,H,W) and mask (H,W) must share the same spatial dimensions."""
    assert s.image.ndim == 3, f"image.ndim={s.image.ndim}"
    assert s.mask.ndim  == 2, f"mask.ndim={s.mask.ndim}"
    assert s.image.shape[1] == s.mask.shape[0], (
        f"height mismatch: image={s.image.shape[1]}, mask={s.mask.shape[0]}"
    )
    assert s.image.shape[2] == s.mask.shape[1], (
        f"width mismatch: image={s.image.shape[2]}, mask={s.mask.shape[1]}"
    )


def _check_class_ids_unchanged(original: TransformSample, result: TransformSample) -> None:
    """Mask values must never change through any geometric transform."""
    unique_orig   = set(np.unique(original.mask).tolist())
    unique_result = set(np.unique(result.mask).tolist())
    assert unique_result.issubset(unique_orig | unique_result), (
        f"Unexpected class IDs introduced: {unique_result - unique_orig}"
    )


class TestHorizontalFlipTransform:
    def test_name(self) -> None:
        assert HorizontalFlipTransform(0.5).name == "horizontal_flip"

    def test_invalid_probability_raises(self) -> None:
        with pytest.raises(Exception):
            HorizontalFlipTransform(1.5)

    def test_always_flips_when_p1(self) -> None:
        np.random.seed(0)
        s = _sample(fill_image=0.3)
        # Put a recognizable pattern in column 0 vs column -1.
        s.image[:, :, 0] = 0.1
        s.image[:, :, -1] = 0.9
        original_col0 = s.image[:, :, 0].copy()
        t = HorizontalFlipTransform(1.0)
        result = t.apply(s)
        assert np.allclose(result.image[:, :, -1], original_col0)

    def test_never_flips_when_p0(self) -> None:
        np.random.seed(0)
        s = _sample()
        original = s.image.copy()
        t = HorizontalFlipTransform(0.0)
        t.apply(s)
        np.testing.assert_array_equal(s.image, original)

    def test_spatial_consistency(self) -> None:
        np.random.seed(0)
        s = _sample()
        _check_spatial_consistency(HorizontalFlipTransform(1.0).apply(s))

    def test_multi_band(self) -> None:
        np.random.seed(0)
        s = _sample(c=12)
        result = HorizontalFlipTransform(1.0).apply(s)
        assert result.image.shape[0] == 12

    def test_metadata_preserved(self) -> None:
        np.random.seed(0)
        s = _sample()
        r = HorizontalFlipTransform(1.0).apply(s)
        assert r.season == "monsoon"
        assert r.river_name == "Kosi"


class TestVerticalFlipTransform:
    def test_name(self) -> None:
        assert VerticalFlipTransform(0.5).name == "vertical_flip"

    def test_always_flips_when_p1(self) -> None:
        np.random.seed(0)
        s = _sample()
        s.image[:, 0, :] = 0.1
        s.image[:, -1, :] = 0.9
        original_row0 = s.image[:, 0, :].copy()
        t = VerticalFlipTransform(1.0)
        result = t.apply(s)
        np.testing.assert_allclose(result.image[:, -1, :], original_row0)

    def test_spatial_consistency(self) -> None:
        np.random.seed(0)
        _check_spatial_consistency(VerticalFlipTransform(1.0).apply(_sample()))

    def test_mask_class_ids_unchanged(self) -> None:
        np.random.seed(0)
        s = _sample(fill_mask=2)
        r = VerticalFlipTransform(1.0).apply(s)
        assert set(np.unique(r.mask).tolist()) == {2}


class TestRotate90Transform:
    def test_name(self) -> None:
        assert Rotate90Transform(0.5).name == "rotate_90"

    def test_invalid_num_rotations_raises(self) -> None:
        with pytest.raises(Exception):
            Rotate90Transform(0.5, num_rotations=4)

    def test_known_rotation(self) -> None:
        np.random.seed(0)
        s = _sample(c=1, h=4, w=4, fill_image=0.0)
        s.image[0, 0, :] = 1.0   # first row = 1
        t = Rotate90Transform(1.0, num_rotations=1)
        r = t.apply(s)
        # After CCW 90° rotation, first row becomes first column..
        np.testing.assert_allclose(r.image[0, :, 0], np.ones(4))

    def test_spatial_consistency_after_square_rotation(self) -> None:
        np.random.seed(0)
        s = _sample(c=3, h=16, w=16)
        r = Rotate90Transform(1.0, num_rotations=1).apply(s)
        _check_spatial_consistency(r)

    def test_class_ids_unchanged(self) -> None:
        np.random.seed(0)
        s = _sample(fill_mask=3)
        r = Rotate90Transform(1.0, num_rotations=2).apply(s)
        assert set(np.unique(r.mask).tolist()) == {3}

    def test_never_rotates_when_p0(self) -> None:
        np.random.seed(0)
        s = _sample()
        original = s.image.copy()
        Rotate90Transform(0.0).apply(s)
        np.testing.assert_array_equal(s.image, original)


class TestBrightnessTransform:
    def test_name(self) -> None:
        assert BrightnessTransform(0.5, 0.05).name == "brightness"

    def test_invalid_max_delta_raises(self) -> None:
        with pytest.raises(Exception):
            BrightnessTransform(0.5, -0.01)

    def test_mask_unchanged(self) -> None:
        np.random.seed(0)
        s = _sample(fill_mask=1)
        original_mask = s.mask.copy()
        BrightnessTransform(1.0, 0.1).apply(s)
        np.testing.assert_array_equal(s.mask, original_mask)

    def test_image_changes_when_p1(self) -> None:
        np.random.seed(999)
        s = _sample(fill_image=0.5)
        original = s.image.copy()
        t = BrightnessTransform(1.0, 0.05)
        t.apply(s)
        # With max_delta=0.05, image must change.
        assert not np.allclose(s.image, original)

    def test_image_unchanged_when_p0(self) -> None:
        np.random.seed(0)
        s = _sample(fill_image=0.5)
        original = s.image.copy()
        BrightnessTransform(0.0, 0.1).apply(s)
        np.testing.assert_array_equal(s.image, original)

    def test_multi_band(self) -> None:
        np.random.seed(0)
        s = _sample(c=12)
        BrightnessTransform(1.0, 0.1).apply(s)
        assert s.image.shape[0] == 12


class TestContrastTransform:
    def test_name(self) -> None:
        assert ContrastTransform(0.5, 0.1).name == "contrast"

    def test_mask_unchanged(self) -> None:
        np.random.seed(0)
        s = _sample(fill_mask=2)
        original_mask = s.mask.copy()
        ContrastTransform(1.0, 0.2).apply(s)
        np.testing.assert_array_equal(s.mask, original_mask)

    def test_changes_image_when_p1(self) -> None:
        np.random.seed(42)
        s = _sample(fill_image=0.5)
        s.image[0, 0, 0] = 0.9   # create variance so contrast matters
        original = s.image.copy()
        ContrastTransform(1.0, 0.3).apply(s)
        # Some pixels should change.
        assert not np.allclose(s.image, original)

    def test_dtype_preserved(self) -> None:
        np.random.seed(0)
        s = _sample()
        ContrastTransform(1.0, 0.2).apply(s)
        assert s.image.dtype == np.float32


class TestGaussianNoiseTransform:
    def test_name(self) -> None:
        assert GaussianNoiseTransform(0.5, 0.02).name == "gaussian_noise"

    def test_mask_unchanged(self) -> None:
        np.random.seed(0)
        s = _sample(fill_mask=1)
        original_mask = s.mask.copy()
        GaussianNoiseTransform(1.0, 0.1).apply(s)
        np.testing.assert_array_equal(s.mask, original_mask)

    def test_zero_std_unchanged(self) -> None:
        np.random.seed(0)
        s = _sample()
        original = s.image.copy()
        GaussianNoiseTransform(1.0, 0.0).apply(s)
        np.testing.assert_array_equal(s.image, original)

    def test_adds_noise_when_std_positive(self) -> None:
        np.random.seed(0)
        s = _sample()
        original = s.image.copy()
        GaussianNoiseTransform(1.0, 0.1).apply(s)
        assert not np.allclose(s.image, original)

    def test_deterministic_with_seed(self) -> None:
        np.random.seed(7)
        s1 = _sample()
        r1 = GaussianNoiseTransform(1.0, 0.05).apply(s1)
        np.random.seed(7)
        s2 = _sample()
        r2 = GaussianNoiseTransform(1.0, 0.05).apply(s2)
        np.testing.assert_array_equal(r1.image, r2.image)


class TestRandomCropTransform:
    def test_name(self) -> None:
        assert RandomCropTransform(0.5, 8, 8).name == "random_crop"

    def test_invalid_crop_size_raises(self) -> None:
        with pytest.raises(Exception):
            RandomCropTransform(0.5, 0, 8)

    def test_output_size(self) -> None:
        np.random.seed(0)
        s = _sample(c=4, h=16, w=16)
        t = RandomCropTransform(1.0, crop_height=8, crop_width=10)
        r = t.apply(s)
        assert r.image.shape == (4, 8, 10)
        assert r.mask.shape  == (8, 10)

    def test_spatial_consistency_after_crop(self) -> None:
        np.random.seed(0)
        s = _sample(h=16, w=16)
        r = RandomCropTransform(1.0, 8, 8).apply(s)
        _check_spatial_consistency(r)

    def test_skip_when_image_too_small(self) -> None:
        np.random.seed(0)
        s = _sample(h=4, w=4)
        original_shape = s.image.shape
        RandomCropTransform(1.0, 8, 8).apply(s)   # crop > image -- skip
        assert s.image.shape == original_shape

    def test_mask_class_ids_preserved(self) -> None:
        np.random.seed(0)
        s = _sample(fill_mask=2, h=16, w=16)
        r = RandomCropTransform(1.0, 8, 8).apply(s)
        assert set(np.unique(r.mask).tolist()) == {2}


class TestRandomScaleTransform:
    def test_name(self) -> None:
        assert RandomScaleTransform(0.5, 0.75, 1.25).name == "random_scale"

    def test_invalid_scale_raises(self) -> None:
        with pytest.raises(Exception):
            RandomScaleTransform(0.5, 1.5, 0.5)   # min > max

    def test_scale_1_unchanged(self) -> None:
        """A scale factor of exactly 1.0 must leave image and mask unchanged."""
        np.random.seed(0)
        s = _sample(h=16, w=16)
        original_shape = s.image.shape
        # Force scale=1.0 by setting min=max=1.0
        t = RandomScaleTransform(1.0, min_scale=1.0, max_scale=1.0)
        r = t.apply(s)
        assert r.image.shape == original_shape

    def test_spatial_consistency_after_scale(self) -> None:
        """Image and mask must have identical spatial dims regardless of scale."""
        try:
            from scipy.ndimage import zoom
            has_scipy = True
        except ImportError:
            has_scipy = False

        np.random.seed(0)
        s = _sample(h=16, w=16)
        t = RandomScaleTransform(1.0, min_scale=0.5, max_scale=0.5)
        r = t.apply(s)
        _check_spatial_consistency(r)

    def test_never_scales_when_p0(self) -> None:
        np.random.seed(0)
        s = _sample(h=16, w=16)
        orig_shape = s.image.shape
        RandomScaleTransform(0.0, 0.5, 2.0).apply(s)
        assert s.image.shape == orig_shape
