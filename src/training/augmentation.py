"""
Augmentation transforms for the Data Transformation and Augmentation Pipeline
(Module 12).

Design constraints (non-negotiable)
-------------------------------------
1. Every geometric transform synchronizes image and mask using the same
   random state.  A single RNG decision is made per sample per transform.
2. Mask operations ALWAYS use nearest-neighbor interpolation.  Class IDs
   are never interpolated; they must remain integers from the valid set.
3. All transforms support arbitrary numbers of input channels.  No transform
   assumes RGB or any specific band ordering.
4. Elastic and perspective distortions are deliberately excluded because they
   distort the characteristic shapes of river channels, sand bars, and
   vegetation patches, degrading the quality of pseudo-labels as supervision
   signals.
5. All parameters come from the transform's constructor.  No defaults are
   hardcoded in the class body; defaults are defined in TransformRegistry or
   in config.
6. All transforms are stateless after construction.  Each call to apply()
   draws fresh random numbers using numpy's default_rng seeded by the
   global seed managed by TransformPipeline.

Registered transforms (auto-discovered by TransformRegistry)
--------------------------------------------------------------
    horizontal_flip    HorizontalFlipTransform
    vertical_flip      VerticalFlipTransform
    rotate_90          Rotate90Transform
    brightness         BrightnessTransform
    contrast           ContrastTransform
    gaussian_noise     GaussianNoiseTransform
    random_crop        RandomCropTransform
    random_scale       RandomScaleTransform
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from src.training.contracts import TransformSample
from src.training.transform import SegmentationTransform

__all__ = [
    "HorizontalFlipTransform",
    "VerticalFlipTransform",
    "Rotate90Transform",
    "BrightnessTransform",
    "ContrastTransform",
    "GaussianNoiseTransform",
    "RandomCropTransform",
    "RandomScaleTransform",
]

_LOGGER: logging.Logger = logging.getLogger(__name__)


# ==============================================================================
# HorizontalFlipTransform
# ==============================================================================

class HorizontalFlipTransform(SegmentationTransform):
    """
    Randomly flips image and mask horizontally (left-right).

    Args:
        probability: Probability of applying the flip [0.0, 1.0].
    """

    _NAME: str = "horizontal_flip"

    def __init__(self, probability: float) -> None:
        if not (0.0 <= probability <= 1.0):
            from src.core.exceptions import InvalidValueError
            raise InvalidValueError(
                field="horizontal_flip.probability",
                value=probability,
                reason="must be in [0.0, 1.0]",
            )
        self._probability = float(probability)

    @property
    def name(self) -> str:
        return self._NAME

    def apply(self, sample: TransformSample) -> TransformSample:
        if np.random.random() < self._probability:
            sample.image = np.ascontiguousarray(np.flip(sample.image, axis=2))
            sample.mask  = np.ascontiguousarray(np.flip(sample.mask,  axis=1))
        return sample


# ==============================================================================
# VerticalFlipTransform
# ==============================================================================

class VerticalFlipTransform(SegmentationTransform):
    """
    Randomly flips image and mask vertically (top-bottom).

    Args:
        probability: Probability of applying the flip [0.0, 1.0].
    """

    _NAME: str = "vertical_flip"

    def __init__(self, probability: float) -> None:
        if not (0.0 <= probability <= 1.0):
            from src.core.exceptions import InvalidValueError
            raise InvalidValueError(
                field="vertical_flip.probability",
                value=probability,
                reason="must be in [0.0, 1.0]",
            )
        self._probability = float(probability)

    @property
    def name(self) -> str:
        return self._NAME

    def apply(self, sample: TransformSample) -> TransformSample:
        if np.random.random() < self._probability:
            sample.image = np.ascontiguousarray(np.flip(sample.image, axis=1))
            sample.mask  = np.ascontiguousarray(np.flip(sample.mask,  axis=0))
        return sample


# ==============================================================================
# Rotate90Transform
# ==============================================================================

class Rotate90Transform(SegmentationTransform):
    """
    Randomly rotates image and mask by a multiple of 90 degrees.

    Suitable for satellite imagery where there is no canonical up direction.

    Args:
        probability:    Probability of applying a rotation [0.0, 1.0].
        num_rotations:  If None, a random multiple of 90 deg is chosen each
                        call.  If an integer (1, 2, or 3), that exact number
                        of 90-degree counter-clockwise rotations is always
                        applied (subject to probability).
    """

    _NAME: str = "rotate_90"

    def __init__(
        self,
        probability:    float,
        num_rotations:  int | None = None,
    ) -> None:
        if not (0.0 <= probability <= 1.0):
            from src.core.exceptions import InvalidValueError
            raise InvalidValueError(
                field="rotate_90.probability",
                value=probability,
                reason="must be in [0.0, 1.0]",
            )
        if num_rotations is not None and num_rotations not in (1, 2, 3):
            from src.core.exceptions import InvalidValueError
            raise InvalidValueError(
                field="rotate_90.num_rotations",
                value=num_rotations,
                reason="must be 1, 2, 3, or None",
            )
        self._probability    = float(probability)
        self._num_rotations  = num_rotations

    @property
    def name(self) -> str:
        return self._NAME

    def apply(self, sample: TransformSample) -> TransformSample:
        if np.random.random() < self._probability:
            k = self._num_rotations if self._num_rotations is not None else np.random.randint(1, 4)
            # np.rot90 for (C, H, W): rotate spatial dims (axes 1 and 2).
            sample.image = np.ascontiguousarray(np.rot90(sample.image, k=k, axes=(1, 2)))
            sample.mask  = np.ascontiguousarray(np.rot90(sample.mask,  k=k, axes=(0, 1)))
        return sample


# ==============================================================================
# BrightnessTransform
# ==============================================================================

class BrightnessTransform(SegmentationTransform):
    """
    Randomly adjusts pixel brightness by adding a uniform delta.

    Applied per-image only; mask is always unchanged.  Suitable for
    multi-band imagery because all bands are shifted by the same delta,
    preserving relative band ratios (important for NDWI, NDVI, etc.).

    For independent per-band adjustment, configure multiple BrightnessTransform
    instances or use per-band scaling in NormalizationTransform instead.

    Args:
        probability:  Probability of applying the adjustment [0.0, 1.0].
        max_delta:     Maximum absolute brightness shift.  The actual shift is
                       drawn uniformly from [-max_delta, max_delta].
                       Units are the same as the input image values (typically
                       reflectance in [0, 1] after normalization).
    """

    _NAME: str = "brightness"

    def __init__(self, probability: float, max_delta: float) -> None:
        if not (0.0 <= probability <= 1.0):
            from src.core.exceptions import InvalidValueError
            raise InvalidValueError(
                field="brightness.probability",
                value=probability,
                reason="must be in [0.0, 1.0]",
            )
        if max_delta < 0.0:
            from src.core.exceptions import InvalidValueError
            raise InvalidValueError(
                field="brightness.max_delta",
                value=max_delta,
                reason="must be >= 0",
            )
        self._probability = float(probability)
        self._max_delta   = float(max_delta)

    @property
    def name(self) -> str:
        return self._NAME

    def apply(self, sample: TransformSample) -> TransformSample:
        if np.random.random() < self._probability:
            delta = float(np.random.uniform(-self._max_delta, self._max_delta))
            sample.image = (sample.image + delta).astype(np.float32)
        return sample


# ==============================================================================
# ContrastTransform
# ==============================================================================

class ContrastTransform(SegmentationTransform):
    """
    Randomly adjusts image contrast by scaling pixel values around the mean.

    The per-image mean is computed across all bands and spatial locations.
    Values are scaled by a factor drawn uniformly from
    [1 - contrast_range, 1 + contrast_range] around the image mean.

    Only the image is affected; the mask is always unchanged.

    Args:
        probability:     Probability of applying [0.0, 1.0].
        contrast_range:  Maximum fractional deviation from 1.0 contrast
                         (e.g. 0.2 -> factor in [0.8, 1.2]).
    """

    _NAME: str = "contrast"

    def __init__(self, probability: float, contrast_range: float) -> None:
        if not (0.0 <= probability <= 1.0):
            from src.core.exceptions import InvalidValueError
            raise InvalidValueError(
                field="contrast.probability",
                value=probability,
                reason="must be in [0.0, 1.0]",
            )
        if contrast_range < 0.0:
            from src.core.exceptions import InvalidValueError
            raise InvalidValueError(
                field="contrast.contrast_range",
                value=contrast_range,
                reason="must be >= 0",
            )
        self._probability     = float(probability)
        self._contrast_range  = float(contrast_range)

    @property
    def name(self) -> str:
        return self._NAME

    def apply(self, sample: TransformSample) -> TransformSample:
        if np.random.random() < self._probability:
            factor  = float(
                np.random.uniform(
                    1.0 - self._contrast_range,
                    1.0 + self._contrast_range,
                )
            )
            mean    = float(np.mean(sample.image))
            sample.image = (mean + factor * (sample.image - mean)).astype(np.float32)
        return sample


# ==============================================================================
# GaussianNoiseTransform
# ==============================================================================

class GaussianNoiseTransform(SegmentationTransform):
    """
    Adds zero-mean Gaussian noise to the image.

    Simulates sensor noise present in real-world multispectral imagery,
    improving model robustness to acquisition artifacts.  Only the image is
    affected; the mask is unchanged.

    Args:
        probability:  Probability of applying [0.0, 1.0].
        std:          Standard deviation of the Gaussian noise.
                      Units are the same as the image pixel values.
    """

    _NAME: str = "gaussian_noise"

    def __init__(self, probability: float, std: float) -> None:
        if not (0.0 <= probability <= 1.0):
            from src.core.exceptions import InvalidValueError
            raise InvalidValueError(
                field="gaussian_noise.probability",
                value=probability,
                reason="must be in [0.0, 1.0]",
            )
        if std < 0.0:
            from src.core.exceptions import InvalidValueError
            raise InvalidValueError(
                field="gaussian_noise.std",
                value=std,
                reason="must be >= 0",
            )
        self._probability = float(probability)
        self._std         = float(std)

    @property
    def name(self) -> str:
        return self._NAME

    def apply(self, sample: TransformSample) -> TransformSample:
        if self._std > 0.0 and np.random.random() < self._probability:
            noise        = np.random.normal(0.0, self._std, sample.image.shape).astype(np.float32)
            sample.image = (sample.image + noise).astype(np.float32)
        return sample


# ==============================================================================
# RandomCropTransform
# ==============================================================================

class RandomCropTransform(SegmentationTransform):
    """
    Randomly crops a fixed-size region from image and mask.

    Both image and mask are cropped with the same top-left corner so they
    remain perfectly synchronized.  The crop size must be smaller than the
    input patch size.

    Args:
        probability:   Probability of applying [0.0, 1.0].
        crop_height:   Height of the output crop in pixels.
        crop_width:    Width of the output crop in pixels.
    """

    _NAME: str = "random_crop"

    def __init__(
        self,
        probability:  float,
        crop_height:   int,
        crop_width:    int,
    ) -> None:
        if not (0.0 <= probability <= 1.0):
            from src.core.exceptions import InvalidValueError
            raise InvalidValueError(
                field="random_crop.probability",
                value=probability,
                reason="must be in [0.0, 1.0]",
            )
        if crop_height < 1 or crop_width < 1:
            from src.core.exceptions import InvalidValueError
            raise InvalidValueError(
                field="random_crop.crop_size",
                value=(crop_height, crop_width),
                reason="crop_height and crop_width must be >= 1",
            )
        self._probability  = float(probability)
        self._crop_height  = int(crop_height)
        self._crop_width   = int(crop_width)

    @property
    def name(self) -> str:
        return self._NAME

    def apply(self, sample: TransformSample) -> TransformSample:
        _, h, w = sample.image.shape
        if (
            np.random.random() < self._probability
            and h >= self._crop_height
            and w >= self._crop_width
        ):
            top  = int(np.random.randint(0, h - self._crop_height + 1))
            left = int(np.random.randint(0, w - self._crop_width  + 1))
            sample.image = sample.image[
                :,
                top: top + self._crop_height,
                left: left + self._crop_width,
            ]
            sample.mask = sample.mask[
                top: top + self._crop_height,
                left: left + self._crop_width,
            ]
        return sample


# ==============================================================================
# RandomScaleTransform
# ==============================================================================

class RandomScaleTransform(SegmentationTransform):
    """
    Randomly scales (resizes) the image and mask by a factor.

    Uses nearest-neighbor interpolation for the mask (mandatory for class ID
    preservation) and bilinear interpolation for the image.

    Requires scipy or PIL for resizing.  Falls back to identity when neither is
    available (logs a WARNING).  This graceful degradation matches the policy
    used by MorphologyProcessor in Module 9.

    Args:
        probability:   Probability of applying [0.0, 1.0].
        min_scale:     Minimum scale factor (e.g. 0.75).
        max_scale:     Maximum scale factor (e.g. 1.25).
    """

    _NAME: str = "random_scale"

    def __init__(
        self,
        probability: float,
        min_scale:    float,
        max_scale:    float,
    ) -> None:
        if not (0.0 <= probability <= 1.0):
            from src.core.exceptions import InvalidValueError
            raise InvalidValueError(
                field="random_scale.probability",
                value=probability,
                reason="must be in [0.0, 1.0]",
            )
        if min_scale <= 0.0 or max_scale <= 0.0 or min_scale > max_scale:
            from src.core.exceptions import InvalidValueError
            raise InvalidValueError(
                field="random_scale.(min_scale, max_scale)",
                value=(min_scale, max_scale),
                reason="must satisfy 0 < min_scale <= max_scale",
            )
        self._probability = float(probability)
        self._min_scale   = float(min_scale)
        self._max_scale   = float(max_scale)

    @property
    def name(self) -> str:
        return self._NAME

    def apply(self, sample: TransformSample) -> TransformSample:
        if np.random.random() >= self._probability:
            return sample

        scale = float(np.random.uniform(self._min_scale, self._max_scale))
        if abs(scale - 1.0) < 1e-6:
            return sample

        _, h, w = sample.image.shape
        new_h   = max(1, int(round(h * scale)))
        new_w   = max(1, int(round(w * scale)))

        sample.image = self._resize_image(sample.image, new_h, new_w)
        sample.mask  = self._resize_mask(sample.mask,   new_h, new_w)
        return sample

    @staticmethod
    def _resize_image(image: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
        """Resize (C, H, W) image using bilinear interpolation via scipy or PIL."""
        try:
            from scipy.ndimage import zoom
            c = image.shape[0]
            h = image.shape[1]
            w = image.shape[2]
            scale_h = new_h / h
            scale_w = new_w / w
            return zoom(image, (1.0, scale_h, scale_w), order=1).astype(np.float32)
        except ImportError:
            pass

        try:
            from PIL import Image as PILImage
            c = image.shape[0]
            bands = []
            for i in range(c):
                pil = PILImage.fromarray(image[i], mode="F")
                pil = pil.resize((new_w, new_h), PILImage.BILINEAR)
                bands.append(np.array(pil, dtype=np.float32))
            return np.stack(bands, axis=0)
        except ImportError:
            pass

        _LOGGER.warning(
            "RandomScaleTransform: neither scipy nor PIL is available; "
            "image scaling is skipped. Install scipy or Pillow."
        )
        return image

    @staticmethod
    def _resize_mask(mask: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
        """
        Resize (H, W) mask using nearest-neighbor interpolation.

        Class IDs are never interpolated; nearest-neighbor is mandatory.
        """
        try:
            from scipy.ndimage import zoom
            h = mask.shape[0]
            w = mask.shape[1]
            scale_h = new_h / h
            scale_w = new_w / w
            return zoom(mask.astype(np.float32), (scale_h, scale_w), order=0).astype(np.uint8)
        except ImportError:
            pass

        try:
            from PIL import Image as PILImage
            pil = PILImage.fromarray(mask, mode="L")
            pil = pil.resize((new_w, new_h), PILImage.NEAREST)
            return np.array(pil, dtype=np.uint8)
        except ImportError:
            pass

        _LOGGER.warning(
            "RandomScaleTransform: neither scipy nor PIL is available; "
            "mask scaling is skipped."
        )
        return mask
