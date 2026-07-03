"""
Transform interface and implementations for the River Morphology training
pipeline (Module 11).

Defines the abstract Transform interface so that RiverMorphologyDataset is
completely decoupled from albumentations. The dataset calls

    image_np, mask_np = self._transform(image_np, mask_np)

without knowing whether the transform is an identity operation, an
albumentations pipeline, or any future implementation.

    Transform (ABC)                 -- interface contract
        IdentityTransform           -- no-op; default for eval splits
        AlbumentationsTransform     -- wraps albumentations.Compose;
                                       handles (C,H,W) <-> (H,W,C)

Module 12 may provide additional Transform implementations (e.g. TTA-based
transforms, mixed augmentation strategies) without any changes to
RiverMorphologyDataset or DataLoaderFactory.

AugmentationPipeline returns:
    - AlbumentationsTransform  when augmentation is enabled
    - IdentityTransform        when augmentation is disabled or for eval
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.core.exceptions import InvalidValueError

__all__ = [
    "Transform",
    "IdentityTransform",
    "AlbumentationsTransform",
    "AugmentationConfig",
    "AugmentationPipeline",
]

_LOGGER: logging.Logger = logging.getLogger(__name__)


# ==============================================================================
# Transform interface
# ==============================================================================

class Transform(ABC):
    """
    Abstract base class for all image/mask transforms.

    Every concrete implementation receives a float32 image array in
    (C, H, W) order and a uint8 mask array in (H, W) order, and returns
    both arrays in the same shapes and dtypes.

    Calling convention (used by RiverMorphologyDataset):

        image_np, mask_np = transform(image_np, mask_np)

    No keywords, no dicts, no albumentations-specific return format.
    All albumentations coupling is encapsulated inside AlbumentationsTransform.
    """

    @abstractmethod
    def __call__(
        self,
        image: np.ndarray,
        mask:  np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply this transform to one (image, mask) pair.

        Args:
            image: float32 ndarray, shape (C, H, W).
            mask:  uint8   ndarray, shape (H, W).

        Returns:
            Tuple of (transformed_image, transformed_mask) with identical
            shapes (C, H, W) and (H, W) and dtypes float32 and uint8.
        """


# ==============================================================================
# IdentityTransform
# ==============================================================================

class IdentityTransform(Transform):
    """
    No-op transform that returns the input arrays unchanged.

    Default transform for:
        - Validation and test splits in DataLoaderFactory.
        - Any RiverMorphologyDataset constructed without an explicit transform.

    Installing IdentityTransform as the default means that __getitem__ calls
    transform(image, mask) unconditionally -- no if-branch needed.
    """

    def __call__(
        self,
        image: np.ndarray,
        mask:  np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (image, mask) unchanged."""
        return image, mask


# ==============================================================================
# AlbumentationsTransform
# ==============================================================================

class AlbumentationsTransform(Transform):
    """
    Wraps an albumentations.Compose pipeline behind the Transform interface.

    All albumentations-specific logic is confined to this class:
        - Transposes (C, H, W) -> (H, W, C) before calling Compose.
        - Transposes result back to (C, H, W).
        - Extracts image and mask from the albumentations result dict.

    Args:
        compose: An albumentations.Compose instance to wrap.

    Raises:
        InvalidValueError: compose is None.
    """

    def __init__(self, compose: Any) -> None:
        if compose is None:
            raise InvalidValueError(
                field="compose",
                value=None,
                reason="albumentations.Compose instance must not be None",
            )
        self._compose = compose
        self._logger: logging.Logger = logging.getLogger(__name__)

    @property
    def compose(self) -> Any:
        """The wrapped albumentations.Compose instance (read-only)."""
        return self._compose

    def __call__(
        self,
        image: np.ndarray,
        mask:  np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply the albumentations pipeline to (image, mask).

        Internal transposition handles the convention mismatch:
            albumentations expects (H, W, C); this pipeline uses (C, H, W).

        Args:
            image: float32 ndarray (C, H, W).
            mask:  uint8   ndarray (H, W).

        Returns:
            Tuple of (transformed_image float32 (C,H,W),
                      transformed_mask  uint8   (H,W)).
        """
        image_hwc = np.transpose(image, (1, 2, 0))
        result    = self._compose(image=image_hwc, mask=mask)
        image_chw = np.transpose(result["image"], (2, 0, 1))
        return image_chw.astype(np.float32), result["mask"].astype(np.uint8)


# ==============================================================================
# AugmentationConfig
# ==============================================================================

@dataclass(frozen=True)
class AugmentationConfig:
    """
    Immutable augmentation settings sourced from config.training.augmentation.

    Attributes:
        enabled:                    Master toggle for all augmentation.
        horizontal_flip:             Random horizontal flip.
        vertical_flip:               Random vertical flip.
        random_rotate_90:             Random 0/90/180/270-degree rotation.
        random_brightness_contrast:   Random brightness and contrast shift.
        brightness_limit:              Maximum brightness shift in [0, 1].
        contrast_limit:                Maximum contrast shift in [0, 1].
    """

    enabled:                    bool  = True
    horizontal_flip:             bool  = True
    vertical_flip:               bool  = True
    random_rotate_90:             bool  = True
    random_brightness_contrast:   bool  = True
    brightness_limit:              float = 0.15
    contrast_limit:                float = 0.15

    @classmethod
    def from_config(cls, config: Any) -> AugmentationConfig:
        """
        Build AugmentationConfig from config.training.augmentation.

        Returns defaults when the section is absent.
        """
        train_cfg = getattr(config, "training", None)
        aug_cfg   = getattr(train_cfg, "augmentation", None)
        if aug_cfg is None:
            _LOGGER.debug("No training.augmentation in config; using defaults.")
            return cls()
        return cls(
            enabled                  = bool(getattr(aug_cfg, "enabled",                    True)),
            horizontal_flip           = bool(getattr(aug_cfg, "horizontal_flip",             True)),
            vertical_flip             = bool(getattr(aug_cfg, "vertical_flip",               True)),
            random_rotate_90           = bool(getattr(aug_cfg, "random_rotate_90",             True)),
            random_brightness_contrast = bool(getattr(aug_cfg, "random_brightness_contrast",   True)),
            brightness_limit           = float(getattr(aug_cfg, "brightness_limit",            0.15)),
            contrast_limit             = float(getattr(aug_cfg, "contrast_limit",              0.15)),
        )


# ==============================================================================
# AugmentationPipeline
# ==============================================================================

class AugmentationPipeline:
    """
    Builds Transform instances for train and eval splits.

    Returns:
        build_train_transform():
            AlbumentationsTransform when augmentation is enabled.
            IdentityTransform when disabled.
        build_eval_transform():
            IdentityTransform always (deterministic, no albumentations).

    Args:
        aug_config: AugmentationConfig controlling active transforms.
    """

    def __init__(self, aug_config: AugmentationConfig) -> None:
        self._aug_config = aug_config
        self._logger: logging.Logger = logging.getLogger(__name__)

    def build_train_transform(self) -> Transform:
        """
        Return a Transform for the training split.

        Returns:
            AlbumentationsTransform wrapping a configured albumentations
            Compose pipeline when augmentation is enabled.
            IdentityTransform when augmentation is disabled (avoids
            requiring albumentations to be installed in that case).

        Raises:
            ImportError: albumentations is not installed and augmentation
                         is enabled.
        """
        if not self._aug_config.enabled:
            self._logger.debug("Augmentation disabled; returning IdentityTransform.")
            return IdentityTransform()

        try:
            import albumentations as A
        except ImportError as exc:
            raise ImportError(
                "albumentations is not installed. "
                "Install with: pip install albumentations==1.4.3"
            ) from exc

        ops: list[Any] = []
        if self._aug_config.horizontal_flip:
            ops.append(A.HorizontalFlip(p=0.5))
        if self._aug_config.vertical_flip:
            ops.append(A.VerticalFlip(p=0.5))
        if self._aug_config.random_rotate_90:
            ops.append(A.RandomRotate90(p=0.5))
        if self._aug_config.random_brightness_contrast:
            ops.append(A.RandomBrightnessContrast(
                brightness_limit=self._aug_config.brightness_limit,
                contrast_limit=self._aug_config.contrast_limit,
                p=0.5,
            ))

        self._logger.debug(
            "AlbumentationsTransform (train): %d op(s).", len(ops)
        )
        return AlbumentationsTransform(A.Compose(ops))

    def build_eval_transform(self) -> Transform:
        """
        Return an IdentityTransform for validation and test splits.

        Evaluation transforms must be deterministic and parameter-free.
        IdentityTransform satisfies both without requiring albumentations.

        Returns:
            IdentityTransform instance.
        """
        self._logger.debug("Eval transform: IdentityTransform (no augmentation).")
        return IdentityTransform()