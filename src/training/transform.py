"""
Abstract transform interface and composition utilities for Module 12.

SegmentationTransform is the stable abstract interface that all transforms
implement.  The interface enforces three invariants:

1. Image and mask geometric transforms are ALWAYS synchronized.
2. Masks ALWAYS use nearest-neighbor interpolation to preserve class IDs.
3. Class IDs are NEVER changed by any transform.

All transforms are stateless after construction; a single instance is safe to
call from multiple DataLoader workers.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from src.training.contracts import TransformSample

__all__ = [
    "SegmentationTransform",
    "ComposedTransform",
    "IdentityTransform",
]

_LOGGER: logging.Logger = logging.getLogger(__name__)


# ==============================================================================
# Abstract interface
# ==============================================================================

class SegmentationTransform(ABC):
    """
    Abstract interface for all segmentation transforms.

    Every concrete implementation must satisfy:
        - apply() returns a TransformSample with the same sample_id, split, and
          all temporal/geographic metadata fields preserved unchanged.
        - Geometric transforms synchronize image and mask identically.
        - Mask interpolation is always nearest-neighbor.
        - Class IDs are never altered.

    Subclasses should be stateless after construction; all stochastic state must
    be captured in the single call to apply(), using the random state provided
    by the caller's RNG, not a module-level random state.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier for this transform (logged in operations_log)."""

    @abstractmethod
    def apply(self, sample: TransformSample) -> TransformSample:
        """
        Apply this transform to one sample.

        Args:
            sample: TransformSample with image (C, H, W) float32 and
                    mask (H, W) uint8.

        Returns:
            Transformed TransformSample with metadata preserved.
        """

    def __call__(self, sample: TransformSample) -> TransformSample:
        """Alias for apply() so transforms are callable like PyTorch transforms."""
        return self.apply(sample)


# ==============================================================================
# ComposedTransform
# ==============================================================================

class ComposedTransform(SegmentationTransform):
    """
    Applies a sequence of transforms in order.

    Equivalent to torchvision.transforms.Compose but operates on
    TransformSample objects and is compatible with multi-band imagery and
    synchronized image-mask transforms.

    Args:
        transforms: Ordered list of SegmentationTransform instances.
                    Executed in the order given.
    """

    _NAME: str = "composed"

    def __init__(self, transforms: list[SegmentationTransform]) -> None:
        self._transforms = list(transforms)

    @property
    def name(self) -> str:
        return self._NAME

    @property
    def transform_names(self) -> tuple[str, ...]:
        """Names of the composed transforms in application order."""
        return tuple(t.name for t in self._transforms)

    def apply(self, sample: TransformSample) -> TransformSample:
        """Apply transforms in order."""
        for transform in self._transforms:
            sample = transform.apply(sample)
        return sample

    def __repr__(self) -> str:
        names = ", ".join(t.name for t in self._transforms)
        return f"ComposedTransform([{names}])"


# ==============================================================================
# IdentityTransform
# ==============================================================================

class IdentityTransform(SegmentationTransform):
    """
    Pass-through transform.  Returns the sample unchanged.

    Used as a placeholder for validation-/test-set pipelines when no
    augmentation is configured, and in tests that require a concrete
    transform without side effects.
    """

    _NAME: str = "identity"

    @property
    def name(self) -> str:
        return self._NAME

    def apply(self, sample: TransformSample) -> TransformSample:
        """Return the sample unchanged."""
        return sample
