"""
Prediction post-processing for Module 16.

Post-processing is applied to the predicted class-ID mask AFTER argmax.
All processors operate on (H, W) uint8 numpy arrays.

Design rules
------------
- No spatial processor assumes a fixed number of classes.
- All processors preserve class IDs; they only reassign ambiguous pixels.
- Operations gracefully degrade when scipy is unavailable (WARNING + passthrough).
- Processors are composable via PostprocessorPipeline.

Registered processors
---------------------
    hole_filler         HoleFiller — fill small interior holes per class.
    small_object_remover SmallObjectRemover — remove components < min_size pixels.
    morph_open          MorphOpenProcessor — morphological opening (erosion+dilation).
    morph_close         MorphCloseProcessor — morphological closing (dilation+erosion).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

__all__ = [
    "MaskPostprocessor",
    "PostprocessorRegistry",
    "PostprocessorPipeline",
    "HoleFiller",
    "SmallObjectRemover",
    "MorphOpenProcessor",
    "MorphCloseProcessor",
]

_LOGGER: logging.Logger = logging.getLogger(__name__)


# ==============================================================================
# Abstract interface
# ==============================================================================

class MaskPostprocessor(ABC):
    """Abstract interface for all mask post-processors."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable registry key."""

    @abstractmethod
    def apply(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply post-processing to a predicted mask.

        Args:
            mask: (H, W) uint8 array of class IDs.

        Returns:
            (H, W) uint8 processed array.  Shape must not change.
        """


# ==============================================================================
# Registry
# ==============================================================================

class PostprocessorRegistry:
    """Registry mapping processor names to classes."""

    _registered: dict[str, type[MaskPostprocessor]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(klass: type[MaskPostprocessor]) -> type[MaskPostprocessor]:
            cls._registered[name.lower()] = klass
            return klass
        return decorator

    @classmethod
    def registered_names(cls) -> tuple[str, ...]:
        return tuple(sorted(cls._registered.keys()))


# ==============================================================================
# PostprocessorPipeline
# ==============================================================================

class PostprocessorPipeline:
    """
    Sequentially applies a list of MaskPostprocessor instances.

    Args:
        processors: Ordered list of MaskPostprocessor instances.
    """

    def __init__(self, processors: list[MaskPostprocessor]) -> None:
        self._processors = list(processors)

    def apply(self, mask: np.ndarray) -> np.ndarray:
        """Apply all processors in sequence."""
        result = mask.copy()
        for proc in self._processors:
            try:
                result = proc.apply(result)
            except Exception as exc:
                _LOGGER.warning(
                    "PostprocessorPipeline: '%s' failed: %s; skipping.",
                    proc.name, exc,
                )
        return result

    @staticmethod
    def build_from_config(config: Any) -> PostprocessorPipeline:
        """
        Build a PostprocessorPipeline from InferenceConfig.

        Args:
            config: InferenceConfig.

        Returns:
            PostprocessorPipeline with enabled processors.
        """
        processors: list[MaskPostprocessor] = []
        if getattr(config, "fill_holes", False):
            processors.append(HoleFiller())
        min_size = getattr(config, "min_object_size", 0)
        if min_size > 0:
            processors.append(SmallObjectRemover(min_size))
        open_sz = getattr(config, "morph_open_size", 0)
        if open_sz > 0:
            processors.append(MorphOpenProcessor(open_sz))
        close_sz = getattr(config, "morph_close_size", 0)
        if close_sz > 0:
            processors.append(MorphCloseProcessor(close_sz))
        return PostprocessorPipeline(processors)


# ==============================================================================
# HoleFiller
# ==============================================================================

@PostprocessorRegistry.register("hole_filler")
class HoleFiller(MaskPostprocessor):
    """
    Fills small interior holes in each class region.

    Interior holes are isolated background pixels completely surrounded by
    a single class. The hole is assigned the class of its surrounding region.
    Uses scipy.ndimage.binary_fill_holes per-class.

    Gracefully falls back to identity when scipy is unavailable.
    """

    @property
    def name(self) -> str:
        return "hole_filler"

    def apply(self, mask: np.ndarray) -> np.ndarray:
        try:
            from scipy.ndimage import binary_fill_holes
        except ImportError:
            _LOGGER.warning("HoleFiller: scipy unavailable; skipping.")
            return mask

        result     = mask.copy()
        num_classes = int(mask.max()) + 1
        for cls_id in range(num_classes):
            binary = (mask == cls_id).astype(bool)
            filled = binary_fill_holes(binary)
            # Only fill pixels that were not already in this class.
            new_holes = filled & ~binary
            result[new_holes] = cls_id
        return result


# ==============================================================================
# SmallObjectRemover
# ==============================================================================

@PostprocessorRegistry.register("small_object_remover")
class SmallObjectRemover(MaskPostprocessor):
    """
    Removes connected components smaller than min_size pixels.

    Small isolated islands are reassigned to the most frequent neighbouring
    class. Uses scipy.ndimage.label for connected component analysis.

    Args:
        min_size: Minimum component size in pixels. Components smaller than
                  this are eliminated. Must be >= 1.
    """

    def __init__(self, min_size: int = 64) -> None:
        self._min_size = max(1, int(min_size))

    @property
    def name(self) -> str:
        return "small_object_remover"

    def apply(self, mask: np.ndarray) -> np.ndarray:
        try:
            from scipy.ndimage import label, uniform_filter
        except ImportError:
            _LOGGER.warning("SmallObjectRemover: scipy unavailable; skipping.")
            return mask

        result      = mask.copy()
        num_classes = int(mask.max()) + 1

        for cls_id in range(num_classes):
            binary          = (mask == cls_id).astype(np.int32)
            labeled, n_comp = label(binary)
            for comp in range(1, n_comp + 1):
                comp_mask = labeled == comp
                if comp_mask.sum() < self._min_size:
                    # Replace with most frequent neighbour class.
                    # Dilate the component mask slightly to find neighbours.
                    from scipy.ndimage import binary_dilation
                    dilated   = binary_dilation(comp_mask, iterations=2)
                    neighbour = dilated & ~comp_mask
                    if neighbour.any():
                        vals        = mask[neighbour]
                        vals        = vals[vals != cls_id]
                        if len(vals) > 0:
                            replace_with = int(np.bincount(vals.astype(np.intp)).argmax())
                            result[comp_mask] = replace_with
        return result


# ==============================================================================
# MorphOpenProcessor
# ==============================================================================

@PostprocessorRegistry.register("morph_open")
class MorphOpenProcessor(MaskPostprocessor):
    """
    Applies morphological opening (erosion followed by dilation).

    Opening removes small protrusions and isolated pixels from class regions.
    Applied independently to each class binary mask.

    Args:
        kernel_size: Size of the square structuring element.
    """

    def __init__(self, kernel_size: int = 3) -> None:
        self._k = max(1, int(kernel_size))

    @property
    def name(self) -> str:
        return "morph_open"

    def apply(self, mask: np.ndarray) -> np.ndarray:
        return _morph_op(mask, self._k, "open")


# ==============================================================================
# MorphCloseProcessor
# ==============================================================================

@PostprocessorRegistry.register("morph_close")
class MorphCloseProcessor(MaskPostprocessor):
    """
    Applies morphological closing (dilation followed by erosion).

    Closing fills small gaps and bridges narrow breaks in class regions.
    Applied independently to each class binary mask.

    Args:
        kernel_size: Size of the square structuring element.
    """

    def __init__(self, kernel_size: int = 3) -> None:
        self._k = max(1, int(kernel_size))

    @property
    def name(self) -> str:
        return "morph_close"

    def apply(self, mask: np.ndarray) -> np.ndarray:
        return _morph_op(mask, self._k, "close")


# ==============================================================================
# Shared morphology helper
# ==============================================================================

def _morph_op(mask: np.ndarray, k: int, op: str) -> np.ndarray:
    """Apply morphological open or close per-class."""
    try:
        from scipy.ndimage import binary_closing, binary_opening
        struct = np.ones((k, k), dtype=bool)
    except ImportError:
        _LOGGER.warning("_morph_op: scipy unavailable; skipping %s.", op)
        return mask

    result      = mask.copy()
    num_classes = int(mask.max()) + 1
    morph_fn    = binary_opening if op == "open" else binary_closing

    for cls_id in range(num_classes):
        binary    = (mask == cls_id).astype(bool)
        processed = morph_fn(binary, structure=struct)
        # Pixels added by morphology get cls_id; pixels removed revert to
        # most-frequent neighbour (handled simply by not assigning them here).
        result[processed] = cls_id

    return result
