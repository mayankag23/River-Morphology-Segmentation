"""
Normalization transforms for the Data Transformation and Augmentation Pipeline
(Module 12).

Design
------
NormalizationTransform applies per-band (channel-wise) standardization:

    output[c] = (input[c] - mean[c]) / std[c]

This matches standard practice for multispectral imagery where different
spectral bands have very different dynamic ranges.

Three normalization sources are supported in order of preference:
    1. Externally supplied statistics (passed directly to the constructor).
    2. Per-split statistics computed from the training data (via statistics.py).
    3. Config-supplied mean/std lists (config.training.normalization.*).

NormalizationTransform is the ONLY radiometric transform that modifies the
relationship between bands.  It must always be the first transform in the
composed pipeline so that all subsequent radiometric adjustments (brightness,
contrast, noise) operate in a consistent normalized space.

Masks are NEVER touched by this transform.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.training.contracts import NormalizationStatistics, TransformSample
from src.training.transform import SegmentationTransform

__all__ = ["NormalizationTransform"]

_LOGGER: logging.Logger = logging.getLogger(__name__)

_EPS: float = 1e-8


class NormalizationTransform(SegmentationTransform):
    """
    Per-band standardization: output = (image - mean) / std.

    All per-band operations are vectorized; no per-pixel Python loops.

    Args:
        stats: NormalizationStatistics with per-band mean and std.
               All std values must be > 0 (guaranteed by NormalizationStatistics
               construction; zeros are replaced with 1.0).
    """

    _NAME: str = "normalization"

    def __init__(self, stats: NormalizationStatistics) -> None:
        if not isinstance(stats, NormalizationStatistics):
            from src.core.exceptions import InvalidValueError
            raise InvalidValueError(
                field="normalization.stats",
                value=type(stats).__name__,
                reason="must be a NormalizationStatistics instance",
            )
        self._stats = stats
        mean_arr, std_arr = stats.as_numpy()
        # Reshape to (C, 1, 1) for broadcasting over (C, H, W) images.
        self._mean = mean_arr.reshape(-1, 1, 1).astype(np.float32)
        self._std  = np.maximum(std_arr, _EPS).reshape(-1, 1, 1).astype(np.float32)

    @property
    def name(self) -> str:
        return self._NAME

    @property
    def stats(self) -> NormalizationStatistics:
        """The NormalizationStatistics this transform applies."""
        return self._stats

    def apply(self, sample: TransformSample) -> TransformSample:
        """
        Normalize the image in-place (modifies sample.image array).

        If the image has more bands than the stats object, only the first
        stats.num_bands bands are normalized and a WARNING is logged.

        Args:
            sample: TransformSample with image (C, H, W) float32.

        Returns:
            Same TransformSample with normalized image.  Mask unchanged.
        """
        c = sample.image.shape[0]
        expected_c = self._mean.shape[0]

        if c != expected_c:
            _LOGGER.warning(
                "NormalizationTransform: image has %d bands but stats have "
                "%d bands; only first %d bands are normalized.",
                c, expected_c, min(c, expected_c),
            )
            n = min(c, expected_c)
            sample.image[:n] = (
                (sample.image[:n] - self._mean[:n]) / self._std[:n]
            ).astype(np.float32)
        else:
            sample.image = (
                (sample.image - self._mean) / self._std
            ).astype(np.float32)

        return sample

    def denormalize(self, image: np.ndarray) -> np.ndarray:
        """
        Reverse the normalization: output = image * std + mean.

        Useful for visualization and inference post-processing.

        Args:
            image: (C, H, W) normalized float32 array.

        Returns:
            Denormalized float32 array in original band value units.
        """
        return (image * self._std + self._mean).astype(np.float32)
