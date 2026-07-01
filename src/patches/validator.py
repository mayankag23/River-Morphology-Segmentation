"""
Patch validity checking for the River Morphology Patch Generation pipeline.

PatchValidator computes the fraction of valid (non-NoData) pixels in a
patch and compares it against a configurable minimum threshold. Patches
below the threshold are excluded from the generated dataset.

A pixel is considered invalid if ANY band at that pixel position is NaN
or equals the configured NoData sentinel value. This conservative policy
treats spectral index bands (derived from optical bands) the same way as
the source optical bands themselves -- a NaN in any one band means the
underlying observation was unreliable at that location.

Pure numpy computation. No I/O.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["PatchValidationResult", "PatchValidator"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PatchValidationResult:
    """
    Immutable result of PatchValidator.validate().

    Attributes:
        is_valid:          True if valid_pixel_ratio >= min_valid_pixel_ratio.
        valid_pixel_ratio: Fraction of pixels considered valid, in [0.0, 1.0].
        total_pixels:      Total pixel count per band (height * width).
        valid_pixels:      Count of pixels considered valid (valid in all bands).
    """

    is_valid:          bool
    valid_pixel_ratio: float
    total_pixels:      int
    valid_pixels:      int


class PatchValidator:
    """
    Computes patch validity based on a NoData / NaN pixel ratio threshold.

    Args:
        nodata_value:           Sentinel value treated as invalid data, in
                                addition to NaN. Sourced from
                                config.patch_generation.nodata_value.
        min_valid_pixel_ratio:  Minimum fraction of valid pixels in [0.0, 1.0]
                                required for a patch to be accepted.
                                Sourced from
                                config.patch_generation.min_valid_pixel_ratio.
    """

    def __init__(
        self,
        nodata_value:           float,
        min_valid_pixel_ratio:  float,
    ) -> None:
        self._nodata_value          = float(nodata_value)
        self._min_valid_pixel_ratio = float(min_valid_pixel_ratio)
        self._logger: logging.Logger = logging.getLogger(__name__)

    @property
    def nodata_value(self) -> float:
        """The configured NoData sentinel value."""
        return self._nodata_value

    @property
    def min_valid_pixel_ratio(self) -> float:
        """The configured minimum valid pixel ratio threshold."""
        return self._min_valid_pixel_ratio

    def validate(self, data: Any) -> PatchValidationResult:
        """
        Compute the valid pixel ratio for a patch and compare against threshold.

        A pixel is invalid if it is NaN or equals nodata_value in ANY band.
        The check is applied across the band axis (axis 0).

        Args:
            data: float32 numpy array, shape (bands, height, width).

        Returns:
            PatchValidationResult with is_valid flag and ratio statistics.
        """
        _, height, width = data.shape
        total_pixels = height * width

        is_nan    = np.isnan(data)
        is_nodata = np.isclose(data, self._nodata_value, equal_nan=False)
        invalid_per_band = is_nan | is_nodata

        # A pixel is invalid if it is invalid in ANY band.
        invalid_pixel_mask = invalid_per_band.any(axis=0)
        valid_pixels = int(total_pixels - int(invalid_pixel_mask.sum()))

        ratio    = valid_pixels / total_pixels if total_pixels > 0 else 0.0
        is_valid = ratio >= self._min_valid_pixel_ratio

        return PatchValidationResult(
            is_valid=is_valid,
            valid_pixel_ratio=ratio,
            total_pixels=total_pixels,
            valid_pixels=valid_pixels,
        )