"""
QA_PIXEL bit-masking for Landsat Collection 2 Level-2 imagery.

Landsat Collection 2 QA_PIXEL bit layout (USGS specification):
    Bit 0  -- Fill (1 = fill data, not a real observation)
    Bit 1  -- Dilated Cloud (expanded cloud buffer)
    Bit 2  -- Cirrus (high-confidence cirrus)
    Bit 3  -- Cloud (high-confidence cloud)
    Bit 4  -- Cloud Shadow
    Bit 5  -- Snow / Ice
    Bit 6  -- Clear (1 = clear observation)
    Bit 7  -- Water

Masking strategy:
    For each enabled bad-quality bit, build a per-pixel boolean flag
    (1 where the bit is set). Combine all flags with bitwise OR to get
    the union "bad pixel" mask. Invert to produce the "good pixel" mask.
    Apply with image.updateMask(), which sets masked pixels to
    nodata -- EE computations ignore masked pixels automatically.

All operations are lazy server-side EE computations. No getInfo() calls
are made in this module. The 'ee' package is imported lazily inside
each method body to support testing without earthengine-api installed.

Usage:

    from src.gee.masking import LandsatQAMasker, QAMaskConfig

    mask_config = QAMaskConfig(mask_cloud=True, mask_cloud_shadow=True)
    masker      = LandsatQAMasker(mask_config)

    masked_collection = masker.apply_to_collection(collection)
    masked_image      = masker.apply_to_image(image)

    # Or build from Config directly:
    masker = LandsatQAMasker.from_preprocessing_config(config)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.core.exceptions import InvalidValueError
from src.gee import GEEAPIError

if TYPE_CHECKING:
    from src.core.config import Config

__all__ = [
    "QAMaskConfig",
    "LandsatQAMasker",
    "QA_BIT_FILL",
    "QA_BIT_DILATED_CLOUD",
    "QA_BIT_CIRRUS",
    "QA_BIT_CLOUD",
    "QA_BIT_CLOUD_SHADOW",
    "QA_BIT_SNOW",
    "QA_BIT_CLEAR",
    "QA_BIT_WATER",
    "QA_PIXEL_BAND",
]

_LOGGER: logging.Logger = logging.getLogger(__name__)

# QA_PIXEL bit positions (Landsat Collection 2 specification).
# These are constants, not configuration, because the USGS specification
# does not vary across Collection 2 sensors.
QA_BIT_FILL:         int = 0
QA_BIT_DILATED_CLOUD: int = 1
QA_BIT_CIRRUS:       int = 2
QA_BIT_CLOUD:        int = 3
QA_BIT_CLOUD_SHADOW: int = 4
QA_BIT_SNOW:         int = 5
QA_BIT_CLEAR:        int = 6
QA_BIT_WATER:        int = 7

QA_PIXEL_BAND: str = "QA_PIXEL"


# ==============================================================================
# QAMaskConfig
# ==============================================================================

@dataclass(frozen=True)
class QAMaskConfig:
    """
    Immutable configuration specifying which QA_PIXEL bits to mask.

    Each flag controls whether the corresponding bad-quality condition
    causes a pixel to be masked (set to nodata) in the output image.
    When a flag is True, pixels with that condition are excluded.

    Attributes:
        mask_fill:          Exclude fill (no-data) pixels.
        mask_dilated_cloud: Exclude pixels in dilated cloud buffer.
        mask_cirrus:        Exclude high-confidence cirrus pixels.
        mask_cloud:         Exclude high-confidence cloud pixels.
        mask_cloud_shadow:  Exclude cloud shadow pixels.
        mask_snow:          Exclude snow/ice pixels. Disabled by default
                            because snow can be confused with sandbars in
                            dry river bed context.
    """

    mask_fill:          bool = True
    mask_dilated_cloud: bool = True
    mask_cirrus:        bool = True
    mask_cloud:         bool = True
    mask_cloud_shadow:  bool = True
    mask_snow:          bool = False

    def any_enabled(self) -> bool:
        """True if at least one masking flag is enabled."""
        return any([
            self.mask_fill,
            self.mask_dilated_cloud,
            self.mask_cirrus,
            self.mask_cloud,
            self.mask_cloud_shadow,
            self.mask_snow,
        ])

    def summary(self) -> str:
        """Return an ASCII summary of active masking flags."""
        flags = {
            "fill":          self.mask_fill,
            "dilated_cloud": self.mask_dilated_cloud,
            "cirrus":        self.mask_cirrus,
            "cloud":         self.mask_cloud,
            "cloud_shadow":  self.mask_cloud_shadow,
            "snow":          self.mask_snow,
        }
        active = [name for name, enabled in flags.items() if enabled]
        return f"QAMaskConfig(active=[{', '.join(active)}])"


# ==============================================================================
# LandsatQAMasker
# ==============================================================================

class LandsatQAMasker:
    """
    Applies QA_PIXEL bit masking to Landsat Collection 2 Level-2 imagery.

    Masks are applied via collection.map() keeping all operations
    server-side. The 'ee' package is imported lazily inside each
    method; this class is fully importable without earthengine-api.

    Args:
        mask_config: QAMaskConfig controlling which quality conditions
                     cause pixels to be excluded.
    """

    def __init__(self, mask_config: QAMaskConfig) -> None:
        self._mask_config = mask_config
        self._logger: logging.Logger = logging.getLogger(__name__)
        self._logger.debug("LandsatQAMasker created. %s", mask_config.summary())

    @classmethod
    def from_preprocessing_config(cls, config: Config) -> LandsatQAMasker:
        """
        Construct a LandsatQAMasker from a Config object.

        Reads masking flags from config.preprocessing. Each flag defaults
        to the QAMaskConfig class default if absent from config.yaml.

        Args:
            config: Fully initialized Config object.

        Returns:
            LandsatQAMasker with flags sourced from config.preprocessing.
        """
        pp = config.preprocessing
        mask_config = QAMaskConfig(
            mask_fill=bool(getattr(pp, "mask_fill", True)),
            mask_dilated_cloud=bool(getattr(pp, "mask_dilated_cloud", True)),
            mask_cirrus=bool(getattr(pp, "mask_cirrus", True)),
            mask_cloud=bool(getattr(pp, "mask_cloud", True)),
            mask_cloud_shadow=bool(getattr(pp, "mask_cloud_shadow", True)),
            mask_snow=bool(getattr(pp, "mask_snow", False)),
        )
        return cls(mask_config)

    @property
    def mask_config(self) -> QAMaskConfig:
        """The active masking configuration."""
        return self._mask_config

    def apply_to_collection(self, collection: Any) -> Any:
        """
        Apply QA masking to every image in a collection via .map().

        If no masking flags are enabled (all False), the collection is
        returned unchanged without creating any unnecessary EE computation.

        Args:
            collection: An ee.ImageCollection to mask.

        Returns:
            The collection with QA masking applied to each image.

        Raises:
            GEEAPIError: collection.map() call failed.
        """
        if not self._mask_config.any_enabled():
            self._logger.debug(
                "No QA masking flags are enabled. "
                "Returning collection unchanged."
            )
            return collection

        self._logger.debug(
            "Applying QA masking to collection. %s",
            self._mask_config.summary(),
        )

        try:
            return collection.map(self.apply_to_image)
        except Exception as exc:
            raise GEEAPIError(
                operation="apply_qa_masking_to_collection",
                reason=f"collection.map() for QA masking failed: {exc}",
            ) from exc

    def apply_to_image(self, image: Any) -> Any:
        """
        Apply QA masking to a single image.

        This method is suitable for use as a function argument to
        collection.map(). It builds the combined quality mask and
        calls image.updateMask() with the inverted bad-pixel flag.

        Args:
            image: An ee.Image with a QA_PIXEL band.

        Returns:
            The image with masked pixels set to nodata.

        Raises:
            GEEAPIError: The masking computation raised an EE exception.
        """
        try:
            good_pixel_mask = self._build_qa_mask(image)
            return image.updateMask(good_pixel_mask)
        except Exception as exc:
            raise GEEAPIError(
                operation="apply_qa_masking_to_image",
                reason=f"QA mask construction or updateMask() failed: {exc}",
            ) from exc

    def _build_qa_mask(self, image: Any) -> Any:
        """
        Build a boolean mask where 1 = valid pixel, 0 = bad pixel.

        Constructs one flag per enabled masking condition using bitwise
        operations on the QA_PIXEL band. Combines all flags with
        bitwise OR to produce the "any bad condition" mask, then inverts
        to produce the "all conditions good" mask.

        The EE expression chain is:
            qa_band.bitwiseAnd(1 << bit_position).neq(0)
        which yields 1 wherever the specified bit is set.

        Args:
            image: An ee.Image with a QA_PIXEL band.

        Returns:
            An ee.Image with values 1 (valid) or 0 (masked).
        """
        try:
            import ee
        except ImportError as exc:
            raise GEEAPIError(
                operation="_build_qa_mask",
                reason="earthengine-api is not installed.",
            ) from exc

        qa = image.select(QA_PIXEL_BAND)

        # Start with all pixels valid (no bad conditions flagged yet).
        bad_pixel_mask = ee.Image.constant(0)

        if self._mask_config.mask_fill:
            bad_pixel_mask = bad_pixel_mask.Or(
                self._bit_flag(qa, QA_BIT_FILL)
            )
        if self._mask_config.mask_dilated_cloud:
            bad_pixel_mask = bad_pixel_mask.Or(
                self._bit_flag(qa, QA_BIT_DILATED_CLOUD)
            )
        if self._mask_config.mask_cirrus:
            bad_pixel_mask = bad_pixel_mask.Or(
                self._bit_flag(qa, QA_BIT_CIRRUS)
            )
        if self._mask_config.mask_cloud:
            bad_pixel_mask = bad_pixel_mask.Or(
                self._bit_flag(qa, QA_BIT_CLOUD)
            )
        if self._mask_config.mask_cloud_shadow:
            bad_pixel_mask = bad_pixel_mask.Or(
                self._bit_flag(qa, QA_BIT_CLOUD_SHADOW)
            )
        if self._mask_config.mask_snow:
            bad_pixel_mask = bad_pixel_mask.Or(
                self._bit_flag(qa, QA_BIT_SNOW)
            )

        # Invert: 1 where pixel is good, 0 where pixel is bad.
        return bad_pixel_mask.Not()

    @staticmethod
    def _bit_flag(qa_band: Any, bit_position: int) -> Any:
        """
        Return an ee.Image that is 1 wherever the specified QA bit is set.

        Args:
            qa_band:      An ee.Image containing the QA_PIXEL band.
            bit_position: Zero-indexed bit position (0–7 for QA_PIXEL).

        Returns:
            ee.Image with values 0 or 1.
        """
        return qa_band.bitwiseAnd(1 << bit_position).neq(0)