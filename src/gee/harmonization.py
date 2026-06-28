"""
Band-name harmonization for mixed Landsat sensor ImageCollections.

Problem:
    Landsat 5 TM and Landsat 7 ETM+ use band numbers SR_B1 through SR_B7
    (with SR_B6 absent, used instead for thermal as ST_B6).
    Landsat 8 OLI and Landsat 9 OLI-2 use SR_B2 through SR_B7 for optical
    and ST_B10 for thermal.
    A merged multi-sensor collection contains images with DIFFERENT band
    names for the same physical wavelength (e.g., SR_B1 = Blue for L5/L7
    but SR_B2 = Blue for L8/L9).

Solution:
    Rename all bands to a common schema using ee.Algorithms.If() for
    server-side conditional logic based on each image's SPACECRAFT_ID
    property. This avoids any getInfo() call and works uniformly across
    single-sensor and mixed-sensor collections.

Common band schema (output):
    Band Name    Wavelength        L5/L7 source    L8/L9 source
    Blue         0.45-0.51 um      SR_B1           SR_B2
    Green        0.53-0.59 um      SR_B2           SR_B3
    Red          0.64-0.67 um      SR_B3           SR_B4
    NIR          0.85-0.88 um      SR_B4           SR_B5
    SWIR1        1.57-1.65 um      SR_B5           SR_B6
    SWIR2        2.11-2.29 um      SR_B7           SR_B7
    Thermal      10.6-12.5 um      ST_B6           ST_B10
    QA_PIXEL     QA band           QA_PIXEL         QA_PIXEL

Usage:

    from src.gee.harmonization import BandHarmonizer

    harmonizer = BandHarmonizer()
    harmonized_collection = harmonizer.harmonize_collection(collection)

    # Or for a single image (specify sensor explicitly):
    renamed = harmonizer.rename_oli_image(image)
    renamed = harmonizer.rename_tm_etm_image(image)
"""

from __future__ import annotations

import logging
from typing import Any

from src.gee import GEEAPIError, GEENotInstalledError

__all__ = [
    "BandHarmonizer",
    "COMMON_BAND_NAMES",
    "OLI_SOURCE_BANDS",
    "TM_ETM_SOURCE_BANDS",
    "OLI_SPACECRAFT_IDS",
    "TM_ETM_SPACECRAFT_IDS",
]

_LOGGER: logging.Logger = logging.getLogger(__name__)

# Common output band schema applied after harmonization.
# All future modules (Module 6+) can rely on these names.
COMMON_BAND_NAMES: tuple[str, ...] = (
    "Blue",
    "Green",
    "Red",
    "NIR",
    "SWIR1",
    "SWIR2",
    "Thermal",
    "QA_PIXEL",
)

# Optical bands used in medoid/composite distance calculations.
OPTICAL_BAND_NAMES: tuple[str, ...] = (
    "Blue",
    "Green",
    "Red",
    "NIR",
    "SWIR1",
    "SWIR2",
)

# Source band names per sensor family, in the same order as COMMON_BAND_NAMES.
OLI_SOURCE_BANDS: tuple[str, ...] = (
    "SR_B2",    # Blue
    "SR_B3",    # Green
    "SR_B4",    # Red
    "SR_B5",    # NIR
    "SR_B6",    # SWIR1
    "SR_B7",    # SWIR2
    "ST_B10",   # Thermal
    "QA_PIXEL", # QA
)

TM_ETM_SOURCE_BANDS: tuple[str, ...] = (
    "SR_B1",    # Blue
    "SR_B2",    # Green
    "SR_B3",    # Red
    "SR_B4",    # NIR
    "SR_B5",    # SWIR1
    "SR_B7",    # SWIR2
    "ST_B6",    # Thermal
    "QA_PIXEL", # QA
)

# SPACECRAFT_ID property values per sensor family.
OLI_SPACECRAFT_IDS: tuple[str, ...] = ("LANDSAT_8", "LANDSAT_9")
TM_ETM_SPACECRAFT_IDS: tuple[str, ...] = ("LANDSAT_5", "LANDSAT_7")


# ==============================================================================
# BandHarmonizer
# ==============================================================================

class BandHarmonizer:
    """
    Renames Landsat bands to the common schema defined in COMMON_BAND_NAMES.

    Supports mixed-sensor collections by using ee.Algorithms.If() to apply
    the correct rename mapping per image based on the SPACECRAFT_ID property.
    All logic is lazy and server-side; no getInfo() calls are made.

    The harmonizer is stateless and reusable across multiple collections.
    """

    def __init__(self) -> None:
        self._logger: logging.Logger = logging.getLogger(__name__)

    def harmonize_collection(self, collection: Any) -> Any:
        """
        Apply band harmonization to every image in a collection via .map().

        Each image's SPACECRAFT_ID property is evaluated server-side to
        select the appropriate rename mapping. Works correctly for both
        single-sensor and mixed-sensor collections.

        Args:
            collection: An ee.ImageCollection (may contain L5/L7/L8/L9 images).

        Returns:
            An ee.ImageCollection where every image uses COMMON_BAND_NAMES.

        Raises:
            GEENotInstalledError: earthengine-api is not installed.
            GEEAPIError:          collection.map() failed.
        """
        self._logger.debug(
            "Harmonizing collection bands to schema: %s",
            list(COMMON_BAND_NAMES),
        )

        try:
            harmonize_fn = self._make_harmonize_function()
            return collection.map(harmonize_fn)
        except GEENotInstalledError:
            # Preserve the original exception.
            raise

        except Exception as exc:
            raise GEEAPIError(
                operation="harmonize_collection",
                reason=f"collection.map() for harmonization failed: {exc}",
            ) from exc

    def harmonize_image(self, image: Any) -> Any:
        """
        Apply band harmonization to a single image using server-side logic.

        Uses ee.Algorithms.If() to select the rename mapping based on the
        SPACECRAFT_ID property. Safe to use on images from any Landsat
        Collection 2 Level-2 sensor.

        Args:
            image: An ee.Image from a Landsat C2 L2 collection.

        Returns:
            The same image with bands renamed to COMMON_BAND_NAMES.

        Raises:
            GEENotInstalledError: earthengine-api is not installed.
            GEEAPIError:          The rename operation failed.
        """
        try:
            harmonize_fn = self._make_harmonize_function()
            return harmonize_fn(image)
        except Exception as exc:
            raise GEEAPIError(
                operation="harmonize_image",
                reason=f"Server-side harmonization failed: {exc}",
            ) from exc

    def rename_oli_image(self, image: Any) -> Any:
        """
        Rename Landsat 8/9 OLI/OLI-2 bands to the common schema.

        Use this method when you know the image is from Landsat 8 or 9
        (e.g., in a single-sensor collection). For mixed collections,
        use harmonize_collection() which handles the sensor detection
        automatically server-side.

        Args:
            image: An ee.Image from a Landsat 8 or 9 C2 L2 collection.

        Returns:
            The image with OLI_SOURCE_BANDS renamed to COMMON_BAND_NAMES.

        Raises:
            GEEAPIError: band selection or rename failed.
        """
        try:
            return image.select(
                list(OLI_SOURCE_BANDS)
            ).rename(list(COMMON_BAND_NAMES))
        except Exception as exc:
            raise GEEAPIError(
                operation="rename_oli_image",
                reason=f"OLI band selection/rename failed: {exc}",
            ) from exc

    def rename_tm_etm_image(self, image: Any) -> Any:
        """
        Rename Landsat 5/7 TM/ETM+ bands to the common schema.

        Use this method when you know the image is from Landsat 5 or 7.
        For mixed collections, use harmonize_collection().

        Args:
            image: An ee.Image from a Landsat 5 or 7 C2 L2 collection.

        Returns:
            The image with TM_ETM_SOURCE_BANDS renamed to COMMON_BAND_NAMES.

        Raises:
            GEEAPIError: band selection or rename failed.
        """
        try:
            return image.select(
                list(TM_ETM_SOURCE_BANDS)
            ).rename(list(COMMON_BAND_NAMES))
        except Exception as exc:
            raise GEEAPIError(
                operation="rename_tm_etm_image",
                reason=f"TM/ETM+ band selection/rename failed: {exc}",
            ) from exc

    def _make_harmonize_function(self) -> Any:
        """
        Return a closure for use with collection.map() that harmonizes bands.

        The closure captures no Python-level conditional logic. All branching
        on spacecraft type is delegated to ee.Algorithms.If() which evaluates
        server-side per image. The Python closure just captures the band
        name lists as constants.

        Returns:
            A callable (image) -> ee.Image suitable for collection.map().

        Raises:
            GEENotInstalledError: earthengine-api is not installed.
        """
        try:
            import ee
        except ImportError as exc:
            raise GEENotInstalledError(
                "earthengine-api is not installed. "
                "Install with: pip install earthengine-api==0.1.390"
            ) from exc

        oli_sources  = list(OLI_SOURCE_BANDS)
        tm_sources   = list(TM_ETM_SOURCE_BANDS)
        common_names = list(COMMON_BAND_NAMES)
        oli_ids      = list(OLI_SPACECRAFT_IDS)

        def _harmonize(image: Any) -> Any:
            spacecraft = ee.String(image.get("SPACECRAFT_ID"))

            # Build is_oli condition: SPACECRAFT_ID is LANDSAT_8 or LANDSAT_9.
            is_l8  = spacecraft.equals(oli_ids[0])
            is_l9  = spacecraft.equals(oli_ids[1])
            is_oli = is_l8.Or(is_l9)

            # Two possible outcomes evaluated lazily server-side.
            oli_renamed = (
                image.select(oli_sources).rename(common_names)
            )
            tm_renamed = (
                image.select(tm_sources).rename(common_names)
            )

            # ee.Algorithms.If chooses server-side without a Python branch.
            return ee.Image(
                ee.Algorithms.If(is_oli, oli_renamed, tm_renamed)
            )

        return _harmonize