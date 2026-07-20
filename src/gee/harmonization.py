# """
# Band-name harmonization for mixed Landsat sensor ImageCollections.

# Problem:
#     Landsat 5 TM and Landsat 7 ETM+ use band numbers SR_B1 through SR_B7
#     (with SR_B6 absent, used instead for thermal as ST_B6).
#     Landsat 8 OLI and Landsat 9 OLI-2 use SR_B2 through SR_B7 for optical
#     and ST_B10 for thermal.
#     A merged multi-sensor collection contains images with DIFFERENT band
#     names for the same physical wavelength (e.g., SR_B1 = Blue for L5/L7
#     but SR_B2 = Blue for L8/L9).

# Solution:
#     Rename all bands to a common schema using ee.Algorithms.If() for
#     server-side conditional logic based on each image's SPACECRAFT_ID
#     property. This avoids any getInfo() call and works uniformly across
#     single-sensor and mixed-sensor collections.

# Common band schema (output):
#     Band Name    Wavelength        L5/L7 source    L8/L9 source
#     Blue         0.45-0.51 um      SR_B1           SR_B2
#     Green        0.53-0.59 um      SR_B2           SR_B3
#     Red          0.64-0.67 um      SR_B3           SR_B4
#     NIR          0.85-0.88 um      SR_B4           SR_B5
#     SWIR1        1.57-1.65 um      SR_B5           SR_B6
#     SWIR2        2.11-2.29 um      SR_B7           SR_B7
#     Thermal      10.6-12.5 um      ST_B6           ST_B10
#     QA_PIXEL     QA band           QA_PIXEL         QA_PIXEL

# Usage:

#     from src.gee.harmonization import BandHarmonizer

#     harmonizer = BandHarmonizer()
#     harmonized_collection = harmonizer.harmonize_collection(collection)

#     # Or for a single image (specify sensor explicitly):
#     renamed = harmonizer.rename_oli_image(image)
#     renamed = harmonizer.rename_tm_etm_image(image)
# """

# from __future__ import annotations

# import logging
# from typing import Any

# from src.gee import GEEAPIError, GEENotInstalledError

# __all__ = [
#     "BandHarmonizer",
#     "COMMON_BAND_NAMES",
#     "OLI_SOURCE_BANDS",
#     "TM_ETM_SOURCE_BANDS",
#     "OLI_SPACECRAFT_IDS",
#     "TM_ETM_SPACECRAFT_IDS",
# ]

# _LOGGER: logging.Logger = logging.getLogger(__name__)

# Common output band schema applied after harmonization.
# All future modules (Module 6+) can rely on these names.

"""
Band Harmonization

Supports

    • Landsat 5
    • Landsat 7
    • Landsat 8
    • Landsat 9
    • Sentinel-2 SR Harmonized

All sensors are converted into one common schema so the remaining
pipeline (features, composite, training, inference, etc.) never
needs to know which satellite produced the image.
"""

from __future__ import annotations

import logging
from typing import Any

from src.gee import (
    GEEAPIError,
    GEENotInstalledError,
)

__all__ = [
    "BandHarmonizer",
    "COMMON_BAND_NAMES",
    "OPTICAL_BAND_NAMES",
    "OLI_SOURCE_BANDS",
    "TM_ETM_SOURCE_BANDS",
    "S2_SOURCE_BANDS",
]

_LOGGER = logging.getLogger(__name__)


# COMMON_BAND_NAMES: tuple[str, ...] = (
#     "Blue",
#     "Green",
#     "Red",
#     "NIR",
#     "SWIR1",
#     "SWIR2",
#     "Thermal",
#     "QA_PIXEL",
# )

# # Optical bands used in medoid/composite distance calculations.
# OPTICAL_BAND_NAMES: tuple[str, ...] = (
#     "Blue",
#     "Green",
#     "Red",
#     "NIR",
#     "SWIR1",
#     "SWIR2",
# )


# ============================================================
# Common harmonized band names
# ============================================================

COMMON_BAND_NAMES = (
    "Blue",
    "Green",
    "Red",
    "NIR",
    "SWIR1",
    "SWIR2",

    #
    # Sentinel has no thermal band.
    # We create a dummy thermal image later so downstream
    # modules do not need changing.
    #
    "Thermal",

    #
    # Generic QA band.
    #
    "QA",
)

OPTICAL_BAND_NAMES = (
    "Blue",
    "Green",
    "Red",
    "NIR",
    "SWIR1",
    "SWIR2",
)

# # Source band names per sensor family, in the same order as COMMON_BAND_NAMES.
# OLI_SOURCE_BANDS: tuple[str, ...] = (
#     "SR_B2",    # Blue
#     "SR_B3",    # Green
#     "SR_B4",    # Red
#     "SR_B5",    # NIR
#     "SR_B6",    # SWIR1
#     "SR_B7",    # SWIR2
#     "ST_B10",   # Thermal
#     "QA_PIXEL", # QA
# )


# ============================================================
# Landsat 8 / 9
# ============================================================

OLI_SOURCE_BANDS = (
    "SR_B2",
    "SR_B3",
    "SR_B4",
    "SR_B5",
    "SR_B6",
    "SR_B7",
    "ST_B10",
    "QA_PIXEL",
)

OLI_SPACECRAFT_IDS = (
    "LANDSAT_8",
    "LANDSAT_9",
)

# TM_ETM_SOURCE_BANDS: tuple[str, ...] = (
#     "SR_B1",    # Blue
#     "SR_B2",    # Green
#     "SR_B3",    # Red
#     "SR_B4",    # NIR
#     "SR_B5",    # SWIR1
#     "SR_B7",    # SWIR2
#     "ST_B6",    # Thermal
#     "QA_PIXEL", # QA
# )

# # SPACECRAFT_ID property values per sensor family.
# OLI_SPACECRAFT_IDS: tuple[str, ...] = ("LANDSAT_8", "LANDSAT_9")
# TM_ETM_SPACECRAFT_IDS: tuple[str, ...] = ("LANDSAT_5", "LANDSAT_7")


# ============================================================
# Landsat 5 / 7
# ============================================================

TM_ETM_SOURCE_BANDS = (
    "SR_B1",
    "SR_B2",
    "SR_B3",
    "SR_B4",
    "SR_B5",
    "SR_B7",
    "ST_B6",
    "QA_PIXEL",
)

TM_ETM_SPACECRAFT_IDS = (
    "LANDSAT_5",
    "LANDSAT_7",
)

# ============================================================
# Sentinel-2
# ============================================================

S2_SOURCE_BANDS = (
    "B2",      # Blue
    "B3",      # Green
    "B4",      # Red
    "B8",      # NIR
    "B11",     # SWIR1
    "B12",     # SWIR2
    "SCL",     # QA
)

S2_SPACECRAFT_IDS = (
    "SENTINEL_2",
)


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

    # def rename_oli_image(self, image: Any) -> Any:
    #     """
    #     Rename Landsat 8/9 OLI/OLI-2 bands to the common schema.

    #     Use this method when you know the image is from Landsat 8 or 9
    #     (e.g., in a single-sensor collection). For mixed collections,
    #     use harmonize_collection() which handles the sensor detection
    #     automatically server-side.

    #     Args:
    #         image: An ee.Image from a Landsat 8 or 9 C2 L2 collection.

    #     Returns:
    #         The image with OLI_SOURCE_BANDS renamed to COMMON_BAND_NAMES.

    #     Raises:
    #         GEEAPIError: band selection or rename failed.
    #     """
    #     try:
    #         return image.select(
    #             list(OLI_SOURCE_BANDS)
    #         ).rename(list(COMMON_BAND_NAMES))
    #     except Exception as exc:
    #         raise GEEAPIError(
    #             operation="rename_oli_image",
    #             reason=f"OLI band selection/rename failed: {exc}",
    #         ) from exc


    def rename_oli_image(
        self,
        image: Any,
    ) -> Any:
        """
        Rename Landsat 8/9 bands to the common schema.
        """

        try:
            return image.select(
                list(OLI_SOURCE_BANDS)
            ).rename(
                list(COMMON_BAND_NAMES)
            )

        except Exception as exc:
            raise GEEAPIError(
                operation="rename_oli_image",
                reason=str(exc),
            ) from exc



    # def rename_tm_etm_image(self, image: Any) -> Any:
    #     """
    #     Rename Landsat 5/7 TM/ETM+ bands to the common schema.

    #     Use this method when you know the image is from Landsat 5 or 7.
    #     For mixed collections, use harmonize_collection().

    #     Args:
    #         image: An ee.Image from a Landsat 5 or 7 C2 L2 collection.

    #     Returns:
    #         The image with TM_ETM_SOURCE_BANDS renamed to COMMON_BAND_NAMES.

    #     Raises:
    #         GEEAPIError: band selection or rename failed.
    #     """
    #     try:
    #         return image.select(
    #             list(TM_ETM_SOURCE_BANDS)
    #         ).rename(list(COMMON_BAND_NAMES))
    #     except Exception as exc:
    #         raise GEEAPIError(
    #             operation="rename_tm_etm_image",
    #             reason=f"TM/ETM+ band selection/rename failed: {exc}",
    #         ) from exc


    def rename_tm_etm_image(
        self,
        image: Any,
    ) -> Any:
        """
        Rename Landsat 5/7 bands.
        """

        try:
            return image.select(
                list(TM_ETM_SOURCE_BANDS)
            ).rename(
                list(COMMON_BAND_NAMES)
            )

        except Exception as exc:
            raise GEEAPIError(
                operation="rename_tm_etm_image",
                reason=str(exc),
            ) from exc


    def rename_sentinel_image(
        self,
        image: Any,
    ) -> Any:
        """
        Rename Sentinel-2 bands into the common schema.

        Sentinel has no thermal band, therefore a constant
        zero-valued band named Thermal is appended.
        """

        try:

            import ee

        except ImportError as exc:

            raise GEENotInstalledError(
                "earthengine-api not installed."
            ) from exc

        try:

            optical = image.select(
                list(S2_SOURCE_BANDS)
            ).rename(
                [
                    "Blue",
                    "Green",
                    "Red",
                    "NIR",
                    "SWIR1",
                    "SWIR2",
                    "QA_PIXEL",
                ]
            )

            thermal = (
                ee.Image.constant(0)
                .rename("Thermal")
                .toFloat()
            )

            return (
                optical
                .addBands(thermal)
                .select(
                    list(COMMON_BAND_NAMES)
                )
            )

        except Exception as exc:

            raise GEEAPIError(
                operation="rename_sentinel_image",
                reason=str(exc),
            ) from exc

    def _make_harmonize_function(
        self,
    ) -> Any:
        """
        Build a server-side harmonization function supporting

            • Landsat 5
            • Landsat 7
            • Landsat 8
            • Landsat 9
            • Sentinel-2
        """

        try:
            import ee

        except ImportError as exc:

            raise GEENotInstalledError(
                "earthengine-api is not installed."
            ) from exc

        oli_sources = list(
            OLI_SOURCE_BANDS
        )

        tm_sources = list(
            TM_ETM_SOURCE_BANDS
        )

        s2_sources = list(
            S2_SOURCE_BANDS
        )

        common = list(
            COMMON_BAND_NAMES
        )

        oli_ids = list(
            OLI_SPACECRAFT_IDS
        )

        tm_ids = list(
            TM_ETM_SPACECRAFT_IDS
        )

        s2_ids = list(
            S2_SPACECRAFT_IDS
        )

        def _harmonize(
            image: Any,
        ) -> Any:

            # spacecraft = ee.String(
            #     image.get(
            #         "SPACECRAFT_ID"
            #     )
            # )
            spacecraft = ee.String(
                ee.Algorithms.If(
                image.propertyNames().contains("SPACECRAFT_ID"),
                image.get("SPACECRAFT_ID"),
                image.get("SPACECRAFT_NAME"),
                )
            )            

            is_oli = ee.List(
                oli_ids
            ).contains(
                spacecraft
            )

            is_tm = ee.List(
                tm_ids
            ).contains(
                spacecraft
            )
            
            is_s2 = spacecraft.match("^Sentinel-2.*")
            # is_s2 = ee.List(
            #     s2_ids
            # ).contains(
            #     spacecraft
            # )

            #
            # Landsat 8 / 9
            #
            oli = (
                image.select(
                    oli_sources
                )
                .rename(
                    common
                )
            )

            #
            # Landsat 5 / 7
            #
            tm = (
                image.select(
                    tm_sources
                )
                .rename(
                    common
                )
            )

            #
            # Sentinel-2
            #
            s2 = (
                image.select(
                    s2_sources
                )
                .rename(
                    [
                        "Blue",
                        "Green",
                        "Red",
                        "NIR",
                        "SWIR1",
                        "SWIR2",
                        "QA",
                    ]
                )
            )

            thermal = (
                ee.Image.constant(0)
                .rename(
                    "Thermal"
                )
                .toFloat()
            )

            s2 = (
                s2
                .addBands(
                    thermal
                )
                .select(
                    common
                )
            )

            return ee.Image(

                ee.Algorithms.If(

                    is_s2,

                    s2,

                    ee.Algorithms.If(

                        is_oli,

                        oli,

                        tm,

                    ),

                )

            )

        return _harmonize