"""
Spectral index computation functions for Landsat imagery.

Each function in this module:
    - Accepts a SINGLE ee.Image with harmonized band names from
      src.gee.harmonization.COMMON_BAND_NAMES.
    - Computes exactly one spectral index.
    - Returns a SINGLE-BAND ee.Image named after the index.
    - Wraps all EE exceptions in GEEAPIError.
    - Has NO side effects and is fully independent and reusable.

Band names assumed present in the input image (from harmonization):
    Blue, Green, Red, NIR, SWIR1, SWIR2, Thermal, QA_PIXEL

No raw Landsat band names (SR_B2, ST_B10, etc.) are referenced here.
All ee imports are deferred to function bodies for testability.

Mathematical formulas and river morphology relevance are documented
in each function's docstring.

Usage:

    from src.gee.indices import compute_mndwi, compute_bsi

    mndwi = compute_mndwi(composite_image)
    bsi   = compute_bsi(composite_image)

    stacked = composite_image.addBands(mndwi).addBands(bsi)
"""

from __future__ import annotations

import logging
from typing import Any

from src.gee import GEEAPIError

__all__ = [
    "compute_ndwi",
    "compute_mndwi",
    "compute_awei_sh",
    "compute_awei_nsh",
    "compute_ndvi",
    "compute_savi",
    "compute_bsi",
    "compute_ndmi",
    "compute_ndbi",
]

_LOGGER: logging.Logger = logging.getLogger(__name__)


def compute_ndwi(image: Any) -> Any:
    """
    Compute the Normalized Difference Water Index (NDWI).

    Formula:
        NDWI = (Green - NIR) / (Green + NIR)

    Reference: McFeeters (1996), Remote Sensing of Environment, 57(2), 167-182.

    River Morphology Relevance:
        Positive values indicate open water. Negative values indicate
        vegetation and dry soil. Less discriminative than MNDWI for
        turbid river water and wet sandbars, but effective for clear,
        deep channels.

    Args:
        image: ee.Image with harmonized bands. Must contain 'Green' and 'NIR'.

    Returns:
        Single-band ee.Image named 'NDWI'. Values in [-1, 1].

    Raises:
        GEEAPIError: EE computation failed.
    """
    _LOGGER.debug("Computing NDWI = (Green - NIR) / (Green + NIR)")
    try:
        return (
            image.normalizedDifference(["Green", "NIR"])
            .rename("NDWI")
        )
    except Exception as exc:
        raise GEEAPIError(
            operation="compute_ndwi",
            reason=f"(Green - NIR) / (Green + NIR) failed: {exc}",
        ) from exc


def compute_mndwi(image: Any) -> Any:
    """
    Compute the Modified Normalized Difference Water Index (MNDWI).

    Formula:
        MNDWI = (Green - SWIR1) / (Green + SWIR1)

    Reference: Xu (2006), International Journal of Remote Sensing, 27(14), 3025-3033.

    River Morphology Relevance:
        PRIMARY discriminator between water and sand/sandbars.
        Sand has high SWIR1 reflectance -> strongly negative MNDWI.
        Water has near-zero SWIR1 reflectance -> positive MNDWI.
        Outperforms NDWI for turbid river water and shallow channels.
        Threshold of MNDWI > 0.2 reliably separates water from sand.

    Args:
        image: ee.Image. Must contain 'Green' and 'SWIR1'.

    Returns:
        Single-band ee.Image named 'MNDWI'. Values in [-1, 1].

    Raises:
        GEEAPIError: EE computation failed.
    """
    _LOGGER.debug("Computing MNDWI = (Green - SWIR1) / (Green + SWIR1)")
    try:
        return (
            image.normalizedDifference(["Green", "SWIR1"])
            .rename("MNDWI")
        )
    except Exception as exc:
        raise GEEAPIError(
            operation="compute_mndwi",
            reason=f"(Green - SWIR1) / (Green + SWIR1) failed: {exc}",
        ) from exc


def compute_awei_sh(image: Any) -> Any:
    """
    Compute the Automated Water Extraction Index (shadow variant, AWEI_sh).

    Formula:
        AWEI_sh = Blue + 2.5*Green - 1.5*(NIR + SWIR1) - 0.25*SWIR2

    Reference: Feyisa et al. (2014), Remote Sensing of Environment, 140, 23-35.

    River Morphology Relevance:
        Specifically designed to distinguish water from shadows caused
        by riparian vegetation, cliffs, and cloud shadows. Captures
        shadowed water channels that MNDWI misses. Positive values
        indicate water including turbid and shadow-affected pixels.

    Args:
        image: ee.Image. Must contain 'Blue', 'Green', 'NIR', 'SWIR1', 'SWIR2'.

    Returns:
        Single-band ee.Image named 'AWEI_sh'. Positive = water.

    Raises:
        GEEAPIError: EE computation failed.
    """
    _LOGGER.debug(
        "Computing AWEI_sh = Blue + 2.5*Green - 1.5*(NIR+SWIR1) - 0.25*SWIR2"
    )
    try:
        blue  = image.select("Blue")
        green = image.select("Green")
        nir   = image.select("NIR")
        swir1 = image.select("SWIR1")
        swir2 = image.select("SWIR2")

        # Blue + 2.5*Green - 1.5*(NIR + SWIR1) - 0.25*SWIR2
        result = (
            blue
            .add(green.multiply(2.5))
            .subtract(nir.add(swir1).multiply(1.5))
            .subtract(swir2.multiply(0.25))
        )
        return result.rename("AWEI_sh")
    except Exception as exc:
        raise GEEAPIError(
            operation="compute_awei_sh",
            reason=(
                f"Blue + 2.5*Green - 1.5*(NIR+SWIR1) - 0.25*SWIR2 "
                f"failed: {exc}"
            ),
        ) from exc


def compute_awei_nsh(image: Any) -> Any:
    """
    Compute the Automated Water Extraction Index (no-shadow variant, AWEI_nsh).

    Formula:
        AWEI_nsh = 4*(Green - SWIR1) - (0.25*NIR + 2.75*SWIR2)

    Reference: Feyisa et al. (2014), Remote Sensing of Environment, 140, 23-35.

    River Morphology Relevance:
        Effective in open riverine environments without significant
        topographic shadowing. Captures wide, turbid river channels
        and exposed floodplain water. Complements AWEI_sh for regions
        with minimal shade. High SWIR2 weighting strongly suppresses
        dry sediment, exposed sandbars, and built-up surfaces.

    Args:
        image: ee.Image. Must contain 'Green', 'NIR', 'SWIR1', 'SWIR2'.

    Returns:
        Single-band ee.Image named 'AWEI_nsh'. Positive = water.

    Raises:
        GEEAPIError: EE computation failed.
    """
    _LOGGER.debug(
        "Computing AWEI_nsh = 4*(Green-SWIR1) - (0.25*NIR + 2.75*SWIR2)"
    )
    try:
        green = image.select("Green")
        nir   = image.select("NIR")
        swir1 = image.select("SWIR1")
        swir2 = image.select("SWIR2")

        # 4*(Green - SWIR1) - (0.25*NIR + 2.75*SWIR2)
        result = (
            green.subtract(swir1).multiply(4.0)
            .subtract(nir.multiply(0.25).add(swir2.multiply(2.75)))
        )
        return result.rename("AWEI_nsh")
    except Exception as exc:
        raise GEEAPIError(
            operation="compute_awei_nsh",
            reason=(
                f"4*(Green-SWIR1) - (0.25*NIR + 2.75*SWIR2) failed: {exc}"
            ),
        ) from exc


def compute_ndvi(image: Any) -> Any:
    """
    Compute the Normalized Difference Vegetation Index (NDVI).

    Formula:
        NDVI = (NIR - Red) / (NIR + Red)

    Reference: Rouse et al. (1973), Third ERTS Symposium, NASA SP-351, 309-317.

    River Morphology Relevance:
        Used as a VEGETATION SUPPRESSOR in river classification.
        Dense riparian vegetation produces high NDVI (> 0.3) and must
        be excluded from the river mask. Negative NDVI values reliably
        indicate water surfaces. Combined with MNDWI, NDVI distinguishes
        water from vegetation in the channel corridor.

    Args:
        image: ee.Image. Must contain 'NIR' and 'Red'.

    Returns:
        Single-band ee.Image named 'NDVI'. Values in [-1, 1].

    Raises:
        GEEAPIError: EE computation failed.
    """
    _LOGGER.debug("Computing NDVI = (NIR - Red) / (NIR + Red)")
    try:
        return (
            image.normalizedDifference(["NIR", "Red"])
            .rename("NDVI")
        )
    except Exception as exc:
        raise GEEAPIError(
            operation="compute_ndvi",
            reason=f"(NIR - Red) / (NIR + Red) failed: {exc}",
        ) from exc


def compute_savi(image: Any, soil_factor: float = 0.5) -> Any:
    """
    Compute the Soil-Adjusted Vegetation Index (SAVI).

    Formula:
        SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L)
        where L is the soil brightness correction factor.

    Reference: Huete (1988), Remote Sensing of Environment, 25(3), 295-309.

    River Morphology Relevance:
        Improves vegetation detection in areas with low vegetation cover
        and bright soil/sand backgrounds. In braided river systems where
        sparse vegetation grows on sandbars, SAVI more accurately
        identifies vegetation than NDVI, reducing false positives in the
        water/sand classification. L = 0.5 is recommended for most
        intermediate vegetation cover conditions.

    Args:
        image:       ee.Image. Must contain 'NIR' and 'Red'.
        soil_factor: Soil brightness correction factor L in [0, 1].
                     L = 0: equivalent to NDVI (dense vegetation).
                     L = 1: minimal soil noise correction.
                     L = 0.5: recommended default (moderate vegetation).

    Returns:
        Single-band ee.Image named 'SAVI'. Values approximately in [-1, 1].

    Raises:
        GEEAPIError: EE computation failed.
    """
    _LOGGER.debug(
        "Computing SAVI = ((NIR-Red)/(NIR+Red+%.1f)) * %.1f",
        soil_factor, 1.0 + soil_factor,
    )
    try:
        nir = image.select("NIR")
        red = image.select("Red")
        l   = float(soil_factor)

        # ((NIR - Red) / (NIR + Red + L)) * (1 + L)
        result = (
            nir.subtract(red)
            .divide(nir.add(red).add(l))
            .multiply(1.0 + l)
        )
        return result.rename("SAVI")
    except Exception as exc:
        raise GEEAPIError(
            operation="compute_savi",
            reason=(
                f"((NIR-Red)/(NIR+Red+{soil_factor})) * "
                f"{1.0 + soil_factor} failed: {exc}"
            ),
        ) from exc


def compute_bsi(image: Any) -> Any:
    """
    Compute the Bare Soil Index (BSI).

    Formula:
        BSI = ((SWIR1 + Red) - (NIR + Blue)) / ((SWIR1 + Red) + (NIR + Blue))

    Reference: Rikimaru et al. (2002), FAO Forestry Department.

    River Morphology Relevance:
        PRIMARY discriminator for exposed sandbars and dry sediment.
        Sand/bare soil: high SWIR1 + high Red -> strongly POSITIVE BSI.
        Vegetation: high NIR -> NEGATIVE BSI.
        Water: low in all optical bands -> near-zero or negative BSI.
        BSI combined with MNDWI creates a three-class discriminator:
        MNDWI > 0.2 -> Water, BSI > 0.1 AND MNDWI < 0.2 -> Sand.

    Args:
        image: ee.Image. Must contain 'Blue', 'Red', 'NIR', 'SWIR1'.

    Returns:
        Single-band ee.Image named 'BSI'. Values in [-1, 1].

    Raises:
        GEEAPIError: EE computation failed.
    """
    _LOGGER.debug(
        "Computing BSI = ((SWIR1+Red)-(NIR+Blue)) / ((SWIR1+Red)+(NIR+Blue))"
    )
    try:
        blue  = image.select("Blue")
        red   = image.select("Red")
        nir   = image.select("NIR")
        swir1 = image.select("SWIR1")

        # Compute shared sub-expressions once to avoid duplicate EE nodes.
        swir1_plus_red = swir1.add(red)
        nir_plus_blue  = nir.add(blue)

        numerator   = swir1_plus_red.subtract(nir_plus_blue)
        denominator = swir1_plus_red.add(nir_plus_blue)

        return numerator.divide(denominator).rename("BSI")
    except Exception as exc:
        raise GEEAPIError(
            operation="compute_bsi",
            reason=(
                f"((SWIR1+Red)-(NIR+Blue)) / ((SWIR1+Red)+(NIR+Blue)) "
                f"failed: {exc}"
            ),
        ) from exc


def compute_ndmi(image: Any) -> Any:
    """
    Compute the Normalized Difference Moisture Index (NDMI).

    Formula:
        NDMI = (NIR - SWIR1) / (NIR + SWIR1)

    Reference: Gao (1996), Remote Sensing of Environment, 58(3), 257-266.

    River Morphology Relevance:
        Measures surface moisture content. High NDMI indicates
        waterlogged soil and wet floodplains adjacent to channels.
        Distinguishes wet sand (moisture-affected, moderate NDMI) from
        dry sand (low NDMI) and open water (MNDWI is more effective for
        pure water pixels). Useful for mapping active floodplains and
        identifying recently inundated areas after flood events.

    Args:
        image: ee.Image. Must contain 'NIR' and 'SWIR1'.

    Returns:
        Single-band ee.Image named 'NDMI'. Values in [-1, 1].

    Raises:
        GEEAPIError: EE computation failed.
    """
    _LOGGER.debug("Computing NDMI = (NIR - SWIR1) / (NIR + SWIR1)")
    try:
        return (
            image.normalizedDifference(["NIR", "SWIR1"])
            .rename("NDMI")
        )
    except Exception as exc:
        raise GEEAPIError(
            operation="compute_ndmi",
            reason=f"(NIR - SWIR1) / (NIR + SWIR1) failed: {exc}",
        ) from exc


def compute_ndbi(image: Any) -> Any:
    """
    Compute the Normalized Difference Built-Up Index (NDBI).

    Formula:
        NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)

    Reference: Zha et al. (2003), International Journal of Remote Sensing,
               24(3), 583-594.

    River Morphology Relevance:
        OPTIONAL index. Positive NDBI indicates built-up/impervious
        surfaces adjacent to river channels (bridges, embankments, urban
        encroachment). Can be used to mask anthropogenic structures from
        morphology maps and to distinguish concrete infrastructure from
        natural sandbars in urban river reaches. Note: NDBI is the
        inverse of NDWI by construction.

    Args:
        image: ee.Image. Must contain 'SWIR1' and 'NIR'.

    Returns:
        Single-band ee.Image named 'NDBI'. Values in [-1, 1].

    Raises:
        GEEAPIError: EE computation failed.
    """
    _LOGGER.debug("Computing NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)")
    try:
        return (
            image.normalizedDifference(["SWIR1", "NIR"])
            .rename("NDBI")
        )
    except Exception as exc:
        raise GEEAPIError(
            operation="compute_ndbi",
            reason=f"(SWIR1 - NIR) / (SWIR1 + NIR) failed: {exc}",
        ) from exc