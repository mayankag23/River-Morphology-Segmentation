"""
Configurable composite generation from preprocessed Landsat collections.

LandsatCompositor reduces an ee.ImageCollection to a single ee.Image using
one of five supported compositing methods. The method is read from
config.composite.method; it can also be provided at call time.

Supported methods:
    MEDIAN:     collection.median()
                Best general-purpose choice for river morphology: resistant
                to remaining cloud contamination and outlier observations.
                Recommended default for most AOIs.

    MEAN:       collection.mean()
                Simpler than median; more sensitive to outliers. Use only
                when cloud masking has been comprehensive.

    MEDOID:     Observation closest to the multi-band median in Euclidean
                distance. Selects a real observed pixel (vs. synthetic median).
                Preserves spectral consistency. Uses qualityMosaic() on
                negative sum-of-squared-differences from the median.
                Best for time-series analysis where pixel authenticity matters.

    MOSAIC:     collection.mosaic()
                Uses the most recent (top of stack) observation per pixel.
                Best when images are sorted by quality and the latest
                image is the best representation.

    PERCENTILE: collection.reduce(ee.Reducer.percentile([value]))
                Configurable quantile. P10 approximates dry-season minimum
                (useful for exposing sandbars). P90 shows maximum extent.

All compositing is done server-side. No getInfo() is called.
The output CompositeResult contains a single ee.Image and provenance metadata.

Input:  ProcessedCollectionResult  (from src.gee.preprocessing)
Output: CompositeResult            (single ee.Image + provenance)

Usage:

    from src.gee.composite import LandsatCompositor, CompositeMethod

    compositor = LandsatCompositor(client, config)

    # Uses method from config.composite.method
    result = compositor.build_composite(processed_result)

    # Override method at call time
    result = compositor.build_composite(
        processed_result,
        method=CompositeMethod.MEDOID,
    )

    # Access composite image
    composite_image = result.image
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from src.core.config import Config
from src.core.exceptions import InvalidValueError
from src.gee import GEEAPIError, GEENotInstalledError
from src.gee.harmonization import COMMON_BAND_NAMES, OPTICAL_BAND_NAMES
from src.gee.preprocessing import ProcessedCollectionResult

if TYPE_CHECKING:
    from src.gee.client import EarthEngineClient

__all__ = [
    "CompositeMethod",
    "CompositeResult",
    "LandsatCompositor",
]

_LOGGER: logging.Logger = logging.getLogger(__name__)

_DEFAULT_COMPOSITE_METHOD:   str = "median"
_DEFAULT_PERCENTILE_VALUE:   int = 50
_MEDOID_SCORE_BAND:          str = "medoid_distance_score"


# ==============================================================================
# CompositeMethod
# ==============================================================================

class CompositeMethod(str, Enum):
    """
    Supported compositing methods.

    As a str Enum, values compare equal to config strings:
        CompositeMethod.MEDIAN == "median"  # True
    """

    MEDIAN     = "median"
    MEAN       = "mean"
    MEDOID     = "medoid"
    MOSAIC     = "mosaic"
    PERCENTILE = "percentile"

    @classmethod
    def from_string(cls, value: str) -> CompositeMethod:
        """
        Convert a config string to CompositeMethod.

        Args:
            value: One of "median", "mean", "medoid", "mosaic", "percentile".

        Returns:
            The corresponding CompositeMethod enum member.

        Raises:
            InvalidValueError: value is not a recognized method string.
        """
        try:
            return cls(value.lower().strip())
        except ValueError:
            raise InvalidValueError(
                field="composite.method",
                value=value,
                reason=(
                    f"must be one of: "
                    f"{[m.value for m in cls]}. "
                    f"Got: '{value}'"
                ),
            )


# ==============================================================================
# CompositeResult
# ==============================================================================

@dataclass(frozen=True)
class CompositeResult:
    """
    Immutable output of LandsatCompositor.build_composite().

    Attributes:
        image:          Single ee.Image composite (lazy server-side graph).
        method:         The CompositeMethod used to produce this composite.
        percentile_value: Percentile value (only relevant for PERCENTILE method).
        source_result:  The ProcessedCollectionResult this was derived from.
        band_names:     Band names in the composite image.
    """

    image:            Any                     # ee.Image
    method:           CompositeMethod
    percentile_value: int | None
    source_result:    ProcessedCollectionResult
    band_names:       tuple[str, ...]

    def summary_lines(self) -> list[str]:
        """Return ASCII-formatted summary lines describing this composite."""
        lines = [
            f"  Composite method: {self.method.value}",
            f"  Band names:       {list(self.band_names)}",
        ]
        if self.method == CompositeMethod.PERCENTILE:
            lines.append(
                f"  Percentile value: {self.percentile_value}"
            )
        if self.source_result.source_result.has_mixed_sensor_families:
            lines.append(
                "  [INFO] Composite derived from a mixed-sensor collection."
            )
        return lines


# ==============================================================================
# LandsatCompositor
# ==============================================================================

class LandsatCompositor:
    """
    Reduces a ProcessedCollectionResult to a single-image CompositeResult.

    Reads the composite method and percentile value from config.composite.
    Both can be overridden at call time via build_composite() parameters.

    The compositor is stateless and reusable across multiple collections.

    Args:
        client: Initialized EarthEngineClient (used for execute_with_retry
                if a composite validation step is added in the future).
        config: Fully initialized Config. Reads from config.composite.
    """

    def __init__(
        self,
        client: EarthEngineClient,
        config: Config,
    ) -> None:
        self._client = client
        self._config = config
        self._logger: logging.Logger = logging.getLogger(__name__)

    def build_composite(
        self,
        processed_result: ProcessedCollectionResult,
        method: CompositeMethod | None = None,
        percentile_value: int | None = None,
    ) -> CompositeResult:
        """
        Reduce the processed collection to a single composite image.

        Method selection priority:
            1. method parameter (if provided)
            2. config.composite.method (from config.yaml)
            3. Default "median"

        Args:
            processed_result: ProcessedCollectionResult from LandsatPreprocessor.
            method:           Override the compositing method.
            percentile_value: Override the percentile value (PERCENTILE method only).

        Returns:
            CompositeResult with a single ee.Image and provenance metadata.

        Raises:
            InvalidValueError:   method string from config is not recognized.
            GEENotInstalledError: earthengine-api is not installed.
            GEEAPIError:          The EE composite operation failed.
        """
        resolved_method = self._resolve_method(method)
        resolved_pct    = self._resolve_percentile(percentile_value, resolved_method)

        collection = processed_result.collection
        band_names = processed_result.band_names

        self._logger.info(
            "Building composite. method=%s, harmonized=%s",
            resolved_method.value,
            processed_result.harmonization_applied,
        )

        image = self._dispatch_composite(
            collection=collection,
            method=resolved_method,
            percentile_value=resolved_pct,
            is_harmonized=processed_result.harmonization_applied,
        )

        result = CompositeResult(
            image=image,
            method=resolved_method,
            percentile_value=resolved_pct,
            source_result=processed_result,
            band_names=band_names,
        )

        for line in result.summary_lines():
            self._logger.info(line)

        return result

    # ------------------------------------------------------------------
    # Private dispatch and composite methods
    # ------------------------------------------------------------------

    def _dispatch_composite(
        self,
        collection: Any,
        method: CompositeMethod,
        percentile_value: int | None,
        is_harmonized: bool,
    ) -> Any:
        """Route to the correct composite implementation based on method."""
        if method == CompositeMethod.MEDIAN:
            return self._median(collection)
        if method == CompositeMethod.MEAN:
            return self._mean(collection)
        if method == CompositeMethod.MEDOID:
            return self._medoid(collection, is_harmonized)
        if method == CompositeMethod.MOSAIC:
            return self._mosaic(collection)
        if method == CompositeMethod.PERCENTILE:
            pct = percentile_value if percentile_value is not None else _DEFAULT_PERCENTILE_VALUE
            return self._percentile(collection, pct)

        raise InvalidValueError(
            field="composite.method",
            value=method,
            reason=f"Unhandled composite method: {method!r}",
        )

    def _median(self, collection: Any) -> Any:
        """
        Compute the per-pixel median across all collection images.

        The median value at each pixel is a synthetic value not observed
        in any single image. It is resistant to outliers from remaining
        cloud contamination and is the recommended default for most cases.

        Returns:
            ee.Image: median composite.
        """
        try:
            return collection.median()
        except Exception as exc:
            raise GEEAPIError(
                operation="composite_median",
                reason=f"collection.median() failed: {exc}",
            ) from exc

    def _mean(self, collection: Any) -> Any:
        """
        Compute the per-pixel mean across all collection images.

        Faster than median but more sensitive to outliers. Suitable when
        cloud masking has been comprehensive.

        Returns:
            ee.Image: mean composite.
        """
        try:
            return collection.mean()
        except Exception as exc:
            raise GEEAPIError(
                operation="composite_mean",
                reason=f"collection.mean() failed: {exc}",
            ) from exc

    def _mosaic(self, collection: Any) -> Any:
        """
        Create a mosaic using the most recent image (top of stack) per pixel.

        Assumes the collection is sorted with the best/most-recent image
        last (default EE order is chronological, so latest = top).
        Useful for post-filtered collections sorted by quality.

        Returns:
            ee.Image: mosaic composite.
        """
        try:
            return collection.mosaic()
        except Exception as exc:
            raise GEEAPIError(
                operation="composite_mosaic",
                reason=f"collection.mosaic() failed: {exc}",
            ) from exc

    def _medoid(self, collection: Any, is_harmonized: bool) -> Any:
        """
        Select the pixel-wise medoid: the observation closest to the median.

        The medoid is a REAL observed pixel (unlike the synthetic median),
        making it spectrally consistent and suitable for time-series analysis.

        Algorithm:
            1. Compute the band-wise median image from the collection.
            2. For each image, compute the per-pixel sum-of-squared differences
               from the median across optical bands.
            3. Negate the distance score (ee.qualityMosaic picks the maximum).
            4. Use qualityMosaic('medoid_distance_score') to select the image
               with the minimum distance (maximum of the negated score).
            5. Remove the temporary score band from the output.

        Only optical bands (not thermal or QA) are used for distance computation
        to avoid the thermal band dominating the distance metric.

        Args:
            collection:    An ee.ImageCollection.
            is_harmonized: If True, use OPTICAL_BAND_NAMES for distance bands.
                           If False, use the SR_B.* band name pattern.

        Returns:
            ee.Image: medoid composite.

        Raises:
            GEENotInstalledError: earthengine-api is not installed.
            GEEAPIError:          EE computation failed.
        """
        try:
            import ee
        except ImportError as exc:
            raise GEENotInstalledError(
                "earthengine-api is not installed."
            ) from exc

        try:
            # Determine which bands to use for the distance metric.
            if is_harmonized:
                metric_band_list = list(OPTICAL_BAND_NAMES)
                metric_collection = collection.select(metric_band_list)
            else:
                # For non-harmonized collections, use all SR_B.* bands.
                metric_collection = collection.select("SR_B.*")

            median = metric_collection.median()

            def _score_image(image: Any) -> Any:
                """
                Add a medoid distance score band to an image.
                Score is negated so qualityMosaic selects minimum distance.
                """
                if is_harmonized:
                    img_metric = image.select(metric_band_list)
                else:
                    img_metric = image.select("SR_B.*")

                squared_diff = img_metric.subtract(median).pow(2)
                distance     = squared_diff.reduce(ee.Reducer.sum())

                # Negate: qualityMosaic picks the HIGHEST value per pixel.
                score = distance.multiply(-1).rename(_MEDOID_SCORE_BAND)
                return image.addBands(score)

            scored  = collection.map(_score_image)
            medoid  = scored.qualityMosaic(_MEDOID_SCORE_BAND)

            # Remove the temporary score band from the output.
            output_bands = scored.first().bandNames().remove(_MEDOID_SCORE_BAND)
            return medoid.select(output_bands)

        except (GEENotInstalledError, GEEAPIError):
            raise
        except Exception as exc:
            raise GEEAPIError(
                operation="composite_medoid",
                reason=f"Medoid composite computation failed: {exc}",
            ) from exc

    def _percentile(self, collection: Any, percentile_value: int) -> Any:
        """
        Compute a per-pixel percentile composite.

        Uses ee.Reducer.percentile([value]) to compute the specified
        quantile. Output band names have '_p{value}' suffix added by EE;
        these are renamed back to the original band names.

        Useful values:
            10  -> dry-season minimum (exposes sandbars)
            25  -> low-water composite
            50  -> equivalent to median
            75  -> high-water composite
            90  -> near-maximum water extent

        Args:
            collection:       An ee.ImageCollection.
            percentile_value: Integer percentile in [0, 100].

        Returns:
            ee.Image: percentile composite with original band names.

        Raises:
            GEENotInstalledError: earthengine-api is not installed.
            GEEAPIError:          EE reduce() call failed.
        """
        try:
            import ee
        except ImportError as exc:
            raise GEENotInstalledError(
                "earthengine-api is not installed."
            ) from exc

        try:
            reduced = collection.reduce(
                ee.Reducer.percentile([int(percentile_value)])
            )

            # EE appends '_p{value}' to band names after percentile reduction.
            # Retrieve the output band names server-side and rename back.
            # We use the first image's band names to build the expected
            # suffixed names, then rename them to the original names.
            original_names = collection.first().bandNames()
            suffixed_names = original_names.map(
                lambda name: ee.String(name).cat(f"_p{percentile_value}")
            )
            return reduced.select(suffixed_names).rename(original_names)

        except (GEENotInstalledError, GEEAPIError):
            raise
        except Exception as exc:
            raise GEEAPIError(
                operation="composite_percentile",
                reason=(
                    f"collection.reduce(percentile=[{percentile_value}]) failed: {exc}"
                ),
            ) from exc

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def _resolve_method(
        self,
        override: CompositeMethod | None,
    ) -> CompositeMethod:
        """
        Determine the composite method to use.

        Priority: override parameter -> config.composite.method -> default.

        Args:
            override: Explicit method from build_composite() caller, or None.

        Returns:
            Resolved CompositeMethod.

        Raises:
            InvalidValueError: Config method string is unrecognized.
        """
        if override is not None:
            return override

        composite_cfg = getattr(self._config, "composite", None)
        if composite_cfg is not None:
            method_str = str(getattr(composite_cfg, "method", _DEFAULT_COMPOSITE_METHOD))
        else:
            method_str = _DEFAULT_COMPOSITE_METHOD

        return CompositeMethod.from_string(method_str)

    def _resolve_percentile(
        self,
        override: int | None,
        method: CompositeMethod,
    ) -> int | None:
        """
        Determine the percentile value when using PERCENTILE method.

        Args:
            override: Explicit value from build_composite() caller, or None.
            method:   The resolved composite method.

        Returns:
            Integer percentile value, or None if method is not PERCENTILE.

        Raises:
            InvalidValueError: Percentile value is outside [0, 100].
        """
        if method != CompositeMethod.PERCENTILE:
            return None

        if override is not None:
            value = int(override)
        else:
            composite_cfg = getattr(self._config, "composite", None)
            if composite_cfg is not None:
                value = int(
                    getattr(composite_cfg, "percentile_value", _DEFAULT_PERCENTILE_VALUE)
                )
            else:
                value = _DEFAULT_PERCENTILE_VALUE

        if not (0 <= value <= 100):
            raise InvalidValueError(
                field="composite.percentile_value",
                value=value,
                reason="must be an integer in the range [0, 100]",
            )

        return value