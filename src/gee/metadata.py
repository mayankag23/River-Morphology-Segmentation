"""
Metadata extraction utilities for EE ImageCollections.

MetadataExtractor retrieves descriptive information about a Landsat
ImageCollection by executing getInfo() calls. All methods in this module
call getInfo() and therefore require the EE client to be initialized and
network connectivity to earthengine.googleapis.com.

Server-side vs client-side distinction:
    - LandsatCollectionBuilder (collections.py) is fully server-side.
    - MetadataExtractor is intentionally client-side: it exists specifically
      to bring EE data into Python for logging, validation, and reporting.
      getInfo() is called only in MetadataExtractor methods, never during
      the collection building or filtering pipeline.

Use client.execute_with_retry() for all getInfo() calls so that transient
EE errors are handled automatically.

Usage:

    from src.gee.metadata import MetadataExtractor, CollectionMetadata

    extractor = MetadataExtractor(client)

    # Individual extractions
    count = extractor.get_image_count(result.collection)
    bands = extractor.get_band_names(result.collection)

    # Full metadata in one call
    meta = extractor.extract_all(result.collection)
    print(meta.image_count)
    print(meta.band_names)
    print(meta.temporal_start, "to", meta.temporal_end)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.gee import GEEAPIError, GEENotInstalledError

if TYPE_CHECKING:
    from src.gee.client import EarthEngineClient

__all__ = [
    "CollectionMetadata",
    "MetadataExtractor",
]

_LOGGER: logging.Logger = logging.getLogger(__name__)


# ==============================================================================
# CollectionMetadata
# ==============================================================================

@dataclass(frozen=True)
class CollectionMetadata:
    """
    Immutable container for extracted ImageCollection metadata.

    All fields are optional (None) when extraction failed or the
    collection was empty. Check individual fields before use.

    Attributes:
        image_count:       Total number of images in the collection.
                           None if extraction failed.
        band_names:        List of band names from the first image.
                           Empty list if collection is empty.
        image_ids:         List of EE system index values (image IDs).
                           Empty list if collection is empty.
        acquisition_dates: List of DATE_ACQUIRED property values.
                           Empty list if collection is empty.
        spacecraft_ids:    List of unique SPACECRAFT_ID values.
                           Empty list if collection is empty.
        temporal_start:    Earliest acquisition date (YYYY-MM-DD).
                           None if collection is empty.
        temporal_end:      Latest acquisition date (YYYY-MM-DD).
                           None if collection is empty.
        crs:               CRS of the first image's first band projection,
                           e.g. "EPSG:32644". None if unavailable.
        scale_meters:      Nominal scale in metres of the first band.
                           None if unavailable.
    """

    image_count:       int | None          = None
    band_names:        list[str]           = field(default_factory=list)
    image_ids:         list[str]           = field(default_factory=list)
    acquisition_dates: list[str]           = field(default_factory=list)
    spacecraft_ids:    list[str]           = field(default_factory=list)
    temporal_start:    str | None          = None
    temporal_end:      str | None          = None
    crs:               str | None          = None
    scale_meters:      float | None        = None

    def summary_lines(self) -> list[str]:
        """
        Return ASCII-formatted summary lines for this metadata.

        Returns:
            List of strings suitable for logging or display.
        """
        def _fmt(value: Any, suffix: str = "") -> str:
            return f"{value}{suffix}" if value is not None else "N/A"

        lines = [
            f"  Image count:     {_fmt(self.image_count)}",
            f"  Band names:      {self.band_names or 'N/A'}",
            f"  Temporal range:  "
            f"{_fmt(self.temporal_start)} to {_fmt(self.temporal_end)}",
            f"  Spacecraft IDs:  {sorted(set(self.spacecraft_ids)) or 'N/A'}",
            f"  CRS:             {_fmt(self.crs)}",
            f"  Scale:           {_fmt(self.scale_meters, ' m')}",
        ]
        return lines


# ==============================================================================
# MetadataExtractor
# ==============================================================================

class MetadataExtractor:
    """
    Extracts metadata from an EE ImageCollection by calling getInfo().

    All methods make server-side EE computations followed by getInfo()
    to materialize the results. Methods use client.execute_with_retry()
    to handle transient EE errors (quota, rate limits, timeouts).

    Args:
        client: Initialized EarthEngineClient. Must have been initialized
                with client.initialize() before any extraction call.
    """

    def __init__(self, client: EarthEngineClient) -> None:
        self._client = client
        self._logger: logging.Logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Individual extraction methods
    # ------------------------------------------------------------------

    def get_image_count(self, collection: Any) -> int:
        """
        Return the total number of images in the collection.

        Calls size().getInfo() once.

        Args:
            collection: An ee.ImageCollection (may be filtered).

        Returns:
            Integer image count.

        Raises:
            GEENotInstalledError: earthengine-api is not installed.
            GEEAPIError:          size().getInfo() failed.
        """
        self._require_ee("get_image_count")
        self._logger.debug("Extracting image count.")
        try:
            count: int = self._client.execute_with_retry(
                collection.size().getInfo
            )
            self._logger.debug("Image count: %d", count)
            return count
        except Exception as exc:
            raise GEEAPIError(
                operation="get_image_count",
                reason=f"size().getInfo() failed: {exc}",
            ) from exc

    def get_band_names(self, collection: Any) -> list[str]:
        """
        Return the band names of the first image in the collection.

        Calls first().bandNames().getInfo() once. Returns an empty list
        if the collection is empty (no images).

        Args:
            collection: An ee.ImageCollection.

        Returns:
            List of band name strings, or empty list if collection is empty.

        Raises:
            GEENotInstalledError: earthengine-api is not installed.
            GEEAPIError:          getInfo() failed for a non-empty-collection reason.
        """
        self._require_ee("get_band_names")
        self._logger.debug("Extracting band names.")
        try:
            result = self._client.execute_with_retry(
                collection.first().bandNames().getInfo
            )
            bands = list(result) if result is not None else []
            self._logger.debug("Band names: %s", bands)
            return bands
        except Exception as exc:
            err_msg = str(exc).lower()
            # EE returns a specific error when the collection has no images
            # and first() produces a null element.
            if any(
                kw in err_msg
                for kw in ("null", "none", "empty", "no elements", "does not exist")
            ):
                self._logger.debug(
                    "get_band_names: collection appears empty. Returning []."
                )
                return []
            raise GEEAPIError(
                operation="get_band_names",
                reason=f"first().bandNames().getInfo() failed: {exc}",
            ) from exc

    def get_image_ids(self, collection: Any) -> list[str]:
        """
        Return the system:index (EE image ID) for each image in the collection.

        Calls aggregate_array('system:index').getInfo() once.

        Args:
            collection: An ee.ImageCollection.

        Returns:
            List of image ID strings, or empty list if collection is empty.

        Raises:
            GEEAPIError: getInfo() failed.
        """
        return self._aggregate_array_property(
            collection,
            "system:index",
            operation="get_image_ids",
        )

    def get_acquisition_dates(self, collection: Any) -> list[str]:
        """
        Return the DATE_ACQUIRED property value for each image.

        Calls aggregate_array('DATE_ACQUIRED').getInfo() once.

        Args:
            collection: An ee.ImageCollection.

        Returns:
            List of date strings (YYYY-MM-DD), or empty list if empty.

        Raises:
            GEEAPIError: getInfo() failed.
        """
        return self._aggregate_array_property(
            collection,
            "DATE_ACQUIRED",
            operation="get_acquisition_dates",
        )

    def get_spacecraft_ids(self, collection: Any) -> list[str]:
        """
        Return the SPACECRAFT_ID property value for each image.

        Calls aggregate_array('SPACECRAFT_ID').getInfo() once.

        Args:
            collection: An ee.ImageCollection.

        Returns:
            List of spacecraft ID strings (e.g. ["LANDSAT_8", "LANDSAT_9"]).
            Empty list if collection is empty.

        Raises:
            GEEAPIError: getInfo() failed.
        """
        return self._aggregate_array_property(
            collection,
            "SPACECRAFT_ID",
            operation="get_spacecraft_ids",
        )

    def get_temporal_coverage(
        self,
        collection: Any,
    ) -> tuple[str | None, str | None]:
        """
        Return the earliest and latest acquisition dates in the collection.

        Uses get_acquisition_dates() and returns (min_date, max_date).
        Returns (None, None) if the collection is empty.

        Args:
            collection: An ee.ImageCollection.

        Returns:
            Tuple of (earliest_date_str, latest_date_str) in YYYY-MM-DD format,
            or (None, None) if the collection contains no images.

        Raises:
            GEEAPIError: Underlying getInfo() call failed.
        """
        self._logger.debug("Extracting temporal coverage.")
        dates = self.get_acquisition_dates(collection)
        if not dates:
            return (None, None)
        sorted_dates = sorted(dates)
        return (sorted_dates[0], sorted_dates[-1])

    def get_crs_and_scale(
        self,
        collection: Any,
    ) -> tuple[str | None, float | None]:
        """
        Return the CRS and nominal scale of the first image's first band.

        Calls first().select(0).projection().getInfo() once to obtain
        both the CRS string and the nominal scale in metres.
        Returns (None, None) if the collection is empty.

        Args:
            collection: An ee.ImageCollection.

        Returns:
            Tuple of (crs_string, scale_in_metres).
            Example: ("EPSG:32644", 30.0)

        Raises:
            GEEAPIError: getInfo() failed for a non-empty-collection reason.
        """
        self._require_ee("get_crs_and_scale")
        self._logger.debug("Extracting CRS and scale.")

        try:
            crs_result: str | None = self._client.execute_with_retry(
                collection.first().select(0).projection().crs().getInfo
            )
            scale_result: float | None = self._client.execute_with_retry(
                collection.first().select(0).projection().nominalScale().getInfo
            )
            self._logger.debug(
                "CRS: %s, scale: %s m", crs_result, scale_result
            )
            return (crs_result, float(scale_result) if scale_result is not None else None)
        except Exception as exc:
            err_msg = str(exc).lower()
            if any(
                kw in err_msg
                for kw in ("null", "none", "empty", "no elements", "does not exist")
            ):
                self._logger.debug(
                    "get_crs_and_scale: collection appears empty. Returning (None, None)."
                )
                return (None, None)
            raise GEEAPIError(
                operation="get_crs_and_scale",
                reason=f"projection getInfo() failed: {exc}",
            ) from exc

    def extract_all(self, collection: Any) -> CollectionMetadata:
        """
        Extract all available metadata in a single structured call.

        Calls each individual extraction method in sequence. If any
        single method fails, its fields in CollectionMetadata are set
        to None or empty (failure is logged but does not abort the others).

        This design ensures that a failure in one field (e.g., CRS for an
        unusual sensor) does not prevent other metadata from being returned.

        Args:
            collection: An ee.ImageCollection.

        Returns:
            CollectionMetadata with all available fields populated.
        """
        self._logger.info("Extracting full collection metadata.")

        image_count = self._safe_extract(
            "image_count", self.get_image_count, collection
        )
        band_names = self._safe_extract(
            "band_names", self.get_band_names, collection, default=[]
        )
        image_ids = self._safe_extract(
            "image_ids", self.get_image_ids, collection, default=[]
        )
        acquisition_dates = self._safe_extract(
            "acquisition_dates", self.get_acquisition_dates, collection, default=[]
        )
        spacecraft_ids = self._safe_extract(
            "spacecraft_ids", self.get_spacecraft_ids, collection, default=[]
        )

        temporal_start: str | None = None
        temporal_end:   str | None = None
        if acquisition_dates:
            try:
                temporal_start, temporal_end = self.get_temporal_coverage(collection)
            except Exception as exc:
                self._logger.warning(
                    "Temporal coverage extraction failed: %s", exc
                )

        crs:          str | None   = None
        scale_meters: float | None = None
        try:
            crs, scale_meters = self.get_crs_and_scale(collection)
        except Exception as exc:
            self._logger.warning("CRS/scale extraction failed: %s", exc)

        meta = CollectionMetadata(
            image_count=image_count,
            band_names=band_names or [],
            image_ids=image_ids or [],
            acquisition_dates=acquisition_dates or [],
            spacecraft_ids=spacecraft_ids or [],
            temporal_start=temporal_start,
            temporal_end=temporal_end,
            crs=crs,
            scale_meters=scale_meters,
        )

        for line in meta.summary_lines():
            self._logger.info(line)

        return meta

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _require_ee(self, operation: str) -> None:
        """
        Verify that earthengine-api is importable.

        Args:
            operation: Name of the calling operation for error messages.

        Raises:
            GEENotInstalledError: ee cannot be imported.
        """
        try:
            import ee  # noqa: F401
        except ImportError as exc:
            raise GEENotInstalledError(
                f"earthengine-api is not installed. "
                f"Cannot call MetadataExtractor.{operation}(). "
                "Install with: pip install earthengine-api==0.1.390"
            ) from exc

    def _aggregate_array_property(
        self,
        collection: Any,
        property_name: str,
        operation: str,
    ) -> list[str]:
        """
        Call aggregate_array(property_name).getInfo() and return as a list.

        Returns an empty list when the collection is empty.

        Args:
            collection:    An ee.ImageCollection.
            property_name: The EE image property to aggregate.
            operation:     Calling operation name for error messages.

        Returns:
            List of property values (as strings), or empty list.

        Raises:
            GEEAPIError: getInfo() failed for a non-empty-collection reason.
        """
        self._logger.debug(
            "Aggregating property '%s' via %s.", property_name, operation
        )
        try:
            result = self._client.execute_with_retry(
                collection.aggregate_array(property_name).getInfo
            )
            values = [str(v) for v in result] if result else []
            self._logger.debug(
                "%s: %d values returned.", operation, len(values)
            )
            return values
        except Exception as exc:
            err_msg = str(exc).lower()
            if any(
                kw in err_msg
                for kw in ("null", "none", "empty", "no elements")
            ):
                return []
            raise GEEAPIError(
                operation=operation,
                reason=(
                    f"aggregate_array('{property_name}').getInfo() failed: {exc}"
                ),
            ) from exc

    def _safe_extract(
        self,
        field_name: str,
        method: Any,
        *args: Any,
        default: Any = None,
    ) -> Any:
        """
        Call a metadata extraction method and return a default on failure.

        Args:
            field_name: Name of the field being extracted (for logging).
            method:     Callable extraction method.
            *args:      Arguments to pass to the method.
            default:    Value to return if the method raises any exception.

        Returns:
            The result of method(*args) or default on any exception.
        """
        try:
            return method(*args)
        except Exception as exc:
            self._logger.warning(
                "Metadata extraction failed for '%s': %s. "
                "Using default: %r.",
                field_name, exc, default,
            )
            return default