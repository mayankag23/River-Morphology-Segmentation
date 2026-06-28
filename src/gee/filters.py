"""
Modular Earth Engine filter functions for Landsat ImageCollections.

Each function is independent, accepts an ee.ImageCollection, applies
exactly one filter, and returns the filtered ee.ImageCollection.

Filters are composable: chain them in any order. The builder in
collections.py uses these functions internally. They can also be used
directly by downstream modules if additional filtering is needed.

No getInfo() calls are made in this module. All operations are lazy
server-side EE computations. All EE imports are deferred to function
bodies so that this module is importable and testable without
earthengine-api installed.

All EE exceptions are wrapped in project-specific types before being
raised. Raw EE exceptions never propagate outside this module.

Constants:
    CLOUD_COVER_PROPERTY    -- EE property name for scene cloud cover
    SPACECRAFT_ID_PROPERTY  -- EE property name for satellite identifier
    DATE_ACQUIRED_PROPERTY  -- EE property name for acquisition date
    SYSTEM_TIME_START       -- EE system property for timestamp (ms)
    SYSTEM_INDEX            -- EE system property for image ID

Usage:

    from src.gee.filters import (
        filter_by_date,
        filter_by_bounds,
        filter_by_cloud_cover,
        filter_by_sensor,
    )

    collection = client.get_image_collection("LANDSAT/LC08/C02/T1_L2")
    collection = filter_by_date(collection, "2023-11-01", "2024-02-28")
    collection = filter_by_bounds(collection, aoi_geometry)
    collection = filter_by_cloud_cover(collection, max_percent=20.0)
    collection = filter_by_sensor(collection, ["LANDSAT_8", "LANDSAT_9"])
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from src.core.exceptions import InvalidValueError, TypeMismatchError
from src.gee import GEEAPIError

__all__ = [
    "filter_by_date",
    "filter_by_bounds",
    "filter_by_cloud_cover",
    "filter_by_sensor",
    "CLOUD_COVER_PROPERTY",
    "SPACECRAFT_ID_PROPERTY",
    "DATE_ACQUIRED_PROPERTY",
    "SYSTEM_TIME_START",
    "SYSTEM_INDEX",
]

_LOGGER: logging.Logger = logging.getLogger(__name__)

# GEE property names for Landsat Collection 2 Level-2 imagery.
# Defined as module constants so downstream modules reference them
# without hardcoding strings.
CLOUD_COVER_PROPERTY:   str = "CLOUD_COVER"
SPACECRAFT_ID_PROPERTY: str = "SPACECRAFT_ID"
DATE_ACQUIRED_PROPERTY: str = "DATE_ACQUIRED"
SYSTEM_TIME_START:      str = "system:time_start"
SYSTEM_INDEX:           str = "system:index"

_DATE_FORMAT: str = "%Y-%m-%d"


# ==============================================================================
# Private validation helpers
# ==============================================================================

def _validate_date_string(date_str: str, field: str) -> None:
    """
    Validate that a string conforms to YYYY-MM-DD format.

    Args:
        date_str: The date string to validate.
        field:    Dot-notation field name for error messages.

    Raises:
        TypeMismatchError:  date_str is not a string.
        InvalidValueError:  date_str does not parse as YYYY-MM-DD.
    """
    if not isinstance(date_str, str):
        raise TypeMismatchError(
            field=field,
            expected_type="str in YYYY-MM-DD format",
            actual_type=type(date_str).__name__,
        )
    try:
        datetime.strptime(date_str, _DATE_FORMAT)
    except ValueError:
        raise InvalidValueError(
            field=field,
            value=date_str,
            reason="must be in YYYY-MM-DD format (e.g. '2023-11-01')",
        )


def _validate_date_range_order(
    start_date: str,
    end_date: str,
) -> None:
    """
    Validate that start_date is strictly before end_date.

    Assumes both strings have already passed _validate_date_string().

    Raises:
        InvalidValueError: start_date >= end_date.
    """
    start_dt = datetime.strptime(start_date, _DATE_FORMAT)
    end_dt   = datetime.strptime(end_date,   _DATE_FORMAT)
    if start_dt >= end_dt:
        raise InvalidValueError(
            field="date_range",
            value=f"start={start_date}, end={end_date}",
            reason="start_date must be strictly before end_date",
        )


def _validate_cloud_cover_percent(value: float, field: str) -> None:
    """
    Validate that a cloud cover percentage is in the range [0, 100].

    Args:
        value: The percentage to validate.
        field: Dot-notation field name for error messages.

    Raises:
        TypeMismatchError:  value is not numeric.
        InvalidValueError:  value is outside [0, 100].
    """
    if not isinstance(value, (int, float)):
        raise TypeMismatchError(
            field=field,
            expected_type="float in range [0.0, 100.0]",
            actual_type=type(value).__name__,
        )
    if not (0.0 <= float(value) <= 100.0):
        raise InvalidValueError(
            field=field,
            value=value,
            reason="must be in range [0.0, 100.0]",
        )


# ==============================================================================
# Public filter functions
# ==============================================================================

def filter_by_date(
    collection: Any,
    start_date: str,
    end_date: str,
) -> Any:
    """
    Filter an ImageCollection to images acquired within [start_date, end_date].

    The interval is inclusive of start_date and exclusive of end_date,
    matching EE's filterDate() semantics. Both arguments must be provided.
    No getInfo() call is made; the returned collection is a lazy EE computation.

    Args:
        collection: An ee.ImageCollection to filter.
        start_date: Acquisition start date (inclusive), YYYY-MM-DD format.
        end_date:   Acquisition end date (exclusive), YYYY-MM-DD format.

    Returns:
        The input collection filtered to the specified date range.

    Raises:
        TypeMismatchError:  A date argument is not a string.
        InvalidValueError:  A date string is malformed or start >= end.
        GEEAPIError:        The EE filterDate() call raised an exception.
    """
    _validate_date_string(start_date, "filter_by_date.start_date")
    _validate_date_string(end_date,   "filter_by_date.end_date")
    _validate_date_range_order(start_date, end_date)

    _LOGGER.debug(
        "Applying date filter: [%s, %s)", start_date, end_date
    )

    try:
        filtered = collection.filterDate(start_date, end_date)
    except Exception as exc:
        raise GEEAPIError(
            operation="filter_by_date",
            reason=(
                f"EE filterDate([{start_date}, {end_date}]) failed: {exc}"
            ),
        ) from exc

    return filtered


def filter_by_bounds(
    collection: Any,
    geometry: Any,
) -> Any:
    """
    Filter an ImageCollection to images that intersect the given geometry.

    Uses EE's filterBounds() which retains images whose footprint
    intersects (not just contains) the geometry. No getInfo() call is made.

    Args:
        collection: An ee.ImageCollection to filter.
        geometry:   An ee.Geometry defining the spatial extent.
                    Must not be None.

    Returns:
        The input collection filtered to images intersecting the geometry.

    Raises:
        InvalidValueError: geometry is None.
        GEEAPIError:       The EE filterBounds() call raised an exception.
    """
    if geometry is None:
        raise InvalidValueError(
            field="filter_by_bounds.geometry",
            value=None,
            reason=(
                "geometry must not be None. "
                "Call client.get_aoi_geometry() or provide an ee.Geometry."
            ),
        )

    _LOGGER.debug("Applying spatial bounds filter.")

    try:
        filtered = collection.filterBounds(geometry)
    except Exception as exc:
        raise GEEAPIError(
            operation="filter_by_bounds",
            reason=f"EE filterBounds() failed: {exc}",
        ) from exc

    return filtered


def filter_by_cloud_cover(
    collection: Any,
    max_percent: float,
    property_name: str = CLOUD_COVER_PROPERTY,
) -> Any:
    """
    Filter an ImageCollection to images with cloud cover at or below max_percent.

    Applies an ee.Filter.lte() on the specified cloud cover property.
    No getInfo() call is made.

    The default property name is CLOUD_COVER, which is the scene-level
    cloud cover percentage in Landsat Collection 2 Level-2 products.

    Args:
        collection:    An ee.ImageCollection to filter.
        max_percent:   Maximum allowable cloud cover in percent [0, 100].
        property_name: EE image property containing the cloud cover value.
                       Defaults to CLOUD_COVER_PROPERTY ("CLOUD_COVER").

    Returns:
        The input collection filtered to images meeting the cloud limit.

    Raises:
        TypeMismatchError:  max_percent is not numeric.
        InvalidValueError:  max_percent is outside [0, 100].
        GEEAPIError:        The EE filter call raised an exception.
    """
    _validate_cloud_cover_percent(max_percent, "filter_by_cloud_cover.max_percent")

    _LOGGER.debug(
        "Applying cloud cover filter: %s <= %.1f%%",
        property_name, max_percent,
    )

    try:
        import ee
        filtered = collection.filter(
            ee.Filter.lte(property_name, float(max_percent))
        )
    except (ImportError, AttributeError) as exc:
        raise GEEAPIError(
            operation="filter_by_cloud_cover",
            reason=f"EE import or filter construction failed: {exc}",
        ) from exc
    except Exception as exc:
        raise GEEAPIError(
            operation="filter_by_cloud_cover",
            reason=(
                f"EE filter(lte('{property_name}', {max_percent})) failed: {exc}"
            ),
        ) from exc

    return filtered


def filter_by_sensor(
    collection: Any,
    spacecraft_ids: list[str],
) -> Any:
    """
    Filter an ImageCollection to images from the specified spacecraft(s).

    Applies an ee.Filter.inList() on the SPACECRAFT_ID property. Use this
    when a merged multi-sensor collection needs to be narrowed to specific
    satellites. No getInfo() call is made.

    Landsat SPACECRAFT_ID values:
        "LANDSAT_5"  -- Landsat 5 TM
        "LANDSAT_7"  -- Landsat 7 ETM+
        "LANDSAT_8"  -- Landsat 8 OLI/TIRS
        "LANDSAT_9"  -- Landsat 9 OLI-2/TIRS-2

    Args:
        collection:     An ee.ImageCollection to filter.
        spacecraft_ids: Non-empty list of SPACECRAFT_ID property values.

    Returns:
        The input collection filtered to the specified spacecraft(s).

    Raises:
        InvalidValueError: spacecraft_ids is empty.
        TypeMismatchError: spacecraft_ids is not a list.
        GEEAPIError:       The EE filter call raised an exception.
    """
    if not isinstance(spacecraft_ids, list):
        raise TypeMismatchError(
            field="filter_by_sensor.spacecraft_ids",
            expected_type="list[str]",
            actual_type=type(spacecraft_ids).__name__,
        )
    if not spacecraft_ids:
        raise InvalidValueError(
            field="filter_by_sensor.spacecraft_ids",
            value=spacecraft_ids,
            reason="spacecraft_ids must be a non-empty list",
        )

    _LOGGER.debug(
        "Applying sensor filter: SPACECRAFT_ID in %s", spacecraft_ids
    )

    try:
        import ee
        filtered = collection.filter(
            ee.Filter.inList(SPACECRAFT_ID_PROPERTY, spacecraft_ids)
        )
    except (ImportError, AttributeError) as exc:
        raise GEEAPIError(
            operation="filter_by_sensor",
            reason=f"EE import or filter construction failed: {exc}",
        ) from exc
    except Exception as exc:
        raise GEEAPIError(
            operation="filter_by_sensor",
            reason=(
                f"EE filter(inList('{SPACECRAFT_ID_PROPERTY}', "
                f"{spacecraft_ids})) failed: {exc}"
            ),
        ) from exc

    return filtered