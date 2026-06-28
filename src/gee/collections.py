"""
Landsat ImageCollection builder for the River Morphology Segmentation System.

LandsatCollectionBuilder is the single entry point for constructing
validated, filtered Landsat ImageCollections. It wraps EE collection
creation and applies composable filters from src.gee.filters.

Supported sensors (all Collection 2 Level-2 Surface Reflectance):
    LandsatSensor.L5  -- Landsat 5 TM  (April 1984 - November 2011)
    LandsatSensor.L7  -- Landsat 7 ETM+ (April 1999 - April 2022)
    LandsatSensor.L8  -- Landsat 8 OLI  (April 2013 - present)
    LandsatSensor.L9  -- Landsat 9 OLI-2 (October 2021 - present)

Band structure compatibility note:
    L5 and L7 use a different band numbering scheme than L8 and L9.
    Merging TM/ETM+ sensors with OLI sensors creates a collection
    with incompatible band structures. CollectionResult.has_mixed_sensor_families
    is True in this case. Module 5 (compositing) must handle harmonization.

Usage (fluent interface):

    from src.gee.collections import LandsatCollectionBuilder, LandsatSensor

    result = (
        LandsatCollectionBuilder(client=client, config=config)
        .with_date_range("2023-11-01", "2024-02-28")
        .with_aoi_from_config()
        .with_cloud_cover(20.0)
        .with_auto_sensors()
        .build()
    )

    collection  = result.collection    # ee.ImageCollection, ready for Module 5
    sensors     = result.sensors       # (LandsatSensor.L8, LandsatSensor.L9)

Usage (manual sensor selection):

    result = (
        LandsatCollectionBuilder(client=client, config=config)
        .with_date_range("2000-01-01", "2002-12-31")
        .with_aoi(custom_geometry)
        .with_cloud_cover(30.0)
        .with_sensors(LandsatSensor.L7)
        .build()
    )

Usage (with empty-collection validation):

    result = builder.build(validate_not_empty=True)
    # Raises GEEAPIError if the filtered collection contains zero images.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

from src.core.config import Config
from src.core.exceptions import InvalidValueError, MissingFieldError, TypeMismatchError
from src.gee import GEEAPIError, GEENotInstalledError
from src.gee.filters import (
    filter_by_bounds,
    filter_by_cloud_cover,
    filter_by_date,
)

if TYPE_CHECKING:
    from src.gee.client import EarthEngineClient

__all__ = [
    "LandsatSensor",
    "SensorAvailabilityPeriod",
    "CollectionResult",
    "LandsatCollectionBuilder",
    "SENSOR_AVAILABILITY",
    "SENSOR_COLLECTION_IDS",
    "SENSOR_SPACECRAFT_IDS",
    "VALID_COLLECTION_IDS",
]

_LOGGER: logging.Logger = logging.getLogger(__name__)

_DATE_FORMAT: str = "%Y-%m-%d"

# Default cloud cover limit used when not explicitly configured.
_DEFAULT_CLOUD_COVER_PCT: float = 20.0


# ==============================================================================
# LandsatSensor enum
# ==============================================================================

class LandsatSensor(str, Enum):
    """
    Landsat sensor identifiers matching the EE SPACECRAFT_ID property.

    As a str enum, values compare equal to the raw EE property strings:
        LandsatSensor.L8 == "LANDSAT_8"  # True
    """

    L5 = "LANDSAT_5"
    L7 = "LANDSAT_7"
    L8 = "LANDSAT_8"
    L9 = "LANDSAT_9"


# ==============================================================================
# Sensor availability periods
# ==============================================================================

@dataclass(frozen=True)
class SensorAvailabilityPeriod:
    """
    Immutable descriptor for a single Landsat sensor's operational period.

    Attributes:
        sensor:            LandsatSensor enum value.
        collection_id:     GEE ImageCollection asset path.
        spacecraft_id:     SPACECRAFT_ID property value in EE.
        operational_start: First date of available Collection 2 L2 imagery.
        operational_end:   Last date of available imagery. None = still operational.
        description:       Human-readable sensor description.
    """

    sensor:            LandsatSensor
    collection_id:     str
    spacecraft_id:     str
    operational_start: date
    operational_end:   date | None
    description:       str

    def overlaps_with(self, start: date, end: date) -> bool:
        """
        Return True if this sensor was operational during [start, end].

        Uses inclusive bounds on both ends. A sensor that ended on the
        query start date is considered overlapping (end-inclusive check).

        Args:
            start: Query date range start (inclusive).
            end:   Query date range end (inclusive).

        Returns:
            True if the sensor's operational period intersects [start, end].
        """
        # Sensor ended strictly before query starts.
        if self.operational_end is not None and self.operational_end < start:
            return False
        # Sensor started strictly after query ends.
        if self.operational_start > end:
            return False
        return True


# ==============================================================================
# Sensor availability registry
# ==============================================================================

# Ordered from oldest to newest. Ordering is used for preferred-sensor
# priority: later entries (newer sensors) are preferred when multiple
# sensors are available for the same date range.
SENSOR_AVAILABILITY: tuple[SensorAvailabilityPeriod, ...] = (
    SensorAvailabilityPeriod(
        sensor=LandsatSensor.L5,
        collection_id="LANDSAT/LT05/C02/T1_L2",
        spacecraft_id="LANDSAT_5",
        operational_start=date(1984, 4, 1),
        operational_end=date(2011, 11, 18),
        description=(
            "Landsat 5 TM C2 L2 (April 1984 - November 2011). "
            "Bands: SR_B1 Blue, SR_B2 Green, SR_B3 Red, "
            "SR_B4 NIR, SR_B5 SWIR1, SR_B7 SWIR2."
        ),
    ),
    SensorAvailabilityPeriod(
        sensor=LandsatSensor.L7,
        collection_id="LANDSAT/LE07/C02/T1_L2",
        spacecraft_id="LANDSAT_7",
        operational_start=date(1999, 4, 15),
        operational_end=date(2022, 4, 6),
        description=(
            "Landsat 7 ETM+ C2 L2 (April 1999 - April 2022). "
            "Scan-line corrector failure May 2003 causes data gaps. "
            "Bands: SR_B1 Blue, SR_B2 Green, SR_B3 Red, "
            "SR_B4 NIR, SR_B5 SWIR1, SR_B7 SWIR2."
        ),
    ),
    SensorAvailabilityPeriod(
        sensor=LandsatSensor.L8,
        collection_id="LANDSAT/LC08/C02/T1_L2",
        spacecraft_id="LANDSAT_8",
        operational_start=date(2013, 4, 11),
        operational_end=None,
        description=(
            "Landsat 8 OLI/TIRS C2 L2 (April 2013 - present). "
            "Bands: SR_B2 Blue, SR_B3 Green, SR_B4 Red, "
            "SR_B5 NIR, SR_B6 SWIR1, SR_B7 SWIR2."
        ),
    ),
    SensorAvailabilityPeriod(
        sensor=LandsatSensor.L9,
        collection_id="LANDSAT/LC09/C02/T1_L2",
        spacecraft_id="LANDSAT_9",
        operational_start=date(2021, 10, 31),
        operational_end=None,
        description=(
            "Landsat 9 OLI-2/TIRS-2 C2 L2 (October 2021 - present). "
            "Bands: SR_B2 Blue, SR_B3 Green, SR_B4 Red, "
            "SR_B5 NIR, SR_B6 SWIR1, SR_B7 SWIR2."
        ),
    ),
)

# Lookup tables derived from SENSOR_AVAILABILITY (single source of truth).
SENSOR_COLLECTION_IDS: dict[LandsatSensor, str] = {
    p.sensor: p.collection_id for p in SENSOR_AVAILABILITY
}

SENSOR_SPACECRAFT_IDS: dict[LandsatSensor, str] = {
    p.sensor: p.spacecraft_id for p in SENSOR_AVAILABILITY
}

VALID_COLLECTION_IDS: frozenset[str] = frozenset(
    p.collection_id for p in SENSOR_AVAILABILITY
)

# Sensor family groups for band structure compatibility checking.
# L5/L7 use SR_B1-SR_B5,SR_B7 numbering; L8/L9 use SR_B2-SR_B7.
_TM_ETM_SENSORS: frozenset[LandsatSensor] = frozenset(
    {LandsatSensor.L5, LandsatSensor.L7}
)
_OLI_SENSORS: frozenset[LandsatSensor] = frozenset(
    {LandsatSensor.L8, LandsatSensor.L9}
)


# ==============================================================================
# CollectionResult
# ==============================================================================

@dataclass(frozen=True)
class CollectionResult:
    """
    Immutable result of LandsatCollectionBuilder.build().

    Contains the ee.ImageCollection and build provenance metadata.
    No getInfo() calls are made to produce this object; all fields
    are derived from the builder configuration.

    Attributes:
        collection:         Filtered ee.ImageCollection, ready for Module 5.
        sensors:            Tuple of LandsatSensor values used in this build.
        collection_ids:     GEE asset paths for each sensor collection.
        start_date:         Applied date filter start (YYYY-MM-DD).
        end_date:           Applied date filter end (YYYY-MM-DD).
        cloud_cover_limit:  Applied cloud cover filter threshold (percent).
        filters_applied:    Human-readable description of each filter applied.
    """

    collection:        Any                     # ee.ImageCollection
    sensors:           tuple[LandsatSensor, ...]
    collection_ids:    tuple[str, ...]
    start_date:        str
    end_date:          str
    cloud_cover_limit: float
    filters_applied:   tuple[str, ...]

    @property
    def has_mixed_sensor_families(self) -> bool:
        """
        True if this result merges TM/ETM+ (L5/L7) with OLI (L8/L9) sensors.

        When True, the collection contains images with incompatible band
        structures. Module 5 must apply band harmonization before computing
        spectral indices that assume a common band naming scheme.

        See: USGS Landsat Collection 2 band mapping documentation.
        """
        sensor_set = set(self.sensors)
        has_tm_etm = bool(sensor_set & _TM_ETM_SENSORS)
        has_oli    = bool(sensor_set & _OLI_SENSORS)
        return has_tm_etm and has_oli

    @property
    def sensor_count(self) -> int:
        """Number of distinct sensors in this collection."""
        return len(self.sensors)

    def summary_lines(self) -> list[str]:
        """
        Return ASCII-formatted summary lines describing this collection build.

        Returns:
            List of strings suitable for logging or display.
        """
        lines = [
            f"  Sensors:       {[s.name for s in self.sensors]}",
            f"  Collections:   {list(self.collection_ids)}",
            f"  Date range:    {self.start_date} to {self.end_date}",
            f"  Cloud cover:   <= {self.cloud_cover_limit:.1f}%",
            f"  Filters:       {len(self.filters_applied)} applied",
        ]
        if self.has_mixed_sensor_families:
            lines.append(
                "  [WARN] Mixed sensor families detected. "
                "Band harmonization required before spectral index computation."
            )
        return lines


# ==============================================================================
# LandsatCollectionBuilder
# ==============================================================================

class LandsatCollectionBuilder:
    """
    Fluent builder for validated, filtered Landsat ImageCollections.

    Sensor selection, date filtering, spatial filtering, and cloud
    cover filtering are each independently configurable. The builder
    validates all parameters before making any EE API calls.

    Construction does not initialize EE or make network calls. EE calls
    happen only inside build().

    Thread safety: not thread-safe. Create one builder per call chain.

    Args:
        client: Initialized EarthEngineClient. Must be initialized before
                calling build() or with_aoi_from_config().
        config: Fully initialized Config object.

    Example (minimal - AOI and dates from config):

        result = (
            LandsatCollectionBuilder(client, config)
            .with_date_range_from_config()
            .with_aoi_from_config()
            .with_cloud_cover_from_config()
            .with_auto_sensors()
            .build()
        )

    Example (explicit parameters):

        result = (
            LandsatCollectionBuilder(client, config)
            .with_date_range("2023-11-01", "2024-02-28")
            .with_aoi(my_geometry)
            .with_cloud_cover(15.0)
            .with_sensors(LandsatSensor.L8, LandsatSensor.L9)
            .build()
        )
    """

    def __init__(
        self,
        client: EarthEngineClient,
        config: Config,
    ) -> None:
        self._client: EarthEngineClient = client
        self._config: Config            = config
        self._logger: logging.Logger    = logging.getLogger(__name__)

        # Mutable builder state. All start as None (unconfigured).
        self._start_date:        str | None            = None
        self._end_date:          str | None            = None
        self._geometry:          Any | None            = None
        self._max_cloud_cover:   float | None          = None
        self._sensors:           list[LandsatSensor] | None = None
        self._auto_sensors:      bool                  = False

    # ------------------------------------------------------------------
    # Fluent configuration methods (return self for chaining)
    # ------------------------------------------------------------------

    def with_date_range(
        self,
        start_date: str,
        end_date: str,
    ) -> LandsatCollectionBuilder:
        """
        Set the date range filter explicitly.

        Args:
            start_date: Start date (inclusive), YYYY-MM-DD format.
            end_date:   End date (exclusive in EE filterDate), YYYY-MM-DD.

        Returns:
            self (for method chaining).

        Raises:
            TypeMismatchError:  A date argument is not a string.
            InvalidValueError:  A date string is malformed or start >= end.
        """
        _validate_date_format(start_date, "with_date_range.start_date")
        _validate_date_format(end_date,   "with_date_range.end_date")
        _validate_date_order(start_date,  end_date)
        self._start_date = start_date
        self._end_date   = end_date
        self._logger.debug(
            "Date range set: [%s, %s)", start_date, end_date
        )
        return self

    def with_date_range_from_config(self) -> LandsatCollectionBuilder:
        """
        Set the date range from config.date_range.

        Raises:
            MissingFieldError: date_range.start or date_range.end is null
                               in config.yaml.
        """
        if not self._config.has_date_range:
            raise MissingFieldError(
                field="date_range.[start, end]",
                context=(
                    "Set date_range.start and date_range.end in config.yaml "
                    "before calling with_date_range_from_config(). "
                    "Example:\n"
                    "  date_range:\n"
                    "    start: '2023-11-01'\n"
                    "    end:   '2024-02-28'"
                ),
            )
        return self.with_date_range(
            str(self._config.date_range.start),
            str(self._config.date_range.end),
        )

    def with_aoi(self, geometry: Any) -> LandsatCollectionBuilder:
        """
        Set the AOI geometry to a pre-built ee.Geometry object.

        Use this when you have already constructed the geometry (e.g.,
        from a previous call to client.get_aoi_geometry() or a custom
        polygon). The geometry is stored and applied in build().

        Args:
            geometry: An ee.Geometry instance. Must not be None.

        Returns:
            self (for method chaining).

        Raises:
            InvalidValueError: geometry is None.
        """
        if geometry is None:
            raise InvalidValueError(
                field="with_aoi.geometry",
                value=None,
                reason=(
                    "geometry must not be None. "
                    "Call client.get_aoi_geometry() or build "
                    "an ee.Geometry object before calling with_aoi()."
                ),
            )
        self._geometry = geometry
        self._logger.debug("AOI geometry set (pre-built).")
        return self

    def with_aoi_from_config(self) -> LandsatCollectionBuilder:
        """
        Set the AOI geometry from config.aoi coordinates via the EE client.

        Calls client.get_aoi_geometry() which reads config.aoi and returns
        an ee.Geometry.Rectangle. Requires the EE client to be initialized
        and EE to be importable.

        Returns:
            self (for method chaining).

        Raises:
            MissingFieldError:  AOI coordinates are null in config.yaml.
            GEENotInstalledError: earthengine-api is not installed.
            GEEGeometryError:    EE rejected the geometry.
        """
        geometry = self._client.get_aoi_geometry()
        self._geometry = geometry
        self._logger.debug("AOI geometry set from config.")
        return self

    def with_cloud_cover(self, max_percent: float) -> LandsatCollectionBuilder:
        """
        Set the maximum allowable scene cloud cover percentage.

        Args:
            max_percent: Maximum cloud cover in percent [0.0, 100.0].

        Returns:
            self (for method chaining).

        Raises:
            TypeMismatchError:  max_percent is not numeric.
            InvalidValueError:  max_percent is outside [0.0, 100.0].
        """
        if not isinstance(max_percent, (int, float)):
            raise TypeMismatchError(
                field="with_cloud_cover.max_percent",
                expected_type="float in range [0.0, 100.0]",
                actual_type=type(max_percent).__name__,
            )
        if not (0.0 <= float(max_percent) <= 100.0):
            raise InvalidValueError(
                field="with_cloud_cover.max_percent",
                value=max_percent,
                reason="must be in range [0.0, 100.0]",
            )
        self._max_cloud_cover = float(max_percent)
        self._logger.debug("Cloud cover limit set: %.1f%%", max_percent)
        return self

    def with_cloud_cover_from_config(self) -> LandsatCollectionBuilder:
        """
        Set the cloud cover limit from config.satellite.max_cloud_cover_percent.

        Returns:
            self (for method chaining).
        """
        max_cc = float(self._config.satellite.max_cloud_cover_percent)
        return self.with_cloud_cover(max_cc)

    def with_auto_sensors(self) -> LandsatCollectionBuilder:
        """
        Enable automatic sensor selection based on the configured date range.

        Selects all Landsat C2 L2 sensors that were operational during the
        configured date range. Sensor selection is deferred to build() so that
        with_date_range() can be called after with_auto_sensors().

        For date ranges before 2013-04-11: L5 and/or L7 are selected.
        For 2013-04-11 to 2021-10-30: L8 only.
        For 2021-10-31 onwards: L8 and L9 (config default).

        Returns:
            self (for method chaining).
        """
        self._auto_sensors = True
        self._sensors      = None
        self._logger.debug("Auto sensor selection enabled.")
        return self

    def with_sensors(
        self,
        *sensors: LandsatSensor,
    ) -> LandsatCollectionBuilder:
        """
        Set the sensors to use explicitly, bypassing auto-selection.

        Args:
            *sensors: One or more LandsatSensor enum values.

        Returns:
            self (for method chaining).

        Raises:
            InvalidValueError: No sensors provided.
        """
        sensor_list = list(sensors)
        if not sensor_list:
            raise InvalidValueError(
                field="with_sensors.sensors",
                value=[],
                reason=(
                    "At least one LandsatSensor must be provided. "
                    "Example: .with_sensors(LandsatSensor.L8, LandsatSensor.L9)"
                ),
            )
        self._sensors      = sensor_list
        self._auto_sensors = False
        self._logger.debug(
            "Manual sensors set: %s", [s.name for s in sensor_list]
        )
        return self

    def with_sensors_from_config(self) -> LandsatCollectionBuilder:
        """
        Set the sensors from config.satellite.collections.

        Resolves each collection ID in config.satellite.collections to a
        LandsatSensor enum value. Raises if any ID is unrecognized.

        Returns:
            self (for method chaining).

        Raises:
            InvalidValueError: A collection ID in config is not a known
                               Landsat C2 L2 collection.
        """
        collection_id_to_sensor: dict[str, LandsatSensor] = {
            p.collection_id: p.sensor for p in SENSOR_AVAILABILITY
        }
        resolved: list[LandsatSensor] = []

        for cid in self._config.satellite.collections:
            sensor = collection_id_to_sensor.get(cid)
            if sensor is None:
                raise InvalidValueError(
                    field="satellite.collections",
                    value=cid,
                    reason=(
                        f"Unrecognized Landsat C2 L2 collection ID. "
                        f"Valid IDs: {sorted(VALID_COLLECTION_IDS)}"
                    ),
                )
            resolved.append(sensor)

        return self.with_sensors(*resolved)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(
        self,
        validate_not_empty: bool = False,
    ) -> CollectionResult:
        """
        Validate all parameters and return a filtered Landsat ImageCollection.

        Build sequence:
            1. Validate that all required fields are set.
            2. Resolve sensors (auto-selection or explicit list).
            3. Create and merge per-sensor ImageCollections (server-side).
            4. Apply date, bounds, and cloud cover filters (server-side).
            5. Optionally verify the filtered collection is non-empty.
            6. Return CollectionResult with collection and provenance metadata.

        getInfo() is called ONLY when validate_not_empty=True. All other
        steps are lazy server-side EE computations.

        Args:
            validate_not_empty: When True, call size().getInfo() to confirm
                                that at least one image passed all filters.
                                Raises GEEAPIError if the collection is empty.

        Returns:
            CollectionResult with the filtered ee.ImageCollection.

        Raises:
            MissingFieldError:   Required fields (date range, AOI) not set.
            GEENotInstalledError: earthengine-api is not installed.
            GEEAPIError:         EE collection creation, merge, or filter failed.
                                 Also raised when validate_not_empty=True and
                                 the collection is empty.
        """
        self._validate_build_requirements()

        sensors = self._resolve_sensors()
        self._logger.info(
            "Building Landsat collection. sensors=%s, date=[%s, %s], "
            "cloud=%.1f%%",
            [s.name for s in sensors],
            self._start_date,
            self._end_date,
            self._max_cloud_cover,
        )

        collection       = self._build_merged_collection(sensors)
        collection, tags = self._apply_all_filters(collection)

        if validate_not_empty:
            self._check_not_empty(collection)

        collection_ids = tuple(SENSOR_COLLECTION_IDS[s] for s in sensors)
        result = CollectionResult(
            collection=collection,
            sensors=tuple(sensors),
            collection_ids=collection_ids,
            start_date=self._start_date,
            end_date=self._end_date,
            cloud_cover_limit=self._max_cloud_cover,
            filters_applied=tuple(tags),
        )

        for line in result.summary_lines():
            self._logger.info(line)

        return result

    # ------------------------------------------------------------------
    # Private build helpers
    # ------------------------------------------------------------------

    def _validate_build_requirements(self) -> None:
        """
        Confirm that all required fields are set before attempting any EE call.

        Raises:
            MissingFieldError: Any required field is absent.
        """
        if self._start_date is None or self._end_date is None:
            raise MissingFieldError(
                field="date_range",
                context=(
                    "Call with_date_range() or with_date_range_from_config() "
                    "before calling build()."
                ),
            )
        if self._geometry is None:
            raise MissingFieldError(
                field="aoi_geometry",
                context=(
                    "Call with_aoi() or with_aoi_from_config() "
                    "before calling build()."
                ),
            )
        if not self._auto_sensors and self._sensors is None:
            raise MissingFieldError(
                field="sensors",
                context=(
                    "Call with_auto_sensors(), with_sensors(), or "
                    "with_sensors_from_config() before calling build()."
                ),
            )
        if self._max_cloud_cover is None:
            self._max_cloud_cover = _DEFAULT_CLOUD_COVER_PCT
            self._logger.debug(
                "Cloud cover not set; using default %.1f%%.",
                _DEFAULT_CLOUD_COVER_PCT,
            )

    def _resolve_sensors(self) -> list[LandsatSensor]:
        """
        Return the list of sensors to use in this build.

        For manual selection: returns the explicitly provided list.
        For auto-selection: finds all sensors operational during the
        configured date range.

        Raises:
            InvalidValueError: Auto-selection finds no sensor for the range.
        """
        if not self._auto_sensors and self._sensors is not None:
            return list(self._sensors)

        # Auto-selection: find sensors that overlap with the date range.
        start_dt = datetime.strptime(self._start_date, _DATE_FORMAT).date()
        end_dt   = datetime.strptime(self._end_date,   _DATE_FORMAT).date()

        available = [
            period.sensor
            for period in SENSOR_AVAILABILITY
            if period.overlaps_with(start_dt, end_dt)
        ]

        if not available:
            raise InvalidValueError(
                field="date_range",
                value=f"{self._start_date} to {self._end_date}",
                reason=(
                    "No Landsat Collection 2 Level-2 sensor was operational "
                    "during this date range. "
                    f"Valid range: {SENSOR_AVAILABILITY[0].operational_start} "
                    f"to present."
                ),
            )

        self._logger.debug(
            "Auto-selected sensors: %s", [s.name for s in available]
        )
        return available

    def _build_merged_collection(
        self,
        sensors: list[LandsatSensor],
    ) -> Any:
        """
        Create and merge per-sensor ImageCollections into a single collection.

        For a single sensor: returns ee.ImageCollection(collection_id).
        For multiple sensors: merges them with .merge() calls.

        Args:
            sensors: Non-empty list of LandsatSensor values.

        Returns:
            An unfiltered merged ee.ImageCollection.

        Raises:
            GEENotInstalledError: earthengine-api is not installed.
            GEEAPIError:          ee.ImageCollection() or .merge() failed.
        """
        try:
            import ee
        except ImportError as exc:
            raise GEENotInstalledError(
                "earthengine-api is not installed. "
                "Install with: pip install earthengine-api==0.1.390"
            ) from exc

        collection_ids = [SENSOR_COLLECTION_IDS[s] for s in sensors]

        try:
            merged = ee.ImageCollection(collection_ids[0])
            for cid in collection_ids[1:]:
                merged = merged.merge(ee.ImageCollection(cid))
        except Exception as exc:
            raise GEEAPIError(
                operation="build_merged_collection",
                reason=(
                    f"Failed to create or merge ImageCollections "
                    f"{collection_ids}: {exc}"
                ),
            ) from exc

        self._logger.debug(
            "Merged %d collection(s): %s", len(collection_ids), collection_ids
        )
        return merged

    def _apply_all_filters(
        self,
        collection: Any,
    ) -> tuple[Any, list[str]]:
        """
        Apply date, bounds, and cloud cover filters to the collection.

        Filters are applied in a fixed order:
            1. Date filter  (reduces collection size early)
            2. Bounds filter (spatial intersection)
            3. Cloud cover filter (image quality)

        Args:
            collection: An unfiltered ee.ImageCollection.

        Returns:
            Tuple of (filtered_collection, list_of_filter_descriptions).

        Raises:
            GEEAPIError: Any individual filter call failed.
        """
        filters_applied: list[str] = []

        collection = filter_by_date(
            collection, self._start_date, self._end_date
        )
        filters_applied.append(
            f"date: [{self._start_date}, {self._end_date})"
        )

        collection = filter_by_bounds(collection, self._geometry)
        filters_applied.append("bounds: AOI geometry")

        collection = filter_by_cloud_cover(
            collection, self._max_cloud_cover
        )
        filters_applied.append(
            f"cloud_cover: <= {self._max_cloud_cover:.1f}%"
        )

        return collection, filters_applied

    def _check_not_empty(self, collection: Any) -> None:
        """
        Verify the filtered collection contains at least one image.

        This is the ONLY getInfo() call in the builder. It is guarded
        behind the validate_not_empty=True parameter of build().

        Args:
            collection: A filtered ee.ImageCollection.

        Raises:
            GEEAPIError: The collection is empty after all filters, or
                         the size().getInfo() call failed.
        """
        try:
            count = self._client.execute_with_retry(
                collection.size().getInfo
            )
        except Exception as exc:
            raise GEEAPIError(
                operation="validate_not_empty",
                reason=f"Failed to retrieve collection size: {exc}",
            ) from exc

        if count == 0:
            raise GEEAPIError(
                operation="validate_not_empty",
                reason=(
                    "The filtered ImageCollection contains zero images. "
                    f"Date range: [{self._start_date}, {self._end_date}), "
                    f"cloud cover: <= {self._max_cloud_cover:.1f}%. "
                    "Consider relaxing the cloud cover limit, widening the "
                    "date range, or checking that imagery exists for the AOI."
                ),
            )

        self._logger.info(
            "Collection validated: %d image(s) available.", count
        )


# ==============================================================================
# Private module-level validation helpers (used by builder methods)
# ==============================================================================

def _validate_date_format(date_str: str, field: str) -> None:
    """Validate YYYY-MM-DD format. Raises TypeMismatchError or InvalidValueError."""
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


def _validate_date_order(start_date: str, end_date: str) -> None:
    """Validate start_date < end_date. Assumes both are valid YYYY-MM-DD strings."""
    start_dt = datetime.strptime(start_date, _DATE_FORMAT)
    end_dt   = datetime.strptime(end_date,   _DATE_FORMAT)
    if start_dt >= end_dt:
        raise InvalidValueError(
            field="date_range",
            value=f"start={start_date}, end={end_date}",
            reason="start_date must be strictly before end_date",
        )