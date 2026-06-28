"""
Unit tests for src/gee/collections.py.

Tests cover:
    - LandsatSensor enum
    - SensorAvailabilityPeriod.overlaps_with()
    - SENSOR_AVAILABILITY registry integrity
    - CollectionResult properties
    - LandsatCollectionBuilder fluent API
    - Auto sensor selection for various date ranges
    - Manual sensor selection
    - Validation errors (missing fields, invalid values)
    - build() success and error paths
    - validate_not_empty integration

Run:
    pytest tests/test_collections.py -v
    pytest tests/test_collections.py -v --cov=src/gee/collections --cov-report=term-missing
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.core.exceptions import InvalidValueError, MissingFieldError
from src.gee import GEEAPIError, GEENotInstalledError
from src.gee.collections import (
    SENSOR_AVAILABILITY,
    SENSOR_COLLECTION_IDS,
    SENSOR_SPACECRAFT_IDS,
    VALID_COLLECTION_IDS,
    CollectionResult,
    LandsatCollectionBuilder,
    LandsatSensor,
    SensorAvailabilityPeriod,
)
from src.core.exceptions import (
    InvalidValueError,
    TypeMismatchError,
)
from tests.conftest import make_valid_config, write_config


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def mock_ee() -> MagicMock:
    ee = MagicMock()
    mock_col = MagicMock()
    mock_col.filterDate.return_value   = mock_col
    mock_col.filterBounds.return_value = mock_col
    mock_col.filter.return_value       = mock_col
    mock_col.merge.return_value        = mock_col
    ee.ImageCollection.return_value    = mock_col
    ee.Filter.lte.return_value         = MagicMock()
    ee.Filter.inList.return_value      = MagicMock()
    return ee


@pytest.fixture
def mock_client() -> MagicMock:
    from src.gee.client import EarthEngineClient
    client = MagicMock(spec=EarthEngineClient)
    client.is_initialized = True
    client.get_aoi_geometry.return_value = MagicMock()
    client.execute_with_retry.side_effect = lambda func, *a, **kw: func(*a, **kw)
    return client


@pytest.fixture
def config_with_aoi_and_dates(tmp_path: Path):
    from src.core.config import Config
    data = make_valid_config()
    data["aoi"].update({
        "min_lon": 87.0, "min_lat": 26.0,
        "max_lon": 87.5, "max_lat": 26.5,
    })
    data["date_range"].update({
        "start": "2023-11-01", "end": "2024-02-28"
    })
    return Config(config_path=write_config(tmp_path, data))


@pytest.fixture
def builder(mock_client, config_with_aoi_and_dates) -> LandsatCollectionBuilder:
    return LandsatCollectionBuilder(
        client=mock_client,
        config=config_with_aoi_and_dates,
    )


@contextmanager
def patch_ee(mock_ee: MagicMock):
    with patch.dict(sys.modules, {"ee": mock_ee}):
        yield


# ==============================================================================
# LandsatSensor enum tests
# ==============================================================================

class TestLandsatSensor:
    """Tests for the LandsatSensor string enum."""

    def test_values_are_ascii(self) -> None:
        for sensor in LandsatSensor:
            assert all(ord(c) < 128 for c in sensor.value)

    def test_is_str_subclass(self) -> None:
        assert isinstance(LandsatSensor.L8, str)

    def test_compares_equal_to_spacecraft_id_strings(self) -> None:
        assert LandsatSensor.L5 == "LANDSAT_5"
        assert LandsatSensor.L7 == "LANDSAT_7"
        assert LandsatSensor.L8 == "LANDSAT_8"
        assert LandsatSensor.L9 == "LANDSAT_9"

    def test_all_four_sensors_defined(self) -> None:
        sensors = {s.name for s in LandsatSensor}
        assert sensors == {"L5", "L7", "L8", "L9"}


# ==============================================================================
# SensorAvailabilityPeriod tests
# ==============================================================================

class TestSensorAvailabilityPeriod:
    """Tests for SensorAvailabilityPeriod.overlaps_with()."""

    def _period(
        self,
        start: date,
        end: date | None,
        sensor: LandsatSensor = LandsatSensor.L8,
    ) -> SensorAvailabilityPeriod:
        return SensorAvailabilityPeriod(
            sensor=sensor,
            collection_id="TEST/COLLECTION",
            spacecraft_id=sensor.value,
            operational_start=start,
            operational_end=end,
            description="Test period",
        )

    def test_fully_inside_range(self) -> None:
        period = self._period(date(2013, 4, 11), None)
        assert period.overlaps_with(date(2023, 1, 1), date(2023, 12, 31)) is True

    def test_range_starts_before_sensor_ends_after(self) -> None:
        period = self._period(date(2013, 4, 11), date(2015, 1, 1))
        assert period.overlaps_with(date(2014, 6, 1), date(2016, 6, 1)) is True

    def test_sensor_ended_before_range_starts(self) -> None:
        period = self._period(date(1984, 4, 1), date(2011, 11, 18))
        assert period.overlaps_with(date(2012, 1, 1), date(2013, 1, 1)) is False

    def test_sensor_started_after_range_ends(self) -> None:
        period = self._period(date(2021, 10, 31), None)
        assert period.overlaps_with(date(2019, 1, 1), date(2020, 12, 31)) is False

    def test_sensor_end_equals_query_start(self) -> None:
        """End-inclusive: sensor ending on query start date overlaps."""
        period = self._period(date(1999, 4, 15), date(2022, 4, 6))
        assert period.overlaps_with(date(2022, 4, 6), date(2022, 12, 31)) is True

    def test_sensor_start_equals_query_end(self) -> None:
        """Start-inclusive: sensor starting on query end date overlaps."""
        period = self._period(date(2021, 10, 31), None)
        assert period.overlaps_with(date(2020, 1, 1), date(2021, 10, 31)) is True

    def test_still_operational_sensor_always_overlaps_from_start(self) -> None:
        period = self._period(date(2013, 4, 11), None)
        assert period.overlaps_with(date(2020, 1, 1), date(2025, 12, 31)) is True


# ==============================================================================
# SENSOR_AVAILABILITY registry tests
# ==============================================================================

class TestSensorAvailabilityRegistry:
    """Tests for the SENSOR_AVAILABILITY constant and derived lookups."""

    def test_has_four_entries(self) -> None:
        assert len(SENSOR_AVAILABILITY) == 4

    def test_all_sensors_represented(self) -> None:
        sensors = {p.sensor for p in SENSOR_AVAILABILITY}
        assert sensors == set(LandsatSensor)

    def test_collection_ids_unique(self) -> None:
        ids = [p.collection_id for p in SENSOR_AVAILABILITY]
        assert len(ids) == len(set(ids))

    def test_spacecraft_ids_unique(self) -> None:
        ids = [p.spacecraft_id for p in SENSOR_AVAILABILITY]
        assert len(ids) == len(set(ids))

    def test_sensor_collection_ids_derived_correctly(self) -> None:
        for period in SENSOR_AVAILABILITY:
            assert SENSOR_COLLECTION_IDS[period.sensor] == period.collection_id

    def test_sensor_spacecraft_ids_derived_correctly(self) -> None:
        for period in SENSOR_AVAILABILITY:
            assert SENSOR_SPACECRAFT_IDS[period.sensor] == period.spacecraft_id

    def test_valid_collection_ids_contains_all(self) -> None:
        for period in SENSOR_AVAILABILITY:
            assert period.collection_id in VALID_COLLECTION_IDS

    def test_descriptions_are_ascii_only(self) -> None:
        for period in SENSOR_AVAILABILITY:
            assert all(ord(c) < 128 for c in period.description)

    def test_l8_collection_id(self) -> None:
        assert SENSOR_COLLECTION_IDS[LandsatSensor.L8] == "LANDSAT/LC08/C02/T1_L2"

    def test_l9_collection_id(self) -> None:
        assert SENSOR_COLLECTION_IDS[LandsatSensor.L9] == "LANDSAT/LC09/C02/T1_L2"

    def test_l5_operational_end_is_not_none(self) -> None:
        """L5 is decommissioned; operational_end must not be None."""
        l5 = next(p for p in SENSOR_AVAILABILITY if p.sensor == LandsatSensor.L5)
        assert l5.operational_end is not None

    def test_l8_operational_end_is_none(self) -> None:
        """L8 is still operational; operational_end must be None."""
        l8 = next(p for p in SENSOR_AVAILABILITY if p.sensor == LandsatSensor.L8)
        assert l8.operational_end is None


# ==============================================================================
# CollectionResult tests
# ==============================================================================

class TestCollectionResult:
    """Tests for CollectionResult computed properties."""

    def _make_result(
        self,
        sensors: tuple[LandsatSensor, ...],
    ) -> CollectionResult:
        return CollectionResult(
            collection=MagicMock(),
            sensors=sensors,
            collection_ids=tuple(SENSOR_COLLECTION_IDS[s] for s in sensors),
            start_date="2023-11-01",
            end_date="2024-02-28",
            cloud_cover_limit=20.0,
            filters_applied=("date", "bounds", "cloud_cover"),
        )

    def test_has_mixed_sensor_families_true(self) -> None:
        result = self._make_result((LandsatSensor.L7, LandsatSensor.L8))
        assert result.has_mixed_sensor_families is True

    def test_has_mixed_sensor_families_false_oli_only(self) -> None:
        result = self._make_result((LandsatSensor.L8, LandsatSensor.L9))
        assert result.has_mixed_sensor_families is False

    def test_has_mixed_sensor_families_false_tm_only(self) -> None:
        result = self._make_result((LandsatSensor.L5, LandsatSensor.L7))
        assert result.has_mixed_sensor_families is False

    def test_sensor_count(self) -> None:
        result = self._make_result((LandsatSensor.L8, LandsatSensor.L9))
        assert result.sensor_count == 2

    def test_sensor_count_single(self) -> None:
        result = self._make_result((LandsatSensor.L8,))
        assert result.sensor_count == 1

    def test_frozen_prevents_mutation(self) -> None:
        result = self._make_result((LandsatSensor.L8,))
        with pytest.raises((AttributeError, TypeError)):
            result.start_date = "2020-01-01"  # type: ignore[misc]

    def test_summary_lines_are_ascii_only(self) -> None:
        result = self._make_result((LandsatSensor.L8, LandsatSensor.L9))
        for line in result.summary_lines():
            assert all(ord(c) < 128 for c in line)

    def test_mixed_sensor_warning_in_summary(self) -> None:
        result = self._make_result((LandsatSensor.L7, LandsatSensor.L8))
        combined = " ".join(result.summary_lines())
        assert "Mixed sensor" in combined or "WARN" in combined


# ==============================================================================
# LandsatCollectionBuilder — configuration method tests
# ==============================================================================

class TestBuilderConfiguration:
    """Tests for the fluent configuration methods on LandsatCollectionBuilder."""

    def test_construction_does_not_call_ee(
        self, mock_client, config_with_aoi_and_dates
    ) -> None:
        """Builder construction must not make any EE calls."""
        builder = LandsatCollectionBuilder(mock_client, config_with_aoi_and_dates)
        mock_client.get_aoi_geometry.assert_not_called()

    def test_with_date_range_returns_self(self, builder) -> None:
        result = builder.with_date_range("2023-11-01", "2024-02-28")
        assert result is builder

    def test_with_aoi_returns_self(self, builder, mock_geometry=MagicMock()) -> None:
        result = builder.with_aoi(mock_geometry)
        assert result is builder

    def test_with_aoi_none_raises(self, builder) -> None:
        with pytest.raises(InvalidValueError, match="must not be None"):
            builder.with_aoi(None)

    def test_with_cloud_cover_returns_self(self, builder) -> None:
        result = builder.with_cloud_cover(20.0)
        assert result is builder

    def test_with_auto_sensors_returns_self(self, builder) -> None:
        result = builder.with_auto_sensors()
        assert result is builder

    def test_with_sensors_returns_self(self, builder) -> None:
        result = builder.with_sensors(LandsatSensor.L8)
        assert result is builder

    def test_with_sensors_empty_raises(self, builder) -> None:
        with pytest.raises(
            InvalidValueError,
            match="At least one LandsatSensor"
        ):
            builder.with_sensors()

    def test_with_date_range_invalid_format_raises(self, builder) -> None:
        with pytest.raises(InvalidValueError, match="YYYY-MM-DD"):
            builder.with_date_range("2023/11/01", "2024-02-28")

    def test_with_date_range_start_after_end_raises(self, builder) -> None:
        with pytest.raises(InvalidValueError, match="strictly before"):
            builder.with_date_range("2024-06-01", "2023-01-01")

    def test_with_cloud_cover_negative_raises(self, builder) -> None:
        with pytest.raises(InvalidValueError, match="range"):
            builder.with_cloud_cover(-5.0)

    def test_with_cloud_cover_over_100_raises(self, builder) -> None:
        with pytest.raises(InvalidValueError, match="range"):
            builder.with_cloud_cover(105.0)

    def test_with_cloud_cover_string_raises(self, builder) -> None:
        with pytest.raises(TypeMismatchError):
            builder.with_cloud_cover("twenty")

    def test_with_date_range_from_config_success(
        self, builder
    ) -> None:
        result = builder.with_date_range_from_config()
        assert result is builder
        assert builder._start_date == "2023-11-01"
        assert builder._end_date   == "2024-02-28"

    def test_with_date_range_from_config_missing_raises(
        self, mock_client, tmp_path: Path
    ) -> None:
        from src.core.config import Config
        data = make_valid_config()  # date_range is null by default
        cfg  = Config(config_path=write_config(tmp_path, data))
        b    = LandsatCollectionBuilder(mock_client, cfg)
        with pytest.raises(MissingFieldError):
            b.with_date_range_from_config()

    def test_with_aoi_from_config_calls_client(
        self, builder, mock_client
    ) -> None:
        builder.with_aoi_from_config()
        mock_client.get_aoi_geometry.assert_called_once()

    def test_with_cloud_cover_from_config(
        self, builder, config_with_aoi_and_dates
    ) -> None:
        builder.with_cloud_cover_from_config()
        expected = float(config_with_aoi_and_dates.satellite.max_cloud_cover_percent)
        assert builder._max_cloud_cover == expected

    def test_with_sensors_from_config_resolves_l8_l9(
        self, builder, config_with_aoi_and_dates
    ) -> None:
        builder.with_sensors_from_config()
        sensor_names = {s.name for s in builder._sensors}
        assert "L8" in sensor_names
        assert "L9" in sensor_names

    def test_with_sensors_from_config_invalid_id_raises(
        self, mock_client, tmp_path: Path
    ) -> None:
        from src.core.config import Config
        data = make_valid_config()
        data["satellite"]["collections"] = ["INVALID/COLLECTION/ID"]
        cfg  = Config(config_path=write_config(tmp_path, data))
        b    = LandsatCollectionBuilder(mock_client, cfg)
        with pytest.raises(InvalidValueError, match="Unrecognized"):
            b.with_sensors_from_config()

    def test_chaining_all_methods(
        self, builder, mock_client, mock_ee: MagicMock
    ) -> None:
        """All configuration methods can be chained fluently."""
        geometry = MagicMock()
        with patch_ee(mock_ee):
            result = (
                builder
                .with_date_range("2023-11-01", "2024-02-28")
                .with_aoi(geometry)
                .with_cloud_cover(20.0)
                .with_sensors(LandsatSensor.L8, LandsatSensor.L9)
                .build()
            )
        assert isinstance(result, CollectionResult)


# ==============================================================================
# LandsatCollectionBuilder — auto sensor selection tests
# ==============================================================================

class TestAutoSensorSelection:
    """Tests for _resolve_sensors() in auto-selection mode."""

    def _builder_with_dates(
        self,
        mock_client,
        config,
        start: str,
        end: str,
    ) -> LandsatCollectionBuilder:
        return (
            LandsatCollectionBuilder(mock_client, config)
            .with_date_range(start, end)
            .with_aoi(MagicMock())
            .with_auto_sensors()
            .with_cloud_cover(20.0)
        )

    def test_l8_l9_selected_for_recent_dates(
        self, mock_client, config_with_aoi_and_dates, mock_ee: MagicMock
    ) -> None:
        b = self._builder_with_dates(
            mock_client, config_with_aoi_and_dates,
            "2023-01-01", "2023-12-31"
        )
        with patch_ee(mock_ee):
            result = b.build()
        assert LandsatSensor.L8 in result.sensors
        assert LandsatSensor.L9 in result.sensors

    def test_l8_only_selected_for_pre_l9_dates(
        self, mock_client, config_with_aoi_and_dates, mock_ee: MagicMock
    ) -> None:
        b = self._builder_with_dates(
            mock_client, config_with_aoi_and_dates,
            "2015-01-01", "2016-12-31"
        )
        with patch_ee(mock_ee):
            result = b.build()
        assert LandsatSensor.L8 in result.sensors
        assert LandsatSensor.L9 not in result.sensors

    def test_l5_and_l7_selected_for_early_2000s(
        self, mock_client, config_with_aoi_and_dates, mock_ee: MagicMock
    ) -> None:
        b = self._builder_with_dates(
            mock_client, config_with_aoi_and_dates,
            "2002-01-01", "2003-12-31"
        )
        with patch_ee(mock_ee):
            result = b.build()
        assert LandsatSensor.L7 in result.sensors

    def test_no_sensor_available_for_invalid_range_raises(
        self, mock_client, config_with_aoi_and_dates, mock_ee: MagicMock
    ) -> None:
        # Before Landsat 5 was operational.
        b = self._builder_with_dates(
            mock_client, config_with_aoi_and_dates,
            "1980-01-01", "1983-12-31"
        )
        with patch_ee(mock_ee):
            with pytest.raises(InvalidValueError, match="No Landsat"):
                b.build()


# ==============================================================================
# LandsatCollectionBuilder — build() tests
# ==============================================================================

class TestBuilderBuild:
    """Tests for build() success and error paths."""

    def test_build_returns_collection_result(
        self, builder, mock_ee: MagicMock
    ) -> None:
        geometry = MagicMock()
        with patch_ee(mock_ee):
            result = (
                builder
                .with_date_range("2023-11-01", "2024-02-28")
                .with_aoi(geometry)
                .with_cloud_cover(20.0)
                .with_sensors(LandsatSensor.L8, LandsatSensor.L9)
                .build()
            )
        assert isinstance(result, CollectionResult)

    def test_build_without_date_range_raises(
        self, builder, mock_ee: MagicMock
    ) -> None:
        builder.with_aoi(MagicMock()).with_sensors(LandsatSensor.L8)
        with patch_ee(mock_ee):
            with pytest.raises(MissingFieldError, match="date_range"):
                builder.build()

    def test_build_without_aoi_raises(
        self, builder, mock_ee: MagicMock
    ) -> None:
        builder.with_date_range("2023-11-01", "2024-02-28")
        builder.with_sensors(LandsatSensor.L8)
        with patch_ee(mock_ee):
            with pytest.raises(MissingFieldError, match="aoi_geometry"):
                builder.build()

    def test_build_without_sensors_set_raises(
        self, mock_client, config_with_aoi_and_dates, mock_ee: MagicMock
    ) -> None:
        b = LandsatCollectionBuilder(mock_client, config_with_aoi_and_dates)
        b.with_date_range("2023-11-01", "2024-02-28")
        b.with_aoi(MagicMock())
        # Neither with_auto_sensors nor with_sensors called.
        with patch_ee(mock_ee):
            with pytest.raises(MissingFieldError, match="sensors"):
                b.build()

    def test_build_creates_image_collection_per_sensor(
        self, builder, mock_ee: MagicMock
    ) -> None:
        geometry = MagicMock()
        with patch_ee(mock_ee):
            builder.with_date_range("2023-11-01", "2024-02-28")
            builder.with_aoi(geometry)
            builder.with_cloud_cover(20.0)
            builder.with_sensors(LandsatSensor.L8, LandsatSensor.L9)
            builder.build()
        assert mock_ee.ImageCollection.call_count >= 2

    def test_build_applies_three_filters(
        self, builder, mock_ee: MagicMock
    ) -> None:
        geometry = MagicMock()
        mock_col = mock_ee.ImageCollection.return_value
        with patch_ee(mock_ee):
            builder.with_date_range("2023-11-01", "2024-02-28")
            builder.with_aoi(geometry)
            builder.with_cloud_cover(20.0)
            builder.with_sensors(LandsatSensor.L8)
            result = builder.build()
        assert len(result.filters_applied) == 3

    def test_build_result_sensors_match_requested(
        self, builder, mock_ee: MagicMock
    ) -> None:
        geometry = MagicMock()
        with patch_ee(mock_ee):
            result = (
                builder
                .with_date_range("2023-11-01", "2024-02-28")
                .with_aoi(geometry)
                .with_cloud_cover(20.0)
                .with_sensors(LandsatSensor.L8)
                .build()
            )
        assert LandsatSensor.L8 in result.sensors
        assert LandsatSensor.L9 not in result.sensors

    def test_build_uses_default_cloud_cover_when_not_set(
        self, mock_client, config_with_aoi_and_dates, mock_ee: MagicMock
    ) -> None:
        b = LandsatCollectionBuilder(mock_client, config_with_aoi_and_dates)
        with patch_ee(mock_ee):
            result = (
                b
                .with_date_range("2023-11-01", "2024-02-28")
                .with_aoi(MagicMock())
                .with_sensors(LandsatSensor.L8)
                .build()
            )
        assert result.cloud_cover_limit == 20.0

    def test_build_ee_not_installed_raises_gee_not_installed_error(
        self, builder
    ) -> None:
        builder.with_date_range("2023-11-01", "2024-02-28")
        builder.with_aoi(MagicMock())
        builder.with_cloud_cover(20.0)
        builder.with_sensors(LandsatSensor.L8)
        modules_without_ee = {k: v for k, v in sys.modules.items() if k != "ee"}
        with patch.dict(sys.modules, modules_without_ee, clear=True):
            with patch("builtins.__import__", side_effect=_block_ee_import):
                with pytest.raises(GEENotInstalledError):
                    builder.build()

    def test_build_ee_api_error_on_collection_creation(
        self, builder, mock_ee: MagicMock
    ) -> None:
        mock_ee.ImageCollection.side_effect = Exception("EE rejected collection")
        builder.with_date_range("2023-11-01", "2024-02-28")
        builder.with_aoi(MagicMock())
        builder.with_cloud_cover(20.0)
        builder.with_sensors(LandsatSensor.L8)
        with patch_ee(mock_ee):
            with pytest.raises(GEEAPIError):
                builder.build()


# ==============================================================================
# validate_not_empty tests
# ==============================================================================

class TestValidateNotEmpty:
    """Tests for build(validate_not_empty=True)."""

    def test_validate_not_empty_passes_when_count_gt_zero(
        self, builder, mock_ee: MagicMock, mock_client
    ) -> None:
        mock_col = mock_ee.ImageCollection.return_value
        mock_col.size.return_value.getInfo.return_value = 5
        mock_client.execute_with_retry.side_effect = lambda f, *a, **kw: f()

        with patch_ee(mock_ee):
            result = (
                builder
                .with_date_range("2023-11-01", "2024-02-28")
                .with_aoi(MagicMock())
                .with_cloud_cover(20.0)
                .with_sensors(LandsatSensor.L8)
                .build(validate_not_empty=True)
            )
        assert isinstance(result, CollectionResult)

    def test_validate_not_empty_raises_when_count_zero(
        self, builder, mock_ee: MagicMock, mock_client
    ) -> None:
        mock_col = mock_ee.ImageCollection.return_value
        mock_col.size.return_value.getInfo.return_value = 0
        mock_client.execute_with_retry.side_effect = lambda f, *a, **kw: f()

        builder.with_date_range("2023-11-01", "2024-02-28")
        builder.with_aoi(MagicMock())
        builder.with_cloud_cover(20.0)
        builder.with_sensors(LandsatSensor.L8)

        with patch_ee(mock_ee):
            with pytest.raises(GEEAPIError, match="zero images"):
                builder.build(validate_not_empty=True)


# ==============================================================================
# Helper
# ==============================================================================

from unittest.mock import patch


@contextmanager
def patch_ee(mock_ee: MagicMock):
    with patch.dict(sys.modules, {"ee": mock_ee}):
        yield


def _block_ee_import(name: str, *args, **kwargs):
    if name == "ee":
        raise ImportError("Simulated: ee not installed")
    import builtins
    return builtins.__import__(name, *args, **kwargs)