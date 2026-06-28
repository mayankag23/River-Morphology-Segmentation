"""
Unit tests for src/gee/filters.py.

All EE operations are mocked. No real authentication required.
Tests cover validation logic, filter application, composability,
and exception wrapping.

Run:
    pytest tests/test_filters.py -v
    pytest tests/test_filters.py -v --cov=src/gee/filters --cov-report=term-missing
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, call

import pytest

from src.core.exceptions import InvalidValueError, TypeMismatchError
from src.gee import GEEAPIError
from src.gee.filters import (
    CLOUD_COVER_PROPERTY,
    SPACECRAFT_ID_PROPERTY,
    filter_by_bounds,
    filter_by_cloud_cover,
    filter_by_date,
    filter_by_sensor,
)


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def mock_ee() -> MagicMock:
    """Return a MagicMock standing in for the earthengine-api module."""
    ee = MagicMock()
    return ee


@pytest.fixture
def mock_collection() -> MagicMock:
    """Return a MagicMock representing an ee.ImageCollection."""
    col = MagicMock()
    col.filterDate.return_value   = col
    col.filterBounds.return_value = col
    col.filter.return_value       = col
    return col


@pytest.fixture
def mock_geometry() -> MagicMock:
    """Return a MagicMock representing an ee.Geometry."""
    return MagicMock()


# ==============================================================================
# Constants tests
# ==============================================================================

class TestConstants:
    """Tests for module-level property name constants."""

    def test_cloud_cover_property_is_ascii(self) -> None:
        assert all(ord(c) < 128 for c in CLOUD_COVER_PROPERTY)

    def test_spacecraft_id_property_is_ascii(self) -> None:
        assert all(ord(c) < 128 for c in SPACECRAFT_ID_PROPERTY)

    def test_cloud_cover_property_value(self) -> None:
        assert CLOUD_COVER_PROPERTY == "CLOUD_COVER"

    def test_spacecraft_id_property_value(self) -> None:
        assert SPACECRAFT_ID_PROPERTY == "SPACECRAFT_ID"


# ==============================================================================
# filter_by_date tests
# ==============================================================================

class TestFilterByDate:
    """Tests for filter_by_date()."""

    def test_returns_filtered_collection(
        self, mock_collection: MagicMock
    ) -> None:
        filtered = filter_by_date(mock_collection, "2023-11-01", "2024-02-28")
        assert filtered is mock_collection.filterDate.return_value

    def test_calls_filter_date_with_correct_args(
        self, mock_collection: MagicMock
    ) -> None:
        filter_by_date(mock_collection, "2023-11-01", "2024-02-28")
        mock_collection.filterDate.assert_called_once_with(
            "2023-11-01", "2024-02-28"
        )

    def test_invalid_start_format_raises_invalid_value_error(
        self, mock_collection: MagicMock
    ) -> None:
        with pytest.raises(InvalidValueError, match="YYYY-MM-DD"):
            filter_by_date(mock_collection, "01/11/2023", "2024-02-28")

    def test_invalid_end_format_raises_invalid_value_error(
        self, mock_collection: MagicMock
    ) -> None:
        with pytest.raises(InvalidValueError, match="YYYY-MM-DD"):
            filter_by_date(mock_collection, "2023-11-01", "28-02-2024")

    def test_start_not_string_raises_type_mismatch(
        self, mock_collection: MagicMock
    ) -> None:
        with pytest.raises(TypeMismatchError):
            filter_by_date(mock_collection, 20231101, "2024-02-28")

    def test_end_not_string_raises_type_mismatch(
        self, mock_collection: MagicMock
    ) -> None:
        with pytest.raises(TypeMismatchError):
            filter_by_date(mock_collection, "2023-11-01", None)

    def test_start_equals_end_raises_invalid_value_error(
        self, mock_collection: MagicMock
    ) -> None:
        with pytest.raises(InvalidValueError, match="strictly before"):
            filter_by_date(mock_collection, "2023-11-01", "2023-11-01")

    def test_start_after_end_raises_invalid_value_error(
        self, mock_collection: MagicMock
    ) -> None:
        with pytest.raises(InvalidValueError, match="strictly before"):
            filter_by_date(mock_collection, "2024-06-01", "2023-01-01")

    def test_ee_exception_wrapped_in_gee_api_error(
        self, mock_collection: MagicMock
    ) -> None:
        mock_collection.filterDate.side_effect = Exception("EE server error")
        with pytest.raises(GEEAPIError, match="filter_by_date"):
            filter_by_date(mock_collection, "2023-11-01", "2024-02-28")

    @pytest.mark.parametrize("start,end", [
        ("1984-04-01", "1984-12-31"),
        ("2013-04-11", "2013-12-31"),
        ("2021-10-31", "2024-12-31"),
    ])
    def test_valid_date_pairs_do_not_raise(
        self, mock_collection: MagicMock, start: str, end: str
    ) -> None:
        result = filter_by_date(mock_collection, start, end)
        assert result is not None


# ==============================================================================
# filter_by_bounds tests
# ==============================================================================

class TestFilterByBounds:
    """Tests for filter_by_bounds()."""

    def test_returns_filtered_collection(
        self, mock_collection: MagicMock, mock_geometry: MagicMock
    ) -> None:
        filtered = filter_by_bounds(mock_collection, mock_geometry)
        assert filtered is mock_collection.filterBounds.return_value

    def test_calls_filter_bounds_with_geometry(
        self, mock_collection: MagicMock, mock_geometry: MagicMock
    ) -> None:
        filter_by_bounds(mock_collection, mock_geometry)
        mock_collection.filterBounds.assert_called_once_with(mock_geometry)

    def test_none_geometry_raises_invalid_value_error(
        self, mock_collection: MagicMock
    ) -> None:
        with pytest.raises(InvalidValueError, match="geometry must not be None"):
            filter_by_bounds(mock_collection, None)

    def test_ee_exception_wrapped_in_gee_api_error(
        self, mock_collection: MagicMock, mock_geometry: MagicMock
    ) -> None:
        mock_collection.filterBounds.side_effect = Exception("EE error")
        with pytest.raises(GEEAPIError, match="filter_by_bounds"):
            filter_by_bounds(mock_collection, mock_geometry)


# ==============================================================================
# filter_by_cloud_cover tests
# ==============================================================================

class TestFilterByCloudCover:
    """Tests for filter_by_cloud_cover()."""

    def test_returns_filtered_collection(
        self, mock_collection: MagicMock, mock_ee: MagicMock
    ) -> None:
        with patch_ee(mock_ee):
            filtered = filter_by_cloud_cover(mock_collection, 20.0)
        assert filtered is mock_collection.filter.return_value

    def test_calls_ee_filter_lte(
        self, mock_collection: MagicMock, mock_ee: MagicMock
    ) -> None:
        with patch_ee(mock_ee):
            filter_by_cloud_cover(mock_collection, 20.0)
        mock_ee.Filter.lte.assert_called_once_with(CLOUD_COVER_PROPERTY, 20.0)

    def test_custom_property_name(
        self, mock_collection: MagicMock, mock_ee: MagicMock
    ) -> None:
        with patch_ee(mock_ee):
            filter_by_cloud_cover(
                mock_collection, 15.0, property_name="CLOUD_COVER_LAND"
            )
        mock_ee.Filter.lte.assert_called_once_with("CLOUD_COVER_LAND", 15.0)

    def test_zero_percent_is_valid(
        self, mock_collection: MagicMock, mock_ee: MagicMock
    ) -> None:
        with patch_ee(mock_ee):
            result = filter_by_cloud_cover(mock_collection, 0.0)
        assert result is not None

    def test_hundred_percent_is_valid(
        self, mock_collection: MagicMock, mock_ee: MagicMock
    ) -> None:
        with patch_ee(mock_ee):
            result = filter_by_cloud_cover(mock_collection, 100.0)
        assert result is not None

    def test_negative_percent_raises_invalid_value_error(
        self, mock_collection: MagicMock, mock_ee: MagicMock
    ) -> None:
        with pytest.raises(InvalidValueError, match="range"):
            with patch_ee(mock_ee):
                filter_by_cloud_cover(mock_collection, -1.0)

    def test_over_hundred_raises_invalid_value_error(
        self, mock_collection: MagicMock, mock_ee: MagicMock
    ) -> None:
        with pytest.raises(InvalidValueError, match="range"):
            with patch_ee(mock_ee):
                filter_by_cloud_cover(mock_collection, 101.0)

    def test_non_numeric_raises_type_mismatch(
        self, mock_collection: MagicMock, mock_ee: MagicMock
    ) -> None:
        with pytest.raises(TypeMismatchError):
            filter_by_cloud_cover(mock_collection, "twenty")

    def test_integer_accepted(
        self, mock_collection: MagicMock, mock_ee: MagicMock
    ) -> None:
        with patch_ee(mock_ee):
            result = filter_by_cloud_cover(mock_collection, 20)
        assert result is not None

    def test_ee_exception_wrapped_in_gee_api_error(
        self, mock_collection: MagicMock, mock_ee: MagicMock
    ) -> None:
        mock_collection.filter.side_effect = Exception("EE filter error")
        with patch_ee(mock_ee):
            with pytest.raises(GEEAPIError, match="filter_by_cloud_cover"):
                filter_by_cloud_cover(mock_collection, 20.0)


# ==============================================================================
# filter_by_sensor tests
# ==============================================================================

class TestFilterBySensor:
    """Tests for filter_by_sensor()."""

    def test_returns_filtered_collection(
        self, mock_collection: MagicMock, mock_ee: MagicMock
    ) -> None:
        with patch_ee(mock_ee):
            filtered = filter_by_sensor(
                mock_collection, ["LANDSAT_8", "LANDSAT_9"]
            )
        assert filtered is mock_collection.filter.return_value

    def test_calls_ee_filter_in_list(
        self, mock_collection: MagicMock, mock_ee: MagicMock
    ) -> None:
        with patch_ee(mock_ee):
            filter_by_sensor(mock_collection, ["LANDSAT_8", "LANDSAT_9"])
        mock_ee.Filter.inList.assert_called_once_with(
            SPACECRAFT_ID_PROPERTY, ["LANDSAT_8", "LANDSAT_9"]
        )

    def test_single_sensor(
        self, mock_collection: MagicMock, mock_ee: MagicMock
    ) -> None:
        with patch_ee(mock_ee):
            result = filter_by_sensor(mock_collection, ["LANDSAT_8"])
        assert result is not None

    def test_empty_list_raises_invalid_value_error(
        self, mock_collection: MagicMock, mock_ee: MagicMock
    ) -> None:
        with pytest.raises(InvalidValueError, match="non-empty"):
            filter_by_sensor(mock_collection, [])

    def test_non_list_raises_type_mismatch(
        self, mock_collection: MagicMock
    ) -> None:
        with pytest.raises(TypeMismatchError):
            filter_by_sensor(mock_collection, "LANDSAT_8")

    def test_ee_exception_wrapped_in_gee_api_error(
        self, mock_collection: MagicMock, mock_ee: MagicMock
    ) -> None:
        mock_collection.filter.side_effect = Exception("EE error")
        with patch_ee(mock_ee):
            with pytest.raises(GEEAPIError, match="filter_by_sensor"):
                filter_by_sensor(mock_collection, ["LANDSAT_8"])


# ==============================================================================
# Filter composability tests
# ==============================================================================

class TestFilterComposability:
    """Tests that filters can be chained in any order."""

    def test_date_then_bounds_then_cloud(
        self,
        mock_collection: MagicMock,
        mock_geometry: MagicMock,
        mock_ee: MagicMock,
    ) -> None:
        """Applying three filters in sequence should work without errors."""
        step1 = MagicMock()
        step2 = MagicMock()
        step3 = MagicMock()
        step1.filterBounds.return_value = step2
        step2.filter.return_value       = step3
        mock_collection.filterDate.return_value = step1

        with patch_ee(mock_ee):
            col = filter_by_date(mock_collection, "2023-11-01", "2024-02-28")
            col = filter_by_bounds(col, mock_geometry)
            col = filter_by_cloud_cover(col, 20.0)

        assert col is step3

    def test_bounds_then_date(
        self,
        mock_collection: MagicMock,
        mock_geometry: MagicMock,
    ) -> None:
        """Filters can be applied in alternate order."""
        step1 = MagicMock()
        step1.filterDate.return_value = MagicMock()
        mock_collection.filterBounds.return_value = step1

        col = filter_by_bounds(mock_collection, mock_geometry)
        col = filter_by_date(col, "2023-11-01", "2024-02-28")
        assert col is step1.filterDate.return_value


# ==============================================================================
# Helper
# ==============================================================================

from contextlib import contextmanager
from unittest.mock import patch


@contextmanager
def patch_ee(mock_ee: MagicMock):
    """Context manager that patches sys.modules['ee'] with mock_ee."""
    with patch.dict(sys.modules, {"ee": mock_ee}):
        yield