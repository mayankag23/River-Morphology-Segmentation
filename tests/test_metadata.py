"""
Unit tests for src/gee/metadata.py.

Tests cover:
    - CollectionMetadata dataclass properties and summary output
    - MetadataExtractor individual extraction methods
    - Empty collection handling (graceful empty returns)
    - GEEAPIError wrapping on failures
    - extract_all() resilience (one failure does not abort others)

Run:
    pytest tests/test_metadata.py -v
    pytest tests/test_metadata.py -v --cov=src/gee/metadata --cov-report=term-missing
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from src.gee import GEEAPIError, GEENotInstalledError
from src.gee.metadata import CollectionMetadata, MetadataExtractor


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def mock_ee() -> MagicMock:
    ee = MagicMock()
    return ee


@pytest.fixture
def mock_client() -> MagicMock:
    from src.gee.client import EarthEngineClient
    client = MagicMock(spec=EarthEngineClient)
    # Default: execute_with_retry calls the function directly.
    client.execute_with_retry.side_effect = lambda f, *a, **kw: f()
    return client


@pytest.fixture
def extractor(mock_client) -> MetadataExtractor:
    return MetadataExtractor(mock_client)


@pytest.fixture
def mock_collection() -> MagicMock:
    """Return a mock ee.ImageCollection with sensible defaults."""
    col = MagicMock()

    # size().getInfo() -> 10
    col.size.return_value.getInfo.return_value = 10

    # first().bandNames().getInfo() -> band list
    col.first.return_value.bandNames.return_value.getInfo.return_value = [
        "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7",
        "ST_B10", "QA_PIXEL",
    ]

    # aggregate_array('system:index').getInfo() -> image IDs
    col.aggregate_array.return_value.getInfo.return_value = [
        "LC08_145040_20231101",
        "LC08_145040_20231117",
        "LC09_145040_20231110",
    ]

    # first().select(0).projection().crs().getInfo() -> EPSG
    col.first.return_value.select.return_value.projection.return_value.crs.return_value.getInfo.return_value = "EPSG:32644"
    col.first.return_value.select.return_value.projection.return_value.nominalScale.return_value.getInfo.return_value = 30.0

    return col


@contextmanager
def patch_ee(mock_ee: MagicMock):
    with patch.dict(sys.modules, {"ee": mock_ee}):
        yield


# ==============================================================================
# CollectionMetadata tests
# ==============================================================================

class TestCollectionMetadata:
    """Tests for the CollectionMetadata frozen dataclass."""

    def test_default_construction(self) -> None:
        meta = CollectionMetadata()
        assert meta.image_count is None
        assert meta.band_names  == []
        assert meta.image_ids   == []

    def test_full_construction(self) -> None:
        meta = CollectionMetadata(
            image_count=10,
            band_names=["SR_B2", "SR_B3"],
            image_ids=["ID_001"],
            acquisition_dates=["2023-11-01"],
            spacecraft_ids=["LANDSAT_8"],
            temporal_start="2023-11-01",
            temporal_end="2024-02-28",
            crs="EPSG:32644",
            scale_meters=30.0,
        )
        assert meta.image_count       == 10
        assert meta.band_names        == ["SR_B2", "SR_B3"]
        assert meta.temporal_start    == "2023-11-01"
        assert meta.crs               == "EPSG:32644"
        assert meta.scale_meters      == 30.0

    def test_frozen_prevents_mutation(self) -> None:
        meta = CollectionMetadata(image_count=5)
        with pytest.raises((AttributeError, TypeError)):
            meta.image_count = 10  # type: ignore[misc]

    def test_summary_lines_returns_list_of_strings(self) -> None:
        meta  = CollectionMetadata(image_count=5, crs="EPSG:32644", scale_meters=30.0)
        lines = meta.summary_lines()
        assert isinstance(lines, list)
        assert all(isinstance(line, str) for line in lines)

    def test_summary_lines_ascii_only(self) -> None:
        meta  = CollectionMetadata(image_count=5, band_names=["SR_B2"])
        lines = meta.summary_lines()
        for line in lines:
            assert all(ord(c) < 128 for c in line), (
                f"Non-ASCII in line: {line!r}"
            )

    def test_summary_lines_contain_image_count(self) -> None:
        meta     = CollectionMetadata(image_count=42)
        combined = " ".join(meta.summary_lines())
        assert "42" in combined

    def test_summary_lines_show_na_for_none(self) -> None:
        meta     = CollectionMetadata()
        combined = " ".join(meta.summary_lines())
        assert "N/A" in combined


# ==============================================================================
# MetadataExtractor.get_image_count tests
# ==============================================================================

class TestGetImageCount:
    """Tests for MetadataExtractor.get_image_count()."""

    def test_returns_integer_count(
        self, extractor: MetadataExtractor, mock_collection: MagicMock, mock_ee: MagicMock
    ) -> None:
        with patch_ee(mock_ee):
            count = extractor.get_image_count(mock_collection)
        assert count == 10

    def test_uses_execute_with_retry(
        self, extractor: MetadataExtractor, mock_collection: MagicMock,
        mock_ee: MagicMock, mock_client: MagicMock
    ) -> None:
        with patch_ee(mock_ee):
            extractor.get_image_count(mock_collection)
        mock_client.execute_with_retry.assert_called()

    def test_ee_not_installed_raises(
        self, extractor: MetadataExtractor, mock_collection: MagicMock
    ) -> None:
        modules_without_ee = {k: v for k, v in sys.modules.items() if k != "ee"}
        with patch.dict(sys.modules, modules_without_ee, clear=True):
            with patch("builtins.__import__", side_effect=_block_ee_import):
                with pytest.raises(GEENotInstalledError):
                    extractor.get_image_count(mock_collection)

    def test_getinfo_failure_raises_gee_api_error(
        self, extractor: MetadataExtractor, mock_collection: MagicMock,
        mock_ee: MagicMock, mock_client: MagicMock
    ) -> None:
        mock_client.execute_with_retry.side_effect = Exception("EE quota exceeded")
        with patch_ee(mock_ee):
            with pytest.raises(GEEAPIError, match="get_image_count"):
                extractor.get_image_count(mock_collection)


# ==============================================================================
# MetadataExtractor.get_band_names tests
# ==============================================================================

class TestGetBandNames:
    """Tests for MetadataExtractor.get_band_names()."""

    def test_returns_list_of_band_names(
        self, extractor: MetadataExtractor, mock_collection: MagicMock, mock_ee: MagicMock
    ) -> None:
        with patch_ee(mock_ee):
            bands = extractor.get_band_names(mock_collection)
        assert "SR_B2" in bands
        assert isinstance(bands, list)

    def test_returns_empty_list_for_empty_collection(
        self, extractor: MetadataExtractor, mock_collection: MagicMock,
        mock_ee: MagicMock, mock_client: MagicMock
    ) -> None:
        mock_client.execute_with_retry.side_effect = Exception("null element")
        with patch_ee(mock_ee):
            bands = extractor.get_band_names(mock_collection)
        assert bands == []

    def test_returns_empty_list_when_getinfo_returns_none(
        self, extractor: MetadataExtractor, mock_collection: MagicMock,
        mock_ee: MagicMock, mock_client: MagicMock
    ) -> None:
        mock_client.execute_with_retry.side_effect = lambda f, *a, **kw: None
        with patch_ee(mock_ee):
            bands = extractor.get_band_names(mock_collection)
        assert bands == []


# ==============================================================================
# MetadataExtractor.get_image_ids tests
# ==============================================================================

class TestGetImageIds:
    """Tests for MetadataExtractor.get_image_ids()."""

    def test_returns_list_of_image_ids(
        self, extractor: MetadataExtractor, mock_collection: MagicMock, mock_ee: MagicMock
    ) -> None:
        with patch_ee(mock_ee):
            ids = extractor.get_image_ids(mock_collection)
        assert isinstance(ids, list)
        assert len(ids) == 3

    def test_returns_strings(
        self, extractor: MetadataExtractor, mock_collection: MagicMock, mock_ee: MagicMock
    ) -> None:
        with patch_ee(mock_ee):
            ids = extractor.get_image_ids(mock_collection)
        for item in ids:
            assert isinstance(item, str)

    def test_returns_empty_for_empty_collection(
        self, extractor: MetadataExtractor, mock_collection: MagicMock,
        mock_ee: MagicMock, mock_client: MagicMock
    ) -> None:
        mock_client.execute_with_retry.side_effect = lambda f, *a, **kw: []
        with patch_ee(mock_ee):
            ids = extractor.get_image_ids(mock_collection)
        assert ids == []


# ==============================================================================
# MetadataExtractor.get_acquisition_dates tests
# ==============================================================================

class TestGetAcquisitionDates:
    """Tests for MetadataExtractor.get_acquisition_dates()."""

    def test_returns_list_of_date_strings(
        self, extractor: MetadataExtractor, mock_collection: MagicMock,
        mock_ee: MagicMock, mock_client: MagicMock
    ) -> None:
        mock_client.execute_with_retry.side_effect = lambda f, *a, **kw: [
            "2023-11-01", "2023-11-17"
        ]
        with patch_ee(mock_ee):
            dates = extractor.get_acquisition_dates(mock_collection)
        assert dates == ["2023-11-01", "2023-11-17"]

    def test_returns_empty_for_null_result(
        self, extractor: MetadataExtractor, mock_collection: MagicMock,
        mock_ee: MagicMock, mock_client: MagicMock
    ) -> None:
        mock_client.execute_with_retry.side_effect = lambda f, *a, **kw: None
        with patch_ee(mock_ee):
            dates = extractor.get_acquisition_dates(mock_collection)
        assert dates == []


# ==============================================================================
# MetadataExtractor.get_spacecraft_ids tests
# ==============================================================================

class TestGetSpacecraftIds:
    """Tests for MetadataExtractor.get_spacecraft_ids()."""

    def test_returns_list_of_spacecraft_ids(
        self, extractor: MetadataExtractor, mock_collection: MagicMock,
        mock_ee: MagicMock, mock_client: MagicMock
    ) -> None:
        mock_client.execute_with_retry.side_effect = lambda f, *a, **kw: [
            "LANDSAT_8", "LANDSAT_9", "LANDSAT_8"
        ]
        with patch_ee(mock_ee):
            ids = extractor.get_spacecraft_ids(mock_collection)
        assert "LANDSAT_8" in ids
        assert "LANDSAT_9" in ids


# ==============================================================================
# MetadataExtractor.get_temporal_coverage tests
# ==============================================================================

class TestGetTemporalCoverage:
    """Tests for MetadataExtractor.get_temporal_coverage()."""

    def test_returns_min_max_dates(
        self, extractor: MetadataExtractor, mock_collection: MagicMock,
        mock_ee: MagicMock, mock_client: MagicMock
    ) -> None:
        mock_client.execute_with_retry.side_effect = lambda f, *a, **kw: [
            "2023-11-17", "2023-11-01", "2024-02-01"
        ]
        with patch_ee(mock_ee):
            start, end = extractor.get_temporal_coverage(mock_collection)
        assert start == "2023-11-01"
        assert end   == "2024-02-01"

    def test_returns_none_none_for_empty_collection(
        self, extractor: MetadataExtractor, mock_collection: MagicMock,
        mock_ee: MagicMock, mock_client: MagicMock
    ) -> None:
        mock_client.execute_with_retry.side_effect = lambda f, *a, **kw: []
        with patch_ee(mock_ee):
            start, end = extractor.get_temporal_coverage(mock_collection)
        assert start is None
        assert end   is None


# ==============================================================================
# MetadataExtractor.get_crs_and_scale tests
# ==============================================================================

class TestGetCrsAndScale:
    """Tests for MetadataExtractor.get_crs_and_scale()."""

    def test_returns_crs_and_scale(
        self, extractor: MetadataExtractor, mock_collection: MagicMock,
        mock_ee: MagicMock, mock_client: MagicMock
    ) -> None:
        call_count = [0]
        def side_effect(f, *a, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                return "EPSG:32644"
            return 30.0
        mock_client.execute_with_retry.side_effect = side_effect

        with patch_ee(mock_ee):
            crs, scale = extractor.get_crs_and_scale(mock_collection)

        assert crs   == "EPSG:32644"
        assert scale == pytest.approx(30.0)

    def test_returns_none_none_for_empty_collection(
        self, extractor: MetadataExtractor, mock_collection: MagicMock,
        mock_ee: MagicMock, mock_client: MagicMock
    ) -> None:
        mock_client.execute_with_retry.side_effect = Exception("null element")
        with patch_ee(mock_ee):
            crs, scale = extractor.get_crs_and_scale(mock_collection)
        assert crs   is None
        assert scale is None

    def test_ee_not_installed_raises(
        self, extractor: MetadataExtractor, mock_collection: MagicMock
    ) -> None:
        modules_without_ee = {k: v for k, v in sys.modules.items() if k != "ee"}
        with patch.dict(sys.modules, modules_without_ee, clear=True):
            with patch("builtins.__import__", side_effect=_block_ee_import):
                with pytest.raises(GEENotInstalledError):
                    extractor.get_crs_and_scale(mock_collection)


# ==============================================================================
# MetadataExtractor.extract_all tests
# ==============================================================================

class TestExtractAll:
    """Tests for MetadataExtractor.extract_all()."""

    def test_returns_collection_metadata(
        self, extractor: MetadataExtractor, mock_collection: MagicMock,
        mock_ee: MagicMock, mock_client: MagicMock
    ) -> None:
        call_results = [
            10,                                      # get_image_count
            ["SR_B2", "SR_B3"],                      # get_band_names
            ["ID_001", "ID_002"],                    # get_image_ids
            ["2023-11-01", "2023-11-17"],            # get_acquisition_dates
            ["LANDSAT_8"],                           # get_spacecraft_ids
        ]
        idx = [0]
        def side_effect(f, *a, **kw):
            if idx[0] < len(call_results):
                result = call_results[idx[0]]
                idx[0] += 1
                return result
            return "EPSG:32644"  # CRS calls after that

        mock_client.execute_with_retry.side_effect = side_effect

        with patch_ee(mock_ee):
            meta = extractor.extract_all(mock_collection)

        assert isinstance(meta, CollectionMetadata)

    def test_returns_metadata_even_if_one_field_fails(
        self, extractor: MetadataExtractor, mock_collection: MagicMock,
        mock_ee: MagicMock, mock_client: MagicMock
    ) -> None:
        """A failure in one extraction should not abort the others."""
        call_count = [0]
        def side_effect(f, *a, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                return 5  # image_count succeeds
            raise Exception("Simulated partial failure")

        mock_client.execute_with_retry.side_effect = side_effect

        with patch_ee(mock_ee):
            meta = extractor.extract_all(mock_collection)

        # image_count should be populated; others should fall back.
        assert meta.image_count == 5
        assert isinstance(meta, CollectionMetadata)

    def test_summary_lines_ascii_only(
        self, extractor: MetadataExtractor, mock_collection: MagicMock,
        mock_ee: MagicMock, mock_client: MagicMock
    ) -> None:
        mock_client.execute_with_retry.side_effect = lambda f, *a, **kw: None

        with patch_ee(mock_ee):
            meta = extractor.extract_all(mock_collection)

        for line in meta.summary_lines():
            assert all(ord(c) < 128 for c in line), (
                f"Non-ASCII in line: {line!r}"
            )


# ==============================================================================
# Helper
# ==============================================================================

def _block_ee_import(name: str, *args, **kwargs):
    if name == "ee":
        raise ImportError("Simulated: ee not installed")
    import builtins
    return builtins.__import__(name, *args, **kwargs)