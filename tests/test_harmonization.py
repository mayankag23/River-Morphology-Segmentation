"""
Unit tests for src/gee/harmonization.py.

Tests cover:
    - Constants: COMMON_BAND_NAMES, OLI_SOURCE_BANDS, TM_ETM_SOURCE_BANDS
    - BandHarmonizer.rename_oli_image() and rename_tm_etm_image()
    - BandHarmonizer.harmonize_collection() via .map()
    - BandHarmonizer.harmonize_image() server-side conditional
    - _make_harmonize_function() produces a callable
    - GEENotInstalledError when ee is absent

Run:
    pytest tests/test_harmonization.py -v
    pytest tests/test_harmonization.py -v --cov=src/gee/harmonization --cov-report=term-missing
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from src.gee import GEEAPIError, GEENotInstalledError
from src.gee.harmonization import (
    COMMON_BAND_NAMES,
    OPTICAL_BAND_NAMES,
    OLI_SOURCE_BANDS,
    OLI_SPACECRAFT_IDS,
    TM_ETM_SOURCE_BANDS,
    TM_ETM_SPACECRAFT_IDS,
    BandHarmonizer,
)


# ==============================================================================
# Helpers
# ==============================================================================

@contextmanager
def patch_ee(mock_ee: MagicMock):
    with patch.dict(sys.modules, {"ee": mock_ee}):
        yield


def _make_mock_ee() -> MagicMock:
    ee = MagicMock()
    ee.String.return_value.equals.return_value.Or.return_value = MagicMock()
    ee.Image.return_value = MagicMock()
    ee.Algorithms.If.return_value = MagicMock()
    return ee


def _make_mock_image(spacecraft_id: str = "LANDSAT_8") -> MagicMock:
    img = MagicMock()
    img.get.return_value = spacecraft_id
    img.select.return_value.rename.return_value = MagicMock()
    return img


def _make_mock_collection() -> MagicMock:
    col = MagicMock()
    col.map.return_value = col
    return col


# ==============================================================================
# Constants tests
# ==============================================================================

class TestConstants:
    """Tests for module-level band name constants."""

    def test_common_band_names_has_eight_entries(self) -> None:
        assert len(COMMON_BAND_NAMES) == 8

    def test_common_band_names_contains_blue(self) -> None:
        assert "Blue" in COMMON_BAND_NAMES

    def test_common_band_names_contains_qa_pixel(self) -> None:
        assert "QA_PIXEL" in COMMON_BAND_NAMES

    def test_common_band_names_contains_thermal(self) -> None:
        assert "Thermal" in COMMON_BAND_NAMES

    def test_common_band_names_is_ascii_only(self) -> None:
        for name in COMMON_BAND_NAMES:
            assert all(ord(c) < 128 for c in name)

    def test_oli_source_bands_has_eight_entries(self) -> None:
        assert len(OLI_SOURCE_BANDS) == 8

    def test_tm_etm_source_bands_has_eight_entries(self) -> None:
        assert len(TM_ETM_SOURCE_BANDS) == 8

    def test_oli_starts_with_sr_b2_for_blue(self) -> None:
        assert OLI_SOURCE_BANDS[0] == "SR_B2"

    def test_tm_etm_starts_with_sr_b1_for_blue(self) -> None:
        assert TM_ETM_SOURCE_BANDS[0] == "SR_B1"

    def test_oli_thermal_is_st_b10(self) -> None:
        assert "ST_B10" in OLI_SOURCE_BANDS

    def test_tm_etm_thermal_is_st_b6(self) -> None:
        assert "ST_B6" in TM_ETM_SOURCE_BANDS

    def test_source_bands_same_length_as_common(self) -> None:
        assert len(OLI_SOURCE_BANDS)    == len(COMMON_BAND_NAMES)
        assert len(TM_ETM_SOURCE_BANDS) == len(COMMON_BAND_NAMES)

    def test_optical_band_names_contains_six_bands(self) -> None:
        assert len(OPTICAL_BAND_NAMES) == 6

    def test_optical_band_names_does_not_contain_thermal(self) -> None:
        assert "Thermal" not in OPTICAL_BAND_NAMES

    def test_optical_band_names_does_not_contain_qa(self) -> None:
        assert "QA_PIXEL" not in OPTICAL_BAND_NAMES

    def test_oli_spacecraft_ids_contain_l8_l9(self) -> None:
        assert "LANDSAT_8" in OLI_SPACECRAFT_IDS
        assert "LANDSAT_9" in OLI_SPACECRAFT_IDS

    def test_tm_etm_spacecraft_ids_contain_l5_l7(self) -> None:
        assert "LANDSAT_5" in TM_ETM_SPACECRAFT_IDS
        assert "LANDSAT_7" in TM_ETM_SPACECRAFT_IDS


# ==============================================================================
# BandHarmonizer.rename_oli_image tests
# ==============================================================================

class TestRenameOLIImage:
    """Tests for BandHarmonizer.rename_oli_image()."""

    def test_calls_select_with_oli_source_bands(self) -> None:
        harmonizer = BandHarmonizer()
        image      = _make_mock_image("LANDSAT_8")
        harmonizer.rename_oli_image(image)
        image.select.assert_called_once_with(list(OLI_SOURCE_BANDS))

    def test_calls_rename_with_common_band_names(self) -> None:
        harmonizer = BandHarmonizer()
        image      = _make_mock_image("LANDSAT_8")
        harmonizer.rename_oli_image(image)
        image.select.return_value.rename.assert_called_once_with(
            list(COMMON_BAND_NAMES)
        )

    def test_returns_renamed_image(self) -> None:
        harmonizer = BandHarmonizer()
        image      = _make_mock_image("LANDSAT_8")
        result     = harmonizer.rename_oli_image(image)
        assert result is image.select.return_value.rename.return_value

    def test_select_failure_raises_gee_api_error(self) -> None:
        harmonizer = BandHarmonizer()
        image      = _make_mock_image()
        image.select.side_effect = Exception("Band not found")
        with pytest.raises(GEEAPIError, match="rename_oli_image"):
            harmonizer.rename_oli_image(image)


# ==============================================================================
# BandHarmonizer.rename_tm_etm_image tests
# ==============================================================================

class TestRenameTMETMImage:
    """Tests for BandHarmonizer.rename_tm_etm_image()."""

    def test_calls_select_with_tm_etm_source_bands(self) -> None:
        harmonizer = BandHarmonizer()
        image      = _make_mock_image("LANDSAT_5")
        harmonizer.rename_tm_etm_image(image)
        image.select.assert_called_once_with(list(TM_ETM_SOURCE_BANDS))

    def test_calls_rename_with_common_band_names(self) -> None:
        harmonizer = BandHarmonizer()
        image      = _make_mock_image("LANDSAT_5")
        harmonizer.rename_tm_etm_image(image)
        image.select.return_value.rename.assert_called_once_with(
            list(COMMON_BAND_NAMES)
        )

    def test_returns_renamed_image(self) -> None:
        harmonizer = BandHarmonizer()
        image      = _make_mock_image("LANDSAT_5")
        result     = harmonizer.rename_tm_etm_image(image)
        assert result is image.select.return_value.rename.return_value

    def test_select_failure_raises_gee_api_error(self) -> None:
        harmonizer = BandHarmonizer()
        image      = _make_mock_image("LANDSAT_7")
        image.select.side_effect = Exception("Band SR_B1 not found")
        with pytest.raises(GEEAPIError, match="rename_tm_etm_image"):
            harmonizer.rename_tm_etm_image(image)


# ==============================================================================
# BandHarmonizer.harmonize_collection tests
# ==============================================================================

class TestHarmonizeCollection:
    """Tests for BandHarmonizer.harmonize_collection()."""

    def test_calls_collection_map(self) -> None:
        harmonizer = BandHarmonizer()
        collection = _make_mock_collection()
        mock_ee    = _make_mock_ee()
        with patch_ee(mock_ee):
            harmonizer.harmonize_collection(collection)
        collection.map.assert_called_once()

    def test_returns_mapped_collection(self) -> None:
        harmonizer = BandHarmonizer()
        collection = _make_mock_collection()
        mock_ee    = _make_mock_ee()
        with patch_ee(mock_ee):
            result = harmonizer.harmonize_collection(collection)
        assert result is collection.map.return_value

    def test_map_receives_callable(self) -> None:
        harmonizer = BandHarmonizer()
        collection = _make_mock_collection()
        mock_ee    = _make_mock_ee()
        with patch_ee(mock_ee):
            harmonizer.harmonize_collection(collection)
        call_arg = collection.map.call_args[0][0]
        assert callable(call_arg)

    def test_ee_not_installed_raises_gee_not_installed(self) -> None:
        harmonizer = BandHarmonizer()
        collection = _make_mock_collection()
        modules_without_ee = {k: v for k, v in sys.modules.items() if k != "ee"}
        with patch.dict(sys.modules, modules_without_ee, clear=True):
            with patch("builtins.__import__", side_effect=_block_ee_import):
                with pytest.raises(GEENotInstalledError):
                    harmonizer.harmonize_collection(collection)

    def test_collection_map_failure_raises_gee_api_error(self) -> None:
        harmonizer = BandHarmonizer()
        collection = _make_mock_collection()
        collection.map.side_effect = Exception("EE map error")
        mock_ee    = _make_mock_ee()
        with patch_ee(mock_ee):
            with pytest.raises(GEEAPIError, match="harmonize_collection"):
                harmonizer.harmonize_collection(collection)


# ==============================================================================
# BandHarmonizer._make_harmonize_function tests
# ==============================================================================

class TestMakeHarmonizeFunction:
    """Tests for the closure returned by _make_harmonize_function()."""

    def test_returns_callable(self) -> None:
        harmonizer = BandHarmonizer()
        mock_ee    = _make_mock_ee()
        with patch_ee(mock_ee):
            fn = harmonizer._make_harmonize_function()
        assert callable(fn)

    def test_function_calls_ee_algorithms_if(self) -> None:
        harmonizer = BandHarmonizer()
        image      = _make_mock_image("LANDSAT_8")
        mock_ee    = _make_mock_ee()
        with patch_ee(mock_ee):
            fn = harmonizer._make_harmonize_function()
            fn(image)
        mock_ee.Algorithms.If.assert_called_once()

    def test_function_reads_spacecraft_id(self) -> None:
        harmonizer = BandHarmonizer()
        image      = _make_mock_image("LANDSAT_9")
        mock_ee    = _make_mock_ee()
        with patch_ee(mock_ee):
            fn = harmonizer._make_harmonize_function()
            fn(image)
        image.get.assert_called_with("SPACECRAFT_ID")

    def test_function_wraps_result_in_ee_image(self) -> None:
        harmonizer = BandHarmonizer()
        image      = _make_mock_image("LANDSAT_8")
        mock_ee    = _make_mock_ee()
        with patch_ee(mock_ee):
            fn     = harmonizer._make_harmonize_function()
            result = fn(image)
        mock_ee.Image.assert_called_once()

    def test_ee_not_installed_raises_gee_not_installed(self) -> None:
        harmonizer = BandHarmonizer()
        modules_without_ee = {k: v for k, v in sys.modules.items() if k != "ee"}
        with patch.dict(sys.modules, modules_without_ee, clear=True):
            with patch("builtins.__import__", side_effect=_block_ee_import):
                with pytest.raises(GEENotInstalledError):
                    harmonizer._make_harmonize_function()


# ==============================================================================
# BandHarmonizer.harmonize_image tests
# ==============================================================================

class TestHarmonizeImage:
    """Tests for BandHarmonizer.harmonize_image()."""

    def test_returns_harmonized_image(self) -> None:
        harmonizer = BandHarmonizer()
        image      = _make_mock_image("LANDSAT_8")
        mock_ee    = _make_mock_ee()
        with patch_ee(mock_ee):
            result = harmonizer.harmonize_image(image)
        assert result is not None

    def test_uses_ee_algorithms_if(self) -> None:
        harmonizer = BandHarmonizer()
        image      = _make_mock_image("LANDSAT_8")
        mock_ee    = _make_mock_ee()
        with patch_ee(mock_ee):
            harmonizer.harmonize_image(image)
        mock_ee.Algorithms.If.assert_called_once()

    def test_failure_raises_gee_api_error(self) -> None:
        harmonizer = BandHarmonizer()
        image      = MagicMock()
        image.get.side_effect = Exception("Property not found")
        mock_ee    = _make_mock_ee()
        with patch_ee(mock_ee):
            with pytest.raises(GEEAPIError, match="harmonize_image"):
                harmonizer.harmonize_image(image)


# ==============================================================================
# Private helper
# ==============================================================================

# def _block_ee_import(name: str, *args, **kwargs):
#     if name == "ee":
#         raise ImportError("Simulated: ee not installed")
#     import builtins
#     return builtins.__import__(name, *args, **kwargs)
import builtins

_ORIGINAL_IMPORT = builtins.__import__

def _block_ee_import(name: str, *args, **kwargs):
    if name == "ee":
        raise ImportError("Simulated: ee not installed")
    return _ORIGINAL_IMPORT(name, *args, **kwargs)