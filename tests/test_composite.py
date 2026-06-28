"""
Unit tests for src/gee/composite.py.

Tests cover:
    - CompositeMethod enum and from_string()
    - CompositeResult properties and summary output
    - LandsatCompositor construction
    - build_composite() with each method
    - Method resolution from config and override
    - Percentile value resolution
    - _medoid() construction and qualityMosaic() call
    - _percentile() band renaming
    - GEEAPIError wrapping

Run:
    pytest tests/test_composite.py -v
    pytest tests/test_composite.py -v --cov=src/gee/composite --cov-report=term-missing
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.core.exceptions import InvalidValueError
from src.gee import GEEAPIError, GEENotInstalledError
from src.gee.collections import CollectionResult, LandsatSensor
from src.gee.composite import (
    CompositeMethod,
    CompositeResult,
    LandsatCompositor,
)
from src.gee.harmonization import COMMON_BAND_NAMES
from src.gee.preprocessing import ProcessedCollectionResult
from tests.conftest import make_valid_config, write_config


# ==============================================================================
# Helpers
# ==============================================================================

@contextmanager
def patch_ee(mock_ee: MagicMock):
    with patch.dict(sys.modules, {"ee": mock_ee}):
        yield


def _make_mock_ee() -> MagicMock:
    ee = MagicMock()
    ee.Reducer.percentile.return_value = MagicMock()
    return ee


def _make_mock_client() -> MagicMock:
    from src.gee.client import EarthEngineClient
    client = MagicMock(spec=EarthEngineClient)
    client.is_initialized = True
    return client


def _make_mock_collection() -> MagicMock:
    col = MagicMock()
    col.median.return_value  = MagicMock()
    col.mean.return_value    = MagicMock()
    col.mosaic.return_value  = MagicMock()
    col.reduce.return_value  = MagicMock()
    col.map.return_value     = col
    # qualityMosaic selects the medoid
    col.qualityMosaic.return_value = MagicMock()
    # first().bandNames() for medoid
    col.first.return_value.bandNames.return_value.remove.return_value = MagicMock()
    # For percentile band renaming
    col.first.return_value.bandNames.return_value.map.return_value = MagicMock()
    col.reduce.return_value.select.return_value.rename.return_value = MagicMock()
    return col


def _make_config(tmp_path: Path, composite_method: str = "median"):
    from src.core.config import Config
    data = make_valid_config()
    data["composite"] = {
        "method":           composite_method,
        "percentile_value": 50,
    }
    data["preprocessing"].update({
        "mask_cloud":           True,
        "mask_cloud_shadow":    True,
        "mask_snow":            False,
        "mask_cirrus":          True,
        "mask_dilated_cloud":   True,
        "mask_fill":            True,
        "thermal_scale_factor": 0.00341802,
        "thermal_offset":       149.0,
        "clip_to_valid_range":  True,
    })
    return Config(config_path=write_config(tmp_path, data))


def _make_processed_result(
    harmonized: bool = True,
    collection: MagicMock | None = None,
) -> ProcessedCollectionResult:
    if collection is None:
        collection = _make_mock_collection()
    source = CollectionResult(
        collection=collection,
        sensors=(LandsatSensor.L8, LandsatSensor.L9),
        collection_ids=("LANDSAT/LC08/C02/T1_L2", "LANDSAT/LC09/C02/T1_L2"),
        start_date="2023-11-01",
        end_date="2024-02-28",
        cloud_cover_limit=20.0,
        filters_applied=("date", "bounds", "cloud_cover"),
    )
    return ProcessedCollectionResult(
        collection=collection,
        source_result=source,
        operations_applied=("scaling", "qa_masking", "harmonization"),
        band_names=COMMON_BAND_NAMES if harmonized else (),
        scale_applied=True,
        masking_applied=True,
        harmonization_applied=harmonized,
    )


# ==============================================================================
# CompositeMethod tests
# ==============================================================================

class TestCompositeMethod:
    """Tests for the CompositeMethod string enum."""

    def test_all_values_are_ascii(self) -> None:
        for method in CompositeMethod:
            assert all(ord(c) < 128 for c in method.value)

    def test_is_str_subclass(self) -> None:
        assert isinstance(CompositeMethod.MEDIAN, str)

    def test_compares_equal_to_lowercase_string(self) -> None:
        assert CompositeMethod.MEDIAN     == "median"
        assert CompositeMethod.MEAN       == "mean"
        assert CompositeMethod.MEDOID     == "medoid"
        assert CompositeMethod.MOSAIC     == "mosaic"
        assert CompositeMethod.PERCENTILE == "percentile"

    def test_from_string_median(self) -> None:
        assert CompositeMethod.from_string("median") == CompositeMethod.MEDIAN

    def test_from_string_case_insensitive(self) -> None:
        assert CompositeMethod.from_string("MEDIAN")    == CompositeMethod.MEDIAN
        assert CompositeMethod.from_string("Percentile") == CompositeMethod.PERCENTILE

    def test_from_string_strips_whitespace(self) -> None:
        assert CompositeMethod.from_string("  mean  ") == CompositeMethod.MEAN

    def test_from_string_invalid_raises_invalid_value_error(self) -> None:
        with pytest.raises(InvalidValueError, match="composite.method"):
            CompositeMethod.from_string("maximum")

    def test_all_five_methods_exist(self) -> None:
        methods = {m.value for m in CompositeMethod}
        assert methods == {"median", "mean", "medoid", "mosaic", "percentile"}


# ==============================================================================
# CompositeResult tests
# ==============================================================================

class TestCompositeResult:
    """Tests for the CompositeResult frozen dataclass."""

    def _make_result(
        self,
        method: CompositeMethod = CompositeMethod.MEDIAN,
    ) -> CompositeResult:
        return CompositeResult(
            image=MagicMock(),
            method=method,
            percentile_value=None,
            source_result=_make_processed_result(),
            band_names=COMMON_BAND_NAMES,
        )

    def test_stores_method(self) -> None:
        result = self._make_result(CompositeMethod.MOSAIC)
        assert result.method == CompositeMethod.MOSAIC

    def test_stores_band_names(self) -> None:
        result = self._make_result()
        assert result.band_names == COMMON_BAND_NAMES

    def test_frozen_prevents_mutation(self) -> None:
        result = self._make_result()
        with pytest.raises((AttributeError, TypeError)):
            result.method = CompositeMethod.MEAN  # type: ignore[misc]

    def test_summary_lines_ascii_only(self) -> None:
        result = self._make_result()
        for line in result.summary_lines():
            assert all(ord(c) < 128 for c in line)

    def test_summary_lines_contain_method(self) -> None:
        result   = self._make_result(CompositeMethod.MEDOID)
        combined = " ".join(result.summary_lines())
        assert "medoid" in combined

    def test_summary_shows_percentile_value_for_pct_method(self) -> None:
        pct_result = CompositeResult(
            image=MagicMock(),
            method=CompositeMethod.PERCENTILE,
            percentile_value=25,
            source_result=_make_processed_result(),
            band_names=COMMON_BAND_NAMES,
        )
        combined = " ".join(pct_result.summary_lines())
        assert "25" in combined


# ==============================================================================
# LandsatCompositor construction and method resolution tests
# ==============================================================================

class TestLandsatCompositorConstruction:
    """Tests for LandsatCompositor.__init__() and method resolution."""

    def test_construction_stores_config(self, tmp_path: Path) -> None:
        cfg    = _make_config(tmp_path)
        client = _make_mock_client()
        comp   = LandsatCompositor(client, cfg)
        assert comp._config is cfg

    def test_resolve_method_uses_override(self, tmp_path: Path) -> None:
        cfg    = _make_config(tmp_path, composite_method="median")
        client = _make_mock_client()
        comp   = LandsatCompositor(client, cfg)
        method = comp._resolve_method(CompositeMethod.MOSAIC)
        assert method == CompositeMethod.MOSAIC

    def test_resolve_method_reads_from_config(self, tmp_path: Path) -> None:
        cfg    = _make_config(tmp_path, composite_method="mean")
        client = _make_mock_client()
        comp   = LandsatCompositor(client, cfg)
        method = comp._resolve_method(None)
        assert method == CompositeMethod.MEAN

    def test_resolve_method_defaults_to_median_when_no_config(
        self, tmp_path: Path
    ) -> None:
        from src.core.config import Config
        data   = make_valid_config()
        cfg    = Config(config_path=write_config(tmp_path, data))
        client = _make_mock_client()
        comp   = LandsatCompositor(client, cfg)
        method = comp._resolve_method(None)
        assert method == CompositeMethod.MEDIAN

    def test_resolve_percentile_none_for_non_percentile_method(
        self, tmp_path: Path
    ) -> None:
        cfg    = _make_config(tmp_path)
        client = _make_mock_client()
        comp   = LandsatCompositor(client, cfg)
        pct    = comp._resolve_percentile(None, CompositeMethod.MEDIAN)
        assert pct is None

    def test_resolve_percentile_reads_from_config(
        self, tmp_path: Path
    ) -> None:
        cfg    = _make_config(tmp_path, composite_method="percentile")
        client = _make_mock_client()
        comp   = LandsatCompositor(client, cfg)
        pct    = comp._resolve_percentile(None, CompositeMethod.PERCENTILE)
        assert pct == 50

    def test_resolve_percentile_uses_override(
        self, tmp_path: Path
    ) -> None:
        cfg    = _make_config(tmp_path, composite_method="percentile")
        client = _make_mock_client()
        comp   = LandsatCompositor(client, cfg)
        pct    = comp._resolve_percentile(25, CompositeMethod.PERCENTILE)
        assert pct == 25

    def test_resolve_percentile_invalid_raises(
        self, tmp_path: Path
    ) -> None:
        cfg    = _make_config(tmp_path)
        client = _make_mock_client()
        comp   = LandsatCompositor(client, cfg)
        with pytest.raises(InvalidValueError, match="percentile_value"):
            comp._resolve_percentile(150, CompositeMethod.PERCENTILE)


# ==============================================================================
# LandsatCompositor.build_composite() tests
# ==============================================================================

class TestBuildComposite:
    """Tests for LandsatCompositor.build_composite() with each method."""

    def _compositor(self, tmp_path: Path, method: str = "median") -> LandsatCompositor:
        cfg    = _make_config(tmp_path, composite_method=method)
        client = _make_mock_client()
        return LandsatCompositor(client, cfg)

    def test_returns_composite_result(self, tmp_path: Path) -> None:
        comp      = self._compositor(tmp_path, "median")
        processed = _make_processed_result()
        mock_ee   = _make_mock_ee()
        with patch_ee(mock_ee):
            result = comp.build_composite(processed)
        assert isinstance(result, CompositeResult)

    def test_median_method_calls_collection_median(
        self, tmp_path: Path
    ) -> None:
        col       = _make_mock_collection()
        processed = _make_processed_result(collection=col)
        comp      = self._compositor(tmp_path, "median")
        mock_ee   = _make_mock_ee()
        with patch_ee(mock_ee):
            comp.build_composite(processed, method=CompositeMethod.MEDIAN)
        col.median.assert_called_once()

    def test_mean_method_calls_collection_mean(
        self, tmp_path: Path
    ) -> None:
        col       = _make_mock_collection()
        processed = _make_processed_result(collection=col)
        comp      = self._compositor(tmp_path, "mean")
        mock_ee   = _make_mock_ee()
        with patch_ee(mock_ee):
            comp.build_composite(processed, method=CompositeMethod.MEAN)
        col.mean.assert_called_once()

    def test_mosaic_method_calls_collection_mosaic(
        self, tmp_path: Path
    ) -> None:
        col       = _make_mock_collection()
        processed = _make_processed_result(collection=col)
        comp      = self._compositor(tmp_path, "mosaic")
        mock_ee   = _make_mock_ee()
        with patch_ee(mock_ee):
            comp.build_composite(processed, method=CompositeMethod.MOSAIC)
        col.mosaic.assert_called_once()

    def test_percentile_method_calls_reduce(
        self, tmp_path: Path
    ) -> None:
        col       = _make_mock_collection()
        processed = _make_processed_result(collection=col)
        comp      = self._compositor(tmp_path, "percentile")
        mock_ee   = _make_mock_ee()
        with patch_ee(mock_ee):
            comp.build_composite(
                processed,
                method=CompositeMethod.PERCENTILE,
                percentile_value=50,
            )
        col.reduce.assert_called_once()

    def test_medoid_method_calls_map_and_quality_mosaic(
        self, tmp_path: Path
    ) -> None:
        col       = _make_mock_collection()
        processed = _make_processed_result(collection=col, harmonized=True)
        comp      = self._compositor(tmp_path, "medoid")
        mock_ee   = _make_mock_ee()
        with patch_ee(mock_ee):
            comp.build_composite(processed, method=CompositeMethod.MEDOID)
        col.map.assert_called_once()
        col.qualityMosaic.assert_called_once_with("medoid_distance_score")

    def test_result_method_matches_requested(
        self, tmp_path: Path
    ) -> None:
        comp      = self._compositor(tmp_path, "mean")
        processed = _make_processed_result()
        mock_ee   = _make_mock_ee()
        with patch_ee(mock_ee):
            result = comp.build_composite(processed, method=CompositeMethod.MEAN)
        assert result.method == CompositeMethod.MEAN

    def test_result_image_is_set(self, tmp_path: Path) -> None:
        comp      = self._compositor(tmp_path, "median")
        processed = _make_processed_result()
        mock_ee   = _make_mock_ee()
        with patch_ee(mock_ee):
            result = comp.build_composite(processed)
        assert result.image is not None

    def test_result_source_result_is_preserved(
        self, tmp_path: Path
    ) -> None:
        comp      = self._compositor(tmp_path, "median")
        processed = _make_processed_result()
        mock_ee   = _make_mock_ee()
        with patch_ee(mock_ee):
            result = comp.build_composite(processed)
        assert result.source_result is processed

    def test_median_failure_raises_gee_api_error(
        self, tmp_path: Path
    ) -> None:
        col       = _make_mock_collection()
        col.median.side_effect = Exception("EE error")
        processed = _make_processed_result(collection=col)
        comp      = self._compositor(tmp_path, "median")
        mock_ee   = _make_mock_ee()
        with patch_ee(mock_ee):
            with pytest.raises(GEEAPIError, match="composite_median"):
                comp.build_composite(processed, method=CompositeMethod.MEDIAN)

    def test_mean_failure_raises_gee_api_error(
        self, tmp_path: Path
    ) -> None:
        col       = _make_mock_collection()
        col.mean.side_effect = Exception("EE error")
        processed = _make_processed_result(collection=col)
        comp      = self._compositor(tmp_path, "mean")
        mock_ee   = _make_mock_ee()
        with patch_ee(mock_ee):
            with pytest.raises(GEEAPIError, match="composite_mean"):
                comp.build_composite(processed, method=CompositeMethod.MEAN)

    def test_mosaic_failure_raises_gee_api_error(
        self, tmp_path: Path
    ) -> None:
        col       = _make_mock_collection()
        col.mosaic.side_effect = Exception("EE error")
        processed = _make_processed_result(collection=col)
        comp      = self._compositor(tmp_path, "mosaic")
        mock_ee   = _make_mock_ee()
        with patch_ee(mock_ee):
            with pytest.raises(GEEAPIError, match="composite_mosaic"):
                comp.build_composite(processed, method=CompositeMethod.MOSAIC)

    def test_medoid_ee_not_installed_raises(
        self, tmp_path: Path
    ) -> None:
        processed        = _make_processed_result()
        comp             = self._compositor(tmp_path, "medoid")
        modules_no_ee    = {k: v for k, v in sys.modules.items() if k != "ee"}
        with patch.dict(sys.modules, modules_no_ee, clear=True):
            with patch("builtins.__import__", side_effect=_block_ee_import):
                with pytest.raises(GEENotInstalledError):
                    comp.build_composite(
                        processed, method=CompositeMethod.MEDOID
                    )

    def test_percentile_ee_not_installed_raises(
        self, tmp_path: Path
    ) -> None:
        processed     = _make_processed_result()
        comp          = self._compositor(tmp_path, "percentile")
        modules_no_ee = {k: v for k, v in sys.modules.items() if k != "ee"}
        with patch.dict(sys.modules, modules_no_ee, clear=True):
            with patch("builtins.__import__", side_effect=_block_ee_import):
                with pytest.raises(GEENotInstalledError):
                    comp.build_composite(
                        processed,
                        method=CompositeMethod.PERCENTILE,
                        percentile_value=50,
                    )

    def test_medoid_uses_optical_bands_for_harmonized(
        self, tmp_path: Path
    ) -> None:
        """When harmonized=True, medoid selects OPTICAL_BAND_NAMES."""
        from src.gee.harmonization import OPTICAL_BAND_NAMES
        col       = _make_mock_collection()
        processed = _make_processed_result(collection=col, harmonized=True)
        comp      = self._compositor(tmp_path, "medoid")
        mock_ee   = _make_mock_ee()

        with patch_ee(mock_ee):
            comp.build_composite(processed, method=CompositeMethod.MEDOID)

        # select() should be called with OPTICAL_BAND_NAMES for distance calc.
        select_calls = [str(c) for c in col.select.call_args_list]
        assert any("Blue" in c or str(list(OPTICAL_BAND_NAMES)) in c
                   for c in select_calls), (
            "medoid should select optical bands for distance computation"
        )

    @pytest.mark.parametrize("method_name,method_enum", [
        ("median",     CompositeMethod.MEDIAN),
        ("mean",       CompositeMethod.MEAN),
        ("mosaic",     CompositeMethod.MOSAIC),
    ])
    def test_config_method_selection(
        self, tmp_path: Path, method_name: str, method_enum: CompositeMethod
    ) -> None:
        """Config-specified method is used when no override is given."""
        comp      = self._compositor(tmp_path, method_name)
        processed = _make_processed_result()
        mock_ee   = _make_mock_ee()
        with patch_ee(mock_ee):
            result = comp.build_composite(processed)
        assert result.method == method_enum


# ==============================================================================
# Private helper
# ==============================================================================

def _block_ee_import(name: str, *args, **kwargs):
    if name == "ee":
        raise ImportError("Simulated: ee not installed")
    import builtins
    return builtins.__import__(name, *args, **kwargs)