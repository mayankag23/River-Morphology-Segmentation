"""
Unit tests for src/gee/preprocessing.py.

Tests cover:
    - ScalingConfig defaults and custom values
    - ProcessedCollectionResult properties and summary output
    - LandsatPreprocessor construction from config
    - process() with all stages enabled
    - process() with individual stages toggled
    - _make_scale_function() closure behaviour
    - _build_scaling_config() reads from config correctly
    - Warning emitted for mixed sensors without harmonization
    - operations_applied reflects completed stages

Run:
    pytest tests/test_preprocessing.py -v
    pytest tests/test_preprocessing.py -v --cov=src/gee/preprocessing --cov-report=term-missing
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from src.gee.collections import CollectionResult, LandsatSensor
from src.gee.harmonization import COMMON_BAND_NAMES
from src.gee.preprocessing import (
    LandsatPreprocessor,
    ProcessedCollectionResult,
    ScalingConfig,
)
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
    return ee


def _make_mock_client() -> MagicMock:
    from src.gee.client import EarthEngineClient
    client = MagicMock(spec=EarthEngineClient)
    client.is_initialized = True
    return client


def _make_mock_collection() -> MagicMock:
    col = MagicMock()
    col.map.return_value = col
    return col


def _make_collection_result(
    sensors: tuple[LandsatSensor, ...] = (LandsatSensor.L8, LandsatSensor.L9),
    mixed: bool = False,
) -> CollectionResult:
    """Build a minimal CollectionResult for testing."""
    if mixed:
        sensors = (LandsatSensor.L7, LandsatSensor.L8)
    return CollectionResult(
        collection=_make_mock_collection(),
        sensors=sensors,
        collection_ids=tuple(
            f"LANDSAT/{s.name}/C02/T1_L2" for s in sensors
        ),
        start_date="2023-11-01",
        end_date="2024-02-28",
        cloud_cover_limit=20.0,
        filters_applied=("date", "bounds", "cloud_cover"),
    )


def _make_config(tmp_path: Path, extra: dict | None = None):
    from src.core.config import Config
    data = make_valid_config()
    data["preprocessing"].update({
        "mask_cloud":         True,
        "mask_cloud_shadow":  True,
        "mask_snow":          False,
        "mask_cirrus":        True,
        "mask_dilated_cloud": True,
        "mask_fill":          True,
        "thermal_scale_factor": 0.00341802,
        "thermal_offset":       149.0,
        "clip_to_valid_range":  True,
    })
    if extra:
        data["preprocessing"].update(extra)
    return Config(config_path=write_config(tmp_path, data))


# ==============================================================================
# ScalingConfig tests
# ==============================================================================

class TestScalingConfig:
    """Tests for the ScalingConfig frozen dataclass."""

    def test_default_sr_scale_factor(self) -> None:
        cfg = ScalingConfig()
        assert cfg.sr_scale_factor == pytest.approx(0.0000275)

    def test_default_sr_offset(self) -> None:
        cfg = ScalingConfig()
        assert cfg.sr_offset == pytest.approx(-0.2)

    def test_default_thermal_scale_factor(self) -> None:
        cfg = ScalingConfig()
        assert cfg.thermal_scale_factor == pytest.approx(0.00341802)

    def test_default_thermal_offset(self) -> None:
        cfg = ScalingConfig()
        assert cfg.thermal_offset == pytest.approx(149.0)

    def test_default_clip_to_valid_range_true(self) -> None:
        cfg = ScalingConfig()
        assert cfg.clip_to_valid_range is True

    def test_custom_values(self) -> None:
        cfg = ScalingConfig(sr_scale_factor=0.0001, clip_to_valid_range=False)
        assert cfg.sr_scale_factor == pytest.approx(0.0001)
        assert cfg.clip_to_valid_range is False

    def test_frozen_prevents_mutation(self) -> None:
        cfg = ScalingConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.sr_offset = 0.0  # type: ignore[misc]


# ==============================================================================
# ProcessedCollectionResult tests
# ==============================================================================

class TestProcessedCollectionResult:
    """Tests for ProcessedCollectionResult properties."""

    def _make_result(
        self,
        scale: bool = True,
        mask: bool = True,
        harmonize: bool = True,
    ) -> ProcessedCollectionResult:
        source = _make_collection_result()
        ops = []
        if scale:     ops.append("scaling")
        if mask:      ops.append("qa_masking")
        if harmonize: ops.append("harmonization")
        return ProcessedCollectionResult(
            collection=MagicMock(),
            source_result=source,
            operations_applied=tuple(ops),
            band_names=COMMON_BAND_NAMES if harmonize else (),
            scale_applied=scale,
            masking_applied=mask,
            harmonization_applied=harmonize,
        )

    def test_scale_applied_true(self) -> None:
        result = self._make_result(scale=True)
        assert result.scale_applied is True

    def test_masking_applied_true(self) -> None:
        result = self._make_result(mask=True)
        assert result.masking_applied is True

    def test_harmonization_applied_true(self) -> None:
        result = self._make_result(harmonize=True)
        assert result.harmonization_applied is True

    def test_band_names_set_to_common_when_harmonized(self) -> None:
        result = self._make_result(harmonize=True)
        assert result.band_names == COMMON_BAND_NAMES

    def test_band_names_empty_when_not_harmonized(self) -> None:
        result = self._make_result(harmonize=False)
        assert result.band_names == ()

    def test_operations_applied_all_three(self) -> None:
        result = self._make_result()
        assert "scaling"        in result.operations_applied
        assert "qa_masking"     in result.operations_applied
        assert "harmonization"  in result.operations_applied

    def test_operations_applied_only_scaling(self) -> None:
        result = self._make_result(scale=True, mask=False, harmonize=False)
        assert result.operations_applied == ("scaling",)

    def test_frozen_prevents_mutation(self) -> None:
        result = self._make_result()
        with pytest.raises((AttributeError, TypeError)):
            result.scale_applied = False  # type: ignore[misc]

    def test_summary_lines_are_ascii_only(self) -> None:
        result = self._make_result()
        for line in result.summary_lines():
            assert all(ord(c) < 128 for c in line)

    def test_summary_lines_contain_scale_info(self) -> None:
        result   = self._make_result(scale=True)
        combined = " ".join(result.summary_lines())
        assert "Scale" in combined

    def test_summary_warns_for_mixed_sensors(self) -> None:
        source = _make_collection_result(mixed=True)
        result = ProcessedCollectionResult(
            collection=MagicMock(),
            source_result=source,
            operations_applied=("scaling",),
            band_names=(),
            scale_applied=True,
            masking_applied=False,
            harmonization_applied=False,
        )
        combined = " ".join(result.summary_lines())
        assert "mixed" in combined.lower() or "harmonization" in combined.lower()


# ==============================================================================
# LandsatPreprocessor construction tests
# ==============================================================================

class TestLandsatPreprocessorConstruction:
    """Tests for LandsatPreprocessor.__init__()."""

    def test_construction_does_not_call_ee(
        self, tmp_path: Path
    ) -> None:
        cfg    = _make_config(tmp_path)
        client = _make_mock_client()
        LandsatPreprocessor(client, cfg)
        # No EE calls should be made at construction time.

    def test_stores_config(self, tmp_path: Path) -> None:
        cfg    = _make_config(tmp_path)
        client = _make_mock_client()
        pp     = LandsatPreprocessor(client, cfg)
        assert pp._config is cfg

    def test_builds_scaling_config_from_satellite(
        self, tmp_path: Path
    ) -> None:
        cfg    = _make_config(tmp_path)
        client = _make_mock_client()
        pp     = LandsatPreprocessor(client, cfg)
        assert pp._scaling_config.sr_scale_factor == pytest.approx(0.0000275)

    def test_builds_scaling_config_with_thermal(
        self, tmp_path: Path
    ) -> None:
        cfg    = _make_config(tmp_path)
        client = _make_mock_client()
        pp     = LandsatPreprocessor(client, cfg)
        assert pp._scaling_config.thermal_scale_factor == pytest.approx(0.00341802)

    def test_masker_reads_cloud_flag(self, tmp_path: Path) -> None:
        cfg    = _make_config(tmp_path, extra={"mask_cloud": True})
        client = _make_mock_client()
        pp     = LandsatPreprocessor(client, cfg)
        assert pp._masker.mask_config.mask_cloud is True

    def test_masker_reads_snow_flag(self, tmp_path: Path) -> None:
        cfg    = _make_config(tmp_path, extra={"mask_snow": True})
        client = _make_mock_client()
        pp     = LandsatPreprocessor(client, cfg)
        assert pp._masker.mask_config.mask_snow is True


# ==============================================================================
# LandsatPreprocessor.process() tests
# ==============================================================================

class TestProcessAllStages:
    """Tests for LandsatPreprocessor.process() with all stages enabled."""

    def test_returns_processed_collection_result(
        self, tmp_path: Path
    ) -> None:
        cfg    = _make_config(tmp_path)
        client = _make_mock_client()
        pp     = LandsatPreprocessor(client, cfg)
        source = _make_collection_result()
        mock_ee = _make_mock_ee()
        with patch_ee(mock_ee):
            result = pp.process(source)
        assert isinstance(result, ProcessedCollectionResult)

    def test_all_three_operations_applied(
        self, tmp_path: Path
    ) -> None:
        cfg    = _make_config(tmp_path)
        client = _make_mock_client()
        pp     = LandsatPreprocessor(client, cfg)
        source = _make_collection_result()
        mock_ee = _make_mock_ee()
        with patch_ee(mock_ee):
            result = pp.process(source)
        assert "scaling"       in result.operations_applied
        assert "qa_masking"    in result.operations_applied
        assert "harmonization" in result.operations_applied

    def test_source_result_preserved_in_output(
        self, tmp_path: Path
    ) -> None:
        cfg    = _make_config(tmp_path)
        client = _make_mock_client()
        pp     = LandsatPreprocessor(client, cfg)
        source = _make_collection_result()
        mock_ee = _make_mock_ee()
        with patch_ee(mock_ee):
            result = pp.process(source)
        assert result.source_result is source

    def test_band_names_set_to_common_after_harmonization(
        self, tmp_path: Path
    ) -> None:
        cfg    = _make_config(tmp_path)
        client = _make_mock_client()
        pp     = LandsatPreprocessor(client, cfg)
        source = _make_collection_result()
        mock_ee = _make_mock_ee()
        with patch_ee(mock_ee):
            result = pp.process(source)
        assert result.band_names == COMMON_BAND_NAMES

    def test_collection_map_called_for_each_stage(
        self, tmp_path: Path
    ) -> None:
        """collection.map() should be called once per active stage."""
        cfg    = _make_config(tmp_path)
        client = _make_mock_client()
        pp     = LandsatPreprocessor(client, cfg)
        source = _make_collection_result()
        col    = source.collection
        mock_ee = _make_mock_ee()
        with patch_ee(mock_ee):
            pp.process(source)
        # Scaling, masking, and harmonization each call collection.map() once.
        assert col.map.call_count == 3


class TestProcessWithToggledStages:
    """Tests for process() with individual stages disabled."""

    def test_scaling_only(self, tmp_path: Path) -> None:
        cfg    = _make_config(tmp_path)
        client = _make_mock_client()
        pp     = LandsatPreprocessor(client, cfg)
        source = _make_collection_result()
        mock_ee = _make_mock_ee()
        with patch_ee(mock_ee):
            result = pp.process(
                source,
                apply_scaling=True,
                apply_masking=False,
                apply_harmonization=False,
            )
        assert result.scale_applied is True
        assert result.masking_applied is False
        assert result.harmonization_applied is False
        assert result.operations_applied == ("scaling",)

    def test_masking_only(self, tmp_path: Path) -> None:
        cfg    = _make_config(tmp_path)
        client = _make_mock_client()
        pp     = LandsatPreprocessor(client, cfg)
        source = _make_collection_result()
        mock_ee = _make_mock_ee()
        with patch_ee(mock_ee):
            result = pp.process(
                source,
                apply_scaling=False,
                apply_masking=True,
                apply_harmonization=False,
            )
        assert result.masking_applied is True
        assert result.scale_applied   is False
        assert result.operations_applied == ("qa_masking",)

    def test_harmonization_only(self, tmp_path: Path) -> None:
        cfg    = _make_config(tmp_path)
        client = _make_mock_client()
        pp     = LandsatPreprocessor(client, cfg)
        source = _make_collection_result()
        mock_ee = _make_mock_ee()
        with patch_ee(mock_ee):
            result = pp.process(
                source,
                apply_scaling=False,
                apply_masking=False,
                apply_harmonization=True,
            )
        assert result.harmonization_applied is True
        assert result.band_names == COMMON_BAND_NAMES

    def test_no_stages_returns_original_collection(
        self, tmp_path: Path
    ) -> None:
        cfg    = _make_config(tmp_path)
        client = _make_mock_client()
        pp     = LandsatPreprocessor(client, cfg)
        source = _make_collection_result()
        mock_ee = _make_mock_ee()
        with patch_ee(mock_ee):
            result = pp.process(
                source,
                apply_scaling=False,
                apply_masking=False,
                apply_harmonization=False,
            )
        assert result.operations_applied == ()
        assert result.collection is source.collection

    def test_map_not_called_when_all_stages_disabled(
        self, tmp_path: Path
    ) -> None:
        cfg    = _make_config(tmp_path)
        client = _make_mock_client()
        pp     = LandsatPreprocessor(client, cfg)
        source = _make_collection_result()
        col    = source.collection
        mock_ee = _make_mock_ee()
        with patch_ee(mock_ee):
            pp.process(
                source,
                apply_scaling=False,
                apply_masking=False,
                apply_harmonization=False,
            )
        col.map.assert_not_called()

    def test_mixed_sensor_without_harmonization_logs_warning(
        self, tmp_path: Path
    ) -> None:
        cfg    = _make_config(tmp_path)
        client = _make_mock_client()
        pp     = LandsatPreprocessor(client, cfg)
        source = _make_collection_result(mixed=True)
        mock_ee = _make_mock_ee()
        with patch_ee(mock_ee):
            with patch.object(pp._logger, "warning") as mock_warn:
                pp.process(
                    source,
                    apply_scaling=False,
                    apply_masking=False,
                    apply_harmonization=False,
                )
        mock_warn.assert_called()


# ==============================================================================
# _make_scale_function tests
# ==============================================================================

class TestMakeScaleFunction:
    """Tests for LandsatPreprocessor._make_scale_function()."""

    def test_returns_callable(self, tmp_path: Path) -> None:
        cfg    = _make_config(tmp_path)
        client = _make_mock_client()
        pp     = LandsatPreprocessor(client, cfg)
        mock_ee = _make_mock_ee()
        with patch_ee(mock_ee):
            fn = pp._make_scale_function()
        assert callable(fn)

    def test_scale_function_calls_select_sr_bands(
        self, tmp_path: Path
    ) -> None:
        cfg    = _make_config(tmp_path)
        client = _make_mock_client()
        pp     = LandsatPreprocessor(client, cfg)
        image  = MagicMock()
        mock_ee = _make_mock_ee()
        with patch_ee(mock_ee):
            fn = pp._make_scale_function()
            fn(image)
        # Check that SR_B.* was selected for optical scaling.
        calls = [str(c) for c in image.select.call_args_list]
        assert any("SR_B" in c for c in calls)

    def test_scale_function_calls_select_st_bands(
        self, tmp_path: Path
    ) -> None:
        cfg    = _make_config(tmp_path)
        client = _make_mock_client()
        pp     = LandsatPreprocessor(client, cfg)
        image  = MagicMock()
        mock_ee = _make_mock_ee()
        with patch_ee(mock_ee):
            fn = pp._make_scale_function()
            fn(image)
        calls = [str(c) for c in image.select.call_args_list]
        assert any("ST_B" in c for c in calls)

    def test_scale_function_calls_multiply_on_sr_result(
        self, tmp_path: Path
    ) -> None:
        cfg    = _make_config(tmp_path)
        client = _make_mock_client()
        pp     = LandsatPreprocessor(client, cfg)
        image  = MagicMock()
        mock_ee = _make_mock_ee()
        with patch_ee(mock_ee):
            fn = pp._make_scale_function()
            fn(image)
        optical_result = image.select.return_value
        optical_result.multiply.assert_called()

    def test_scale_function_adds_bands_with_overwrite(
        self, tmp_path: Path
    ) -> None:
        cfg    = _make_config(tmp_path)
        client = _make_mock_client()
        pp     = LandsatPreprocessor(client, cfg)
        image  = MagicMock()
        mock_ee = _make_mock_ee()
        with patch_ee(mock_ee):
            fn = pp._make_scale_function()
            fn(image)
        # addBands(overwrite=True) should be called.
        calls = image.addBands.call_args_list
        assert any(
            kw.get("overwrite") is True or (args and len(args) >= 3 and args[2] is True)
            for args, kw in calls
        )


# ==============================================================================
# _build_scaling_config tests
# ==============================================================================

class TestBuildScalingConfig:
    """Tests for LandsatPreprocessor._build_scaling_config()."""

    def test_reads_sr_scale_factor_from_satellite_config(
        self, tmp_path: Path
    ) -> None:
        cfg    = _make_config(tmp_path)
        client = _make_mock_client()
        pp     = LandsatPreprocessor(client, cfg)
        assert pp._scaling_config.sr_scale_factor == pytest.approx(0.0000275)

    def test_reads_sr_offset_from_satellite_config(
        self, tmp_path: Path
    ) -> None:
        cfg    = _make_config(tmp_path)
        client = _make_mock_client()
        pp     = LandsatPreprocessor(client, cfg)
        assert pp._scaling_config.sr_offset == pytest.approx(-0.2)

    def test_reads_thermal_scale_from_preprocessing(
        self, tmp_path: Path
    ) -> None:
        cfg    = _make_config(tmp_path)
        client = _make_mock_client()
        pp     = LandsatPreprocessor(client, cfg)
        assert pp._scaling_config.thermal_scale_factor == pytest.approx(0.00341802)

    def test_reads_clip_to_valid_range_from_preprocessing(
        self, tmp_path: Path
    ) -> None:
        cfg    = _make_config(tmp_path)
        client = _make_mock_client()
        pp     = LandsatPreprocessor(client, cfg)
        assert pp._scaling_config.clip_to_valid_range is True

    def test_defaults_used_when_thermal_keys_absent(
        self, tmp_path: Path
    ) -> None:
        """When thermal keys are not in config, defaults apply."""
        from src.core.config import Config
        data   = make_valid_config()
        cfg    = Config(config_path=write_config(tmp_path, data))
        client = _make_mock_client()
        pp     = LandsatPreprocessor(client, cfg)
        # Default thermal scale factor should be applied.
        assert pp._scaling_config.thermal_scale_factor == pytest.approx(0.00341802)