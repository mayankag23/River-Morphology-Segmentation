"""
Unit tests for src/gee/masking.py.

Tests cover:
    - QAMaskConfig defaults and frozen behaviour
    - LandsatQAMasker construction and from_preprocessing_config()
    - apply_to_image() with specific bit conditions
    - apply_to_collection() delegation
    - _build_qa_mask() bit logic
    - _bit_flag() helper
    - No-op path when all masking flags disabled

Run:
    pytest tests/test_masking.py -v
    pytest tests/test_masking.py -v --cov=src/gee/masking --cov-report=term-missing
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from src.gee.masking import (
    QA_BIT_CIRRUS,
    QA_BIT_CLOUD,
    QA_BIT_CLOUD_SHADOW,
    QA_BIT_DILATED_CLOUD,
    QA_BIT_FILL,
    QA_BIT_SNOW,
    QA_PIXEL_BAND,
    LandsatQAMasker,
    QAMaskConfig,
)
from src.gee import GEEAPIError
from tests.conftest import make_valid_config, write_config


# ==============================================================================
# Helpers
# ==============================================================================

@contextmanager
def patch_ee(mock_ee: MagicMock):
    with patch.dict(sys.modules, {"ee": mock_ee}):
        yield


def _make_config(tmp_path: Path, extra_preprocessing: dict | None = None):
    from src.core.config import Config
    data = make_valid_config()
    if extra_preprocessing:
        data["preprocessing"].update(extra_preprocessing)
    return Config(config_path=write_config(tmp_path, data))


def _make_mock_ee() -> MagicMock:
    ee = MagicMock()
    # Image.constant(0) returns a mock image
    const_img = MagicMock()
    const_img.Or.return_value = const_img
    const_img.Not.return_value = const_img
    ee.Image.constant.return_value = const_img
    return ee


def _make_mock_image() -> MagicMock:
    img = MagicMock()
    qa  = MagicMock()
    qa.bitwiseAnd.return_value.neq.return_value = MagicMock()
    img.select.return_value = qa
    img.updateMask.return_value = img
    return img


def _make_mock_collection() -> MagicMock:
    col = MagicMock()
    col.map.return_value = col
    return col


# ==============================================================================
# QAMaskConfig tests
# ==============================================================================

class TestQAMaskConfig:
    """Tests for the QAMaskConfig frozen dataclass."""

    def test_default_mask_cloud_true(self) -> None:
        cfg = QAMaskConfig()
        assert cfg.mask_cloud is True

    def test_default_mask_cloud_shadow_true(self) -> None:
        cfg = QAMaskConfig()
        assert cfg.mask_cloud_shadow is True

    def test_default_mask_snow_false(self) -> None:
        cfg = QAMaskConfig()
        assert cfg.mask_snow is False

    def test_default_mask_fill_true(self) -> None:
        cfg = QAMaskConfig()
        assert cfg.mask_fill is True

    def test_default_mask_cirrus_true(self) -> None:
        cfg = QAMaskConfig()
        assert cfg.mask_cirrus is True

    def test_default_mask_dilated_cloud_true(self) -> None:
        cfg = QAMaskConfig()
        assert cfg.mask_dilated_cloud is True

    def test_custom_values_applied(self) -> None:
        cfg = QAMaskConfig(
            mask_cloud=False, mask_snow=True, mask_fill=False
        )
        assert cfg.mask_cloud is False
        assert cfg.mask_snow is True
        assert cfg.mask_fill is False

    def test_frozen_prevents_mutation(self) -> None:
        cfg = QAMaskConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.mask_cloud = False  # type: ignore[misc]

    def test_any_enabled_true_with_defaults(self) -> None:
        assert QAMaskConfig().any_enabled() is True

    def test_any_enabled_false_when_all_disabled(self) -> None:
        cfg = QAMaskConfig(
            mask_fill=False, mask_dilated_cloud=False, mask_cirrus=False,
            mask_cloud=False, mask_cloud_shadow=False, mask_snow=False,
        )
        assert cfg.any_enabled() is False

    def test_any_enabled_true_when_one_enabled(self) -> None:
        cfg = QAMaskConfig(
            mask_fill=False, mask_dilated_cloud=False, mask_cirrus=False,
            mask_cloud=True, mask_cloud_shadow=False, mask_snow=False,
        )
        assert cfg.any_enabled() is True

    def test_summary_is_ascii_only(self) -> None:
        cfg = QAMaskConfig()
        summary = cfg.summary()
        assert all(ord(c) < 128 for c in summary)

    def test_summary_contains_enabled_flags(self) -> None:
        cfg     = QAMaskConfig(mask_snow=True)
        summary = cfg.summary()
        assert "snow" in summary

    def test_summary_does_not_contain_disabled_flags(self) -> None:
        cfg     = QAMaskConfig(mask_snow=False)
        summary = cfg.summary()
        assert "snow" not in summary


# ==============================================================================
# LandsatQAMasker construction tests
# ==============================================================================

class TestLandsatQAMaskerConstruction:
    """Tests for LandsatQAMasker construction and from_preprocessing_config()."""

    def test_construction_stores_config(self) -> None:
        mask_cfg = QAMaskConfig(mask_snow=True)
        masker   = LandsatQAMasker(mask_cfg)
        assert masker.mask_config.mask_snow is True

    def test_from_preprocessing_config_reads_flags(
        self, tmp_path: Path
    ) -> None:
        cfg = _make_config(tmp_path, extra_preprocessing={
            "mask_cloud": True,
            "mask_snow":  True,
            "mask_fill":  False,
        })
        masker = LandsatQAMasker.from_preprocessing_config(cfg)
        assert masker.mask_config.mask_cloud is True
        assert masker.mask_config.mask_snow  is True
        assert masker.mask_config.mask_fill  is False

    def test_from_preprocessing_config_defaults_when_keys_absent(
        self, tmp_path: Path
    ) -> None:
        """Keys absent from config.yaml fall back to QAMaskConfig defaults."""
        cfg    = _make_config(tmp_path)
        masker = LandsatQAMasker.from_preprocessing_config(cfg)
        assert masker.mask_config.mask_cloud        is True
        assert masker.mask_config.mask_cloud_shadow is True
        assert masker.mask_config.mask_snow         is False

    def test_mask_config_property_returns_config(self) -> None:
        cfg    = QAMaskConfig()
        masker = LandsatQAMasker(cfg)
        assert masker.mask_config is cfg


# ==============================================================================
# LandsatQAMasker.apply_to_collection tests
# ==============================================================================

class TestApplyToCollection:
    """Tests for LandsatQAMasker.apply_to_collection()."""

    def test_calls_collection_map(self) -> None:
        masker     = LandsatQAMasker(QAMaskConfig())
        collection = _make_mock_collection()
        mock_ee    = _make_mock_ee()
        with patch_ee(mock_ee):
            masker.apply_to_collection(collection)
        collection.map.assert_called_once()

    def test_returns_mapped_collection(self) -> None:
        masker     = LandsatQAMasker(QAMaskConfig())
        collection = _make_mock_collection()
        mock_ee    = _make_mock_ee()
        with patch_ee(mock_ee):
            result = masker.apply_to_collection(collection)
        assert result is collection.map.return_value

    def test_no_map_call_when_all_flags_disabled(self) -> None:
        """When no masking is enabled, the original collection is returned."""
        all_off = QAMaskConfig(
            mask_fill=False, mask_dilated_cloud=False, mask_cirrus=False,
            mask_cloud=False, mask_cloud_shadow=False, mask_snow=False,
        )
        masker     = LandsatQAMasker(all_off)
        collection = _make_mock_collection()
        result     = masker.apply_to_collection(collection)
        collection.map.assert_not_called()
        assert result is collection

    def test_collection_map_failure_raises_gee_api_error(self) -> None:
        masker     = LandsatQAMasker(QAMaskConfig())
        collection = _make_mock_collection()
        collection.map.side_effect = Exception("EE map failed")
        mock_ee    = _make_mock_ee()
        with patch_ee(mock_ee):
            with pytest.raises(GEEAPIError, match="apply_qa_masking_to_collection"):
                masker.apply_to_collection(collection)


# ==============================================================================
# LandsatQAMasker.apply_to_image tests
# ==============================================================================

class TestApplyToImage:
    """Tests for LandsatQAMasker.apply_to_image()."""

    def test_calls_update_mask(self) -> None:
        masker = LandsatQAMasker(QAMaskConfig())
        image  = _make_mock_image()
        mock_ee = _make_mock_ee()
        with patch_ee(mock_ee):
            masker.apply_to_image(image)
        image.updateMask.assert_called_once()

    def test_returns_updated_image(self) -> None:
        masker = LandsatQAMasker(QAMaskConfig())
        image  = _make_mock_image()
        mock_ee = _make_mock_ee()
        with patch_ee(mock_ee):
            result = masker.apply_to_image(image)
        assert result is image.updateMask.return_value

    def test_selects_qa_pixel_band(self) -> None:
        masker  = LandsatQAMasker(QAMaskConfig())
        image   = _make_mock_image()
        mock_ee = _make_mock_ee()
        with patch_ee(mock_ee):
            masker.apply_to_image(image)
        image.select.assert_called_with(QA_PIXEL_BAND)

    def test_update_mask_failure_raises_gee_api_error(self) -> None:
        masker = LandsatQAMasker(QAMaskConfig())
        image  = _make_mock_image()
        image.updateMask.side_effect = Exception("EE error")
        mock_ee = _make_mock_ee()
        with patch_ee(mock_ee):
            with pytest.raises(GEEAPIError, match="apply_qa_masking_to_image"):
                masker.apply_to_image(image)


# ==============================================================================
# LandsatQAMasker._build_qa_mask tests
# ==============================================================================

class TestBuildQAMask:
    """Tests for the _build_qa_mask() bit logic."""

    def test_cloud_flag_calls_bitwise_and_with_cloud_bit(self) -> None:
        masker = LandsatQAMasker(QAMaskConfig(
            mask_cloud=True, mask_fill=False, mask_dilated_cloud=False,
            mask_cirrus=False, mask_cloud_shadow=False, mask_snow=False,
        ))
        image  = _make_mock_image()
        qa     = image.select.return_value
        mock_ee = _make_mock_ee()
        with patch_ee(mock_ee):
            masker._build_qa_mask(image)
        # Bit 3 = cloud. 1 << 3 = 8.
        qa.bitwiseAnd.assert_called_with(1 << QA_BIT_CLOUD)

    def test_snow_flag_calls_bitwise_and_with_snow_bit(self) -> None:
        masker = LandsatQAMasker(QAMaskConfig(
            mask_cloud=False, mask_fill=False, mask_dilated_cloud=False,
            mask_cirrus=False, mask_cloud_shadow=False, mask_snow=True,
        ))
        image  = _make_mock_image()
        qa     = image.select.return_value
        mock_ee = _make_mock_ee()
        with patch_ee(mock_ee):
            masker._build_qa_mask(image)
        qa.bitwiseAnd.assert_called_with(1 << QA_BIT_SNOW)

    def test_no_bitwise_and_calls_when_all_disabled(self) -> None:
        masker = LandsatQAMasker(QAMaskConfig(
            mask_fill=False, mask_dilated_cloud=False, mask_cirrus=False,
            mask_cloud=False, mask_cloud_shadow=False, mask_snow=False,
        ))
        image  = _make_mock_image()
        qa     = image.select.return_value
        mock_ee = _make_mock_ee()
        with patch_ee(mock_ee):
            masker._build_qa_mask(image)
        qa.bitwiseAnd.assert_not_called()

    def test_result_is_not_inverted_bad_mask(self) -> None:
        """The mask returned should call .Not() on the bad-pixel mask."""
        masker  = LandsatQAMasker(QAMaskConfig(mask_cloud=True))
        image   = _make_mock_image()
        mock_ee = _make_mock_ee()
        const   = mock_ee.Image.constant.return_value
        with patch_ee(mock_ee):
            result = masker._build_qa_mask(image)
        # The chain ends with bad_mask.Not()
        const.Not.assert_called_once()

    def test_multiple_flags_combine_with_or(self) -> None:
        masker = LandsatQAMasker(QAMaskConfig(
            mask_cloud=True, mask_cloud_shadow=True,
            mask_fill=False, mask_dilated_cloud=False,
            mask_cirrus=False, mask_snow=False,
        ))
        image   = _make_mock_image()
        mock_ee = _make_mock_ee()
        const   = mock_ee.Image.constant.return_value
        with patch_ee(mock_ee):
            masker._build_qa_mask(image)
        # .Or() should be called twice (once per enabled flag).
        assert const.Or.call_count == 2


# ==============================================================================
# LandsatQAMasker._bit_flag tests
# ==============================================================================

class TestBitFlag:
    """Tests for the static _bit_flag() helper."""

    def test_calls_bitwise_and_with_shifted_bit(self) -> None:
        qa  = MagicMock()
        LandsatQAMasker._bit_flag(qa, 3)
        qa.bitwiseAnd.assert_called_once_with(1 << 3)

    def test_calls_neq_zero_on_result(self) -> None:
        qa      = MagicMock()
        and_res = qa.bitwiseAnd.return_value
        LandsatQAMasker._bit_flag(qa, 3)
        and_res.neq.assert_called_once_with(0)

    @pytest.mark.parametrize("bit", [0, 1, 2, 3, 4, 5, 6, 7])
    def test_all_valid_bit_positions(self, bit: int) -> None:
        qa     = MagicMock()
        result = LandsatQAMasker._bit_flag(qa, bit)
        assert result is qa.bitwiseAnd.return_value.neq.return_value