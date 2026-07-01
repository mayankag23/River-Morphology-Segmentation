"""
Unit tests for src/patches/validator.py.

Pure numpy, no I/O.

Run:
    pytest tests/patches/test_patch_validator.py -v
    pytest tests/patches/test_patch_validator.py -v \
        --cov=src/patches/validator --cov-report=term-missing
"""

from __future__ import annotations

import numpy as np
import pytest

from src.patches.validator import PatchValidationResult, PatchValidator


@pytest.fixture
def validator() -> PatchValidator:
    return PatchValidator(nodata_value=-9999.0, min_valid_pixel_ratio=0.7)


# ==============================================================================
# PatchValidationResult tests
# ==============================================================================

class TestPatchValidationResult:
    """Tests for the frozen PatchValidationResult dataclass."""

    def test_frozen(self) -> None:
        r = PatchValidationResult(
            is_valid=True, valid_pixel_ratio=1.0, total_pixels=4, valid_pixels=4
        )
        with pytest.raises((AttributeError, TypeError)):
            r.is_valid = False  # type: ignore[misc]


# ==============================================================================
# PatchValidator construction tests
# ==============================================================================

class TestPatchValidatorConstruction:
    """Tests for PatchValidator.__init__() and properties."""

    def test_stores_nodata_value(self, validator: PatchValidator) -> None:
        assert validator.nodata_value == pytest.approx(-9999.0)

    def test_stores_min_valid_ratio(self, validator: PatchValidator) -> None:
        assert validator.min_valid_pixel_ratio == pytest.approx(0.7)


# ==============================================================================
# PatchValidator.validate() tests
# ==============================================================================

class TestPatchValidatorValidate:
    """Tests for PatchValidator.validate() with various pixel conditions."""

    def test_all_valid_pixels(self, validator: PatchValidator) -> None:
        data = np.random.rand(3, 4, 4).astype(np.float32)
        result = validator.validate(data)
        assert result.is_valid is True
        assert result.valid_pixel_ratio == pytest.approx(1.0)

    def test_all_nan_pixels(self, validator: PatchValidator) -> None:
        data = np.full((3, 4, 4), np.nan, dtype=np.float32)
        result = validator.validate(data)
        assert result.is_valid is False
        assert result.valid_pixel_ratio == pytest.approx(0.0)

    def test_all_nodata_sentinel_pixels(self, validator: PatchValidator) -> None:
        data = np.full((3, 4, 4), -9999.0, dtype=np.float32)
        result = validator.validate(data)
        assert result.is_valid is False
        assert result.valid_pixel_ratio == pytest.approx(0.0)

    def test_mixed_nan_and_valid_above_threshold(self, validator: PatchValidator) -> None:
        data = np.random.rand(3, 4, 4).astype(np.float32)
        data[:, 0, 0] = np.nan       # invalid pixel 1
        data[:, 0, 1] = -9999.0      # invalid pixel 2
        result = validator.validate(data)
        # 14 of 16 pixels valid -> 0.875 >= 0.7
        assert result.is_valid is True
        assert result.valid_pixel_ratio == pytest.approx(14 / 16)

    def test_mixed_below_threshold(self, validator: PatchValidator) -> None:
        data = np.random.rand(3, 4, 4).astype(np.float32)
        for i in range(6):
            r, c = divmod(i, 4)
            data[:, r, c] = np.nan
        result = validator.validate(data)
        # 10 of 16 pixels valid -> 0.625 < 0.7
        assert result.is_valid is False
        assert result.valid_pixel_ratio == pytest.approx(10 / 16)

    def test_threshold_boundary_exact_match_is_valid(self) -> None:
        validator = PatchValidator(nodata_value=-9999.0, min_valid_pixel_ratio=0.75)
        data = np.random.rand(1, 2, 2).astype(np.float32)
        data[:, 0, 0] = np.nan  # 1 of 4 invalid -> exactly 0.75
        result = validator.validate(data)
        assert result.is_valid is True  # >= threshold (inclusive)

    def test_pixel_invalid_if_any_band_invalid(self, validator: PatchValidator) -> None:
        data = np.random.rand(3, 2, 2).astype(np.float32)
        data[1, 0, 0] = np.nan  # only band index 1 is NaN at this pixel
        result = validator.validate(data)
        assert result.valid_pixels == 3  # 4 total - 1 invalid pixel

    def test_total_pixels_correct(self, validator: PatchValidator) -> None:
        data = np.random.rand(5, 8, 6).astype(np.float32)
        result = validator.validate(data)
        assert result.total_pixels == 48

    def test_zero_min_ratio_always_valid(self) -> None:
        validator = PatchValidator(nodata_value=-9999.0, min_valid_pixel_ratio=0.0)
        data = np.full((1, 2, 2), np.nan, dtype=np.float32)
        result = validator.validate(data)
        assert result.is_valid is True

    def test_one_min_ratio_requires_perfect_patch(self) -> None:
        validator = PatchValidator(nodata_value=-9999.0, min_valid_pixel_ratio=1.0)
        data = np.random.rand(1, 2, 2).astype(np.float32)
        data[:, 0, 0] = np.nan
        result = validator.validate(data)
        assert result.is_valid is False

    def test_negative_nodata_does_not_flag_positive_values(self) -> None:
        validator = PatchValidator(nodata_value=-9999.0, min_valid_pixel_ratio=1.0)
        data = np.full((1, 2, 2), 9999.0, dtype=np.float32)  # positive, not nodata
        result = validator.validate(data)
        assert result.is_valid is True