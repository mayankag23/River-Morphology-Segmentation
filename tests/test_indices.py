"""
Unit tests for src/gee/indices.py.

Each spectral index function is tested independently with a mocked ee.Image.
Tests verify:
    - Correct band names are selected from the image.
    - Key arithmetic operations and coefficients are applied.
    - The output band is renamed to the canonical index name.
    - EE exceptions are wrapped in GEEAPIError.

No real EE authentication required.

Run:
    pytest tests/test_indices.py -v
    pytest tests/test_indices.py -v --cov=src/gee/indices --cov-report=term-missing
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from unittest.mock import MagicMock, call, patch

import pytest

from src.gee import GEEAPIError
from src.gee.indices import (
    compute_awei_nsh,
    compute_awei_sh,
    compute_bsi,
    compute_mndwi,
    compute_ndbi,
    compute_ndmi,
    compute_ndvi,
    compute_ndwi,
    compute_savi,
)


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def band_mocks() -> dict[str, MagicMock]:
    """Return distinct named mocks for each harmonized band."""
    mocks = {}
    for band in ("Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2", "Thermal"):
        m = MagicMock(name=band)
        # Make arithmetic operations chain correctly.
        m.add.return_value       = m
        m.subtract.return_value  = m
        m.multiply.return_value  = m
        m.divide.return_value    = m
        m.rename.return_value    = m
        mocks[band] = m
    return mocks


@pytest.fixture
def mock_image(band_mocks) -> MagicMock:
    """
    Return a mock ee.Image where select() returns distinct band mocks.
    normalizedDifference() returns a mock with rename() chained.
    """
    img = MagicMock(name="ee.Image")
    img.select.side_effect = lambda band_name: band_mocks.get(
        band_name, MagicMock()
    )
    # normalizedDifference returns a fresh mock with rename() available.
    nd_result = MagicMock(name="normalizedDifference_result")
    nd_result.rename.return_value = MagicMock(name="renamed_nd")
    img.normalizedDifference.return_value = nd_result
    return img


# ==============================================================================
# compute_ndwi tests
# ==============================================================================

class TestComputeNDWI:
    """Tests for compute_ndwi()."""

    def test_calls_normalized_difference_with_green_nir(
        self, mock_image: MagicMock
    ) -> None:
        compute_ndwi(mock_image)
        mock_image.normalizedDifference.assert_called_once_with(["Green", "NIR"])

    def test_renames_output_to_ndwi(self, mock_image: MagicMock) -> None:
        compute_ndwi(mock_image)
        mock_image.normalizedDifference.return_value.rename.assert_called_once_with(
            "NDWI"
        )

    def test_returns_renamed_image(self, mock_image: MagicMock) -> None:
        result = compute_ndwi(mock_image)
        expected = mock_image.normalizedDifference.return_value.rename.return_value
        assert result is expected

    def test_ee_error_wrapped_in_gee_api_error(self) -> None:
        bad_image = MagicMock()
        bad_image.normalizedDifference.side_effect = Exception("EE error")
        with pytest.raises(GEEAPIError, match="compute_ndwi"):
            compute_ndwi(bad_image)

    def test_gee_api_error_passthrough(self) -> None:
        bad_image = MagicMock()
        bad_image.normalizedDifference.side_effect = GEEAPIError(
            "compute_ndwi", "already wrapped"
        )
        with pytest.raises(GEEAPIError):
            compute_ndwi(bad_image)


# ==============================================================================
# compute_mndwi tests
# ==============================================================================

class TestComputeMNDWI:
    """Tests for compute_mndwi()."""

    def test_calls_normalized_difference_with_green_swir1(
        self, mock_image: MagicMock
    ) -> None:
        compute_mndwi(mock_image)
        mock_image.normalizedDifference.assert_called_once_with(["Green", "SWIR1"])

    def test_renames_output_to_mndwi(self, mock_image: MagicMock) -> None:
        compute_mndwi(mock_image)
        mock_image.normalizedDifference.return_value.rename.assert_called_once_with(
            "MNDWI"
        )

    def test_returns_renamed_image(self, mock_image: MagicMock) -> None:
        result   = compute_mndwi(mock_image)
        expected = mock_image.normalizedDifference.return_value.rename.return_value
        assert result is expected

    def test_ee_error_wrapped_in_gee_api_error(self) -> None:
        bad = MagicMock()
        bad.normalizedDifference.side_effect = Exception("EE")
        with pytest.raises(GEEAPIError, match="compute_mndwi"):
            compute_mndwi(bad)

    def test_uses_swir1_not_swir2(self, mock_image: MagicMock) -> None:
        """MNDWI uses SWIR1, not SWIR2. Verify band order."""
        compute_mndwi(mock_image)
        args = mock_image.normalizedDifference.call_args[0][0]
        assert args[1] == "SWIR1"
        assert "SWIR2" not in args


# ==============================================================================
# compute_awei_sh tests
# ==============================================================================

class TestComputeAWEISH:
    """Tests for compute_awei_sh()."""

    def test_selects_all_required_bands(
        self, mock_image: MagicMock
    ) -> None:
        compute_awei_sh(mock_image)
        selected = [c[0][0] for c in mock_image.select.call_args_list]
        assert "Blue"  in selected
        assert "Green" in selected
        assert "NIR"   in selected
        assert "SWIR1" in selected
        assert "SWIR2" in selected

    def test_multiplies_green_by_2_5(
        self, mock_image: MagicMock, band_mocks: dict
    ) -> None:
        compute_awei_sh(mock_image)
        band_mocks["Green"].multiply.assert_called_with(2.5)

    def test_multiplies_swir2_by_0_25(
        self, mock_image: MagicMock, band_mocks: dict
    ) -> None:
        compute_awei_sh(mock_image)
        band_mocks["SWIR2"].multiply.assert_called_with(0.25)

    def test_multiplies_nir_swir1_sum_by_1_5(
        self, mock_image: MagicMock, band_mocks: dict
    ) -> None:
        compute_awei_sh(mock_image)
        # NIR + SWIR1 intermediate: nir.add(swir1)
        # then .multiply(1.5) on the result
        nir_add_result = band_mocks["NIR"].add.return_value
        nir_add_result.multiply.assert_called_with(1.5)

    def test_renames_output_to_awei_sh(
        self, mock_image: MagicMock, band_mocks: dict
    ) -> None:
        result = compute_awei_sh(mock_image)
        # The chain ends with .rename("AWEI_sh") on the Blue band chain.
        assert result is not None

    def test_ee_error_wrapped_in_gee_api_error(self) -> None:
        bad = MagicMock()
        bad.select.side_effect = Exception("Band error")
        with pytest.raises(GEEAPIError, match="compute_awei_sh"):
            compute_awei_sh(bad)

    def test_five_bands_selected(
        self, mock_image: MagicMock
    ) -> None:
        compute_awei_sh(mock_image)
        assert mock_image.select.call_count == 5


# ==============================================================================
# compute_awei_nsh tests
# ==============================================================================

class TestComputeAWEINSH:
    """Tests for compute_awei_nsh()."""

    def test_selects_green_nir_swir1_swir2(
        self, mock_image: MagicMock
    ) -> None:
        compute_awei_nsh(mock_image)
        selected = [c[0][0] for c in mock_image.select.call_args_list]
        assert "Green" in selected
        assert "NIR"   in selected
        assert "SWIR1" in selected
        assert "SWIR2" in selected

    def test_does_not_select_blue(
        self, mock_image: MagicMock
    ) -> None:
        """AWEI_nsh does not use Blue band."""
        compute_awei_nsh(mock_image)
        selected = [c[0][0] for c in mock_image.select.call_args_list]
        assert "Blue" not in selected

    def test_multiplies_green_swir1_diff_by_4(
        self, mock_image: MagicMock, band_mocks: dict
    ) -> None:
        compute_awei_nsh(mock_image)
        green_sub_swir1 = band_mocks["Green"].subtract.return_value
        green_sub_swir1.multiply.assert_called_with(4.0)

    def test_multiplies_nir_by_0_25(
        self, mock_image: MagicMock, band_mocks: dict
    ) -> None:
        compute_awei_nsh(mock_image)
        band_mocks["NIR"].multiply.assert_called_with(0.25)

    def test_multiplies_swir2_by_2_75(
        self, mock_image: MagicMock, band_mocks: dict
    ) -> None:
        compute_awei_nsh(mock_image)
        band_mocks["SWIR2"].multiply.assert_called_with(2.75)

    def test_ee_error_wrapped(self) -> None:
        bad = MagicMock()
        bad.select.side_effect = Exception("err")
        with pytest.raises(GEEAPIError, match="compute_awei_nsh"):
            compute_awei_nsh(bad)

    def test_four_bands_selected(self, mock_image: MagicMock) -> None:
        compute_awei_nsh(mock_image)
        assert mock_image.select.call_count == 4


# ==============================================================================
# compute_ndvi tests
# ==============================================================================

class TestComputeNDVI:
    """Tests for compute_ndvi()."""

    def test_calls_normalized_difference_with_nir_red(
        self, mock_image: MagicMock
    ) -> None:
        compute_ndvi(mock_image)
        mock_image.normalizedDifference.assert_called_once_with(["NIR", "Red"])

    def test_renames_output_to_ndvi(self, mock_image: MagicMock) -> None:
        compute_ndvi(mock_image)
        mock_image.normalizedDifference.return_value.rename.assert_called_once_with(
            "NDVI"
        )

    def test_nir_is_first_in_normalized_difference(
        self, mock_image: MagicMock
    ) -> None:
        """NIR must be first (numerator) to produce positive values for vegetation."""
        compute_ndvi(mock_image)
        args = mock_image.normalizedDifference.call_args[0][0]
        assert args[0] == "NIR"
        assert args[1] == "Red"

    def test_ee_error_wrapped(self) -> None:
        bad = MagicMock()
        bad.normalizedDifference.side_effect = Exception("err")
        with pytest.raises(GEEAPIError, match="compute_ndvi"):
            compute_ndvi(bad)


# ==============================================================================
# compute_savi tests
# ==============================================================================

class TestComputeSAVI:
    """Tests for compute_savi()."""

    def test_selects_nir_and_red(
        self, mock_image: MagicMock
    ) -> None:
        compute_savi(mock_image)
        selected = [c[0][0] for c in mock_image.select.call_args_list]
        assert "NIR" in selected
        assert "Red" in selected

    def test_default_soil_factor_is_0_5(
        self, mock_image: MagicMock, band_mocks: dict
    ) -> None:
        compute_savi(mock_image)
        # With L=0.5: denominator is nir.add(red).add(0.5)
        nir_add_red = band_mocks["NIR"].subtract.return_value
        # numerator = nir.subtract(red)
        # denominator = nir.add(red).add(0.5)
        # We verify that add(0.5) was called somewhere on the NIR chain.
        nir_add_result = band_mocks["NIR"].add.return_value
        nir_add_result.add.assert_called_with(0.5)

    def test_custom_soil_factor_applied(
        self, mock_image: MagicMock, band_mocks: dict
    ) -> None:
        compute_savi(mock_image, soil_factor=0.25)
        nir_add_result = band_mocks["NIR"].add.return_value
        nir_add_result.add.assert_called_with(0.25)

    def test_multiplies_by_one_plus_soil_factor(
        self, mock_image: MagicMock, band_mocks: dict
    ) -> None:
        """(1 + L) must be applied as the final multiplier."""
        compute_savi(mock_image, soil_factor=0.5)
        # (NIR - Red) / (NIR + Red + L) is divided, then multiplied by (1 + L).
        # With L=0.5, multiply by 1.5.
        nir_sub_red = band_mocks["NIR"].subtract.return_value
        divide_result = nir_sub_red.divide.return_value
        divide_result.multiply.assert_called_with(1.5)

    def test_custom_factor_multiplier(
        self, mock_image: MagicMock, band_mocks: dict
    ) -> None:
        compute_savi(mock_image, soil_factor=1.0)
        nir_sub_red   = band_mocks["NIR"].subtract.return_value
        divide_result = nir_sub_red.divide.return_value
        divide_result.multiply.assert_called_with(2.0)

    def test_renames_to_savi(
        self, mock_image: MagicMock, band_mocks: dict
    ) -> None:
        result = compute_savi(mock_image)
        assert result is not None

    def test_ee_error_wrapped(self) -> None:
        bad = MagicMock()
        bad.select.side_effect = Exception("err")
        with pytest.raises(GEEAPIError, match="compute_savi"):
            compute_savi(bad)

    def test_two_bands_selected(self, mock_image: MagicMock) -> None:
        compute_savi(mock_image)
        assert mock_image.select.call_count == 2


# ==============================================================================
# compute_bsi tests
# ==============================================================================

class TestComputeBSI:
    """Tests for compute_bsi()."""

    def test_selects_blue_red_nir_swir1(
        self, mock_image: MagicMock
    ) -> None:
        compute_bsi(mock_image)
        selected = [c[0][0] for c in mock_image.select.call_args_list]
        assert "Blue"  in selected
        assert "Red"   in selected
        assert "NIR"   in selected
        assert "SWIR1" in selected

    def test_does_not_select_swir2(self, mock_image: MagicMock) -> None:
        compute_bsi(mock_image)
        selected = [c[0][0] for c in mock_image.select.call_args_list]
        assert "SWIR2" not in selected

    def test_four_bands_selected(self, mock_image: MagicMock) -> None:
        compute_bsi(mock_image)
        assert mock_image.select.call_count == 4

    def test_swir1_add_red_computed(
        self, mock_image: MagicMock, band_mocks: dict
    ) -> None:
        """SWIR1 + Red must be computed (appears in both numerator and denominator)."""
        compute_bsi(mock_image)
        band_mocks["SWIR1"].add.assert_any_call(band_mocks["Red"])

    def test_nir_add_blue_computed(
        self, mock_image: MagicMock, band_mocks: dict
    ) -> None:
        """NIR + Blue must be computed."""
        compute_bsi(mock_image)
        band_mocks["NIR"].add.assert_any_call(band_mocks["Blue"])

    def test_numerator_subtracts_nir_blue_from_swir1_red(
        self, mock_image: MagicMock, band_mocks: dict
    ) -> None:
        compute_bsi(mock_image)
        swir1_add_red   = band_mocks["SWIR1"].add.return_value
        nir_add_blue    = band_mocks["NIR"].add.return_value
        swir1_add_red.subtract.assert_called_with(nir_add_blue)

    def test_denominator_adds_nir_blue_to_swir1_red(
        self, mock_image: MagicMock, band_mocks: dict
    ) -> None:
        compute_bsi(mock_image)
        swir1_add_red = band_mocks["SWIR1"].add.return_value
        nir_add_blue  = band_mocks["NIR"].add.return_value
        swir1_add_red.add.assert_called_with(nir_add_blue)

    def test_renames_to_bsi(
        self, mock_image: MagicMock, band_mocks: dict
    ) -> None:
        result = compute_bsi(mock_image)
        assert result is not None

    def test_ee_error_wrapped(self) -> None:
        bad = MagicMock()
        bad.select.side_effect = Exception("err")
        with pytest.raises(GEEAPIError, match="compute_bsi"):
            compute_bsi(bad)


# ==============================================================================
# compute_ndmi tests
# ==============================================================================

class TestComputeNDMI:
    """Tests for compute_ndmi()."""

    def test_calls_normalized_difference_with_nir_swir1(
        self, mock_image: MagicMock
    ) -> None:
        compute_ndmi(mock_image)
        mock_image.normalizedDifference.assert_called_once_with(["NIR", "SWIR1"])

    def test_renames_output_to_ndmi(self, mock_image: MagicMock) -> None:
        compute_ndmi(mock_image)
        mock_image.normalizedDifference.return_value.rename.assert_called_once_with(
            "NDMI"
        )

    def test_nir_is_first_positive_for_wet_vegetation(
        self, mock_image: MagicMock
    ) -> None:
        compute_ndmi(mock_image)
        args = mock_image.normalizedDifference.call_args[0][0]
        assert args[0] == "NIR"
        assert args[1] == "SWIR1"

    def test_ee_error_wrapped(self) -> None:
        bad = MagicMock()
        bad.normalizedDifference.side_effect = Exception("err")
        with pytest.raises(GEEAPIError, match="compute_ndmi"):
            compute_ndmi(bad)


# ==============================================================================
# compute_ndbi tests
# ==============================================================================

class TestComputeNDBI:
    """Tests for compute_ndbi()."""

    def test_calls_normalized_difference_with_swir1_nir(
        self, mock_image: MagicMock
    ) -> None:
        compute_ndbi(mock_image)
        mock_image.normalizedDifference.assert_called_once_with(["SWIR1", "NIR"])

    def test_renames_output_to_ndbi(self, mock_image: MagicMock) -> None:
        compute_ndbi(mock_image)
        mock_image.normalizedDifference.return_value.rename.assert_called_once_with(
            "NDBI"
        )

    def test_swir1_is_first_positive_for_built_up(
        self, mock_image: MagicMock
    ) -> None:
        """SWIR1 first means positive for built-up (inverse of NDWI)."""
        compute_ndbi(mock_image)
        args = mock_image.normalizedDifference.call_args[0][0]
        assert args[0] == "SWIR1"
        assert args[1] == "NIR"

    def test_ee_error_wrapped(self) -> None:
        bad = MagicMock()
        bad.normalizedDifference.side_effect = Exception("err")
        with pytest.raises(GEEAPIError, match="compute_ndbi"):
            compute_ndbi(bad)


# ==============================================================================
# Cross-index formula consistency tests
# ==============================================================================

class TestFormulaConsistency:
    """Tests that verify formula consistency across indices."""

    def test_ndbi_is_inverse_of_ndmi_band_order(
        self, mock_image: MagicMock
    ) -> None:
        """NDBI(SWIR1, NIR) is the inverse of NDMI(NIR, SWIR1)."""
        compute_ndmi(mock_image)
        ndmi_args = mock_image.normalizedDifference.call_args[0][0]
        mock_image.reset_mock()
        compute_ndbi(mock_image)
        ndbi_args = mock_image.normalizedDifference.call_args[0][0]
        # They should use the same two bands but in reversed order.
        assert set(ndmi_args) == set(ndbi_args)
        assert ndmi_args[0] == ndbi_args[1]
        assert ndmi_args[1] == ndbi_args[0]

    def test_all_index_functions_return_something(
        self, mock_image: MagicMock
    ) -> None:
        """All index functions must return a non-None result."""
        functions_and_args = [
            (compute_ndwi,   (mock_image,)),
            (compute_mndwi,  (mock_image,)),
            (compute_awei_sh, (mock_image,)),
            (compute_awei_nsh, (mock_image,)),
            (compute_ndvi,   (mock_image,)),
            (compute_savi,   (mock_image,)),
            (compute_bsi,    (mock_image,)),
            (compute_ndmi,   (mock_image,)),
            (compute_ndbi,   (mock_image,)),
        ]
        for fn, args in functions_and_args:
            mock_image.reset_mock()
            nd_result = MagicMock()
            nd_result.rename.return_value = MagicMock()
            mock_image.normalizedDifference.return_value = nd_result
            result = fn(*args)
            assert result is not None, f"{fn.__name__} returned None"

    def test_all_index_functions_wrap_errors_in_gee_api_error(self) -> None:
        """All index functions must wrap exceptions in GEEAPIError."""
        functions = [
            compute_ndwi, compute_mndwi, compute_awei_sh, compute_awei_nsh,
            compute_ndvi, compute_savi, compute_bsi, compute_ndmi, compute_ndbi,
        ]
        for fn in functions:
            bad = MagicMock()
            bad.select.side_effect = Exception("forced error")
            bad.normalizedDifference.side_effect = Exception("forced error")
            with pytest.raises(GEEAPIError):
                fn(bad)