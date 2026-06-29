"""
Unit tests for src/gee/feature_stack.py.

Tests cover:
    - FeatureStackResult frozen dataclass and properties
    - FeatureStackAssembler construction and add_index()
    - FeatureStackAssembler.build() band name accumulation
    - list_available_features() completeness
    - validate_harmonization() pass and fail cases
    - describe_feature_stack() ASCII output

Run:
    pytest tests/test_feature_stack.py -v
    pytest tests/test_feature_stack.py -v \
        --cov=src/gee/feature_stack --cov-report=term-missing
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.core.exceptions import InvalidValueError
from src.gee import GEEAPIError
from src.gee.feature_stack import (
    FeatureStackAssembler,
    FeatureStackResult,
    describe_feature_stack,
    list_available_features,
    validate_harmonization,
)
from src.gee.harmonization import COMMON_BAND_NAMES


# ==============================================================================
# Helpers
# ==============================================================================

def _make_composite_result(band_names=None) -> MagicMock:
    """Return a mock CompositeResult with configurable band_names."""
    result = MagicMock()
    result.band_names = band_names if band_names is not None else COMMON_BAND_NAMES
    result.image      = MagicMock(name="composite_ee_image")
    result.image.addBands.return_value = result.image
    return result


def _make_feature_config() -> MagicMock:
    """Return a mock FeatureConfig."""
    cfg = MagicMock()
    cfg.savi_soil_factor = 0.5
    return cfg


def _make_feature_stack_result(
    computed: tuple = ("NDWI", "MNDWI", "BSI"),
    skipped:  tuple = ("NDBI",),
) -> FeatureStackResult:
    composite_bands = COMMON_BAND_NAMES
    index_bands     = computed
    return FeatureStackResult(
        image=MagicMock(),
        features_computed=computed,
        features_skipped=skipped,
        composite_band_names=composite_bands,
        index_band_names=index_bands,
        all_band_names=composite_bands + index_bands,
        source_composite=_make_composite_result(),
        feature_config=_make_feature_config(),
    )


# ==============================================================================
# FeatureStackResult tests
# ==============================================================================

class TestFeatureStackResult:
    """Tests for the FeatureStackResult frozen dataclass."""

    def test_frozen_prevents_mutation(self) -> None:
        result = _make_feature_stack_result()
        with pytest.raises((AttributeError, TypeError)):
            result.features_computed = ("NDWI",)  # type: ignore[misc]

    def test_features_computed_stored_correctly(self) -> None:
        result = _make_feature_stack_result(computed=("NDWI", "BSI"))
        assert result.features_computed == ("NDWI", "BSI")

    def test_features_skipped_stored_correctly(self) -> None:
        result = _make_feature_stack_result(skipped=("NDBI", "SAVI"))
        assert result.features_skipped == ("NDBI", "SAVI")

    def test_all_band_names_is_composite_plus_index(self) -> None:
        composite = COMMON_BAND_NAMES
        index     = ("NDWI", "BSI")
        result    = FeatureStackResult(
            image=MagicMock(),
            features_computed=index,
            features_skipped=(),
            composite_band_names=composite,
            index_band_names=index,
            all_band_names=composite + index,
            source_composite=_make_composite_result(),
            feature_config=_make_feature_config(),
        )
        assert result.all_band_names == composite + index

    def test_summary_lines_returns_list_of_strings(self) -> None:
        result = _make_feature_stack_result()
        lines  = result.summary_lines()
        assert isinstance(lines, list)
        assert all(isinstance(line, str) for line in lines)

    def test_summary_lines_are_ascii_only(self) -> None:
        result = _make_feature_stack_result()
        for line in result.summary_lines():
            assert all(ord(c) < 128 for c in line), (
                f"Non-ASCII in: {line!r}"
            )

    def test_summary_lines_show_computed_count(self) -> None:
        result   = _make_feature_stack_result(computed=("NDWI", "MNDWI"))
        combined = " ".join(result.summary_lines())
        assert "NDWI" in combined

    def test_summary_shows_total_band_count(self) -> None:
        result   = _make_feature_stack_result()
        combined = " ".join(result.summary_lines())
        expected_total = str(len(result.all_band_names))
        assert expected_total in combined


# ==============================================================================
# FeatureStackAssembler tests
# ==============================================================================

class TestFeatureStackAssembler:
    """Tests for the FeatureStackAssembler class."""

    def test_construction_stores_base_image(self) -> None:
        base    = MagicMock()
        assembler = FeatureStackAssembler(base, COMMON_BAND_NAMES)
        assert assembler._image is base

    def test_construction_stores_base_bands(self) -> None:
        assembler = FeatureStackAssembler(MagicMock(), COMMON_BAND_NAMES)
        assert assembler._base_bands == COMMON_BAND_NAMES

    def test_initial_index_count_is_zero(self) -> None:
        assembler = FeatureStackAssembler(MagicMock(), COMMON_BAND_NAMES)
        assert assembler.index_count == 0

    def test_add_index_calls_add_bands(self) -> None:
        base         = MagicMock()
        assembler    = FeatureStackAssembler(base, COMMON_BAND_NAMES)
        index_image  = MagicMock()
        assembler.add_index(index_image, "NDWI")
        base.addBands.assert_called_once_with(index_image)

    def test_add_index_tracks_band_name(self) -> None:
        base      = MagicMock()
        assembler = FeatureStackAssembler(base, COMMON_BAND_NAMES)
        assembler.add_index(MagicMock(), "NDWI")
        assert "NDWI" in assembler._index_bands

    def test_add_multiple_indices_tracks_all_names(self) -> None:
        base      = MagicMock()
        assembler = FeatureStackAssembler(base, COMMON_BAND_NAMES)
        assembler.add_index(MagicMock(), "NDWI")
        assembler.add_index(MagicMock(), "MNDWI")
        assembler.add_index(MagicMock(), "BSI")
        assert assembler.index_count == 3
        assert "NDWI"  in assembler._index_bands
        assert "MNDWI" in assembler._index_bands
        assert "BSI"   in assembler._index_bands

    def test_add_index_updates_image_reference(self) -> None:
        base             = MagicMock()
        new_image        = MagicMock()
        base.addBands.return_value = new_image
        assembler        = FeatureStackAssembler(base, COMMON_BAND_NAMES)
        assembler.add_index(MagicMock(), "NDWI")
        assert assembler._image is new_image

    def test_add_index_failure_raises_gee_api_error(self) -> None:
        base             = MagicMock()
        base.addBands.side_effect = Exception("EE addBands failed")
        assembler        = FeatureStackAssembler(base, COMMON_BAND_NAMES)
        with pytest.raises(GEEAPIError, match="add_index_band_NDWI"):
            assembler.add_index(MagicMock(), "NDWI")

    def test_build_returns_tuple_of_three(self) -> None:
        assembler = FeatureStackAssembler(MagicMock(), COMMON_BAND_NAMES)
        result    = assembler.build()
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_build_returns_correct_all_band_names(self) -> None:
        base      = MagicMock()
        base.addBands.return_value = base
        assembler = FeatureStackAssembler(base, COMMON_BAND_NAMES)
        assembler.add_index(MagicMock(), "NDWI")
        assembler.add_index(MagicMock(), "BSI")
        _, all_bands, _ = assembler.build()
        assert "NDWI" in all_bands
        assert "BSI"  in all_bands
        # All composite bands must also be present.
        for band in COMMON_BAND_NAMES:
            assert band in all_bands

    def test_build_returns_correct_index_band_names(self) -> None:
        base      = MagicMock()
        base.addBands.return_value = base
        assembler = FeatureStackAssembler(base, COMMON_BAND_NAMES)
        assembler.add_index(MagicMock(), "MNDWI")
        _, _, index_bands = assembler.build()
        assert index_bands == ("MNDWI",)

    def test_build_returns_correct_image(self) -> None:
        base      = MagicMock()
        assembler = FeatureStackAssembler(base, COMMON_BAND_NAMES)
        image, _, _ = assembler.build()
        assert image is base

    def test_build_with_no_indices_returns_base(self) -> None:
        base      = MagicMock()
        assembler = FeatureStackAssembler(base, COMMON_BAND_NAMES)
        image, all_bands, index_bands = assembler.build()
        assert image is base
        assert index_bands == ()
        assert all_bands   == COMMON_BAND_NAMES

    def test_index_band_order_preserved(self) -> None:
        base      = MagicMock()
        base.addBands.return_value = base
        assembler = FeatureStackAssembler(base, COMMON_BAND_NAMES)
        for name in ("NDWI", "MNDWI", "NDVI", "BSI"):
            assembler.add_index(MagicMock(), name)
        _, _, index_bands = assembler.build()
        assert index_bands == ("NDWI", "MNDWI", "NDVI", "BSI")


# ==============================================================================
# list_available_features tests
# ==============================================================================

class TestListAvailableFeatures:
    """Tests for list_available_features()."""

    def test_returns_list(self) -> None:
        assert isinstance(list_available_features(), list)

    def test_contains_nine_features(self) -> None:
        assert len(list_available_features()) == 9

    def test_is_sorted(self) -> None:
        features = list_available_features()
        assert features == sorted(features)

    def test_contains_ndwi(self) -> None:
        assert "NDWI" in list_available_features()

    def test_contains_bsi(self) -> None:
        assert "BSI" in list_available_features()

    def test_contains_ndbi(self) -> None:
        assert "NDBI" in list_available_features()

    def test_all_names_are_ascii(self) -> None:
        for name in list_available_features():
            assert all(ord(c) < 128 for c in name)


# ==============================================================================
# validate_harmonization tests
# ==============================================================================

class TestValidateHarmonization:
    """Tests for validate_harmonization()."""

    def test_passes_for_common_band_names(self) -> None:
        composite = _make_composite_result(band_names=COMMON_BAND_NAMES)
        # Should not raise
        validate_harmonization(composite)

    def test_passes_for_empty_band_names_with_warning(self) -> None:
        """Empty band_names: warn but do not raise."""
        composite = _make_composite_result(band_names=())
        validate_harmonization(composite)  # Should not raise

    def test_raises_for_incompatible_band_names(self) -> None:
        bad_bands = ("SR_B2", "SR_B3", "SR_B4", "SR_B5")
        composite = _make_composite_result(band_names=bad_bands)
        with pytest.raises(InvalidValueError, match="harmonized band names"):
            validate_harmonization(composite)

    def test_raises_includes_expected_names(self) -> None:
        composite = _make_composite_result(band_names=("SR_B2",))
        with pytest.raises(InvalidValueError) as exc_info:
            validate_harmonization(composite)
        assert "Blue" in str(exc_info.value) or "COMMON" in str(exc_info.value)


# ==============================================================================
# describe_feature_stack tests
# ==============================================================================

class TestDescribeFeatureStack:
    """Tests for describe_feature_stack()."""

    def test_returns_string(self) -> None:
        result = _make_feature_stack_result()
        assert isinstance(describe_feature_stack(result), str)

    def test_output_is_ascii_only(self) -> None:
        result      = _make_feature_stack_result()
        description = describe_feature_stack(result)
        assert all(ord(c) < 128 for c in description)

    def test_contains_feature_stack_result_header(self) -> None:
        result      = _make_feature_stack_result()
        description = describe_feature_stack(result)
        assert "FeatureStackResult" in description

    def test_contains_computed_feature_names(self) -> None:
        result      = _make_feature_stack_result(computed=("NDWI", "BSI"))
        description = describe_feature_stack(result)
        assert "NDWI" in description
        assert "BSI"  in description