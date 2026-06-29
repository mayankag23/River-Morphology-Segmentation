"""
Unit tests for src/gee/registry.py.

Tests cover:
    - IndexMetadata frozen dataclass
    - BUILT_IN_INDICES registry completeness
    - FeatureRegistry.get() for all registered indices
    - FeatureRegistry.get_function() returns callables
    - FeatureRegistry.names() content
    - FeatureRegistry.get_by_category() filtering
    - FeatureRegistry.is_registered() membership
    - FeatureRegistry.get_default_enabled() filtering
    - Invalid name handling

Run:
    pytest tests/test_registry.py -v
    pytest tests/test_registry.py -v --cov=src/gee/registry --cov-report=term-missing
"""

from __future__ import annotations

import pytest

from src.core.exceptions import InvalidValueError
from src.gee.registry import (
    BUILT_IN_INDICES,
    INDEX_CATEGORIES,
    FeatureRegistry,
    IndexMetadata,
)


# ==============================================================================
# INDEX_CATEGORIES tests
# ==============================================================================

class TestIndexCategories:
    """Tests for the INDEX_CATEGORIES constant."""

    def test_is_frozenset(self) -> None:
        assert isinstance(INDEX_CATEGORIES, frozenset)

    def test_contains_water(self) -> None:
        assert "water" in INDEX_CATEGORIES

    def test_contains_vegetation(self) -> None:
        assert "vegetation" in INDEX_CATEGORIES

    def test_contains_bare_soil(self) -> None:
        assert "bare_soil" in INDEX_CATEGORIES

    def test_contains_moisture(self) -> None:
        assert "moisture" in INDEX_CATEGORIES

    def test_contains_built_up(self) -> None:
        assert "built_up" in INDEX_CATEGORIES

    def test_all_values_are_ascii_only(self) -> None:
        for cat in INDEX_CATEGORIES:
            assert all(ord(c) < 128 for c in cat)


# ==============================================================================
# IndexMetadata tests
# ==============================================================================

class TestIndexMetadata:
    """Tests for the IndexMetadata frozen dataclass."""

    def _sample(self) -> IndexMetadata:
        return IndexMetadata(
            name="TEST",
            output_band_name="TEST",
            formula_str="(A - B) / (A + B)",
            description="Test index.",
            river_relevance="Useful for testing.",
            bands_required=("NIR", "Red"),
            category="water",
            reference="Author (2024).",
            is_optional=False,
            config_key="test",
        )

    def test_frozen_prevents_mutation(self) -> None:
        meta = self._sample()
        with pytest.raises((AttributeError, TypeError)):
            meta.name = "OTHER"  # type: ignore[misc]

    def test_attributes_set_correctly(self) -> None:
        meta = self._sample()
        assert meta.name           == "TEST"
        assert meta.category       == "water"
        assert meta.is_optional    is False
        assert meta.config_key     == "test"
        assert meta.bands_required == ("NIR", "Red")

    def test_formula_str_is_ascii(self) -> None:
        for meta in BUILT_IN_INDICES:
            assert all(ord(c) < 128 for c in meta.formula_str), (
                f"Non-ASCII in {meta.name}.formula_str"
            )

    def test_description_is_ascii(self) -> None:
        for meta in BUILT_IN_INDICES:
            assert all(ord(c) < 128 for c in meta.description)

    def test_river_relevance_is_ascii(self) -> None:
        for meta in BUILT_IN_INDICES:
            assert all(ord(c) < 128 for c in meta.river_relevance)


# ==============================================================================
# BUILT_IN_INDICES tests
# ==============================================================================

class TestBuiltInIndices:
    """Tests for the BUILT_IN_INDICES constant."""

    def test_has_nine_entries(self) -> None:
        assert len(BUILT_IN_INDICES) == 9

    def test_contains_ndwi(self) -> None:
        names = {m.name for m in BUILT_IN_INDICES}
        assert "NDWI" in names

    def test_contains_mndwi(self) -> None:
        names = {m.name for m in BUILT_IN_INDICES}
        assert "MNDWI" in names

    def test_contains_awei_sh(self) -> None:
        names = {m.name for m in BUILT_IN_INDICES}
        assert "AWEI_sh" in names

    def test_contains_awei_nsh(self) -> None:
        names = {m.name for m in BUILT_IN_INDICES}
        assert "AWEI_nsh" in names

    def test_contains_ndvi(self) -> None:
        names = {m.name for m in BUILT_IN_INDICES}
        assert "NDVI" in names

    def test_contains_savi(self) -> None:
        names = {m.name for m in BUILT_IN_INDICES}
        assert "SAVI" in names

    def test_contains_bsi(self) -> None:
        names = {m.name for m in BUILT_IN_INDICES}
        assert "BSI" in names

    def test_contains_ndmi(self) -> None:
        names = {m.name for m in BUILT_IN_INDICES}
        assert "NDMI" in names

    def test_contains_ndbi(self) -> None:
        names = {m.name for m in BUILT_IN_INDICES}
        assert "NDBI" in names

    def test_ndbi_is_optional(self) -> None:
        ndbi = next(m for m in BUILT_IN_INDICES if m.name == "NDBI")
        assert ndbi.is_optional is True

    def test_mndwi_is_not_optional(self) -> None:
        mndwi = next(m for m in BUILT_IN_INDICES if m.name == "MNDWI")
        assert mndwi.is_optional is False

    def test_all_names_unique(self) -> None:
        names = [m.name for m in BUILT_IN_INDICES]
        assert len(names) == len(set(names))

    def test_all_config_keys_unique(self) -> None:
        keys = [m.config_key for m in BUILT_IN_INDICES]
        assert len(keys) == len(set(keys))

    def test_all_categories_in_valid_set(self) -> None:
        for meta in BUILT_IN_INDICES:
            assert meta.category in INDEX_CATEGORIES, (
                f"{meta.name}.category '{meta.category}' not in INDEX_CATEGORIES"
            )

    def test_all_bands_required_are_from_common_names(self) -> None:
        from src.gee.harmonization import COMMON_BAND_NAMES
        common_set = set(COMMON_BAND_NAMES)
        for meta in BUILT_IN_INDICES:
            for band in meta.bands_required:
                assert band in common_set, (
                    f"{meta.name} requires band '{band}' not in COMMON_BAND_NAMES"
                )

    def test_mndwi_uses_green_and_swir1(self) -> None:
        mndwi = next(m for m in BUILT_IN_INDICES if m.name == "MNDWI")
        assert "Green" in mndwi.bands_required
        assert "SWIR1" in mndwi.bands_required

    def test_bsi_uses_four_bands(self) -> None:
        bsi = next(m for m in BUILT_IN_INDICES if m.name == "BSI")
        assert len(bsi.bands_required) == 4

    def test_awei_sh_uses_five_bands(self) -> None:
        awei_sh = next(m for m in BUILT_IN_INDICES if m.name == "AWEI_sh")
        assert len(awei_sh.bands_required) == 5


# ==============================================================================
# FeatureRegistry.get() tests
# ==============================================================================

class TestFeatureRegistryGet:
    """Tests for FeatureRegistry.get()."""

    @pytest.mark.parametrize("name", [
        "NDWI", "MNDWI", "AWEI_sh", "AWEI_nsh",
        "NDVI", "SAVI", "BSI", "NDMI", "NDBI",
    ])
    def test_get_all_registered_indices(self, name: str) -> None:
        meta = FeatureRegistry.get(name)
        assert meta.name == name

    def test_get_invalid_name_raises_invalid_value_error(self) -> None:
        with pytest.raises(InvalidValueError, match="not a registered"):
            FeatureRegistry.get("INVALID_INDEX")

    def test_get_returns_index_metadata(self) -> None:
        meta = FeatureRegistry.get("NDWI")
        assert isinstance(meta, IndexMetadata)

    def test_get_mndwi_formula(self) -> None:
        meta = FeatureRegistry.get("MNDWI")
        assert "Green" in meta.formula_str
        assert "SWIR1" in meta.formula_str

    def test_get_bsi_category(self) -> None:
        meta = FeatureRegistry.get("BSI")
        assert meta.category == "bare_soil"

    def test_get_ndbi_is_optional(self) -> None:
        meta = FeatureRegistry.get("NDBI")
        assert meta.is_optional is True

    def test_get_error_message_includes_available_indices(self) -> None:
        with pytest.raises(InvalidValueError) as exc_info:
            FeatureRegistry.get("UNKNOWN")
        assert "NDWI" in str(exc_info.value)


# ==============================================================================
# FeatureRegistry.get_function() tests
# ==============================================================================

class TestFeatureRegistryGetFunction:
    """Tests for FeatureRegistry.get_function()."""

    @pytest.mark.parametrize("name", [
        "NDWI", "MNDWI", "AWEI_sh", "AWEI_nsh",
        "NDVI", "SAVI", "BSI", "NDMI", "NDBI",
    ])
    def test_returns_callable_for_all_indices(self, name: str) -> None:
        fn = FeatureRegistry.get_function(name)
        assert callable(fn)

    def test_invalid_name_raises_invalid_value_error(self) -> None:
        with pytest.raises(InvalidValueError):
            FeatureRegistry.get_function("NOT_REGISTERED")

    def test_function_names_match_module_functions(self) -> None:
        from src.gee import indices
        fn = FeatureRegistry.get_function("NDWI")
        assert fn is indices.compute_ndwi

    def test_bsi_function_is_compute_bsi(self) -> None:
        from src.gee.indices import compute_bsi
        assert FeatureRegistry.get_function("BSI") is compute_bsi

    def test_savi_function_is_compute_savi(self) -> None:
        from src.gee.indices import compute_savi
        assert FeatureRegistry.get_function("SAVI") is compute_savi


# ==============================================================================
# FeatureRegistry.names() tests
# ==============================================================================

class TestFeatureRegistryNames:
    """Tests for FeatureRegistry.names()."""

    def test_returns_tuple(self) -> None:
        assert isinstance(FeatureRegistry.names(), tuple)

    def test_contains_all_nine_indices(self) -> None:
        names = FeatureRegistry.names()
        assert len(names) == 9

    def test_contains_all_expected_names(self) -> None:
        names = set(FeatureRegistry.names())
        expected = {"NDWI", "MNDWI", "AWEI_sh", "AWEI_nsh", "NDVI",
                    "SAVI", "BSI", "NDMI", "NDBI"}
        assert names == expected

    def test_all_names_are_ascii(self) -> None:
        for name in FeatureRegistry.names():
            assert all(ord(c) < 128 for c in name)


# ==============================================================================
# FeatureRegistry.get_by_category() tests
# ==============================================================================

class TestFeatureRegistryGetByCategory:
    """Tests for FeatureRegistry.get_by_category()."""

    def test_returns_tuple(self) -> None:
        result = FeatureRegistry.get_by_category("water")
        assert isinstance(result, tuple)

    def test_water_category_has_four_indices(self) -> None:
        result = FeatureRegistry.get_by_category("water")
        assert len(result) == 4

    def test_water_category_contains_mndwi(self) -> None:
        result = FeatureRegistry.get_by_category("water")
        names  = {m.name for m in result}
        assert "MNDWI" in names

    def test_vegetation_category_has_two_indices(self) -> None:
        result = FeatureRegistry.get_by_category("vegetation")
        assert len(result) == 2

    def test_bare_soil_category_has_one_index(self) -> None:
        result = FeatureRegistry.get_by_category("bare_soil")
        assert len(result) == 1
        assert result[0].name == "BSI"

    def test_moisture_category_has_one_index(self) -> None:
        result = FeatureRegistry.get_by_category("moisture")
        assert len(result) == 1
        assert result[0].name == "NDMI"

    def test_built_up_category_has_one_index(self) -> None:
        result = FeatureRegistry.get_by_category("built_up")
        assert len(result) == 1
        assert result[0].name == "NDBI"

    def test_invalid_category_raises_invalid_value_error(self) -> None:
        with pytest.raises(InvalidValueError, match="valid category"):
            FeatureRegistry.get_by_category("invalid_category")


# ==============================================================================
# FeatureRegistry.is_registered() tests
# ==============================================================================

class TestFeatureRegistryIsRegistered:
    """Tests for FeatureRegistry.is_registered()."""

    def test_returns_true_for_ndwi(self) -> None:
        assert FeatureRegistry.is_registered("NDWI") is True

    def test_returns_true_for_bsi(self) -> None:
        assert FeatureRegistry.is_registered("BSI") is True

    def test_returns_false_for_unknown(self) -> None:
        assert FeatureRegistry.is_registered("UNKNOWN") is False

    def test_returns_false_for_empty_string(self) -> None:
        assert FeatureRegistry.is_registered("") is False

    def test_case_sensitive(self) -> None:
        assert FeatureRegistry.is_registered("ndwi")  is False
        assert FeatureRegistry.is_registered("NDWI")  is True


# ==============================================================================
# FeatureRegistry.get_default_enabled() tests
# ==============================================================================

class TestFeatureRegistryGetDefaultEnabled:
    """Tests for FeatureRegistry.get_default_enabled()."""

    def test_returns_tuple(self) -> None:
        assert isinstance(FeatureRegistry.get_default_enabled(), tuple)

    def test_does_not_contain_ndbi(self) -> None:
        """NDBI is optional and should not be in default enabled."""
        assert "NDBI" not in FeatureRegistry.get_default_enabled()

    def test_contains_mndwi(self) -> None:
        assert "MNDWI" in FeatureRegistry.get_default_enabled()

    def test_contains_bsi(self) -> None:
        assert "BSI" in FeatureRegistry.get_default_enabled()

    def test_count_is_eight(self) -> None:
        """8 non-optional indices, 1 optional (NDBI)."""
        assert len(FeatureRegistry.get_default_enabled()) == 8


# ==============================================================================
# FeatureRegistry.get_config_key_map() tests
# ==============================================================================

class TestGetConfigKeyMap:
    """Tests for FeatureRegistry.get_config_key_map()."""

    def test_returns_dict(self) -> None:
        assert isinstance(FeatureRegistry.get_config_key_map(), dict)

    def test_has_nine_entries(self) -> None:
        assert len(FeatureRegistry.get_config_key_map()) == 9

    def test_ndwi_key_maps_to_ndwi_name(self) -> None:
        assert FeatureRegistry.get_config_key_map()["ndwi"] == "NDWI"

    def test_bsi_key_maps_to_bsi_name(self) -> None:
        assert FeatureRegistry.get_config_key_map()["bsi"] == "BSI"

    def test_all_values_are_registered_names(self) -> None:
        for name in FeatureRegistry.get_config_key_map().values():
            assert FeatureRegistry.is_registered(name)