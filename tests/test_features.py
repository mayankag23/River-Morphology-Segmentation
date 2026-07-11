"""
Unit tests for src/gee/features.py.

Tests cover:
    - FeatureConfig defaults, from_config(), enabled_features(), disabled_features()
    - SpectralFeatureGenerator construction
    - generate() with all features enabled
    - generate() with specific features disabled
    - generate() result band names
    - generate() preserves source_composite
    - generate_single_index()
    - Harmonization warning path
    - _build_function_map() SAVI partial handling

Run:
    pytest tests/test_features.py -v
    pytest tests/test_features.py -v \
        --cov=src/gee/features --cov-report=term-missing
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.gee.harmonization import OPTICAL_BAND_NAMES

import pytest

from src.core.exceptions import InvalidValueError
from src.gee import GEEAPIError
from src.gee.composite import CompositeMethod, CompositeResult
from src.gee.features import FeatureConfig, SpectralFeatureGenerator
from src.gee.feature_stack import FeatureStackResult
from src.gee.harmonization import COMMON_BAND_NAMES
from src.gee.preprocessing import ProcessedCollectionResult
from tests.conftest import make_valid_config, write_config
from src.gee import feature_stack


# ==============================================================================
# Fixtures
# ==============================================================================

def _make_mock_client() -> MagicMock:
    from src.gee.client import EarthEngineClient
    client = MagicMock(spec=EarthEngineClient)
    client.is_initialized = True
    return client


def _make_composite_result(
    harmonized: bool = True,
    image: MagicMock | None = None,
) -> CompositeResult:
    """Build a minimal CompositeResult for testing."""
    if image is None:
        img = MagicMock(name="composite_image")
        img.addBands.return_value = img
    else:
        img = image

    mock_processed = MagicMock(spec=ProcessedCollectionResult)
    mock_processed.harmonization_applied = harmonized
    mock_processed.source_result         = MagicMock()
    mock_processed.source_result.has_mixed_sensor_families = False

    return CompositeResult(
        image=img,
        method=CompositeMethod.MEDIAN,
        percentile_value=None,
        source_result=mock_processed,
        band_names=COMMON_BAND_NAMES if harmonized else (),
    )


def _make_config(tmp_path: Path, features: dict | None = None):
    from src.core.config import Config
    data = make_valid_config()
    if features is not None:
        data["features"] = features
    return Config(config_path=write_config(tmp_path, data))


# ==============================================================================
# FeatureConfig tests
# ==============================================================================

class TestFeatureConfig:
    """Tests for the FeatureConfig frozen dataclass."""

    def test_default_ndwi_enabled(self) -> None:
        assert FeatureConfig().ndwi is True

    def test_default_mndwi_enabled(self) -> None:
        assert FeatureConfig().mndwi is True

    def test_default_bsi_enabled(self) -> None:
        assert FeatureConfig().bsi is True

    def test_default_ndbi_disabled(self) -> None:
        """NDBI is optional and off by default."""
        assert FeatureConfig().ndbi is False

    def test_default_savi_soil_factor(self) -> None:
        assert FeatureConfig().savi_soil_factor == pytest.approx(0.5)

    def test_frozen_prevents_mutation(self) -> None:
        cfg = FeatureConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.ndwi = False  # type: ignore[misc]

    def test_custom_values_applied(self) -> None:
        cfg = FeatureConfig(ndwi=False, ndbi=True, savi_soil_factor=0.25)
        assert cfg.ndwi             is False
        assert cfg.ndbi             is True
        assert cfg.savi_soil_factor == pytest.approx(0.25)

    def test_is_enabled_true_for_ndwi(self) -> None:
        cfg = FeatureConfig(ndwi=True)
        assert cfg.is_enabled("ndwi") is True

    def test_is_enabled_false_for_ndbi(self) -> None:
        cfg = FeatureConfig(ndbi=False)
        assert cfg.is_enabled("ndbi") is False

    def test_enabled_features_excludes_disabled(self) -> None:
        cfg = FeatureConfig(ndwi=False, mndwi=True, ndbi=False)
        enabled = cfg.enabled_features()
        assert "NDWI" not in enabled
        assert "MNDWI" in enabled
        assert "NDBI"  not in enabled

    def test_enabled_features_in_registration_order(self) -> None:
        cfg = FeatureConfig()
        enabled = cfg.enabled_features()
        # NDWI is registered before MNDWI.
        assert enabled.index("NDWI") < enabled.index("MNDWI")

    def test_disabled_features_contains_ndbi_by_default(self) -> None:
        cfg = FeatureConfig()
        assert "NDBI" in cfg.disabled_features()

    def test_all_disabled_when_all_flags_false(self) -> None:
        cfg = FeatureConfig(
            ndwi=False, mndwi=False, awei_sh=False, awei_nsh=False,
            ndvi=False, savi=False, bsi=False, ndmi=False, ndbi=False,
        )
        assert cfg.enabled_features() == []
        assert len(cfg.disabled_features()) == 9

    def test_all_enabled_when_all_flags_true(self) -> None:
        cfg = FeatureConfig(
            ndwi=True, mndwi=True, awei_sh=True, awei_nsh=True,
            ndvi=True, savi=True, bsi=True, ndmi=True, ndbi=True,
        )
        assert len(cfg.enabled_features()) == 9
        assert cfg.disabled_features() == []


# ==============================================================================
# FeatureConfig.from_config() tests
# ==============================================================================

class TestFeatureConfigFromConfig:
    """Tests for FeatureConfig.from_config()."""

    def test_returns_defaults_when_no_features_section(
        self, tmp_path: Path
    ) -> None:
        cfg        = _make_config(tmp_path)  # no features section
        fc         = FeatureConfig.from_config(cfg)
        assert fc.ndwi  is True
        assert fc.ndbi  is False
        assert fc.mndwi is True

    def test_reads_ndwi_flag(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path, features={"ndwi": False, "mndwi": True})
        fc  = FeatureConfig.from_config(cfg)
        assert fc.ndwi  is False
        assert fc.mndwi is True

    def test_reads_ndbi_flag(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path, features={"ndbi": True})
        fc  = FeatureConfig.from_config(cfg)
        assert fc.ndbi is True

    def test_reads_savi_soil_factor(self, tmp_path: Path) -> None:
        cfg = _make_config(
            tmp_path, features={"savi_soil_factor": 0.25}
        )
        fc = FeatureConfig.from_config(cfg)
        assert fc.savi_soil_factor == pytest.approx(0.25)

    def test_partial_features_section_uses_defaults(
        self, tmp_path: Path
    ) -> None:
        """Keys absent from features section fall back to FeatureConfig defaults."""
        cfg = _make_config(tmp_path, features={"ndwi": False})
        fc  = FeatureConfig.from_config(cfg)
        assert fc.ndwi  is False
        assert fc.mndwi is True  # default


# ==============================================================================
# SpectralFeatureGenerator construction tests
# ==============================================================================

class TestSpectralFeatureGeneratorConstruction:
    """Tests for SpectralFeatureGenerator.__init__()."""

    def test_construction_stores_config(self, tmp_path: Path) -> None:
        cfg    = _make_config(tmp_path)
        client = _make_mock_client()
        gen    = SpectralFeatureGenerator(client, cfg)
        assert gen._config is cfg

    def test_construction_builds_default_feature_config(
        self, tmp_path: Path
    ) -> None:
        cfg    = _make_config(tmp_path)
        client = _make_mock_client()
        gen    = SpectralFeatureGenerator(client, cfg)
        assert isinstance(gen._default_config, FeatureConfig)

    def test_construction_does_not_call_ee(
        self, tmp_path: Path
    ) -> None:
        cfg    = _make_config(tmp_path)
        client = _make_mock_client()
        SpectralFeatureGenerator(client, cfg)
        # No EE calls should occur during construction.


# ==============================================================================
# SpectralFeatureGenerator.generate() tests
# ==============================================================================

class TestGenerateAllFeatures:
    """Tests for generate() with various feature configurations."""

    def test_returns_feature_stack_result(self, tmp_path: Path) -> None:
        cfg    = _make_config(tmp_path)
        client = _make_mock_client()
        gen    = SpectralFeatureGenerator(client, cfg)
        comp   = _make_composite_result()

        result = gen.generate(comp)
        assert isinstance(result, FeatureStackResult)

    def test_preserves_source_composite(self, tmp_path: Path) -> None:
        cfg    = _make_config(tmp_path)
        client = _make_mock_client()
        gen    = SpectralFeatureGenerator(client, cfg)
        comp   = _make_composite_result()

        result = gen.generate(comp)
        assert result.source_composite is comp

    def test_all_default_enabled_features_are_computed(
        self, tmp_path: Path
    ) -> None:
        """8 non-optional indices must be computed with default config."""
        cfg    = _make_config(tmp_path)
        client = _make_mock_client()
        gen    = SpectralFeatureGenerator(client, cfg)
        comp   = _make_composite_result()

        result = gen.generate(comp)
        assert len(result.features_computed) == 8
        assert "NDWI"  in result.features_computed
        assert "MNDWI" in result.features_computed
        assert "BSI"   in result.features_computed
        assert "NDBI"  not in result.features_computed

    def test_ndbi_computed_when_enabled(self, tmp_path: Path) -> None:
        cfg    = _make_config(tmp_path, features={"ndbi": True})
        client = _make_mock_client()
        gen    = SpectralFeatureGenerator(client, cfg)
        comp   = _make_composite_result()

        custom = FeatureConfig(ndbi=True)
        result = gen.generate(comp, feature_config=custom)
        assert "NDBI" in result.features_computed

    def test_disabled_feature_appears_in_skipped(
        self, tmp_path: Path
    ) -> None:
        cfg    = _make_config(tmp_path)
        client = _make_mock_client()
        gen    = SpectralFeatureGenerator(client, cfg)
        comp   = _make_composite_result()

        custom = FeatureConfig(ndwi=False)
        result = gen.generate(comp, feature_config=custom)
        assert "NDWI" not in result.features_computed
        assert "NDWI" in result.features_skipped

    def test_feature_override_config_used_not_default(
        self, tmp_path: Path
    ) -> None:
        cfg    = _make_config(tmp_path)
        client = _make_mock_client()
        gen    = SpectralFeatureGenerator(client, cfg)
        comp   = _make_composite_result()

        # Only compute BSI
        only_bsi = FeatureConfig(
            ndwi=False, mndwi=False, awei_sh=False, awei_nsh=False,
            ndvi=False, savi=False, bsi=True, ndmi=False, ndbi=False,
        )
        result = gen.generate(comp, feature_config=only_bsi)
        assert result.features_computed == ("BSI",)

    def test_all_disabled_produces_empty_features_computed(
        self, tmp_path: Path
    ) -> None:
        cfg    = _make_config(tmp_path)
        client = _make_mock_client()
        gen    = SpectralFeatureGenerator(client, cfg)
        comp   = _make_composite_result()

        none_config = FeatureConfig(
            ndwi=False, mndwi=False, awei_sh=False, awei_nsh=False,
            ndvi=False, savi=False, bsi=False, ndmi=False, ndbi=False,
        )
        result = gen.generate(comp, feature_config=none_config)
        assert result.features_computed == ()
        assert len(result.features_skipped) == 9

    def test_composite_band_names_preserved_in_result(
        self, tmp_path: Path
    ) -> None:
        cfg    = _make_config(tmp_path)
        client = _make_mock_client()
        gen    = SpectralFeatureGenerator(client, cfg)
        comp   = _make_composite_result()

        result = gen.generate(comp)
        for band in COMMON_BAND_NAMES:
            assert band in result.composite_band_names

    def test_all_band_names_includes_composite_and_index_bands(
        self, tmp_path: Path
    ) -> None:
        cfg    = _make_config(tmp_path)
        client = _make_mock_client()
        gen    = SpectralFeatureGenerator(client, cfg)
        comp   = _make_composite_result()

        result = gen.generate(comp)
        all_bands = result.all_band_names

        for band in OPTICAL_BAND_NAMES:
            assert band in all_bands

        assert "Thermal" not in all_bands
        assert "QA_PIXEL" not in all_bands

        for feature_name in result.features_computed:
            assert feature_name in all_bands

        for idx in result.features_computed:
            assert idx in all_bands

    def test_index_bands_are_subset_of_all_band_names(
        self, tmp_path: Path
    ) -> None:
        cfg    = _make_config(tmp_path)
        client = _make_mock_client()
        gen    = SpectralFeatureGenerator(client, cfg)
        comp   = _make_composite_result()

        result = gen.generate(comp)
        for band in result.index_band_names:
            assert band in result.all_band_names

    def test_feature_config_stored_in_result(self, tmp_path: Path) -> None:
        cfg    = _make_config(tmp_path)
        client = _make_mock_client()
        gen    = SpectralFeatureGenerator(client, cfg)
        comp   = _make_composite_result()

        custom = FeatureConfig(ndbi=True)
        result = gen.generate(comp, feature_config=custom)
        assert result.feature_config is custom

    def test_features_computed_in_registration_order(
        self, tmp_path: Path
    ) -> None:
        """Computed features must follow BUILT_IN_INDICES registration order."""
        from src.gee.registry import BUILT_IN_INDICES
        cfg    = _make_config(tmp_path)
        client = _make_mock_client()
        gen    = SpectralFeatureGenerator(client, cfg)
        comp   = _make_composite_result()

        result       = gen.generate(comp)
        expected_order = [
            m.name for m in BUILT_IN_INDICES
            if m.name in set(result.features_computed)
        ]
        assert list(result.features_computed) == expected_order

    # def test_add_bands_called_for_each_enabled_feature(
    #     self, tmp_path: Path
    # ) -> None:
    #     cfg    = _make_config(tmp_path)
    #     client = _make_mock_client()
    #     gen    = SpectralFeatureGenerator(client, cfg)
    #     img    = MagicMock(name="composite_image")
    #     img.addBands.return_value = img
    #     comp   = _make_composite_result(image=img)

    #     result = gen.generate(comp)
    #     # addBands should be called once per computed feature.
    #     # assert img.addBands.call_count == len(result.features_computed)

    #     selected_img = img.select.return_value

    #     # assert img.select.call_count == 1
    #     # img.select.assert_called_once_with(list(OPTICAL_BAND_NAMES))

    #     assert selected_img.addBands.call_count == len(
    #         result.features_computed
    #     )
    def test_add_bands_called_for_each_enabled_feature(
        self, tmp_path: Path
    ) -> None:
        cfg = _make_config(tmp_path)
        client = _make_mock_client()
        gen = SpectralFeatureGenerator(client, cfg)

        img = MagicMock(name="composite_image")

        # Use one stable mock for the model stack so chained addBands()
        # calls are recorded on the same object.
        selected_img = MagicMock(name="model_base_image")
        selected_img.addBands.return_value = selected_img

        img.select.return_value = selected_img

        comp = _make_composite_result(image=img)

        result = gen.generate(comp)

        assert selected_img.addBands.call_count == len(
            result.features_computed
        )


# ==============================================================================
# SpectralFeatureGenerator.generate() — SAVI soil factor tests
# ==============================================================================

class TestGenerateSAVI:
    """Tests that SAVI uses the configured soil factor."""

    def test_savi_uses_configured_soil_factor(
        self, tmp_path: Path
    ) -> None:
        cfg    = _make_config(tmp_path)
        client = _make_mock_client()
        gen    = SpectralFeatureGenerator(client, cfg)
        comp   = _make_composite_result()

        custom = FeatureConfig(
            ndwi=False, mndwi=False, awei_sh=False, awei_nsh=False,
            ndvi=False, savi=True, bsi=False, ndmi=False, ndbi=False,
            savi_soil_factor=0.25,
        )

        img = comp.image
        # Track what multiply was called with on the NIR band.
        nir_mock = MagicMock(name="NIR")
        nir_mock.subtract.return_value = nir_mock
        nir_mock.add.return_value      = nir_mock
        nir_mock.divide.return_value   = nir_mock
        nir_mock.multiply.return_value = nir_mock
        img.select.side_effect = lambda b: nir_mock

        result = gen.generate(comp, feature_config=custom)
        # With soil_factor=0.25, multiply should have been called with 1.25
        multiply_calls = [
            c[0][0] for c in nir_mock.multiply.call_args_list
        ]
        assert 1.25 in multiply_calls

    def test_savi_default_soil_factor_0_5_used(
        self, tmp_path: Path
    ) -> None:
        cfg    = _make_config(tmp_path)
        client = _make_mock_client()
        gen    = SpectralFeatureGenerator(client, cfg)
        comp   = _make_composite_result()

        custom = FeatureConfig(
            ndwi=False, mndwi=False, awei_sh=False, awei_nsh=False,
            ndvi=False, savi=True, bsi=False, ndmi=False, ndbi=False,
            savi_soil_factor=0.5,
        )

        img = comp.image
        nir_mock = MagicMock(name="NIR")
        nir_mock.subtract.return_value = nir_mock
        nir_mock.add.return_value      = nir_mock
        nir_mock.divide.return_value   = nir_mock
        nir_mock.multiply.return_value = nir_mock
        img.select.side_effect = lambda b: nir_mock

        gen.generate(comp, feature_config=custom)
        multiply_calls = [
            c[0][0] for c in nir_mock.multiply.call_args_list
        ]
        # With L=0.5: final multiplier is 1.5
        assert 1.5 in multiply_calls


# ==============================================================================
# SpectralFeatureGenerator.generate_single_index() tests
# ==============================================================================

class TestGenerateSingleIndex:
    """Tests for SpectralFeatureGenerator.generate_single_index()."""

    def test_returns_single_band_image(self, tmp_path: Path) -> None:
        cfg    = _make_config(tmp_path)
        client = _make_mock_client()
        gen    = SpectralFeatureGenerator(client, cfg)
        comp   = _make_composite_result()

        result = gen.generate_single_index(comp, "BSI")
        assert result is not None

    def test_invalid_index_name_raises(self, tmp_path: Path) -> None:
        cfg    = _make_config(tmp_path)
        client = _make_mock_client()
        gen    = SpectralFeatureGenerator(client, cfg)
        comp   = _make_composite_result()

        with pytest.raises(InvalidValueError):
            gen.generate_single_index(comp, "INVALID_INDEX")

    def test_savi_uses_provided_soil_factor(
        self, tmp_path: Path
    ) -> None:
        cfg    = _make_config(tmp_path)
        client = _make_mock_client()
        gen    = SpectralFeatureGenerator(client, cfg)

        img      = MagicMock()
        nir_mock = MagicMock()
        nir_mock.subtract.return_value = nir_mock
        nir_mock.add.return_value      = nir_mock
        nir_mock.divide.return_value   = nir_mock
        nir_mock.multiply.return_value = nir_mock
        img.select.side_effect = lambda b: nir_mock
        img.addBands.return_value = img

        selected_img = MagicMock(name="model_base_image")
        selected_img.addBands.return_value = selected_img
        img.select.return_value = selected_img
        comp = _make_composite_result(image=img)

        gen.generate_single_index(comp, "SAVI", soil_factor=0.1)
        multiply_calls = [c[0][0] for c in nir_mock.multiply.call_args_list]
        assert 1.1 in multiply_calls


# ==============================================================================
# SpectralFeatureGenerator._build_function_map() tests
# ==============================================================================

class TestBuildFunctionMap:
    """Tests for the SAVI partial function wrapping."""

    def test_all_nine_functions_in_map(self, tmp_path: Path) -> None:
        cfg    = _make_config(tmp_path)
        client = _make_mock_client()
        gen    = SpectralFeatureGenerator(client, cfg)
        cfg_obj = FeatureConfig()
        fn_map  = gen._build_function_map(cfg_obj)
        assert len(fn_map) == 9

    def test_savi_is_partial_in_map(self, tmp_path: Path) -> None:
        import functools
        cfg    = _make_config(tmp_path)
        client = _make_mock_client()
        gen    = SpectralFeatureGenerator(client, cfg)
        fn_map = gen._build_function_map(FeatureConfig(savi_soil_factor=0.5))
        savi_fn = fn_map["SAVI"]
        assert isinstance(savi_fn, functools.partial)

    def test_savi_partial_has_correct_soil_factor(
        self, tmp_path: Path
    ) -> None:
        import functools
        cfg    = _make_config(tmp_path)
        client = _make_mock_client()
        gen    = SpectralFeatureGenerator(client, cfg)
        fn_map = gen._build_function_map(FeatureConfig(savi_soil_factor=0.25))
        savi_fn = fn_map["SAVI"]
        assert isinstance(savi_fn, functools.partial)
        assert savi_fn.keywords.get("soil_factor") == pytest.approx(0.25)

    def test_ndwi_is_not_partial(self, tmp_path: Path) -> None:
        import functools
        from src.gee.indices import compute_ndwi
        cfg    = _make_config(tmp_path)
        client = _make_mock_client()
        gen    = SpectralFeatureGenerator(client, cfg)
        fn_map = gen._build_function_map(FeatureConfig())
        assert not isinstance(fn_map["NDWI"], functools.partial)
        assert fn_map["NDWI"] is compute_ndwi


# ==============================================================================
# Harmonization warning tests
# ==============================================================================

class TestHarmonizationWarning:
    """Tests for harmonization validation in generate()."""

#     def test_warns_when_harmonization_not_applied(
#         self, tmp_path: Path
#     ) -> None:
#         cfg    = _make_config(tmp_path)
#         client = _make_mock_client()
#         gen    = SpectralFeatureGenerator(client, cfg)
#         comp   = _make_composite_result(harmonized=False)

#         with patch.object(gen._logger, "warning") as mock_warn:
#             gen.generate(comp, feature_config=FeatureConfig(
#                 ndwi=False, mndwi=False, awei_sh=False, awei_nsh=False,
#                 ndvi=False, savi=False, bsi=False, ndmi=False, ndbi=False,
#             ))
#         # Warning should mention harmonization when band_names is empty.
#         mock_warn.assert_called()
    def test_warns_when_harmonization_not_applied(
        self, tmp_path: Path
    ) -> None:
        cfg = _make_config(tmp_path)
        client = _make_mock_client()
        gen = SpectralFeatureGenerator(client, cfg)
        comp = _make_composite_result(harmonized=False)

        with patch.object(feature_stack._LOGGER, "warning") as mock_warn:
            gen.generate(
                comp,
                feature_config=FeatureConfig(
                    ndwi=False,
                    mndwi=False,
                    awei_sh=False,
                    awei_nsh=False,
                    ndvi=False,
                    savi=False,
                    bsi=False,
                    ndmi=False,
                    ndbi=False,
                ),
            )

            mock_warn.assert_called_once()