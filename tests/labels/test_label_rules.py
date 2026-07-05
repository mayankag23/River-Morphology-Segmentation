"""
Unit tests for src/labels/rules.py.

Pure numpy, no I/O, no Earth Engine.

Key invariants tested:
- _soft_threshold() is the stable protected interface (delegates to sigmoid).
- BackgroundRule is always last in RuleEngine.rule_names.
- RuleRegistry.clear() is called in teardown for test isolation.

Run:
    pytest tests/labels/test_label_rules.py -v \
        --cov=src/labels/rules --cov-report=term-missing
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.labels.contracts import SpectralBandData
from src.labels.rules import (
    BackgroundRule,
    ClassificationRule,
    RuleEngine,
    RuleRegistry,
    SandRule,
    VegetationRule,
    WaterRule,
)
from tests.conftest import make_valid_config, write_config


def _make_config(tmp_path: Path, **rule_overrides):
    from src.core.config import Config
    data = make_valid_config()
    data["labels"] = {
        "nodata_value": 255, "mask_filename_pattern": "{patch_id}_mask.tif",
        "default_label_version": "1.0.0", "default_annotator": "spectral_rule_engine",
        "default_confidence": 0.7, "default_confidence_source": "automatic",
        "min_distinct_classes": 1, "reject_single_class_masks": False,
        "max_nodata_ratio": 0.5, "output_formats": ["csv"],
        "ratios": {}, "bare_sediment_numerator": [], "bare_sediment_denominator": [],
        "rules": {
            "water": {
                "enabled": True, "mndwi_threshold": 0.2, "ndwi_threshold": 0.0,
                "awei_nsh_threshold": 0.0, "awei_sh_threshold": 0.0,
                "ndvi_max_threshold": 0.05, "mndwi_weight": 0.40,
                "ndwi_weight": 0.20, "awei_nsh_weight": 0.15, "awei_sh_weight": 0.05,
                "ndvi_weight": 0.20, "confidence_scale": 0.30, "min_confidence": 0.40,
            },
            "sand": {
                "enabled": True, "mndwi_max_threshold": 0.0, "bsi_threshold": 0.0,
                "ndvi_max_threshold": 0.25, "ndwi_max_threshold": 0.0,
                "mndwi_weight": 0.30, "bsi_weight": 0.35, "ndvi_weight": 0.25,
                "ndwi_weight": 0.10, "confidence_scale": 0.30, "min_confidence": 0.30,
            },
            "vegetation": {
                "enabled": True, "ndvi_threshold": 0.20, "savi_threshold": 0.10,
                "ndmi_threshold": 0.00, "ndvi_weight": 0.50, "savi_weight": 0.30,
                "ndmi_weight": 0.20, "confidence_scale": 0.25, "min_confidence": 0.35,
            },
            "background": {"enabled": True, "min_confidence": 0.10},
        },
        "conflict_resolution": {"strategy": "highest_confidence",
                                "water_priority": 0, "vegetation_priority": 1,
                                "sand_priority": 2, "background_priority": 3},
        "morphology": {"enabled": False},
        "quality": {"min_valid_pixel_ratio": 0.1, "min_quality_score": 0.1,
                    "max_unclassified_ratio": 0.9, "min_class_pixels": 1},
        "confidence": {"min_pixel_confidence": 0.1, "min_mask_confidence": 0.1},
        "generation": {"pseudo_label_version": "1.0.0",
                       "rule_engine_version": "1.0.0",
                       "generation_method": "spectral_rules",
                       "strategy_type": "spectral_rules"},
    }
    return Config(config_path=write_config(tmp_path, data))


def _band_data(H=4, W=4, **band_values) -> SpectralBandData:
    bands = {name: np.full((H, W), val, dtype=np.float32)
             for name, val in band_values.items()}
    return SpectralBandData(
        bands=bands, height=H, width=W, crs="EPSG:4326", transform=None,
        band_names=tuple(band_values.keys()),
    )


# ==============================================================================
# _soft_threshold -- stable protected interface
# ==============================================================================

class TestSoftThreshold:
    """
    _soft_threshold() is the stable interface. It delegates internally to
    _sigmoid_evidence(). Tests call _soft_threshold() directly.
    """

    def test_greater_high_value_gives_high_confidence(self) -> None:
        arr = np.array([[0.8, 0.9]], dtype=np.float32)
        result = ClassificationRule._soft_threshold(arr, 0.2, 0.3, "greater")
        assert (result > 0.8).all()

    def test_greater_low_value_gives_low_confidence(self) -> None:
        arr = np.array([[0.0, 0.1]], dtype=np.float32)
        result = ClassificationRule._soft_threshold(arr, 0.5, 0.3, "greater")
        assert (result < 0.2).all()

    def test_less_direction_inverts(self) -> None:
        arr = np.array([[-0.5]], dtype=np.float32)
        result = ClassificationRule._soft_threshold(arr, 0.0, 0.3, "less")
        assert result[0, 0] > 0.8

    def test_nan_becomes_zero(self) -> None:
        arr = np.array([[np.nan]], dtype=np.float32)
        result = ClassificationRule._soft_threshold(arr, 0.0, 0.3, "greater")
        assert result[0, 0] == pytest.approx(0.0)

    def test_at_threshold_is_approximately_half(self) -> None:
        """Sigmoid at the threshold value produces ~0.5 evidence."""
        arr = np.array([[0.2]], dtype=np.float32)
        result = ClassificationRule._soft_threshold(arr, 0.2, 0.3, "greater")
        assert result[0, 0] == pytest.approx(0.5, abs=0.01)

    def test_output_is_float32(self) -> None:
        arr = np.array([[0.5]], dtype=np.float32)
        result = ClassificationRule._soft_threshold(arr, 0.0, 0.3, "greater")
        assert result.dtype == np.float32

    def test_output_in_zero_one(self) -> None:
        arr = np.linspace(-1.0, 1.0, 100, dtype=np.float32).reshape(10, 10)
        result = ClassificationRule._soft_threshold(arr, 0.0, 0.3, "greater")
        assert result.min() >= 0.0
        assert result.max() <= 1.0


# ==============================================================================
# WaterRule
# ==============================================================================

class TestWaterRule:
    def test_water_detected_when_mndwi_high(self, tmp_path: Path) -> None:
        cfg  = _make_config(tmp_path)
        rule = WaterRule(cfg)
        bd   = _band_data(MNDWI=0.6, NDWI=0.3, AWEI_nsh=0.5, NDVI=-0.1)
        res  = rule.apply(bd)
        assert res.pixel_mask.all()
        assert res.confidence.mean() > 0.4

    def test_water_not_detected_when_mndwi_low(self, tmp_path: Path) -> None:
        cfg  = _make_config(tmp_path)
        rule = WaterRule(cfg)
        bd   = _band_data(MNDWI=-0.3, NDWI=-0.2, NDVI=0.3)
        res  = rule.apply(bd)
        assert not res.pixel_mask.any()

    def test_class_id_is_1(self, tmp_path: Path) -> None:
        assert WaterRule(_make_config(tmp_path)).class_id == 1

    def test_class_name_is_water(self, tmp_path: Path) -> None:
        assert WaterRule(_make_config(tmp_path)).class_name == "water"

    def test_missing_bands_handled(self, tmp_path: Path) -> None:
        cfg  = _make_config(tmp_path)
        rule = WaterRule(cfg)
        bd   = _band_data(MNDWI=0.6)   # only MNDWI present
        res  = rule.apply(bd)
        assert "NDWI" in res.bands_missing

    def test_context_ignored_currently(self, tmp_path: Path) -> None:
        from src.labels.contracts import ClassificationContext
        cfg  = _make_config(tmp_path)
        rule = WaterRule(cfg)
        bd   = _band_data(MNDWI=0.6, NDWI=0.3, NDVI=-0.1)
        ctx  = ClassificationContext(season="monsoon")
        res  = rule.apply(bd, context=ctx)
        assert isinstance(res.confidence, np.ndarray)


# ==============================================================================
# SandRule
# ==============================================================================

class TestSandRule:
    def test_sand_detected(self, tmp_path: Path) -> None:
        cfg  = _make_config(tmp_path)
        rule = SandRule(cfg)
        bd   = _band_data(MNDWI=-0.4, BSI=0.2, NDVI=0.1, NDWI=-0.3)
        res  = rule.apply(bd)
        assert res.pixel_mask.all()

    def test_sand_not_detected_near_water(self, tmp_path: Path) -> None:
        cfg  = _make_config(tmp_path)
        rule = SandRule(cfg)
        bd   = _band_data(MNDWI=0.5, BSI=0.1, NDVI=0.05)
        res  = rule.apply(bd)
        assert not res.pixel_mask.any()

    def test_class_id_is_2(self, tmp_path: Path) -> None:
        assert SandRule(_make_config(tmp_path)).class_id == 2

    def test_context_ignored_currently(self, tmp_path: Path) -> None:
        from src.labels.contracts import ClassificationContext
        cfg  = _make_config(tmp_path)
        rule = SandRule(cfg)
        bd   = _band_data(MNDWI=-0.4, BSI=0.2, NDVI=0.1, NDWI=-0.3)
        ctx  = ClassificationContext(season="winter")
        res  = rule.apply(bd, context=ctx)
        assert isinstance(res.confidence, np.ndarray)


# ==============================================================================
# VegetationRule
# ==============================================================================

class TestVegetationRule:
    def test_vegetation_detected(self, tmp_path: Path) -> None:
        cfg  = _make_config(tmp_path)
        rule = VegetationRule(cfg)
        bd   = _band_data(NDVI=0.5, SAVI=0.35, NDMI=0.2)
        res  = rule.apply(bd)
        assert res.pixel_mask.all()

    def test_vegetation_not_detected_low_ndvi(self, tmp_path: Path) -> None:
        cfg  = _make_config(tmp_path)
        rule = VegetationRule(cfg)
        bd   = _band_data(NDVI=0.05, SAVI=0.02, NDMI=-0.1)
        res  = rule.apply(bd)
        assert not res.pixel_mask.any()

    def test_class_id_is_3(self, tmp_path: Path) -> None:
        assert VegetationRule(_make_config(tmp_path)).class_id == 3


# ==============================================================================
# BackgroundRule
# ==============================================================================

class TestBackgroundRule:
    def test_background_is_fallback(self, tmp_path: Path) -> None:
        cfg  = _make_config(tmp_path)
        rule = BackgroundRule(cfg)
        bd   = _band_data()
        res  = rule.apply(bd)
        assert res.pixel_mask.all()
        assert res.confidence.mean() == pytest.approx(0.10)

    def test_class_id_is_0(self, tmp_path: Path) -> None:
        assert BackgroundRule(_make_config(tmp_path)).class_id == 0


# ==============================================================================
# RuleRegistry
# ==============================================================================

class TestRuleRegistry:
    def test_registered_names_include_all_four(self, tmp_path: Path) -> None:
        names = RuleRegistry.registered_names()
        assert "water" in names
        assert "sand" in names
        assert "vegetation" in names
        assert "background" in names

    def test_background_registered_last(self) -> None:
        """Background must always be the last registered rule."""
        names = RuleRegistry.registered_names()
        assert names[-1] == "background"

    def test_clear_removes_all(self) -> None:
        # Save existing state to restore after the test.
        saved = dict(RuleRegistry._registered)
        RuleRegistry.clear()
        assert RuleRegistry.registered_names() == ()
        # Restore.
        RuleRegistry._registered.update(saved)

    def test_register_external(self) -> None:
        from src.labels.contracts import RuleResult

        class _TestRule(ClassificationRule):
            _CLASS_NAME = "test_external"
            _CLASS_ID   = 99

            @property
            def class_id(self) -> int:
                return self._CLASS_ID

            @property
            def class_name(self) -> str:
                return self._CLASS_NAME

            @property
            def is_enabled(self) -> bool:
                return True

            def apply(self, band_data, context=None) -> RuleResult:
                h, w = band_data.height, band_data.width
                return RuleResult(
                    class_id=self._CLASS_ID, class_name=self._CLASS_NAME,
                    confidence=np.zeros((h, w), dtype=np.float32),
                    pixel_mask=np.zeros((h, w), dtype=bool),
                    bands_used=(), bands_missing=(),
                )

        saved = dict(RuleRegistry._registered)
        try:
            RuleRegistry.register_external(_TestRule)
            assert "test_external" in RuleRegistry.registered_names()
        finally:
            RuleRegistry._registered.clear()
            RuleRegistry._registered.update(saved)


# ==============================================================================
# RuleEngine
# ==============================================================================

class TestRuleEngine:
    def test_from_config_builds_engine(self, tmp_path: Path) -> None:
        engine = RuleEngine.from_config(_make_config(tmp_path))
        assert engine.num_rules == 4

    def test_apply_all_returns_results_for_each_rule(self, tmp_path: Path) -> None:
        engine  = RuleEngine.from_config(_make_config(tmp_path))
        bd      = _band_data(MNDWI=0.5, NDWI=0.3, NDVI=-0.1, BSI=0.1, SAVI=0.2, NDMI=0.1)
        results = engine.apply_all(bd)
        assert len(results) == 4

    def test_rule_names_in_correct_order(self, tmp_path: Path) -> None:
        engine = RuleEngine.from_config(_make_config(tmp_path))
        names  = engine.rule_names
        assert "water" in names
        assert "background" in names
        assert names[-1] == "background"

    def test_water_wins_over_background(self, tmp_path: Path) -> None:
        engine  = RuleEngine.from_config(_make_config(tmp_path))
        bd      = _band_data(MNDWI=0.6, NDWI=0.4, AWEI_nsh=0.3, NDVI=-0.1)
        results = engine.apply_all(bd)
        water_r = next(r for r in results if r.class_name == "water")
        bg_r    = next(r for r in results if r.class_name == "background")
        assert water_r.confidence.mean() > bg_r.confidence.mean()

    def test_context_forwarded_to_rules(self, tmp_path: Path) -> None:
        from src.labels.contracts import ClassificationContext
        engine = RuleEngine.from_config(_make_config(tmp_path))
        bd     = _band_data(MNDWI=0.5, NDWI=0.3, NDVI=-0.1)
        ctx    = ClassificationContext(season="monsoon", acquisition_date="2023-07-15")
        # Must not raise; context is accepted and currently ignored by all rules.
        results = engine.apply_all(bd, context=ctx)
        assert len(results) == 4

    def test_explicit_rule_list_accepted(self) -> None:
        """RuleEngine constructor accepts explicit rule list (for testing)."""
        engine = RuleEngine(rules=[])
        assert engine.num_rules == 0
        assert engine.rule_names == ()


# """
# Unit tests for src/labels/rules.py.

# Pure numpy, no I/O, no EE.

# Run:
#     pytest tests/labels/test_label_rules.py -v \
#         --cov=src/labels/rules --cov-report=term-missing
# """

# from __future__ import annotations

# from pathlib import Path
# from unittest.mock import MagicMock

# import numpy as np
# import pytest

# from src.labels.contracts import SpectralBandData
# from src.labels.rules import (
#     BackgroundRule,
#     ClassificationRule,
#     RuleEngine,
#     SandRule,
#     VegetationRule,
#     WaterRule,
# )
# from tests.conftest import make_valid_config, write_config


# def _make_config(tmp_path: Path, **rule_overrides):
#     from src.core.config import Config
#     data = make_valid_config()
#     data["labels"] = {
#         "nodata_value": 255, "mask_filename_pattern": "{patch_id}_mask.tif",
#         "default_label_version": "1.0.0", "default_annotator": "spectral_rule_engine",
#         "default_confidence": 0.7, "default_confidence_source": "automatic",
#         "min_distinct_classes": 1, "reject_single_class_masks": False,
#         "max_nodata_ratio": 0.5, "output_formats": ["csv"],
#         "ratios": {}, "bare_sediment_numerator": [], "bare_sediment_denominator": [],
#         "rules": {
#             "water": {
#                 "enabled": True, "mndwi_threshold": 0.2, "ndwi_threshold": 0.0,
#                 "awei_nsh_threshold": 0.0, "awei_sh_threshold": 0.0,
#                 "ndvi_max_threshold": 0.05, "mndwi_weight": 0.40,
#                 "ndwi_weight": 0.20, "awei_nsh_weight": 0.15, "awei_sh_weight": 0.05,
#                 "ndvi_weight": 0.20, "confidence_scale": 0.30, "min_confidence": 0.40,
#             },
#             "sand": {
#                 "enabled": True, "mndwi_max_threshold": 0.0, "bsi_threshold": 0.0,
#                 "ndvi_max_threshold": 0.25, "ndwi_max_threshold": 0.0,
#                 "mndwi_weight": 0.30, "bsi_weight": 0.35, "ndvi_weight": 0.25,
#                 "ndwi_weight": 0.10, "confidence_scale": 0.30, "min_confidence": 0.30,
#             },
#             "vegetation": {
#                 "enabled": True, "ndvi_threshold": 0.20, "savi_threshold": 0.10,
#                 "ndmi_threshold": 0.00, "ndvi_weight": 0.50, "savi_weight": 0.30,
#                 "ndmi_weight": 0.20, "confidence_scale": 0.25, "min_confidence": 0.35,
#             },
#             "background": {"enabled": True, "min_confidence": 0.10},
#         },
#         "conflict_resolution": {"strategy": "highest_confidence",
#                                 "water_priority": 0, "vegetation_priority": 1,
#                                 "sand_priority": 2, "background_priority": 3},
#         "morphology": {"enabled": False},
#         "quality": {"min_valid_pixel_ratio": 0.1, "min_quality_score": 0.1,
#                     "max_unclassified_ratio": 0.9, "min_class_pixels": 1},
#         "confidence": {"min_pixel_confidence": 0.1, "min_mask_confidence": 0.1},
#         "generation": {"pseudo_label_version": "1.0.0",
#                        "rule_engine_version": "1.0.0",
#                        "generation_method": "spectral_rules"},
#     }
#     return Config(config_path=write_config(tmp_path, data))


# def _band_data(H=4, W=4, **band_values) -> SpectralBandData:
#     bands = {name: np.full((H, W), val, dtype=np.float32)
#              for name, val in band_values.items()}
#     return SpectralBandData(
#         bands=bands, height=H, width=W, crs="EPSG:4326", transform=None,
#         band_names=tuple(band_values.keys()),
#     )


# class TestSoftThreshold:
#     def test_greater_full_confidence(self) -> None:
#         arr = np.array([[0.8, 0.9]], dtype=np.float32)
#         result = ClassificationRule._soft_threshold(arr, 0.2, 0.3, "greater")
#         assert (result == 1.0).all()

#     def test_greater_zero_confidence_below(self) -> None:
#         arr = np.array([[0.0, 0.1]], dtype=np.float32)
#         result = ClassificationRule._soft_threshold(arr, 0.5, 0.3, "greater")
#         assert (result == 0.0).all()

#     def test_less_direction(self) -> None:
#         arr = np.array([[-0.5]], dtype=np.float32)
#         result = ClassificationRule._soft_threshold(arr, 0.0, 0.3, "less")
#         assert result[0, 0] == pytest.approx(1.0)

#     def test_nan_becomes_zero(self) -> None:
#         arr = np.array([[np.nan]], dtype=np.float32)
#         result = ClassificationRule._soft_threshold(arr, 0.0, 0.3, "greater")
#         assert result[0, 0] == pytest.approx(0.0)


# class TestWaterRule:
#     def test_water_detected_when_mndwi_high(self, tmp_path: Path) -> None:
#         cfg  = _make_config(tmp_path)
#         rule = WaterRule(cfg)
#         bd   = _band_data(MNDWI=0.6, NDWI=0.3, AWEI_nsh=0.5, NDVI=-0.1)
#         res  = rule.apply(bd)
#         assert res.pixel_mask.all()
#         assert res.confidence.mean() > 0.4

#     def test_water_not_detected_when_mndwi_low(self, tmp_path: Path) -> None:
#         cfg  = _make_config(tmp_path)
#         rule = WaterRule(cfg)
#         bd   = _band_data(MNDWI=-0.3, NDWI=-0.2, NDVI=0.3)
#         res  = rule.apply(bd)
#         assert not res.pixel_mask.any()

#     def test_class_id_is_1(self, tmp_path: Path) -> None:
#         assert WaterRule(_make_config(tmp_path)).class_id == 1

#     def test_class_name_is_water(self, tmp_path: Path) -> None:
#         assert WaterRule(_make_config(tmp_path)).class_name == "water"

#     def test_missing_bands_handled(self, tmp_path: Path) -> None:
#         cfg = _make_config(tmp_path)
#         rule = WaterRule(cfg)
#         bd   = _band_data(MNDWI=0.6)   # only MNDWI present
#         res  = rule.apply(bd)
#         assert "NDWI" in res.bands_missing

#     def test_disabled_rule_still_returns_result(self, tmp_path: Path) -> None:
#         from src.core.config import Config
#         data = make_valid_config()
#         data["labels"] = _make_config(tmp_path)._node   # reuse config
#         cfg  = _make_config(tmp_path)
#         # Manually set enabled=False check
#         rule = WaterRule(cfg)
#         assert isinstance(rule, WaterRule)


# class TestSandRule:
#     def test_sand_detected(self, tmp_path: Path) -> None:
#         cfg  = _make_config(tmp_path)
#         rule = SandRule(cfg)
#         bd   = _band_data(MNDWI=-0.4, BSI=0.2, NDVI=0.1, NDWI=-0.3)
#         res  = rule.apply(bd)
#         assert res.pixel_mask.all()

#     def test_sand_not_detected_near_water(self, tmp_path: Path) -> None:
#         cfg  = _make_config(tmp_path)
#         rule = SandRule(cfg)
#         bd   = _band_data(MNDWI=0.5, BSI=0.1, NDVI=0.05)
#         res  = rule.apply(bd)
#         assert not res.pixel_mask.any()

#     def test_class_id_is_2(self, tmp_path: Path) -> None:
#         assert SandRule(_make_config(tmp_path)).class_id == 2


# class TestVegetationRule:
#     def test_vegetation_detected(self, tmp_path: Path) -> None:
#         cfg  = _make_config(tmp_path)
#         rule = VegetationRule(cfg)
#         bd   = _band_data(NDVI=0.5, SAVI=0.35, NDMI=0.2)
#         res  = rule.apply(bd)
#         assert res.pixel_mask.all()

#     def test_vegetation_not_detected_low_ndvi(self, tmp_path: Path) -> None:
#         cfg  = _make_config(tmp_path)
#         rule = VegetationRule(cfg)
#         bd   = _band_data(NDVI=0.05, SAVI=0.02, NDMI=-0.1)
#         res  = rule.apply(bd)
#         assert not res.pixel_mask.any()

#     def test_class_id_is_3(self, tmp_path: Path) -> None:
#         assert VegetationRule(_make_config(tmp_path)).class_id == 3


# class TestBackgroundRule:
#     def test_background_is_fallback(self, tmp_path: Path) -> None:
#         cfg  = _make_config(tmp_path)
#         rule = BackgroundRule(cfg)
#         bd   = _band_data()
#         res  = rule.apply(bd)
#         assert res.pixel_mask.all()
#         assert res.confidence.mean() == pytest.approx(0.10)

#     def test_class_id_is_0(self, tmp_path: Path) -> None:
#         assert BackgroundRule(_make_config(tmp_path)).class_id == 0


# class TestRuleEngine:
#     def test_from_config_builds_engine(self, tmp_path: Path) -> None:
#         engine = RuleEngine.from_config(_make_config(tmp_path))
#         assert engine.num_rules == 4

#     def test_apply_all_returns_results_for_each_rule(self, tmp_path: Path) -> None:
#         engine = RuleEngine.from_config(_make_config(tmp_path))
#         bd     = _band_data(MNDWI=0.5, NDWI=0.3, NDVI=-0.1, BSI=0.1, SAVI=0.2, NDMI=0.1)
#         results = engine.apply_all(bd)
#         assert len(results) == 4

#     def test_rule_names_in_correct_order(self, tmp_path: Path) -> None:
#         engine = RuleEngine.from_config(_make_config(tmp_path))
#         names  = engine.rule_names
#         assert "water" in names
#         assert "background" in names
#         assert names[-1] == "background"

#     def test_water_wins_over_background(self, tmp_path: Path) -> None:
#         engine  = RuleEngine.from_config(_make_config(tmp_path))
#         bd      = _band_data(MNDWI=0.6, NDWI=0.4, AWEI_nsh=0.3, NDVI=-0.1)
#         results = engine.apply_all(bd)
#         water_r = next(r for r in results if r.class_name == "water")
#         bg_r    = next(r for r in results if r.class_name == "background")
#         assert water_r.confidence.mean() > bg_r.confidence.mean()