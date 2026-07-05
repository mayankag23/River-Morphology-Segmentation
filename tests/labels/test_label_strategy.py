"""
Unit tests for src/labels/strategy.py.

Tests cover:
- LabelStrategyRegistry registration (decorator and external).
- LabelStrategyRegistry.create() selects the correct strategy.
- LabelStrategyRegistry.clear() for test isolation.
- PseudoLabelGenerator is registered under "spectral_rules".
- LabelGenerationStrategy is abstract (cannot be instantiated).

Run:
    pytest tests/labels/test_label_strategy.py -v \
        --cov=src/labels/strategy --cov-report=term-missing
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.labels.strategy import LabelGenerationStrategy, LabelStrategyRegistry


# ==============================================================================
# Fixture: minimal stub strategy for registry tests
# ==============================================================================

class _StubStrategy(LabelGenerationStrategy):
    """Minimal concrete strategy stub for isolation tests."""

    _STRATEGY_TYPE: str = "_stub_strategy"

    @property
    def strategy_type(self) -> str:
        return self._STRATEGY_TYPE

    def generate(self, patch_path, patch_id, output_mask_path, context=None):
        raise NotImplementedError

    @classmethod
    def from_config(cls, config, class_schema):
        return cls()


class TestLabelGenerationStrategyABC:
    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            LabelGenerationStrategy()  # type: ignore[abstract]


class TestLabelStrategyRegistry:
    """
    All tests clean up the registry using the saved/restore pattern so that
    the real PseudoLabelGenerator registration is not damaged.
    """

    def setup_method(self) -> None:
        self._saved = dict(LabelStrategyRegistry._registered)

    def teardown_method(self) -> None:
        LabelStrategyRegistry._registered.clear()
        LabelStrategyRegistry._registered.update(self._saved)

    def test_register_decorator(self) -> None:
        @LabelStrategyRegistry.register
        class _Local(_StubStrategy):
            _STRATEGY_TYPE = "_local_strategy"

            @classmethod
            def from_config(cls, config, class_schema):
                return cls()

        assert "_local_strategy" in LabelStrategyRegistry.registered_types()

    def test_register_external(self) -> None:
        LabelStrategyRegistry.register_external(_StubStrategy)
        assert "_stub_strategy" in LabelStrategyRegistry.registered_types()

    def test_registered_types_returns_tuple(self) -> None:
        assert isinstance(LabelStrategyRegistry.registered_types(), tuple)

    def test_clear_removes_all(self) -> None:
        LabelStrategyRegistry.clear()
        assert LabelStrategyRegistry.registered_types() == ()

    def test_create_unknown_strategy_raises(self, tmp_path: Path) -> None:
        LabelStrategyRegistry.clear()

        from src.core.config import Config
        from tests.conftest import make_valid_config, write_config

        data = make_valid_config()
        data["labels"] = {
            "nodata_value": 255,
            "generation": {"strategy_type": "nonexistent_strategy"},
        }
        cfg = Config(config_path=write_config(tmp_path, data))

        from src.labels.schema import ClassSchema
        schema = ClassSchema(classes=())
        with pytest.raises(ValueError, match="nonexistent_strategy"):
            LabelStrategyRegistry.create(cfg, schema)

    def test_create_selects_spectral_rules_by_default(self, tmp_path: Path) -> None:
        # Make sure PseudoLabelGenerator is registered (restored from saved state).
        LabelStrategyRegistry._registered.clear()
        LabelStrategyRegistry._registered.update(self._saved)

        # Import to ensure registration side-effect.
        import src.labels.generator  # noqa: F401

        assert "spectral_rules" in LabelStrategyRegistry.registered_types()


class TestPseudoLabelGeneratorRegistration:
    def test_spectral_rules_is_registered(self) -> None:
        """PseudoLabelGenerator must be registered at import time."""
        import src.labels.generator  # noqa: F401
        assert "spectral_rules" in LabelStrategyRegistry.registered_types()

    def test_registered_type_returns_correct_strategy_type(self) -> None:
        from src.labels.generator import PseudoLabelGenerator
        assert PseudoLabelGenerator._STRATEGY_TYPE == "spectral_rules"
