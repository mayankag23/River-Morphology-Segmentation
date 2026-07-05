"""
Tests for src/training/models/registry.py

Run:
    pytest tests/training/models/test_model_registry.py -v \
        --cov=src/training/models/registry --cov-report=term-missing
"""

from __future__ import annotations

import pytest

from src.training.models.registry import ModelRegistry
from src.training.models.base import SegmentationModel


class TestModelRegistry:
    def setup_method(self) -> None:
        """Save and restore registry state for test isolation."""
        self._saved = dict(ModelRegistry._registered)

    def teardown_method(self) -> None:
        ModelRegistry._registered.clear()
        ModelRegistry._registered.update(self._saved)

    def _make_stub(self, name: str) -> type:
        """Create a minimal SegmentationModel stub for registration tests."""
        class _Stub(SegmentationModel):
            model_name = name
            def forward(self, x):
                return x
        _Stub.__name__ = f"Stub_{name}"
        return _Stub

    def test_unetplusplus_is_registered_by_default(self) -> None:
        assert "unetplusplus" in ModelRegistry.registered_names()

    def test_register_decorator(self) -> None:
        Stub = self._make_stub("_test_reg_a")
        ModelRegistry.register(Stub)
        assert "_test_reg_a" in ModelRegistry.registered_names()

    def test_register_external(self) -> None:
        Stub = self._make_stub("_test_reg_b")
        ModelRegistry.register_external(Stub)
        assert "_test_reg_b" in ModelRegistry.registered_names()

    def test_get_returns_correct_class(self) -> None:
        Stub = self._make_stub("_test_reg_c")
        ModelRegistry.register(Stub)
        assert ModelRegistry.get("_test_reg_c") is Stub

    def test_get_unknown_raises_keyerror(self) -> None:
        with pytest.raises(KeyError, match="not registered"):
            ModelRegistry.get("does_not_exist_xyz")

    def test_is_registered_true(self) -> None:
        Stub = self._make_stub("_test_reg_d")
        ModelRegistry.register(Stub)
        assert ModelRegistry.is_registered("_test_reg_d") is True

    def test_is_registered_false(self) -> None:
        assert ModelRegistry.is_registered("never_registered_xyz") is False

    def test_registered_names_is_sorted(self) -> None:
        names = ModelRegistry.registered_names()
        assert list(names) == sorted(names)

    def test_register_empty_name_raises(self) -> None:
        class _Bad(SegmentationModel):
            model_name = ""
            def forward(self, x): return x
        with pytest.raises(ValueError, match="model_name"):
            ModelRegistry.register(_Bad)

    def test_clear_removes_all(self) -> None:
        Stub = self._make_stub("_test_reg_e")
        ModelRegistry.register(Stub)
        ModelRegistry.clear()
        assert ModelRegistry.registered_names() == ()

    def test_register_external_empty_name_raises(self) -> None:
        class _Bad(SegmentationModel):
            model_name = ""
            def forward(self, x): return x
        with pytest.raises(ValueError, match="model_name"):
            ModelRegistry.register_external(_Bad)

    def test_overwrite_registration(self) -> None:
        """Re-registering the same name replaces the previous entry."""
        Stub1 = self._make_stub("_overwrite_test")
        Stub2 = self._make_stub("_overwrite_test")
        ModelRegistry.register(Stub1)
        ModelRegistry.register(Stub2)
        assert ModelRegistry.get("_overwrite_test") is Stub2
