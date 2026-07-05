"""
Tests for src/training/transform.py

Run:
    pytest tests/training/test_training_transform.py -v \
        --cov=src/training/transform --cov-report=term-missing
"""

from __future__ import annotations

import numpy as np
import pytest

from src.training.contracts import TransformSample
from src.training.transform import ComposedTransform, IdentityTransform, SegmentationTransform


def _sample(c: int = 3, h: int = 8, w: int = 8, split: str = "train") -> TransformSample:
    return TransformSample(
        image=np.random.rand(c, h, w).astype(np.float32),
        mask=np.zeros((h, w), dtype=np.uint8),
        sample_id="test_001",
        split=split,
    )


class TestIdentityTransform:
    def test_name(self) -> None:
        assert IdentityTransform().name == "identity"

    def test_returns_same_object(self) -> None:
        t = IdentityTransform()
        s = _sample()
        assert t.apply(s) is s

    def test_callable(self) -> None:
        t = IdentityTransform()
        s = _sample()
        assert t(s) is s


class TestComposedTransform:
    def test_name(self) -> None:
        assert ComposedTransform([]).name == "composed"

    def test_empty_composed_is_identity(self) -> None:
        t  = ComposedTransform([])
        s  = _sample()
        s2 = t.apply(s)
        assert s2 is s

    def test_applies_in_order(self) -> None:
        log: list[str] = []

        class _A(SegmentationTransform):
            _NAME = "_a"

            @property
            def name(self) -> str:
                return self._NAME

            def apply(self, sample: TransformSample) -> TransformSample:
                log.append("A")
                return sample

        class _B(SegmentationTransform):
            _NAME = "_b"

            @property
            def name(self) -> str:
                return self._NAME

            def apply(self, sample: TransformSample) -> TransformSample:
                log.append("B")
                return sample

        t = ComposedTransform([_A(), _B()])
        t.apply(_sample())
        assert log == ["A", "B"]

    def test_transform_names(self) -> None:
        t = ComposedTransform([IdentityTransform(), IdentityTransform()])
        assert t.transform_names == ("identity", "identity")

    def test_repr_is_ascii(self) -> None:
        r = repr(ComposedTransform([IdentityTransform()]))
        assert all(ord(c) < 128 for c in r)

    def test_metadata_preserved_through_composition(self) -> None:
        s = _sample()
        s.acquisition_date   = "2023-07-15"
        s.season             = "monsoon"
        s.hydrological_year  = 2023
        s.river_name         = "Kosi"
        t = ComposedTransform([IdentityTransform()])
        s2 = t.apply(s)
        assert s2.acquisition_date   == "2023-07-15"
        assert s2.season             == "monsoon"
        assert s2.hydrological_year  == 2023
        assert s2.river_name         == "Kosi"
