"""
Unit tests for src/labels/morphology.py.

Run:
    pytest tests/labels/test_label_morphology.py -v \
        --cov=src/labels/morphology --cov-report=term-missing
"""

from __future__ import annotations

import numpy as np
import pytest

from src.labels.contracts import ClassificationResult, MorphologyResult
from src.labels.morphology import MorphologyConfig, MorphologyProcessor


def _simple_classification(class_map: np.ndarray) -> ClassificationResult:
    h, w = class_map.shape
    return ClassificationResult(
        class_map=class_map,
        confidence_map=np.where(class_map > 0, 0.8, 0.0).astype(np.float32),
        rule_results=[],
        unclassified_mask=(class_map == 0),
        nodata_mask=np.zeros((h, w), dtype=bool),
    )


class TestMorphologyConfig:
    def test_defaults(self) -> None:
        cfg = MorphologyConfig()
        assert cfg.enabled is True
        assert cfg.opening_radius == 1

    def test_frozen(self) -> None:
        cfg = MorphologyConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.enabled = False  # type: ignore[misc]


class TestMorphologyProcessorDisabled:
    def test_disabled_returns_unchanged_class_map(self) -> None:
        cfg = MorphologyConfig(enabled=False)
        proc = MorphologyProcessor(cfg)
        class_map = np.array([[1, 2], [0, 1]], dtype=np.uint8)
        clf    = _simple_classification(class_map)
        result = proc.process(clf)
        assert isinstance(result, MorphologyResult)
        np.testing.assert_array_equal(result.class_map, class_map)
        assert "morphology_disabled" in result.operations_applied


class TestMorphologyProcessorEnabled:
    def test_result_has_same_shape(self) -> None:
        cfg = MorphologyConfig(
            enabled=True, opening_radius=1, closing_radius=1,
            min_object_size=0, min_hole_size=0, majority_filter_size=0,
        )
        proc = MorphologyProcessor(cfg)
        class_map = np.zeros((8, 8), dtype=np.uint8)
        class_map[2:6, 2:6] = 1
        clf    = _simple_classification(class_map)
        result = proc.process(clf)
        assert result.class_map.shape == (8, 8)
        assert result.class_map.dtype == np.uint8

    def test_operations_log_non_empty(self) -> None:
        cfg  = MorphologyConfig(enabled=True, opening_radius=1, closing_radius=1)
        proc = MorphologyProcessor(cfg)
        class_map = np.zeros((8, 8), dtype=np.uint8)
        class_map[2:6, 2:6] = 1
        clf    = _simple_classification(class_map)
        result = proc.process(clf)
        assert len(result.operations_applied) > 0


class TestRemoveSmallObjects:
    def test_removes_isolated_pixels(self) -> None:
        small = np.zeros((10, 10), dtype=bool)
        small[0, 0] = True   # isolated
        small[5:8, 5:8] = True  # large region
        result = MorphologyProcessor._remove_small_objects(small, min_size=5)
        assert not result[0, 0]
        assert result[5:8, 5:8].all()

    def test_keeps_large_objects(self) -> None:
        large = np.zeros((10, 10), dtype=bool)
        large[1:9, 1:9] = True
        result = MorphologyProcessor._remove_small_objects(large, min_size=5)
        assert result[1:9, 1:9].all()