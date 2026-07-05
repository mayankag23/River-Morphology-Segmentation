"""
Unit tests for src/labels/conflicts.py.

Tests verify both strategies and that resolved_confidence_map (Level 3)
is correctly populated on the returned ClassificationResult.

Run:
    pytest tests/labels/test_label_conflicts.py -v \
        --cov=src/labels/conflicts --cov-report=term-missing
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.core.exceptions import InvalidValueError
from src.labels.contracts import ClassificationResult, RuleResult
from src.labels.conflicts import ConflictResolver


def _make_classification(h=4, w=4, class_map=None, confidence_map=None,
                          rule_results=None) -> ClassificationResult:
    if class_map is None:
        class_map = np.zeros((h, w), dtype=np.uint8)
    if confidence_map is None:
        confidence_map = np.zeros((h, w), dtype=np.float32)

    shape = class_map.shape
    return ClassificationResult(
        class_map=class_map, confidence_map=confidence_map,
        rule_results=rule_results or [],
        unclassified_mask=(confidence_map == 0.0),
        nodata_mask=np.zeros(shape, dtype=bool),
        resolved_confidence_map=None,
    )


class TestConflictResolverConstruction:
    def test_invalid_strategy_raises(self) -> None:
        with pytest.raises(InvalidValueError, match="strategy"):
            ConflictResolver(strategy="unknown", priority_order=[1, 2, 3, 0])

    def test_highest_confidence_accepted(self) -> None:
        ConflictResolver("highest_confidence", [1, 3, 2, 0])

    def test_priority_order_accepted(self) -> None:
        ConflictResolver("priority_order", [1, 3, 2, 0])


class TestHighestConfidenceStrategy:
    def test_highest_confidence_passes_class_map_through(self) -> None:
        class_map      = np.array([[1, 2], [3, 0]], dtype=np.uint8)
        confidence_map = np.array([[0.8, 0.5], [0.7, 0.1]], dtype=np.float32)
        clf   = _make_classification(class_map=class_map, confidence_map=confidence_map)
        res   = ConflictResolver("highest_confidence", [1, 3, 2, 0]).resolve(clf)
        np.testing.assert_array_equal(res.class_map, class_map)

    def test_highest_confidence_sets_resolved_equal_to_confidence(self) -> None:
        """For highest_confidence, Level 3 == Level 2."""
        conf_map  = np.array([[0.8, 0.5], [0.7, 0.1]], dtype=np.float32)
        class_map = np.array([[1, 2], [3, 0]], dtype=np.uint8)
        clf       = _make_classification(class_map=class_map, confidence_map=conf_map)
        res       = ConflictResolver("highest_confidence", [1, 3, 2, 0]).resolve(clf)
        assert res.resolved_confidence_map is not None
        np.testing.assert_array_equal(res.resolved_confidence_map, conf_map)


class TestPriorityOrderStrategy:
    def _make_with_rules(self) -> ClassificationResult:
        h, w = 4, 4
        water_conf = np.full((h, w), 0.5, dtype=np.float32)
        water_mask = np.ones((h, w), dtype=bool)
        sand_conf  = np.full((h, w), 0.4, dtype=np.float32)
        sand_mask  = np.ones((h, w), dtype=bool)

        rule_water = RuleResult(
            class_id=1, class_name="water",
            confidence=water_conf, pixel_mask=water_mask,
            bands_used=("MNDWI",), bands_missing=(),
        )
        rule_sand  = RuleResult(
            class_id=2, class_name="sand",
            confidence=sand_conf, pixel_mask=sand_mask,
            bands_used=("BSI",), bands_missing=(),
        )

        class_map      = np.full((h, w), 1, dtype=np.uint8)
        confidence_map = water_conf.copy()

        return ClassificationResult(
            class_map=class_map, confidence_map=confidence_map,
            rule_results=[rule_water, rule_sand],
            unclassified_mask=np.zeros((h, w), dtype=bool),
            nodata_mask=np.zeros((h, w), dtype=bool),
            resolved_confidence_map=None,
        )

    def test_water_wins_over_sand_with_priority(self) -> None:
        resolver = ConflictResolver("priority_order", priority_order=[1, 3, 2, 0])
        clf      = self._make_with_rules()
        result   = resolver.resolve(clf)
        assert (result.class_map == 1).all()

    def test_priority_order_populates_resolved_confidence(self) -> None:
        """Level 3 resolved_confidence_map must be populated for priority_order."""
        resolver = ConflictResolver("priority_order", priority_order=[1, 3, 2, 0])
        clf      = self._make_with_rules()
        result   = resolver.resolve(clf)
        assert result.resolved_confidence_map is not None
        assert result.resolved_confidence_map.shape == clf.class_map.shape


class TestConflictResolverFromConfig:
    def test_from_config_creates_resolver(self, tmp_path: Path) -> None:
        from src.core.config import Config
        from tests.conftest import make_valid_config, write_config
        data = make_valid_config()
        data["labels"] = {
            "nodata_value": 255,
            "conflict_resolution": {
                "strategy": "highest_confidence",
                "water_priority": 0, "vegetation_priority": 1,
                "sand_priority": 2, "background_priority": 3,
            },
        }
        cfg      = Config(config_path=write_config(tmp_path, data))
        resolver = ConflictResolver.from_config(cfg)
        assert resolver._strategy == "highest_confidence"


# """
# Unit tests for src/labels/conflicts.py.

# Run:
#     pytest tests/labels/test_label_conflicts.py -v \
#         --cov=src/labels/conflicts --cov-report=term-missing
# """

# from __future__ import annotations

# from pathlib import Path

# import numpy as np
# import pytest

# from src.core.exceptions import InvalidValueError
# from src.labels.contracts import ClassificationResult, RuleResult
# from src.labels.conflicts import ConflictResolver


# def _make_classification(h=4, w=4, class_map=None, confidence_map=None) -> ClassificationResult:
#     if class_map is None:
#         class_map = np.zeros((h, w), dtype=np.uint8)
#     if confidence_map is None:
#         confidence_map = np.zeros((h, w), dtype=np.float32)
#     return ClassificationResult(
#         class_map=class_map, confidence_map=confidence_map,
#         rule_results=[], unclassified_mask=(confidence_map == 0.0),
#         nodata_mask=np.zeros((h, w), dtype=bool),
#     )


# class TestConflictResolverConstruction:
#     def test_invalid_strategy_raises(self) -> None:
#         with pytest.raises(InvalidValueError, match="strategy"):
#             ConflictResolver(strategy="unknown", priority_order=[1, 2, 3, 0])

#     def test_highest_confidence_accepted(self) -> None:
#         ConflictResolver("highest_confidence", [1, 3, 2, 0])

#     def test_priority_order_accepted(self) -> None:
#         ConflictResolver("priority_order", [1, 3, 2, 0])


# class TestHighestConfidenceStrategy:
#     def test_highest_confidence_passes_through(self) -> None:
#         """highest_confidence is already applied by classifier; resolver is a pass-through."""
#         class_map      = np.array([[1, 2], [3, 0]], dtype=np.uint8)
#         confidence_map = np.array([[0.8, 0.5], [0.7, 0.1]], dtype=np.float32)
#         clf   = _make_classification(class_map=class_map, confidence_map=confidence_map)
#         res   = ConflictResolver("highest_confidence", [1, 3, 2, 0]).resolve(clf)
#         np.testing.assert_array_equal(res.class_map, class_map)


# class TestPriorityOrderStrategy:
#     def _make_with_rules(self) -> ClassificationResult:
#         h, w = 4, 4
#         # Water rule covers all pixels
#         water_conf  = np.full((h, w), 0.5, dtype=np.float32)
#         water_mask  = np.ones((h, w), dtype=bool)
#         # Sand rule also covers all pixels with slightly lower confidence
#         sand_conf   = np.full((h, w), 0.4, dtype=np.float32)
#         sand_mask   = np.ones((h, w), dtype=bool)

#         rule_water = RuleResult(
#             class_id=1, class_name="water",
#             confidence=water_conf, pixel_mask=water_mask,
#             bands_used=("MNDWI",), bands_missing=(),
#         )
#         rule_sand  = RuleResult(
#             class_id=2, class_name="sand",
#             confidence=sand_conf, pixel_mask=sand_mask,
#             bands_used=("BSI",), bands_missing=(),
#         )

#         # Confidence already set to water (highest_confidence applied first)
#         class_map      = np.full((h, w), 1, dtype=np.uint8)
#         confidence_map = water_conf.copy()

#         return ClassificationResult(
#             class_map=class_map, confidence_map=confidence_map,
#             rule_results=[rule_water, rule_sand],
#             unclassified_mask=np.zeros((h, w), dtype=bool),
#             nodata_mask=np.zeros((h, w), dtype=bool),
#         )

#     def test_water_wins_over_sand_with_priority(self) -> None:
#         # priority_order: water=0 (highest) > sand=2
#         resolver = ConflictResolver("priority_order", priority_order=[1, 3, 2, 0])
#         clf      = self._make_with_rules()
#         result   = resolver.resolve(clf)
#         # Water (priority 0) should dominate
#         assert (result.class_map == 1).all()


# class TestConflictResolverFromConfig:
#     def test_from_config_creates_resolver(self, tmp_path: Path) -> None:
#         from src.core.config import Config
#         from tests.conftest import make_valid_config, write_config
#         data = make_valid_config()
#         data["labels"] = {
#             "nodata_value": 255,
#             "conflict_resolution": {
#                 "strategy": "highest_confidence",
#                 "water_priority": 0, "vegetation_priority": 1,
#                 "sand_priority": 2, "background_priority": 3,
#             },
#         }
#         cfg      = Config(config_path=write_config(tmp_path, data))
#         resolver = ConflictResolver.from_config(cfg)
#         assert resolver._strategy == "highest_confidence"