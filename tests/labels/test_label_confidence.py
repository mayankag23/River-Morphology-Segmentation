"""
Unit tests for src/labels/confidence.py.

Tests verify:
- Level 3 resolved_confidence_map is used when available.
- Falls back to Level 2 confidence_map when resolved is None.
- min_mask_confidence is accessible as a public property.
- component_scores is populated.

Run:
    pytest tests/labels/test_label_confidence.py -v \
        --cov=src/labels/confidence --cov-report=term-missing
"""

from __future__ import annotations

import numpy as np
import pytest

from src.labels.confidence import ConfidenceConfig, ConfidenceEstimator
from src.labels.contracts import ClassificationResult, RuleResult


def _make_classification(
    class_map: np.ndarray,
    confidence_map: np.ndarray,
    rule_results: list | None = None,
    resolved_confidence_map: np.ndarray | None = None,
) -> ClassificationResult:
    h, w = class_map.shape
    return ClassificationResult(
        class_map=class_map,
        confidence_map=confidence_map,
        rule_results=rule_results or [],
        unclassified_mask=(confidence_map == 0.0),
        nodata_mask=np.zeros((h, w), dtype=bool),
        resolved_confidence_map=resolved_confidence_map,
    )


class TestConfidenceConfig:
    def test_defaults(self) -> None:
        cfg = ConfidenceConfig()
        assert cfg.min_pixel_confidence == pytest.approx(0.20)
        assert cfg.min_mask_confidence  == pytest.approx(0.30)

    def test_frozen(self) -> None:
        cfg = ConfidenceConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.min_mask_confidence = 0.5  # type: ignore[misc]


class TestConfidenceEstimator:
    def _estimator(self, min_mask: float = 0.3) -> ConfidenceEstimator:
        return ConfidenceEstimator(
            ConfidenceConfig(min_pixel_confidence=0.2, min_mask_confidence=min_mask)
        )

    def test_min_mask_confidence_public_property(self) -> None:
        est = self._estimator(min_mask=0.45)
        assert est.min_mask_confidence == pytest.approx(0.45)

    def test_mask_confidence_from_valid_pixels(self) -> None:
        conf_map  = np.full((4, 4), 0.8, dtype=np.float32)
        class_map = np.ones((4, 4), dtype=np.uint8)
        clf       = _make_classification(class_map, conf_map)
        result    = self._estimator().estimate(clf, class_map)
        assert result.mask_confidence == pytest.approx(0.8, abs=1e-3)

    def test_uses_resolved_confidence_when_available(self) -> None:
        """Level 3 resolved_confidence_map takes priority over Level 2."""
        conf_map     = np.full((4, 4), 0.5, dtype=np.float32)
        resolved_map = np.full((4, 4), 0.9, dtype=np.float32)
        class_map    = np.ones((4, 4), dtype=np.uint8)
        clf          = _make_classification(
            class_map, conf_map, resolved_confidence_map=resolved_map
        )
        result = self._estimator().estimate(clf, class_map)
        # mask_confidence should be ~0.9 (from resolved), not ~0.5 (from raw).
        assert result.mask_confidence == pytest.approx(0.9, abs=1e-2)

    def test_falls_back_to_confidence_map_when_resolved_is_none(self) -> None:
        """Falls back to Level 2 when resolved_confidence_map is None."""
        conf_map  = np.full((4, 4), 0.6, dtype=np.float32)
        class_map = np.ones((4, 4), dtype=np.uint8)
        clf       = _make_classification(class_map, conf_map, resolved_confidence_map=None)
        result    = self._estimator().estimate(clf, class_map)
        assert result.mask_confidence == pytest.approx(0.6, abs=1e-2)

    def test_nodata_pixels_excluded_from_mask_confidence(self) -> None:
        conf_map  = np.full((4, 4), 0.8, dtype=np.float32)
        class_map = np.full((4, 4), 255, dtype=np.uint8)
        clf       = _make_classification(class_map, conf_map)
        clf.nodata_mask[:] = True
        result    = self._estimator().estimate(clf, class_map)
        assert result.mask_confidence == pytest.approx(0.0)

    def test_zero_confidence_when_no_valid_pixels(self) -> None:
        conf_map  = np.zeros((4, 4), dtype=np.float32)
        class_map = np.full((4, 4), 255, dtype=np.uint8)
        clf       = _make_classification(class_map, conf_map)
        result    = self._estimator().estimate(clf, class_map)
        assert result.mask_confidence == pytest.approx(0.0)

    def test_agreement_score_between_0_and_1(self) -> None:
        conf_map  = np.full((4, 4), 0.7, dtype=np.float32)
        class_map = np.ones((4, 4), dtype=np.uint8)
        r1 = RuleResult(1, "water", conf_map.copy(), class_map.astype(bool),
                        ("MNDWI",), ())
        r2 = RuleResult(1, "water", conf_map.copy() * 0.5, class_map.astype(bool),
                        ("NDWI",), ())
        clf    = _make_classification(class_map, conf_map, rule_results=[r1, r2])
        result = self._estimator().estimate(clf, class_map)
        assert 0.0 <= result.agreement_score <= 1.0

    def test_pixel_confidence_shape_preserved(self) -> None:
        conf_map  = np.random.rand(6, 8).astype(np.float32)
        class_map = np.ones((6, 8), dtype=np.uint8)
        clf       = _make_classification(class_map, conf_map)
        result    = self._estimator().estimate(clf, class_map)
        assert result.pixel_confidence.shape == (6, 8)

    def test_component_scores_populated(self) -> None:
        conf_map  = np.full((4, 4), 0.7, dtype=np.float32)
        class_map = np.ones((4, 4), dtype=np.uint8)
        clf       = _make_classification(class_map, conf_map)
        result    = self._estimator().estimate(clf, class_map)
        assert "mask_confidence" in result.component_scores
        assert "agreement_score" in result.component_scores
        assert result.component_scores["mask_confidence"] == pytest.approx(0.7, abs=1e-2)


# """
# Unit tests for src/labels/confidence.py.

# Run:
#     pytest tests/labels/test_label_confidence.py -v \
#         --cov=src/labels/confidence --cov-report=term-missing
# """

# from __future__ import annotations

# import numpy as np
# import pytest

# from src.labels.confidence import ConfidenceConfig, ConfidenceEstimator
# from src.labels.contracts import ClassificationResult, RuleResult


# def _make_classification(
#     class_map: np.ndarray,
#     confidence_map: np.ndarray,
#     rule_results: list | None = None,
# ) -> ClassificationResult:
#     h, w = class_map.shape
#     return ClassificationResult(
#         class_map=class_map, confidence_map=confidence_map,
#         rule_results=rule_results or [],
#         unclassified_mask=(confidence_map == 0.0),
#         nodata_mask=np.zeros((h, w), dtype=bool),
#     )


# class TestConfidenceConfig:
#     def test_defaults(self) -> None:
#         cfg = ConfidenceConfig()
#         assert cfg.min_pixel_confidence == pytest.approx(0.20)
#         assert cfg.min_mask_confidence  == pytest.approx(0.30)

#     def test_frozen(self) -> None:
#         cfg = ConfidenceConfig()
#         with pytest.raises((AttributeError, TypeError)):
#             cfg.min_mask_confidence = 0.5  # type: ignore[misc]


# class TestConfidenceEstimator:
#     def _estimator(self, min_mask: float = 0.3) -> ConfidenceEstimator:
#         return ConfidenceEstimator(
#             ConfidenceConfig(min_pixel_confidence=0.2, min_mask_confidence=min_mask)
#         )

#     def test_mask_confidence_from_valid_pixels(self) -> None:
#         conf_map = np.full((4, 4), 0.8, dtype=np.float32)
#         class_map = np.ones((4, 4), dtype=np.uint8)
#         clf    = _make_classification(class_map, conf_map)
#         result = self._estimator().estimate(clf, class_map)
#         assert result.mask_confidence == pytest.approx(0.8, abs=1e-3)

#     def test_nodata_pixels_excluded_from_mask_confidence(self) -> None:
#         conf_map = np.full((4, 4), 0.8, dtype=np.float32)
#         class_map = np.full((4, 4), 255, dtype=np.uint8)  # all nodata
#         clf       = _make_classification(class_map, conf_map)
#         # Set nodata_mask = True everywhere
#         clf.nodata_mask[:] = True
#         result    = self._estimator().estimate(clf, class_map)
#         assert result.mask_confidence == pytest.approx(0.0)

#     def test_zero_confidence_when_no_valid_pixels(self) -> None:
#         conf_map  = np.zeros((4, 4), dtype=np.float32)
#         class_map = np.full((4, 4), 255, dtype=np.uint8)
#         clf       = _make_classification(class_map, conf_map)
#         result    = self._estimator().estimate(clf, class_map)
#         assert result.mask_confidence == pytest.approx(0.0)

#     def test_agreement_score_between_0_and_1(self) -> None:
#         conf_map  = np.full((4, 4), 0.7, dtype=np.float32)
#         class_map = np.ones((4, 4), dtype=np.uint8)
#         r1 = RuleResult(1, "water", conf_map.copy(), class_map.astype(bool),
#                         ("MNDWI",), ())
#         r2 = RuleResult(1, "water", conf_map.copy() * 0.5, class_map.astype(bool),
#                         ("NDWI",), ())
#         clf    = _make_classification(class_map, conf_map, rule_results=[r1, r2])
#         result = self._estimator().estimate(clf, class_map)
#         assert 0.0 <= result.agreement_score <= 1.0

#     def test_pixel_confidence_shape_preserved(self) -> None:
#         conf_map  = np.random.rand(6, 8).astype(np.float32)
#         class_map = np.ones((6, 8), dtype=np.uint8)
#         clf       = _make_classification(class_map, conf_map)
#         result    = self._estimator().estimate(clf, class_map)
#         assert result.pixel_confidence.shape == (6, 8)