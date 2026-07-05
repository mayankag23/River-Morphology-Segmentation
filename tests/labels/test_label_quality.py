"""
Unit tests for src/labels/quality.py.

Tests verify that:
- metric_scores dict is populated with component sub-scores.
- QualityResult.issues is a mutable list (not frozen).

Run:
    pytest tests/labels/test_label_quality.py -v \
        --cov=src/labels/quality --cov-report=term-missing
"""

from __future__ import annotations

import numpy as np
import pytest

from src.labels.contracts import MorphologyResult
from src.labels.quality import QualityAssessment, QualityConfig
from src.labels.schema import ClassDefinition, ClassSchema


def _schema() -> ClassSchema:
    return ClassSchema(classes=(
        ClassDefinition(0, "background", (128, 128, 128)),
        ClassDefinition(1, "water",      (0, 119, 190)),
        ClassDefinition(2, "sand",       (255, 200, 87)),
        ClassDefinition(3, "vegetation", (34, 139, 34)),
    ))


def _morph_result(class_map: np.ndarray) -> MorphologyResult:
    return MorphologyResult(class_map=class_map, operations_applied=["test"])


class TestQualityConfig:
    def test_defaults(self) -> None:
        cfg = QualityConfig()
        assert cfg.min_valid_pixel_ratio == pytest.approx(0.5)

    def test_frozen(self) -> None:
        cfg = QualityConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.min_quality_score = 0.9  # type: ignore[misc]


class TestQualityAssessment:
    def _assessor(self, **kwargs) -> QualityAssessment:
        cfg = QualityConfig(**kwargs)
        return QualityAssessment(cfg, _schema(), nodata_value=255)

    def test_all_valid_pixels_passes(self) -> None:
        qa   = self._assessor(min_valid_pixel_ratio=0.5, min_quality_score=0.0,
                              max_unclassified_ratio=0.5, min_class_pixels=1)
        cmap = np.ones((8, 8), dtype=np.uint8)
        res  = qa.assess(_morph_result(cmap))
        assert res.valid_pixel_ratio == pytest.approx(1.0)
        assert res.is_acceptable is True

    def test_all_nodata_fails(self) -> None:
        qa   = self._assessor(min_valid_pixel_ratio=0.5)
        cmap = np.full((8, 8), 255, dtype=np.uint8)
        res  = qa.assess(_morph_result(cmap))
        assert res.valid_pixel_ratio == pytest.approx(0.0)
        assert res.is_acceptable is False

    def test_quality_score_in_range(self) -> None:
        qa   = self._assessor()
        cmap = np.zeros((8, 8), dtype=np.uint8)
        cmap[:4, :] = 1
        cmap[4:, :] = 2
        res  = qa.assess(_morph_result(cmap))
        assert 0.0 <= res.quality_score <= 1.0

    def test_num_classes_present_correct(self) -> None:
        qa   = self._assessor(min_class_pixels=1)
        cmap = np.zeros((8, 8), dtype=np.uint8)
        cmap[:2, :] = 1
        cmap[2:4, :] = 2
        res  = qa.assess(_morph_result(cmap))
        assert res.num_classes_present == 3  # background + water + sand

    def test_excessive_unclassified_fails(self) -> None:
        qa   = self._assessor(max_unclassified_ratio=0.1)
        cmap = np.full((8, 8), 255, dtype=np.uint8)
        res  = qa.assess(_morph_result(cmap))
        assert res.is_acceptable is False
        assert any("unclassified" in i for i in res.issues)

    def test_quality_result_issues_is_mutable_list(self) -> None:
        """QualityResult.issues must be a list, not a tuple (test invariant)."""
        qa   = self._assessor()
        cmap = np.ones((4, 4), dtype=np.uint8)
        res  = qa.assess(_morph_result(cmap))
        assert isinstance(res.issues, list)

    def test_metric_scores_populated(self) -> None:
        qa   = self._assessor(min_valid_pixel_ratio=0.5, min_quality_score=0.0,
                              max_unclassified_ratio=0.5, min_class_pixels=1)
        cmap = np.ones((8, 8), dtype=np.uint8)
        res  = qa.assess(_morph_result(cmap))
        assert "valid_pixel_score" in res.metric_scores
        assert "unclassified_score" in res.metric_scores
        assert "class_coverage_score" in res.metric_scores
        for key, val in res.metric_scores.items():
            assert 0.0 <= val <= 1.0, f"metric_scores['{key}'] = {val} out of range"

    def test_metric_scores_consistent_with_quality_score(self) -> None:
        qa   = self._assessor(min_valid_pixel_ratio=0.5, min_quality_score=0.0,
                              max_unclassified_ratio=0.5, min_class_pixels=1)
        cmap = np.ones((8, 8), dtype=np.uint8)
        res  = qa.assess(_morph_result(cmap))
        expected = (
            0.5 * res.metric_scores["valid_pixel_score"]
            + 0.3 * res.metric_scores["unclassified_score"]
            + 0.2 * res.metric_scores["class_coverage_score"]
        )
        assert res.quality_score == pytest.approx(min(expected, 1.0), abs=1e-3)


# """
# Unit tests for src/labels/quality.py.

# Run:
#     pytest tests/labels/test_label_quality.py -v \
#         --cov=src/labels/quality --cov-report=term-missing
# """

# from __future__ import annotations

# import numpy as np
# import pytest

# from src.labels.contracts import MorphologyResult
# from src.labels.quality import QualityAssessment, QualityConfig
# from src.labels.schema import ClassDefinition, ClassSchema


# def _schema() -> ClassSchema:
#     return ClassSchema(classes=(
#         ClassDefinition(0, "background", (128, 128, 128)),
#         ClassDefinition(1, "water",      (0, 119, 190)),
#         ClassDefinition(2, "sand",       (255, 200, 87)),
#         ClassDefinition(3, "vegetation", (34, 139, 34)),
#     ))


# def _morph_result(class_map: np.ndarray) -> MorphologyResult:
#     return MorphologyResult(class_map=class_map, operations_applied=["test"])


# class TestQualityConfig:
#     def test_defaults(self) -> None:
#         cfg = QualityConfig()
#         assert cfg.min_valid_pixel_ratio == pytest.approx(0.5)

#     def test_frozen(self) -> None:
#         cfg = QualityConfig()
#         with pytest.raises((AttributeError, TypeError)):
#             cfg.min_quality_score = 0.9  # type: ignore[misc]


# class TestQualityAssessment:
#     def _assessor(self, **kwargs) -> QualityAssessment:
#         cfg = QualityConfig(**kwargs)
#         return QualityAssessment(cfg, _schema(), nodata_value=255)

#     def test_all_valid_pixels_passes(self) -> None:
#         qa    = self._assessor(min_valid_pixel_ratio=0.5, min_quality_score=0.0,
#                                max_unclassified_ratio=0.5, min_class_pixels=1)
#         cmap  = np.ones((8, 8), dtype=np.uint8)  # all water
#         res   = qa.assess(_morph_result(cmap))
#         assert res.valid_pixel_ratio == pytest.approx(1.0)
#         assert res.is_acceptable is True

#     def test_all_nodata_fails(self) -> None:
#         qa    = self._assessor(min_valid_pixel_ratio=0.5)
#         cmap  = np.full((8, 8), 255, dtype=np.uint8)
#         res   = qa.assess(_morph_result(cmap))
#         assert res.valid_pixel_ratio == pytest.approx(0.0)
#         assert res.is_acceptable is False

#     def test_quality_score_in_range(self) -> None:
#         qa   = self._assessor()
#         cmap = np.zeros((8, 8), dtype=np.uint8)
#         cmap[:4, :] = 1; cmap[4:, :] = 2
#         res  = qa.assess(_morph_result(cmap))
#         assert 0.0 <= res.quality_score <= 1.0

#     def test_num_classes_present_correct(self) -> None:
#         qa   = self._assessor(min_class_pixels=1)
#         cmap = np.zeros((8, 8), dtype=np.uint8)
#         cmap[:2, :] = 1  # water
#         cmap[2:4, :] = 2  # sand
#         res  = qa.assess(_morph_result(cmap))
#         assert res.num_classes_present == 3  # background + water + sand

#     def test_excessive_unclassified_fails(self) -> None:
#         qa   = self._assessor(max_unclassified_ratio=0.1)
#         cmap = np.full((8, 8), 255, dtype=np.uint8)  # all unclassified
#         res  = qa.assess(_morph_result(cmap))
#         assert res.is_acceptable is False
#         assert any("unclassified" in i for i in res.issues)

#     def test_quality_result_is_not_frozen(self) -> None:
#         qa   = self._assessor()
#         cmap = np.ones((4, 4), dtype=np.uint8)
#         res  = qa.assess(_morph_result(cmap))
#         assert isinstance(res.issues, list)