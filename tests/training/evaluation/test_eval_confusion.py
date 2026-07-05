"""Tests for src/training/evaluation/confusion.py"""
from __future__ import annotations
import numpy as np
import pytest
from src.training.evaluation.confusion import ConfusionMatrixAccumulator


def _cm(n=4, ignore=255):
    return ConfusionMatrixAccumulator(num_classes=n, ignore_index=ignore,
                                      class_names=tuple(f"c{i}" for i in range(n)))


class TestConfusionMatrixAccumulator:
    def test_invalid_num_classes_raises(self):
        with pytest.raises(ValueError):
            ConfusionMatrixAccumulator(num_classes=0)

    def test_empty_matrix_is_zeros(self):
        cm = _cm()
        assert cm.matrix.sum() == 0

    def test_perfect_predictions_fills_diagonal(self):
        cm = _cm(n=2)
        preds   = np.array([0, 0, 1, 1])
        targets = np.array([0, 0, 1, 1])
        cm.update(preds, targets)
        mat = cm.matrix
        assert mat[0, 0] == 2
        assert mat[1, 1] == 2
        assert mat[0, 1] == 0
        assert mat[1, 0] == 0

    def test_wrong_predictions_fill_off_diagonal(self):
        cm = _cm(n=2)
        preds   = np.array([1, 1])
        targets = np.array([0, 0])
        cm.update(preds, targets)
        assert cm.matrix[0, 1] == 2   # true=0, pred=1
        assert cm.matrix[0, 0] == 0

    def test_ignore_index_excluded(self):
        cm = _cm(n=2, ignore=255)
        preds   = np.array([0, 1, 255])
        targets = np.array([0, 1, 255])
        cm.update(preds, targets)
        assert cm.total_pixels == 2   # 255 excluded

    def test_ignore_index_in_targets_excludes_pixel(self):
        cm = _cm(n=2, ignore=255)
        preds   = np.array([0, 1])
        targets = np.array([0, 255])   # second pixel is ignored
        cm.update(preds, targets)
        assert cm.total_pixels == 1

    def test_reset_clears_matrix(self):
        cm = _cm(n=2)
        cm.update(np.array([0]), np.array([0]))
        cm.reset()
        assert cm.total_pixels == 0

    def test_multiple_updates_accumulate(self):
        cm = _cm(n=2)
        cm.update(np.array([0, 0]), np.array([0, 0]))
        cm.update(np.array([1, 1]), np.array([1, 1]))
        assert cm.total_pixels == 4

    def test_compute_returns_confusion_matrix_object(self):
        from src.training.evaluation.contracts import ConfusionMatrix
        cm = _cm(n=2)
        cm.update(np.array([0, 1]), np.array([0, 1]))
        result = cm.compute()
        assert isinstance(result, ConfusionMatrix)

    def test_compute_normalised_rows_sum_to_one(self):
        cm = _cm(n=3)
        preds   = np.array([0, 0, 1, 1, 2, 2])
        targets = np.array([0, 0, 1, 1, 2, 2])
        cm.update(preds, targets)
        obj = cm.compute()
        for row in obj.normalized:
            assert abs(sum(row) - 1.0) < 1e-6

    def test_per_class_counts_tp_correct(self):
        cm = _cm(n=2)
        cm.update(np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1]))
        counts = cm.per_class_counts()
        assert counts["c0"]["tp"] == 2
        assert counts["c1"]["tp"] == 2
        assert counts["c0"]["fp"] == 0
        assert counts["c0"]["fn"] == 0

    def test_per_class_counts_fp_fn_correct(self):
        cm = _cm(n=2)
        # true=0, pred=1 -> contributes fp to class 1, fn to class 0
        cm.update(np.array([1]), np.array([0]))
        counts = cm.per_class_counts()
        assert counts["c0"]["fn"] == 1
        assert counts["c1"]["fp"] == 1

    def test_2d_input_flattened(self):
        cm = _cm(n=2)
        preds   = np.array([[0, 1], [1, 0]])
        targets = np.array([[0, 1], [1, 0]])
        cm.update(preds, targets)
        assert cm.total_pixels == 4

    def test_out_of_range_predictions_excluded(self):
        """Predictions >= num_classes must be silently excluded."""
        cm = _cm(n=2)
        preds   = np.array([0, 1, 99])  # 99 is out of range
        targets = np.array([0, 1, 0])
        cm.update(preds, targets)
        assert cm.total_pixels == 2   # 99 excluded
