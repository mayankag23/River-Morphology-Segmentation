"""Tests for src/training/evaluation/metrics.py"""
from __future__ import annotations
import numpy as np
import pytest
from src.training.evaluation.metrics import (
    MetricRegistry, compute_all_metrics,
    _pixel_accuracy, _mean_pixel_accuracy, _precision, _recall,
    _f1, _dice, _iou, _fw_iou, _kappa, _balanced_accuracy,
)


def _perfect_cm(n: int = 4) -> np.ndarray:
    """Diagonal confusion matrix — perfect predictions."""
    cm = np.zeros((n, n), dtype=np.int64)
    np.fill_diagonal(cm, 100)
    return cm


def _zero_cm(n: int = 4) -> np.ndarray:
    return np.zeros((n, n), dtype=np.int64)


def _random_cm(n: int = 4, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 50, (n, n), dtype=np.int64)


class TestMetricRegistry:
    def test_all_builtins_registered(self):
        names = MetricRegistry.registered_names()
        expected = {"pixel_accuracy", "mean_pixel_accuracy", "precision", "recall",
                    "f1", "dice", "iou", "mean_iou", "fw_iou", "kappa", "balanced_accuracy"}
        assert expected.issubset(set(names))

    def test_register_custom(self):
        @MetricRegistry.register("_test_custom_metric")
        def _my(cm): return {"_test_custom_metric": 0.42}
        assert "_test_custom_metric" in MetricRegistry.registered_names()
        result = MetricRegistry.get("_test_custom_metric")(_zero_cm())
        assert result["_test_custom_metric"] == pytest.approx(0.42)
        # cleanup
        del MetricRegistry._registry["_test_custom_metric"]

    def test_unknown_metric_raises(self):
        with pytest.raises(KeyError):
            MetricRegistry.get("nonexistent_xyz")


class TestPixelAccuracy:
    def test_perfect_is_one(self):
        r = _pixel_accuracy(_perfect_cm())
        assert r["pixel_accuracy"] == pytest.approx(1.0)

    def test_all_wrong_is_zero(self):
        cm = np.zeros((2, 2), dtype=np.int64)
        cm[0, 1] = 50; cm[1, 0] = 50   # all misclassified
        r = _pixel_accuracy(cm)
        assert r["pixel_accuracy"] == pytest.approx(0.0)

    def test_zero_matrix_returns_zero(self):
        assert _pixel_accuracy(_zero_cm())["pixel_accuracy"] == 0.0

    def test_value_in_0_1(self):
        r = _pixel_accuracy(_random_cm())
        assert 0.0 <= r["pixel_accuracy"] <= 1.0


class TestMeanPixelAccuracy:
    def test_perfect_is_one(self):
        r = _mean_pixel_accuracy(_perfect_cm())
        assert r["mean_pixel_accuracy"] == pytest.approx(1.0)

    def test_zero_matrix_returns_zero(self):
        assert _mean_pixel_accuracy(_zero_cm())["mean_pixel_accuracy"] == 0.0

    def test_returns_per_class_array(self):
        r = _mean_pixel_accuracy(_perfect_cm(4))
        assert "_per_class_pixel_accuracy" in r
        assert len(r["_per_class_pixel_accuracy"]) == 4


class TestPrecision:
    def test_perfect_is_one(self):
        r = _precision(_perfect_cm())
        assert r["mean_precision"] == pytest.approx(1.0)

    def test_zero_matrix_returns_zero(self):
        assert _precision(_zero_cm())["mean_precision"] == 0.0

    def test_no_nan(self):
        r = _precision(_random_cm())
        assert not np.isnan(r["mean_precision"])


class TestRecall:
    def test_perfect_is_one(self):
        r = _recall(_perfect_cm())
        assert r["mean_recall"] == pytest.approx(1.0)

    def test_zero_matrix_returns_zero(self):
        assert _recall(_zero_cm())["mean_recall"] == 0.0

    def test_no_nan(self):
        r = _recall(_random_cm())
        assert not np.isnan(r["mean_recall"])


class TestF1:
    def test_perfect_is_one(self):
        r = _f1(_perfect_cm())
        assert r["mean_f1"] == pytest.approx(1.0)

    def test_zero_matrix_returns_zero(self):
        assert _f1(_zero_cm())["mean_f1"] == 0.0

    def test_f1_equals_harmonic_mean_of_p_r(self):
        """With perfect predictions F1 == 1.0."""
        cm = _perfect_cm(2)
        r  = _f1(cm)
        assert r["mean_f1"] == pytest.approx(1.0)


class TestDice:
    def test_perfect_is_one(self):
        assert _dice(_perfect_cm())["mean_dice"] == pytest.approx(1.0)

    def test_dice_equals_f1_numerically(self):
        """Dice and F1 are algebraically equivalent for segmentation."""
        cm    = _random_cm(4)
        d_val = _dice(cm)["mean_dice"]
        f_val = _f1(cm)["mean_f1"]
        assert d_val == pytest.approx(f_val, rel=1e-5)

    def test_zero_returns_zero(self):
        assert _dice(_zero_cm())["mean_dice"] == 0.0


class TestIoU:
    def test_perfect_is_one(self):
        assert _iou(_perfect_cm())["mean_iou"] == pytest.approx(1.0)

    def test_zero_returns_zero(self):
        assert _iou(_zero_cm())["mean_iou"] == 0.0

    def test_iou_leq_f1(self):
        """IoU is always <= F1 (tighter bound)."""
        cm    = _random_cm()
        iou_v = _iou(cm)["mean_iou"]
        f1_v  = _f1(cm)["mean_f1"]
        assert iou_v <= f1_v + 1e-6


class TestFwIoU:
    def test_perfect_is_one(self):
        assert _fw_iou(_perfect_cm())["fw_iou"] == pytest.approx(1.0)

    def test_zero_returns_zero(self):
        assert _fw_iou(_zero_cm())["fw_iou"] == 0.0

    def test_value_in_0_1(self):
        v = _fw_iou(_random_cm())["fw_iou"]
        assert 0.0 <= v <= 1.0 + 1e-6


class TestKappa:
    def test_perfect_is_one(self):
        assert _kappa(_perfect_cm())["kappa"] == pytest.approx(1.0)

    def test_zero_matrix_returns_zero(self):
        assert _kappa(_zero_cm())["kappa"] == 0.0

    def test_value_in_neg1_to_1(self):
        v = _kappa(_random_cm())["kappa"]
        assert -1.0 <= v <= 1.0 + 1e-6


class TestBalancedAccuracy:
    def test_perfect_is_one(self):
        assert _balanced_accuracy(_perfect_cm())["balanced_accuracy"] == pytest.approx(1.0)

    def test_equals_mean_recall(self):
        cm    = _random_cm()
        ba    = _balanced_accuracy(cm)["balanced_accuracy"]
        mr    = _recall(cm)["mean_recall"]
        assert ba == pytest.approx(mr)


class TestComputeAllMetrics:
    def test_returns_all_aggregate_keys(self):
        r = compute_all_metrics(_perfect_cm())
        assert "pixel_accuracy"      in r
        assert "mean_iou"            in r
        assert "mean_dice"           in r
        assert "kappa"               in r
        assert "balanced_accuracy"   in r

    def test_subset_when_names_specified(self):
        r = compute_all_metrics(_perfect_cm(), ("pixel_accuracy",))
        assert "pixel_accuracy" in r

    def test_no_nan_in_perfect_predictions(self):
        r = compute_all_metrics(_perfect_cm())
        for k, v in r.items():
            if isinstance(v, float):
                assert not np.isnan(v), f"{k} is NaN"

    def test_no_nan_in_zero_matrix(self):
        r = compute_all_metrics(_zero_cm())
        for k, v in r.items():
            if isinstance(v, float):
                assert not np.isnan(v), f"{k} is NaN"

    def test_no_nan_in_random_matrix(self):
        r = compute_all_metrics(_random_cm())
        for k, v in r.items():
            if isinstance(v, float):
                assert not np.isnan(v), f"{k} is NaN"

    def test_all_background_mask(self):
        """All pixels are class 0 — other classes have zero support."""
        cm = np.zeros((4, 4), dtype=np.int64)
        cm[0, 0] = 1000   # all predicted correctly as class 0
        r  = compute_all_metrics(cm)
        assert not any(np.isnan(v) for v in r.values() if isinstance(v, float))

    def test_single_class_predictions(self):
        """Model always predicts class 1 regardless of truth."""
        cm = np.zeros((4, 4), dtype=np.int64)
        cm[:, 1] = 100    # all predicted as class 1
        np.fill_diagonal(cm, 0)
        cm[1, 1] = 100
        r = compute_all_metrics(cm)
        assert not any(np.isnan(v) for v in r.values() if isinstance(v, float))
