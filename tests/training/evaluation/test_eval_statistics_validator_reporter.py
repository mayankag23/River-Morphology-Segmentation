"""Tests for statistics.py, validator.py, and reporter.py"""
from __future__ import annotations
import csv
import json
import numpy as np
import pytest
from pathlib import Path

from src.training.evaluation.statistics import PredictionStatisticsAccumulator
from src.training.evaluation.validator import EvaluationValidator, EvaluationValidationResult
from src.training.evaluation.contracts import (
    ClassMetrics, ConfusionMatrix, EvaluationConfig, EvaluationResult,
    PredictionStatistics,
)
from src.training.evaluation.reporter import EvaluationReporter


# ==============================================================================
# PredictionStatisticsAccumulator
# ==============================================================================

class TestPredictionStatisticsAccumulator:
    def _cm(self) -> np.ndarray:
        cm = np.zeros((2, 2), dtype=np.int64)
        cm[0, 0] = 80;  cm[0, 1] = 20   # class 0: 100 GT, 80 correct
        cm[1, 0] = 10;  cm[1, 1] = 90   # class 1: 100 GT, 90 correct
        return cm

    def test_total_pixels(self):
        acc = PredictionStatisticsAccumulator(("bg", "water"))
        ps  = acc.compute(self._cm())
        assert ps.total_pixels == 200

    def test_class_pixel_counts(self):
        acc = PredictionStatisticsAccumulator(("bg", "water"))
        ps  = acc.compute(self._cm())
        assert ps.class_pixel_counts["bg"]    == 100
        assert ps.class_pixel_counts["water"] == 100

    def test_pred_pixel_counts(self):
        acc = PredictionStatisticsAccumulator(("bg", "water"))
        ps  = acc.compute(self._cm())
        # col 0: bg predicted = 80+10 = 90; col 1: water predicted = 20+90 = 110
        assert ps.pred_pixel_counts["bg"]    == 90
        assert ps.pred_pixel_counts["water"] == 110

    def test_class_frequencies_sum_to_one(self):
        acc = PredictionStatisticsAccumulator(("bg", "water"))
        ps  = acc.compute(self._cm())
        total_freq = sum(ps.class_frequencies.values())
        assert total_freq == pytest.approx(1.0)

    def test_sample_counter(self):
        acc = PredictionStatisticsAccumulator(("bg",))
        acc.increment_samples(5)
        acc.increment_samples(3)
        cm = np.array([[100]], dtype=np.int64)
        ps = acc.compute(cm)
        assert ps.total_samples == 8

    def test_empty_matrix_no_crash(self):
        acc = PredictionStatisticsAccumulator(("bg",))
        cm  = np.zeros((1, 1), dtype=np.int64)
        ps  = acc.compute(cm)
        assert ps.total_pixels == 0


# ==============================================================================
# EvaluationValidator
# ==============================================================================

class TestEvaluationValidator:
    def _validator(self):
        return EvaluationValidator()

    def test_valid_batch_passes(self):
        v      = self._validator()
        preds  = np.array([0, 1, 2, 3])
        tgts   = np.array([0, 1, 2, 3])
        result = v.validate_batch(preds, tgts, num_classes=4, ignore_index=255)
        assert result.is_valid

    def test_shape_mismatch_detected(self):
        v = self._validator()
        r = v.validate_batch(np.array([0, 1]), np.array([0]), num_classes=4, ignore_index=255)
        assert not r.is_valid
        assert any("shape" in i for i in r.issues)

    def test_invalid_class_id_detected(self):
        v = self._validator()
        r = v.validate_batch(
            np.array([0, 1]), np.array([0, 99]),
            num_classes=4, ignore_index=255,
        )
        assert not r.is_valid

    def test_ignore_index_not_flagged_as_invalid(self):
        v = self._validator()
        r = v.validate_batch(
            np.array([0, 1]), np.array([0, 255]),
            num_classes=4, ignore_index=255,
        )
        assert r.is_valid

    def test_nan_in_predictions_detected(self):
        v     = self._validator()
        preds = np.array([0.0, float("nan")])
        tgts  = np.array([0.0, 1.0])
        r     = v.validate_batch(preds, tgts, num_classes=4, ignore_index=255)
        assert not r.is_valid

    def test_valid_config(self):
        import types
        v          = self._validator()
        cfg        = EvaluationConfig(split="test", batch_size=8)
        model      = types.SimpleNamespace()
        data       = types.SimpleNamespace(num_classes=4, test_dataset=[], validation_dataset=[], train_dataset=[])
        result     = v.validate_config(cfg, model, data)
        assert result.is_valid

    def test_invalid_split_detected(self):
        import types
        v    = self._validator()
        cfg  = EvaluationConfig(split="holdout")
        data = types.SimpleNamespace(num_classes=4, test_dataset=[])
        r    = v.validate_config(cfg, types.SimpleNamespace(), data)
        assert not r.is_valid

    def test_none_model_detected(self):
        import types
        v    = self._validator()
        cfg  = EvaluationConfig()
        data = types.SimpleNamespace(num_classes=4, test_dataset=[])
        r    = v.validate_config(cfg, None, data)
        assert not r.is_valid

    def test_validation_result_issues_is_copy(self):
        r = EvaluationValidationResult(["a"])
        r.issues.append("b")
        assert len(r.issues) == 1


# ==============================================================================
# EvaluationReporter
# ==============================================================================

def _make_result(split: str = "test") -> EvaluationResult:
    cm = ConfusionMatrix(
        matrix=((90, 10), (5, 95)),
        normalized=((0.9, 0.1), (0.05, 0.95)),
        class_names=("bg", "water"), num_classes=2, total_pixels=200,
    )
    ps = PredictionStatistics(
        total_pixels=200, total_samples=10,
        class_pixel_counts={"bg": 100, "water": 100},
        pred_pixel_counts={"bg": 95, "water": 105},
        class_frequencies={"bg": 0.5, "water": 0.5},
        pred_frequencies={"bg": 0.475, "water": 0.525},
    )
    per_class = {
        "bg":    ClassMetrics(class_id=0, class_name="bg",    precision=0.9, recall=0.9, f1=0.9, dice=0.9, iou=0.8, pixel_accuracy=0.9, tp=90, fp=5, fn=10, tn=95, num_pixels=100, num_predicted=95, support=100),
        "water": ClassMetrics(class_id=1, class_name="water", precision=0.95, recall=0.9, f1=0.92, dice=0.92, iou=0.86, pixel_accuracy=0.95, tp=95, fp=10, fn=5, tn=90, num_pixels=100, num_predicted=105, support=100),
    }
    return EvaluationResult(
        pixel_accuracy=0.925, mean_pixel_accuracy=0.925, mean_iou=0.83, fw_iou=0.83,
        mean_dice=0.91, mean_precision=0.925, mean_recall=0.9, mean_f1=0.91,
        kappa=0.85, balanced_accuracy=0.9, per_class=per_class,
        confusion_matrix=cm, statistics=ps, split=split, architecture="unetplusplus",
        num_classes=2, ignore_index=255, total_samples=10, total_pixels=200,
        evaluation_time_s=3.5, operations_log=("step1",), class_names=("bg", "water"),
    )


class TestEvaluationReporter:
    def test_save_json_creates_file(self, tmp_path):
        r        = EvaluationReporter(tmp_path)
        result   = _make_result()
        path     = r.save_json(result)
        assert path.exists()

    def test_save_json_valid_json(self, tmp_path):
        r      = EvaluationReporter(tmp_path)
        path   = r.save_json(_make_result())
        with open(path) as f:
            data = json.load(f)
        assert "mean_iou" in data
        assert "per_class" in data

    def test_save_csv_creates_file(self, tmp_path):
        r    = EvaluationReporter(tmp_path)
        path = r.save_csv(_make_result())
        assert path.exists()

    def test_save_csv_has_correct_columns(self, tmp_path):
        r    = EvaluationReporter(tmp_path)
        path = r.save_csv(_make_result())
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
        assert "iou" in fieldnames
        assert "class_name" in fieldnames
        assert "precision" in fieldnames

    def test_save_csv_rows_match_classes(self, tmp_path):
        r    = EvaluationReporter(tmp_path)
        path = r.save_csv(_make_result())
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2   # bg and water

    def test_save_all_returns_paths(self, tmp_path):
        r     = EvaluationReporter(tmp_path)
        paths = r.save_all(_make_result())
        assert "json" in paths and "csv" in paths

    def test_creates_output_dir(self, tmp_path):
        new_dir = tmp_path / "nested" / "reports"
        r       = EvaluationReporter(new_dir)
        r.save_json(_make_result())
        assert new_dir.exists()
