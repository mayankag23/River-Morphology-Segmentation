"""Tests for src/training/evaluation/contracts.py"""
from __future__ import annotations
import pytest
from src.training.evaluation.contracts import (
    ClassMetrics, ConfusionMatrix, EvaluationConfig,
    EvaluationResult, PredictionStatistics,
)


class TestEvaluationConfig:
    def test_frozen(self):
        cfg = EvaluationConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.split = "train"  # type: ignore[misc]

    def test_defaults(self):
        cfg = EvaluationConfig()
        assert cfg.split == "test"
        assert cfg.batch_size == 8
        assert cfg.ignore_index == 255

    def test_from_config_reads_values(self):
        class _Eval:
            split="validation"; batch_size=16; num_workers=2; device="cpu"
            ignore_index=255; output_dir=""; save_json=True; save_csv=False
            metrics=("pixel_accuracy","mean_iou"); pin_memory=False
        class _Cfg:
            evaluation = _Eval()
        cfg = EvaluationConfig.from_config(_Cfg())
        assert cfg.split == "validation"
        assert cfg.batch_size == 16
        assert "pixel_accuracy" in cfg.metrics

    def test_from_config_no_section_returns_defaults(self):
        class _Cfg: pass
        assert EvaluationConfig.from_config(_Cfg()) == EvaluationConfig()


class TestClassMetrics:
    def test_frozen(self):
        cm = ClassMetrics(class_id=0, class_name="bg")
        with pytest.raises((AttributeError, TypeError)):
            cm.iou = 0.9  # type: ignore[misc]

    def test_as_dict_has_all_keys(self):
        cm = ClassMetrics(class_id=1, class_name="water", precision=0.9, recall=0.8,
                          f1=0.85, dice=0.85, iou=0.74, pixel_accuracy=0.92,
                          tp=100, fp=10, fn=20, tn=1000, num_pixels=120, num_predicted=110, support=120)
        d = cm.as_dict()
        assert "precision" in d and "iou" in d and "tp" in d

    def test_defaults_are_zero(self):
        cm = ClassMetrics(class_id=0, class_name="bg")
        assert cm.precision == 0.0
        assert cm.iou == 0.0


class TestConfusionMatrix:
    def _make(self):
        return ConfusionMatrix(
            matrix=((90, 10), (5, 95)),
            normalized=((0.9, 0.1), (0.05, 0.95)),
            class_names=("bg", "water"),
            num_classes=2,
            total_pixels=200,
        )

    def test_frozen(self):
        cm = self._make()
        with pytest.raises((AttributeError, TypeError)):
            cm.total_pixels = 999  # type: ignore[misc]

    def test_as_dict_serialisable(self):
        import json
        d = self._make().as_dict()
        assert json.dumps(d)  # must not raise


class TestPredictionStatistics:
    def test_frozen(self):
        ps = PredictionStatistics(
            total_pixels=100, total_samples=5,
            class_pixel_counts={"bg": 80, "water": 20},
            pred_pixel_counts={"bg": 78, "water": 22},
            class_frequencies={"bg": 0.8, "water": 0.2},
            pred_frequencies={"bg": 0.78, "water": 0.22},
        )
        with pytest.raises((AttributeError, TypeError)):
            ps.total_pixels = 999  # type: ignore[misc]

    def test_as_dict(self):
        ps = PredictionStatistics(
            total_pixels=100, total_samples=5,
            class_pixel_counts={"bg": 80},
            pred_pixel_counts={"bg": 78},
            class_frequencies={"bg": 0.8},
            pred_frequencies={"bg": 0.78},
        )
        assert "total_pixels" in ps.as_dict()


class TestEvaluationResultSummary:
    def _make(self):
        cm = ConfusionMatrix(
            matrix=((90, 10), (5, 95)),
            normalized=((0.9, 0.1), (0.05, 0.95)),
            class_names=("bg", "water"), num_classes=2, total_pixels=200,
        )
        ps = PredictionStatistics(
            total_pixels=200, total_samples=10,
            class_pixel_counts={}, pred_pixel_counts={},
            class_frequencies={}, pred_frequencies={},
        )
        return EvaluationResult(
            pixel_accuracy=0.9, mean_pixel_accuracy=0.9, mean_iou=0.8, fw_iou=0.85,
            mean_dice=0.82, mean_precision=0.88, mean_recall=0.87, mean_f1=0.87,
            kappa=0.78, balanced_accuracy=0.87, per_class={},
            confusion_matrix=cm, statistics=ps, split="test", architecture="unetplusplus",
            num_classes=2, ignore_index=255, total_samples=10, total_pixels=200,
            evaluation_time_s=5.2, operations_log=("a", "b"), class_names=("bg", "water"),
        )

    def test_frozen(self):
        r = self._make()
        with pytest.raises((AttributeError, TypeError)):
            r.mean_iou = 0.99  # type: ignore[misc]

    def test_summary_lines_ascii(self):
        lines = self._make().summary_lines()
        assert len(lines) > 0
        assert all(ord(c) < 128 for l in lines for c in l)

    def test_as_dict_json_serialisable(self):
        import json
        d = self._make().as_dict()
        assert json.dumps(d)  # must not raise
