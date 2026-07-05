"""
Integration smoke tests for evaluator.py and engine.py.

Uses a tiny synthetic model and synthetic DataLoader — no GeoTIFF, no torch
DataLoader, no rasterio. Tests that the full evaluation pipeline produces
valid, non-NaN EvaluationResult objects.
"""
from __future__ import annotations
import types
import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.evaluation.contracts import EvaluationConfig, EvaluationResult
from src.training.evaluation.engine import EvaluationEngine
from src.training.evaluation.evaluator import Evaluator


# ==============================================================================
# Tiny test model and dataset helpers
# ==============================================================================

def _tiny_model(in_ch: int = 4, n_cls: int = 4) -> nn.Module:
    class _M(nn.Module):
        def __init__(self): super().__init__(); self.conv = nn.Conv2d(in_ch, n_cls, 1)
        def forward(self, x): return self.conv(x)
    return _M().eval()


def _tiny_dataloader(
    n_samples: int = 8,
    in_ch:     int = 4,
    n_cls:     int = 4,
    h: int = 8,
    w: int = 8,
) -> DataLoader:
    images = torch.randn(n_samples, in_ch, h, w)
    masks  = torch.randint(0, n_cls, (n_samples, h, w), dtype=torch.long)
    return DataLoader(TensorDataset(images, masks), batch_size=4, shuffle=False)


def _tiny_training_result(in_ch: int = 4, n_cls: int = 4) -> object:
    return types.SimpleNamespace(
        model=_tiny_model(in_ch, n_cls),
        architecture="unetplusplus",
        num_parameters=100,
        in_channels=in_ch,
        num_classes=n_cls,
    )


def _tiny_data_result(n_cls: int = 4, in_ch: int = 4, n: int = 8) -> object:
    images = torch.randn(n, in_ch, 8, 8)
    masks  = torch.randint(0, n_cls, (n, 8, 8), dtype=torch.long)
    ds     = TensorDataset(images, masks)
    return types.SimpleNamespace(
        train_dataset=ds, validation_dataset=ds, test_dataset=ds,
        num_classes=n_cls, num_bands=in_ch, is_valid=True,
        num_train_samples=n, num_val_samples=n, num_test_samples=n,
    )


CLASS_NAMES = ("background", "water", "sand", "vegetation")


# ==============================================================================
# Evaluator tests
# ==============================================================================

class TestEvaluator:
    def _evaluator(self, n_cls=4):
        cfg = EvaluationConfig(batch_size=4, ignore_index=255, device="cpu")
        return Evaluator(
            config=cfg, num_classes=n_cls,
            class_names=CLASS_NAMES[:n_cls], device=torch.device("cpu"),
        )

    def test_run_returns_dict_with_required_keys(self):
        ev     = self._evaluator()
        loader = _tiny_dataloader()
        model  = _tiny_model()
        result = ev.run(model, loader)
        assert "cm_accumulator"    in result
        assert "stats_accumulator" in result
        assert "per_class"         in result
        assert "aggregate"         in result

    def test_per_class_has_all_classes(self):
        ev     = self._evaluator(n_cls=4)
        result = ev.run(_tiny_model(), _tiny_dataloader())
        assert set(result["per_class"].keys()) == set(CLASS_NAMES)

    def test_no_nan_in_aggregate_metrics(self):
        ev     = self._evaluator()
        result = ev.run(_tiny_model(), _tiny_dataloader())
        for k, v in result["aggregate"].items():
            if isinstance(v, float):
                assert not np.isnan(v), f"NaN in {k}"

    def test_total_pixels_positive(self):
        ev     = self._evaluator()
        result = ev.run(_tiny_model(), _tiny_dataloader())
        assert result["cm_accumulator"].total_pixels > 0

    def test_perfect_model_high_accuracy(self):
        """A model that always outputs class 0 on an all-class-0 dataset."""
        class _ZeroModel(nn.Module):
            def __init__(self): super().__init__()
            def forward(self, x):
                B, C, H, W = x.shape[0], 4, x.shape[2], x.shape[3]
                logits = torch.full((B, C, H, W), -100.0)
                logits[:, 0] = 100.0   # always class 0
                return logits

        images = torch.randn(4, 4, 8, 8)
        masks  = torch.zeros(4, 8, 8, dtype=torch.long)   # all class 0
        loader = DataLoader(TensorDataset(images, masks), batch_size=4)

        cfg = EvaluationConfig(batch_size=4, ignore_index=255, device="cpu")
        ev  = Evaluator(config=cfg, num_classes=4, class_names=CLASS_NAMES, device=torch.device("cpu"))
        result = ev.run(_ZeroModel(), loader)
        pa = result["aggregate"].get("pixel_accuracy", 0.0)
        assert pa == pytest.approx(1.0)


# ==============================================================================
# EvaluationEngine tests
# ==============================================================================

class TestEvaluationEngine:
    # def _engine(self, **kw):
    #     cfg = EvaluationConfig(split="test", batch_size=4, device="cpu", **kw)
    #     return EvaluationEngine(cfg)
    def _engine(self, **kw):
        cfg = EvaluationConfig(
            split=kw.pop("split", "test"),
            batch_size=kw.pop("batch_size", 4),
            device=kw.pop("device", "cpu"),
            **kw,
        )
        return EvaluationEngine(cfg)

    def test_evaluate_returns_evaluation_result(self):
        engine  = self._engine()
        result  = engine.evaluate(
            _tiny_training_result(), _tiny_data_result(),
            class_names=CLASS_NAMES, num_classes=4,
        )
        assert isinstance(result, EvaluationResult)

    def test_result_is_frozen(self):
        engine = self._engine()
        result = engine.evaluate(
            _tiny_training_result(), _tiny_data_result(),
            class_names=CLASS_NAMES, num_classes=4,
        )
        with pytest.raises((AttributeError, TypeError)):
            result.mean_iou = 0.99  # type: ignore[misc]

    def test_per_class_keys_match_class_names(self):
        engine = self._engine()
        result = engine.evaluate(
            _tiny_training_result(), _tiny_data_result(),
            class_names=CLASS_NAMES, num_classes=4,
        )
        assert set(result.per_class.keys()) == set(CLASS_NAMES)

    def test_no_nan_in_aggregate_metrics(self):
        engine = self._engine()
        result = engine.evaluate(
            _tiny_training_result(), _tiny_data_result(),
            class_names=CLASS_NAMES, num_classes=4,
        )
        for attr in ("pixel_accuracy", "mean_iou", "mean_dice",
                     "mean_f1", "kappa", "balanced_accuracy"):
            val = getattr(result, attr)
            assert not np.isnan(val), f"{attr} is NaN"

    def test_metrics_in_0_1(self):
        engine = self._engine()
        result = engine.evaluate(
            _tiny_training_result(), _tiny_data_result(),
            class_names=CLASS_NAMES, num_classes=4,
        )
        for attr in ("pixel_accuracy", "mean_iou", "mean_dice", "mean_f1"):
            val = getattr(result, attr)
            assert 0.0 <= val <= 1.0 + 1e-6, f"{attr}={val} out of [0,1]"

    def test_class_names_in_result(self):
        engine = self._engine()
        result = engine.evaluate(
            _tiny_training_result(), _tiny_data_result(),
            class_names=CLASS_NAMES, num_classes=4,
        )
        assert result.class_names == CLASS_NAMES

    def test_total_samples_correct(self):
        engine = self._engine()
        result = engine.evaluate(
            _tiny_training_result(), _tiny_data_result(n=8),
            class_names=CLASS_NAMES, num_classes=4,
        )
        assert result.total_samples == 8

    def test_architecture_from_training_result(self):
        engine = self._engine()
        result = engine.evaluate(
            _tiny_training_result(), _tiny_data_result(),
            class_names=CLASS_NAMES, num_classes=4,
        )
        assert result.architecture == "unetplusplus"

    def test_split_recorded(self):
        engine = self._engine(split="validation")
        result = engine.evaluate(
            _tiny_training_result(), _tiny_data_result(),
            class_names=CLASS_NAMES, num_classes=4,
        )
        assert result.split == "validation"

    def test_confusion_matrix_total_pixels_positive(self):
        engine = self._engine()
        result = engine.evaluate(
            _tiny_training_result(), _tiny_data_result(),
            class_names=CLASS_NAMES, num_classes=4,
        )
        assert result.confusion_matrix.total_pixels > 0

    def test_operations_log_non_empty(self):
        engine = self._engine()
        result = engine.evaluate(
            _tiny_training_result(), _tiny_data_result(),
            class_names=CLASS_NAMES, num_classes=4,
        )
        assert len(result.operations_log) > 0

    def test_evaluation_time_positive(self):
        engine = self._engine()
        result = engine.evaluate(
            _tiny_training_result(), _tiny_data_result(),
            class_names=CLASS_NAMES, num_classes=4,
        )
        assert result.evaluation_time_s >= 0.0

    def test_save_reports_when_output_dir_set(self, tmp_path):
        engine = EvaluationEngine(EvaluationConfig(
            split="test", batch_size=4, device="cpu",
            output_dir=str(tmp_path), save_json=True, save_csv=True,
        ))
        engine.evaluate(
            _tiny_training_result(), _tiny_data_result(),
            class_names=CLASS_NAMES, num_classes=4,
        )
        json_files = list(tmp_path.glob("*.json"))
        csv_files  = list(tmp_path.glob("*.csv"))
        assert len(json_files) >= 1
        assert len(csv_files)  >= 1

    def test_all_ignore_index_masks(self):
        """All masks are ignore_index — result must not crash and metrics should be 0."""
        images = torch.randn(4, 4, 8, 8)
        masks  = torch.full((4, 8, 8), 255, dtype=torch.long)   # all ignored
        ds     = TensorDataset(images, masks)
        dr     = types.SimpleNamespace(
            test_dataset=ds, validation_dataset=ds, train_dataset=ds,
            num_classes=4, num_bands=4, is_valid=True,
            num_train_samples=4, num_val_samples=4, num_test_samples=4,
        )
        engine = self._engine()
        result = engine.evaluate(
            _tiny_training_result(), dr, class_names=CLASS_NAMES, num_classes=4,
        )
        assert isinstance(result, EvaluationResult)
        assert not np.isnan(result.mean_iou)

    def test_as_dict_json_serialisable(self):
        import json
        engine = self._engine()
        result = engine.evaluate(
            _tiny_training_result(), _tiny_data_result(),
            class_names=CLASS_NAMES, num_classes=4,
        )
        assert json.dumps(result.as_dict())   # must not raise
