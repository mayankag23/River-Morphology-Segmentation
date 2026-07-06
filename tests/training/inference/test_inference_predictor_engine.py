"""
Integration smoke tests for predictor.py, exporter.py, and engine.py.
Uses synthetic tiny model and dataset — no rasterio, no GeoTIFF, no real data.
"""
from __future__ import annotations
import types
import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.inference.contracts import (
    InferenceConfig, InferenceResult, SamplePrediction,
)
from src.training.inference.confidence import MaxProbabilityStrategy, EntropyStrategy
from src.training.inference.predictor import Predictor


# ==============================================================================
# Tiny test fixtures
# ==============================================================================

C, N_CLS, H, W = 4, 4, 8, 8
CLASS_NAMES = ("background", "water", "sand", "vegetation")


def _model() -> nn.Module:
    class _M(nn.Module):
        def __init__(self): super().__init__(); self.conv = nn.Conv2d(C, N_CLS, 1)
        def forward(self, x): return self.conv(x)
    return _M().eval()


def _cfg(**kw) -> InferenceConfig:
    defaults = dict(
        checkpoint_path="", checkpoint_strategy="best",
        checkpoint_dir="checkpoints", device="cpu", batch_size=4, num_workers=0,
        mixed_precision=False, deterministic=False, probability_mode="softmax",
        confidence_strategy="max_probability", output_dir="predictions",
        export_numpy=True, export_geotiff=False, export_png=False,
        postprocess=False, fill_holes=False, min_object_size=0,
        morph_open_size=0, morph_close_size=0, ignore_index=255, seed=42,
        pin_memory=False,
    )
    defaults.update(kw)
    return InferenceConfig(**defaults)


def _predictor(strategy="max_probability", **kw) -> Predictor:
    strat = MaxProbabilityStrategy() if strategy == "max_probability" else EntropyStrategy()
    return Predictor(
        config=_cfg(**kw), model=_model(), confidence_strategy=strat,
        device=torch.device("cpu"), class_names=CLASS_NAMES,
    )


def _loader(n=8) -> DataLoader:
    images = torch.randn(n, C, H, W)
    masks  = torch.randint(0, N_CLS, (n, H, W), dtype=torch.long)
    return DataLoader(TensorDataset(images, masks), batch_size=4, shuffle=False)


def _training_result(model=None) -> object:
    return types.SimpleNamespace(
        model=model or _model(), architecture="unetplusplus",
        num_parameters=1000, in_channels=C, num_classes=N_CLS,
        best_checkpoint=None,
    )


def _data_result(n=8) -> object:
    images = torch.randn(n, C, H, W)
    masks  = torch.randint(0, N_CLS, (n, H, W), dtype=torch.long)
    ds     = TensorDataset(images, masks)
    return types.SimpleNamespace(
        train_dataset=ds, validation_dataset=ds, test_dataset=ds,
        num_classes=N_CLS, num_bands=C, is_valid=True,
        num_train_samples=n, num_val_samples=n, num_test_samples=n,
    )


# ==============================================================================
# Predictor
# ==============================================================================

class TestPredictor:
    def test_predict_batch_returns_list(self):
        pred = _predictor()
        imgs = torch.randn(2, C, H, W)
        results = pred.predict_batch(imgs)
        assert len(results) == 2

    def test_sample_prediction_shapes(self):
        pred    = _predictor()
        imgs    = torch.randn(1, C, H, W)
        results = pred.predict_batch(imgs)
        sp      = results[0]
        assert sp.predicted_mask.shape == (H, W)
        assert sp.probabilities.shape  == (N_CLS, H, W)
        assert sp.confidence.shape     == (H, W)

    def test_probabilities_sum_to_one(self):
        pred    = _predictor()
        imgs    = torch.randn(2, C, H, W)
        results = pred.predict_batch(imgs)
        for sp in results:
            sums = sp.probabilities.sum(axis=0)
            np.testing.assert_allclose(sums, 1.0, atol=1e-5)

    def test_predicted_mask_dtype_uint8(self):
        pred = _predictor()
        imgs = torch.randn(1, C, H, W)
        sp   = pred.predict_batch(imgs)[0]
        assert sp.predicted_mask.dtype == np.uint8

    def test_confidence_in_0_1(self):
        pred = _predictor()
        imgs = torch.randn(2, C, H, W)
        for sp in pred.predict_batch(imgs):
            assert sp.confidence.min() >= 0.0
            assert sp.confidence.max() <= 1.0 + 1e-6

    def test_no_nan_in_probabilities(self):
        pred = _predictor()
        imgs = torch.randn(2, C, H, W)
        for sp in pred.predict_batch(imgs):
            assert not np.isnan(sp.probabilities).any()

    def test_entropy_strategy_works(self):
        pred    = _predictor(strategy="entropy")
        imgs    = torch.randn(1, C, H, W)
        results = pred.predict_batch(imgs)
        assert results[0].confidence.shape == (H, W)

    def test_sigmoid_mode_works(self):
        pred    = _predictor(probability_mode="sigmoid")
        imgs    = torch.randn(1, C, H, W)
        results = pred.predict_batch(imgs)
        assert results[0].probabilities.shape == (N_CLS, H, W)

    def test_predict_single_image(self):
        pred  = _predictor()
        image = np.random.randn(C, H, W).astype(np.float32)
        sp    = pred.predict_single(image, {"sample_id": "test_001"})
        assert sp.sample_id == "test_001"
        assert sp.predicted_mask.shape == (H, W)

    def test_metadata_preserved(self):
        pred   = _predictor()
        imgs   = torch.randn(1, C, H, W)
        meta   = {"sample_id": "r001", "river_name": "Kosi",
                  "acquisition_date": "2023-07-15", "season": "monsoon",
                  "hydrological_year": 2023, "sensor": "L8",
                  "reach_id": "R1", "basin_id": "B1", "aoi_id": "A1",
                  "scene_id": "SC1", "year": 2023, "month": 7,
                  "patch_path": "", "mask_path": ""}
        sp = pred.predict_batch(imgs, [meta])[0]
        assert sp.river_name == "Kosi"
        assert sp.season     == "monsoon"
        assert sp.hydrological_year == 2023
        assert sp.acquisition_date  == "2023-07-15"

    def test_predict_dataset(self):
        pred    = _predictor()
        results = pred.predict_dataset(_loader(n=8))
        assert len(results) == 8

    def test_deterministic_inference(self):
        """Same input with same seed should give identical output."""
        torch.manual_seed(0)
        pred = _predictor()
        imgs = torch.randn(1, C, H, W)
        sp1  = pred.predict_batch(imgs)[0]
        sp2  = pred.predict_batch(imgs)[0]
        np.testing.assert_array_equal(sp1.predicted_mask, sp2.predicted_mask)
        np.testing.assert_allclose(sp1.probabilities, sp2.probabilities)


# ==============================================================================
# PredictionExporter
# ==============================================================================

class TestPredictionExporter:
    def _sp(self):
        return SamplePrediction(
            sample_id="patch_001",
            predicted_mask=np.zeros((8, 8), dtype=np.uint8),
            probabilities=np.ones((4, 8, 8), dtype=np.float32) * 0.25,
            confidence=np.ones((8, 8), dtype=np.float32) * 0.8,
        )

    def test_export_numpy_creates_file(self, tmp_path):
        from src.training.inference.exporter import PredictionExporter
        cfg  = _cfg(output_dir=str(tmp_path), export_numpy=True)
        exp  = PredictionExporter(cfg, CLASS_NAMES)
        paths = exp.export(self._sp())
        assert len(paths) == 1
        assert paths[0].endswith(".npy")

    def test_export_numpy_file_loadable(self, tmp_path):
        from src.training.inference.exporter import PredictionExporter
        cfg  = _cfg(output_dir=str(tmp_path), export_numpy=True)
        exp  = PredictionExporter(cfg, CLASS_NAMES)
        paths = exp.export(self._sp())
        arr   = np.load(paths[0])
        assert arr.shape == (8, 8)

    def test_export_all_updates_exported_paths(self, tmp_path):
        from src.training.inference.exporter import PredictionExporter
        cfg   = _cfg(output_dir=str(tmp_path), export_numpy=True)
        exp   = PredictionExporter(cfg, CLASS_NAMES)
        sp    = self._sp()
        preds = exp.export_all([sp])
        assert len(preds[0].exported_paths) >= 1

    def test_no_export_when_all_disabled(self, tmp_path):
        from src.training.inference.exporter import PredictionExporter
        cfg   = _cfg(output_dir=str(tmp_path), export_numpy=False,
                     export_geotiff=False, export_png=False)
        exp   = PredictionExporter(cfg, CLASS_NAMES)
        paths = exp.export(self._sp())
        assert len(paths) == 0


# ==============================================================================
# InferenceEngine integration
# ==============================================================================

class TestInferenceEngine:
    def _save_checkpoint(self, tmp_path, model: nn.Module) -> None:
        import torch
        payload = {
            "version": "1.0", "epoch": 3, "train_loss": 0.3, "val_loss": 0.4,
            "model_state": model.state_dict(), "optimizer_state": {},
            "scheduler_state": None, "scaler_state": None, "rng_state": {},
            "architecture": "unetplusplus", "num_classes": N_CLS, "in_channels": C,
        }
        torch.save(payload, tmp_path / "checkpoint_best.pt")

    def test_returns_inference_result(self, tmp_path):
        from src.training.inference.engine import InferenceEngine
        model = _model()
        self._save_checkpoint(tmp_path, model)
        cfg    = _cfg(checkpoint_strategy="best", checkpoint_dir=str(tmp_path),
                      output_dir=str(tmp_path), export_numpy=False)
        engine = InferenceEngine(cfg)
        result = engine.predict(
            _training_result(model), _data_result(n=4),
            class_names=CLASS_NAMES, num_classes=N_CLS,
        )
        assert isinstance(result, InferenceResult)

    def test_result_is_frozen(self, tmp_path):
        from src.training.inference.engine import InferenceEngine
        model = _model()
        self._save_checkpoint(tmp_path, model)
        cfg    = _cfg(checkpoint_strategy="best", checkpoint_dir=str(tmp_path),
                      output_dir=str(tmp_path), export_numpy=False)
        result = InferenceEngine(cfg).predict(
            _training_result(model), _data_result(n=4),
            class_names=CLASS_NAMES, num_classes=N_CLS,
        )
        with pytest.raises((AttributeError, TypeError)):
            result.num_samples = 99  # type: ignore[misc]

    def test_correct_num_samples(self, tmp_path):
        from src.training.inference.engine import InferenceEngine
        model = _model()
        self._save_checkpoint(tmp_path, model)
        cfg    = _cfg(checkpoint_strategy="best", checkpoint_dir=str(tmp_path),
                      output_dir=str(tmp_path), export_numpy=False)
        result = InferenceEngine(cfg).predict(
            _training_result(model), _data_result(n=8),
            class_names=CLASS_NAMES, num_classes=N_CLS,
        )
        assert result.num_samples == 8

    def test_class_names_recorded(self, tmp_path):
        from src.training.inference.engine import InferenceEngine
        model = _model()
        self._save_checkpoint(tmp_path, model)
        cfg    = _cfg(checkpoint_strategy="best", checkpoint_dir=str(tmp_path),
                      output_dir=str(tmp_path), export_numpy=False)
        result = InferenceEngine(cfg).predict(
            _training_result(model), _data_result(n=4),
            class_names=CLASS_NAMES, num_classes=N_CLS,
        )
        assert result.class_names == CLASS_NAMES

    def test_checkpoint_meta_populated(self, tmp_path):
        from src.training.inference.engine import InferenceEngine
        model = _model()
        self._save_checkpoint(tmp_path, model)
        cfg    = _cfg(checkpoint_strategy="best", checkpoint_dir=str(tmp_path),
                      output_dir=str(tmp_path), export_numpy=False)
        result = InferenceEngine(cfg).predict(
            _training_result(model), _data_result(n=4),
            class_names=CLASS_NAMES, num_classes=N_CLS,
        )
        assert result.checkpoint_meta.epoch == 3

    def test_no_data_result_gives_zero_predictions(self, tmp_path):
        from src.training.inference.engine import InferenceEngine
        model = _model()
        self._save_checkpoint(tmp_path, model)
        cfg    = _cfg(checkpoint_strategy="best", checkpoint_dir=str(tmp_path),
                      output_dir=str(tmp_path), export_numpy=False)
        result = InferenceEngine(cfg).predict(
            _training_result(model), data_result=None,
            class_names=CLASS_NAMES, num_classes=N_CLS,
        )
        assert result.num_samples == 0

    def test_mean_confidence_in_0_1(self, tmp_path):
        from src.training.inference.engine import InferenceEngine
        model = _model()
        self._save_checkpoint(tmp_path, model)
        cfg    = _cfg(checkpoint_strategy="best", checkpoint_dir=str(tmp_path),
                      output_dir=str(tmp_path), export_numpy=False)
        result = InferenceEngine(cfg).predict(
            _training_result(model), _data_result(n=4),
            class_names=CLASS_NAMES, num_classes=N_CLS,
        )
        assert 0.0 <= result.mean_confidence <= 1.0

    def test_class_pixel_counts_present(self, tmp_path):
        from src.training.inference.engine import InferenceEngine
        model = _model()
        self._save_checkpoint(tmp_path, model)
        cfg    = _cfg(checkpoint_strategy="best", checkpoint_dir=str(tmp_path),
                      output_dir=str(tmp_path), export_numpy=False)
        result = InferenceEngine(cfg).predict(
            _training_result(model), _data_result(n=4),
            class_names=CLASS_NAMES, num_classes=N_CLS,
        )
        assert set(result.class_pixel_counts.keys()) == set(CLASS_NAMES)

    def test_numpy_export_creates_files(self, tmp_path):
        from src.training.inference.engine import InferenceEngine
        model = _model()
        self._save_checkpoint(tmp_path, model)
        cfg    = _cfg(checkpoint_strategy="best", checkpoint_dir=str(tmp_path),
                      output_dir=str(tmp_path), export_numpy=True)
        result = InferenceEngine(cfg).predict(
            _training_result(model), _data_result(n=4),
            class_names=CLASS_NAMES, num_classes=N_CLS,
        )
        npy_files = list(tmp_path.glob("*.npy"))
        assert len(npy_files) == 4   # one per sample

    def test_as_dict_json_serialisable(self, tmp_path):
        import json
        from src.training.inference.engine import InferenceEngine
        model = _model()
        self._save_checkpoint(tmp_path, model)
        cfg    = _cfg(checkpoint_strategy="best", checkpoint_dir=str(tmp_path),
                      output_dir=str(tmp_path), export_numpy=False)
        result = InferenceEngine(cfg).predict(
            _training_result(model), _data_result(n=4),
            class_names=CLASS_NAMES, num_classes=N_CLS,
        )
        assert json.dumps(result.as_dict())   # must not raise

    def test_operations_log_non_empty(self, tmp_path):
        from src.training.inference.engine import InferenceEngine
        model = _model()
        self._save_checkpoint(tmp_path, model)
        cfg    = _cfg(checkpoint_strategy="best", checkpoint_dir=str(tmp_path),
                      output_dir=str(tmp_path), export_numpy=False)
        result = InferenceEngine(cfg).predict(
            _training_result(model), _data_result(n=4),
            class_names=CLASS_NAMES, num_classes=N_CLS,
        )
        assert len(result.operations_log) > 0
