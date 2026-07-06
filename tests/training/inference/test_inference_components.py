"""Tests for loader.py, confidence.py, postprocessing.py, validator.py"""
from __future__ import annotations
import numpy as np
import pytest
from pathlib import Path

from src.training.inference.contracts import InferenceConfig, CheckpointMetadata
from src.training.inference.confidence import (
    ConfidenceRegistry, EntropyStrategy, MaxProbabilityStrategy,
)
from src.training.inference.postprocessing import (
    HoleFiller, MorphCloseProcessor, MorphOpenProcessor,
    PostprocessorPipeline, SmallObjectRemover,
)
from src.training.inference.validator import InferenceValidator, InferenceValidationResult


# ==============================================================================
# CheckpointLoader
# ==============================================================================

class TestCheckpointLoader:
    def _cfg(self, **kw) -> InferenceConfig:
        defaults = dict(
            checkpoint_path="", checkpoint_strategy="best",
            checkpoint_dir="checkpoints", device="cpu", batch_size=4,
            num_workers=0, mixed_precision=False, deterministic=False,
            probability_mode="softmax", confidence_strategy="max_probability",
            output_dir="predictions", export_numpy=True, export_geotiff=False,
            export_png=False, postprocess=False, fill_holes=False,
            min_object_size=0, morph_open_size=0, morph_close_size=0,
            ignore_index=255, seed=42, pin_memory=False,
        )
        defaults.update(kw)
        return InferenceConfig(**defaults)

    def test_resolve_best_path(self, tmp_path):
        from src.training.inference.loader import CheckpointLoader
        best = tmp_path / "checkpoint_best.pt"
        best.touch()
        cfg    = self._cfg(checkpoint_strategy="best", checkpoint_dir=str(tmp_path))
        loader = CheckpointLoader(cfg)
        assert loader.resolve_path() == best.resolve()

    def test_resolve_latest_path(self, tmp_path):
        from src.training.inference.loader import CheckpointLoader
        latest = tmp_path / "checkpoint_latest.pt"
        latest.touch()
        cfg    = self._cfg(checkpoint_strategy="latest", checkpoint_dir=str(tmp_path))
        assert CheckpointLoader(cfg).resolve_path() == latest.resolve()

    def test_resolve_explicit_path(self, tmp_path):
        from src.training.inference.loader import CheckpointLoader
        p = tmp_path / "my.pt"
        p.touch()
        cfg = self._cfg(checkpoint_strategy="explicit", checkpoint_path=str(p))
        assert CheckpointLoader(cfg).resolve_path() == p.resolve()

    def test_missing_file_raises(self, tmp_path):
        from src.training.inference.loader import CheckpointLoader
        cfg = self._cfg(checkpoint_strategy="best", checkpoint_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            CheckpointLoader(cfg).resolve_path()

    def test_unknown_strategy_raises(self, tmp_path):
        from src.training.inference.loader import CheckpointLoader
        cfg = self._cfg(checkpoint_strategy="random")
        with pytest.raises(ValueError):
            CheckpointLoader(cfg).resolve_path()

    def test_load_and_restore_model(self, tmp_path):
        torch = pytest.importorskip("torch")
        import torch.nn as nn
        import torch.optim as optim
        from src.training.inference.loader import CheckpointLoader

        model = nn.Linear(2, 2)
        model.weight.data.fill_(3.14)
        payload = {
            "version": "1.0", "epoch": 5, "train_loss": 0.3, "val_loss": 0.4,
            "model_state": model.state_dict(), "optimizer_state": {},
            "scheduler_state": None, "scaler_state": None, "rng_state": {},
            "architecture": "unetplusplus", "num_classes": 4, "in_channels": 12,
        }
        path = tmp_path / "checkpoint_best.pt"
        torch.save(payload, path)

        cfg    = self._cfg(checkpoint_strategy="best", checkpoint_dir=str(tmp_path))
        loader = CheckpointLoader(cfg)
        ckpt_path = loader.resolve_path()
        loaded    = loader.load(ckpt_path)

        fresh = nn.Linear(2, 2)
        fresh.weight.data.zero_()
        loader.restore_model(fresh, loaded)
        assert torch.allclose(fresh.weight.data, torch.full_like(fresh.weight.data, 3.14))

    def test_extract_metadata(self, tmp_path):
        from src.training.inference.loader import CheckpointLoader
        payload = {
            "version": "1.0", "epoch": 7, "train_loss": 0.2, "val_loss": 0.3,
            "architecture": "unetplusplus", "num_classes": 4, "in_channels": 12,
        }
        meta = CheckpointLoader.extract_metadata(tmp_path / "ckpt.pt", payload)
        assert meta.epoch == 7
        assert meta.architecture == "unetplusplus"
        assert meta.num_classes == 4


# ==============================================================================
# Confidence strategies
# ==============================================================================

def _probs(C=4, H=8, W=8, seed=0):
    rng   = np.random.default_rng(seed)
    logits = rng.standard_normal((C, H, W)).astype(np.float32)
    exp    = np.exp(logits - logits.max(axis=0, keepdims=True))
    return (exp / exp.sum(axis=0, keepdims=True)).astype(np.float32)


class TestMaxProbabilityStrategy:
    def test_name(self):
        assert MaxProbabilityStrategy().name == "max_probability"

    def test_output_shape(self):
        p = _probs()
        c = MaxProbabilityStrategy().compute(p)
        assert c.shape == (8, 8)

    def test_output_dtype_float32(self):
        assert MaxProbabilityStrategy().compute(_probs()).dtype == np.float32

    def test_value_in_0_1(self):
        c = MaxProbabilityStrategy().compute(_probs())
        assert c.min() >= 0.0 and c.max() <= 1.0

    def test_uniform_probs_low_confidence(self):
        C, H, W = 4, 4, 4
        p = np.full((C, H, W), 1.0 / C, dtype=np.float32)
        c = MaxProbabilityStrategy().compute(p)
        assert np.allclose(c, 1.0 / C)

    def test_certain_probs_high_confidence(self):
        C, H, W = 4, 4, 4
        p = np.zeros((C, H, W), dtype=np.float32)
        p[0] = 1.0
        c = MaxProbabilityStrategy().compute(p)
        assert np.allclose(c, 1.0)


class TestEntropyStrategy:
    def test_name(self):
        assert EntropyStrategy().name == "entropy"

    def test_output_shape(self):
        assert EntropyStrategy().compute(_probs()).shape == (8, 8)

    def test_value_in_0_1(self):
        c = EntropyStrategy().compute(_probs())
        assert c.min() >= 0.0 and c.max() <= 1.0 + 1e-6

    def test_uniform_probs_zero_confidence(self):
        C, H, W = 4, 4, 4
        p = np.full((C, H, W), 1.0 / C, dtype=np.float32)
        c = EntropyStrategy().compute(p)
        assert np.allclose(c, 0.0, atol=1e-5)

    def test_certain_probs_one_confidence(self):
        C, H, W = 4, 4, 4
        p = np.zeros((C, H, W), dtype=np.float32)
        p[0] = 1.0
        c = EntropyStrategy().compute(p)
        assert np.allclose(c, 1.0, atol=1e-5)

    def test_no_nan(self):
        c = EntropyStrategy().compute(_probs())
        assert not np.isnan(c).any()


class TestConfidenceRegistry:
    def test_registered_names(self):
        names = ConfidenceRegistry.registered_names()
        assert "max_probability" in names
        assert "entropy" in names

    def test_build_max_probability(self):
        s = ConfidenceRegistry.build("max_probability")
        assert isinstance(s, MaxProbabilityStrategy)

    def test_build_entropy(self):
        assert isinstance(ConfidenceRegistry.build("entropy"), EntropyStrategy)

    def test_unknown_raises(self):
        with pytest.raises(KeyError):
            ConfidenceRegistry.build("nonexistent_xyz")


# ==============================================================================
# Post-processing
# ==============================================================================

def _mask(h=16, w=16, fill=0):
    return np.full((h, w), fill, dtype=np.uint8)


class TestPostprocessorPipeline:
    def test_empty_pipeline_identity(self):
        mask = _mask(fill=1)
        pipe = PostprocessorPipeline([])
        result = pipe.apply(mask)
        np.testing.assert_array_equal(result, mask)

    def test_build_from_config_empty(self):
        cfg  = InferenceConfig(postprocess=False)
        pipe = PostprocessorPipeline.build_from_config(cfg)
        mask = _mask(fill=2)
        np.testing.assert_array_equal(pipe.apply(mask), mask)

    def test_single_processor_applied(self):
        class _Double:
            name = "double"
            def apply(self, mask): return (mask * 0 + 1).astype(np.uint8)
        pipe   = PostprocessorPipeline([_Double()])
        result = pipe.apply(_mask(fill=0))
        assert (result == 1).all()


class TestHoleFiller:
    def test_name(self):
        assert HoleFiller().name == "hole_filler"

    def test_output_shape_unchanged(self):
        scipy = pytest.importorskip("scipy")
        mask   = _mask(8, 8, fill=1)
        result = HoleFiller().apply(mask)
        assert result.shape == mask.shape

    def test_fills_interior_hole(self):
        scipy = pytest.importorskip("scipy")
        mask = np.ones((8, 8), dtype=np.uint8)
        mask[3, 3] = 0   # small hole in class 1 region
        result = HoleFiller().apply(mask)
        assert result[3, 3] == 1


class TestSmallObjectRemover:
    def test_name(self):
        assert SmallObjectRemover().name == "small_object_remover"

    def test_output_shape_unchanged(self):
        scipy = pytest.importorskip("scipy")
        mask   = _mask(16, 16, fill=0)
        result = SmallObjectRemover(min_size=4).apply(mask)
        assert result.shape == mask.shape


class TestMorphProcessors:
    def test_morph_open_name(self):
        assert MorphOpenProcessor().name == "morph_open"

    def test_morph_close_name(self):
        assert MorphCloseProcessor().name == "morph_close"

    def test_open_output_shape(self):
        scipy = pytest.importorskip("scipy")
        mask   = _mask(16, 16, fill=1)
        result = MorphOpenProcessor(3).apply(mask)
        assert result.shape == mask.shape

    def test_close_output_shape(self):
        scipy = pytest.importorskip("scipy")
        mask   = _mask(16, 16, fill=0)
        result = MorphCloseProcessor(3).apply(mask)
        assert result.shape == mask.shape


# ==============================================================================
# InferenceValidator
# ==============================================================================

class TestInferenceValidator:
    def _cfg(self, **kw):
        defaults = dict(
            checkpoint_path="", checkpoint_strategy="best",
            checkpoint_dir="checkpoints", device="cpu", batch_size=4,
            num_workers=0, mixed_precision=False, deterministic=False,
            probability_mode="softmax", confidence_strategy="max_probability",
            output_dir="predictions", export_numpy=True, export_geotiff=False,
            export_png=False, postprocess=False, fill_holes=False,
            min_object_size=0, morph_open_size=0, morph_close_size=0,
            ignore_index=255, seed=42, pin_memory=False,
        )
        defaults.update(kw)
        return InferenceConfig(**defaults)

    def test_valid_config_passes(self):
        v = InferenceValidator()
        r = v.validate_config(self._cfg())
        assert r.is_valid

    def test_zero_batch_size_detected(self):
        v = InferenceValidator()
        r = v.validate_config(self._cfg(batch_size=0))
        assert not r.is_valid

    def test_invalid_probability_mode_detected(self):
        v = InferenceValidator()
        r = v.validate_config(self._cfg(probability_mode="argmax"))
        assert not r.is_valid

    def test_explicit_strategy_without_path_detected(self):
        v = InferenceValidator()
        r = v.validate_config(self._cfg(checkpoint_strategy="explicit", checkpoint_path=""))
        assert not r.is_valid

    def test_valid_prediction_passes(self):
        v     = InferenceValidator()
        mask  = np.zeros((4, 4), dtype=np.uint8)
        probs = np.ones((4, 4, 4), dtype=np.float32) * 0.25
        conf  = np.ones((4, 4), dtype=np.float32) * 0.5
        r = v.validate_prediction(mask, probs, conf, num_classes=4)
        assert r.is_valid

    def test_nan_in_probs_detected(self):
        v     = InferenceValidator()
        probs = np.full((4, 4, 4), np.nan, dtype=np.float32)
        r = v.validate_prediction(
            np.zeros((4, 4), dtype=np.uint8), probs,
            np.zeros((4, 4), dtype=np.float32), 4,
        )
        assert not r.is_valid

    def test_wrong_num_classes_detected(self):
        v     = InferenceValidator()
        probs = np.ones((2, 4, 4), dtype=np.float32)   # 2 classes
        r = v.validate_prediction(
            np.zeros((4, 4), dtype=np.uint8), probs,
            np.zeros((4, 4), dtype=np.float32),
            num_classes=4,  # expect 4
        )
        assert not r.is_valid

    def test_issues_are_copy(self):
        r = InferenceValidationResult(["a"])
        r.issues.append("b")
        assert len(r.issues) == 1
