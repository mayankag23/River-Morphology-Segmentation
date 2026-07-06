"""Tests for src/training/inference/contracts.py"""
from __future__ import annotations
import numpy as np
import pytest
from src.training.inference.contracts import (
    CheckpointMetadata, InferenceConfig, InferenceResult, SamplePrediction,
)


class TestInferenceConfig:
    def test_frozen(self):
        cfg = InferenceConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.device = "cuda"  # type: ignore[misc]

    def test_defaults(self):
        cfg = InferenceConfig()
        assert cfg.checkpoint_strategy == "best"
        assert cfg.device == "cpu"
        assert cfg.probability_mode == "softmax"
        assert cfg.confidence_strategy == "max_probability"

    def test_from_config_reads_values(self):
        class _Inf:
            checkpoint_path=""; checkpoint_strategy="latest"
            checkpoint_dir="ckpts"; device="cpu"; batch_size=16; num_workers=2
            mixed_precision=False; deterministic=True; probability_mode="softmax"
            confidence_strategy="entropy"; output_dir="out"; export_numpy=True
            export_geotiff=False; export_png=False; postprocess=False
            fill_holes=False; min_object_size=0; morph_open_size=0
            morph_close_size=0; ignore_index=255; seed=7; pin_memory=False
        class _Cfg:
            inference = _Inf()
        cfg = InferenceConfig.from_config(_Cfg())
        assert cfg.checkpoint_strategy == "latest"
        assert cfg.batch_size == 16
        assert cfg.confidence_strategy == "entropy"

    def test_from_config_no_section_returns_defaults(self):
        class _Cfg: pass
        assert InferenceConfig.from_config(_Cfg()) == InferenceConfig()


class TestCheckpointMetadata:
    def test_frozen(self):
        cm = CheckpointMetadata("path", "1.0", 10, 0.3, 0.4, "unetplusplus", 4, 12)
        with pytest.raises((AttributeError, TypeError)):
            cm.epoch = 99  # type: ignore[misc]

    def test_as_dict(self):
        cm = CheckpointMetadata("path", "1.0", 10, 0.3, 0.4, "unetplusplus", 4, 12)
        d  = cm.as_dict()
        assert d["epoch"] == 10
        assert d["architecture"] == "unetplusplus"


class TestSamplePrediction:
    def _make(self):
        return SamplePrediction(
            sample_id      = "p001",
            predicted_mask = np.zeros((8, 8), dtype=np.uint8),
            probabilities  = np.ones((4, 8, 8), dtype=np.float32) * 0.25,
            confidence     = np.ones((8, 8), dtype=np.float32) * 0.9,
        )

    def test_not_frozen(self):
        sp = self._make()
        sp.sample_id = "new_id"   # should not raise
        assert sp.sample_id == "new_id"

    def test_defaults_empty(self):
        sp = self._make()
        assert sp.acquisition_date == ""
        assert sp.river_name == ""

    def test_summary_dict_keys(self):
        d = self._make().summary_dict()
        assert "sample_id" in d
        assert "confidence_mean" in d
        assert "prob_shape" in d

    def test_exported_paths_mutable(self):
        sp = self._make()
        sp.exported_paths.append("/tmp/test.npy")
        assert len(sp.exported_paths) == 1


class TestInferenceResult:
    def _ckpt(self):
        return CheckpointMetadata("p", "1.0", 5, 0.3, 0.4, "unetplusplus", 4, 12)

    def _make(self):
        sp = SamplePrediction(
            sample_id="p1",
            predicted_mask=np.zeros((4, 4), dtype=np.uint8),
            probabilities=np.ones((4, 4, 4), dtype=np.float32) * 0.25,
            confidence=np.ones((4, 4), dtype=np.float32) * 0.8,
        )
        return InferenceResult(
            predictions=(sp,), num_samples=1, architecture="unetplusplus",
            num_classes=4, class_names=("bg","water","sand","veg"),
            checkpoint_meta=self._ckpt(), inference_config=InferenceConfig(),
            device_used="cpu", total_inference_s=1.5, per_sample_ms=1500.0,
            operations_log=("a","b"), mean_confidence=0.8,
            class_pixel_counts={"bg":10,"water":3,"sand":2,"veg":1},
        )

    def test_frozen(self):
        r = self._make()
        with pytest.raises((AttributeError, TypeError)):
            r.num_samples = 99  # type: ignore[misc]

    def test_summary_lines_ascii(self):
        lines = self._make().summary_lines()
        assert all(ord(c) < 128 for l in lines for c in l)

    def test_as_dict_json_serialisable(self):
        import json
        d = self._make().as_dict()
        assert json.dumps(d)
