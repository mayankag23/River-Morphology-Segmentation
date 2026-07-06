"""
Additional tests for src/training/inference/factory.py.

Target branches not covered by existing tests:
- build() with postprocess=True (PostprocessorPipeline.build_from_config path)
- build() with deterministic=True (_seed called with deterministic=True)
- _resolve_device() with cuda string but cuda unavailable -> CPU fallback
- _seed() with deterministic=True branch (torch.use_deterministic_algorithms)
- _SimpleBatchLoader.__iter__ and __len__
- build_dataloader() with pin_memory=True on CPU (pin_memory stays False)
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from src.training.inference.contracts import InferenceConfig
from src.training.inference.factory import InferenceFactory, _SimpleBatchLoader


# ==============================================================================
# Helpers
# ==============================================================================

def _cfg(**kw) -> InferenceConfig:
    defaults = dict(
        checkpoint_path="", checkpoint_strategy="best",
        checkpoint_dir="ckpts", device="cpu", batch_size=4, num_workers=0,
        mixed_precision=False, deterministic=False, probability_mode="softmax",
        confidence_strategy="max_probability", output_dir="predictions",
        export_numpy=False, export_geotiff=False, export_png=False,
        postprocess=False, fill_holes=False, min_object_size=0,
        morph_open_size=0, morph_close_size=0, ignore_index=255,
        seed=42, pin_memory=False,
    )
    defaults.update(kw)
    return InferenceConfig(**defaults)


def _tiny_model():
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    class _M(nn.Module):
        def __init__(self): super().__init__(); self.conv = nn.Conv2d(4, 4, 1)
        def forward(self, x): return self.conv(x)
    return _M().eval()


# ==============================================================================
# _resolve_device
# ==============================================================================

class TestResolveDevice:
    def test_cpu_device_returns_cpu(self):
        import torch
        cfg = _cfg(device="cpu")
        dev = InferenceFactory._resolve_device(cfg)
        assert str(dev) == "cpu"

    def test_cuda_device_falls_back_to_cpu_when_unavailable(self):
        import torch
        cfg = _cfg(device="cuda")
        # On a machine without CUDA, this must return CPU without raising.
        dev = InferenceFactory._resolve_device(cfg)
        if not torch.cuda.is_available():
            assert str(dev) == "cpu"
        else:
            assert "cuda" in str(dev)


# ==============================================================================
# _seed
# ==============================================================================

class TestSeed:
    def test_seed_non_deterministic_is_reproducible(self):
        import random, numpy as np
        InferenceFactory._seed(42, deterministic=False)
        a = random.random()
        InferenceFactory._seed(42, deterministic=False)
        b = random.random()
        assert a == b

    def test_seed_deterministic_does_not_raise(self):
        """deterministic=True path must not raise on CPU."""
        # torch.use_deterministic_algorithms(True) may raise on some ops,
        # but the _seed call itself must succeed.
        try:
            InferenceFactory._seed(7, deterministic=True)
        except Exception as exc:
            # Acceptable only if it's about unsupported ops, not about _seed logic.
            assert "deterministic" in str(exc).lower() or "algorithm" in str(exc).lower()
        # Reset to non-deterministic for other tests.
        try:
            import torch
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass

    def test_seed_numpy_reproducible(self):
        import numpy as np
        InferenceFactory._seed(99, deterministic=False)
        arr1 = np.random.rand(5)
        InferenceFactory._seed(99, deterministic=False)
        arr2 = np.random.rand(5)
        np.testing.assert_array_equal(arr1, arr2)


# ==============================================================================
# build() — postprocess=True branch
# ==============================================================================

class TestBuildWithPostprocess:
    def test_build_with_postprocess_true_returns_context(self):
        torch = pytest.importorskip("torch")
        cfg   = _cfg(postprocess=True, fill_holes=False, min_object_size=0,
                     morph_open_size=0, morph_close_size=0)
        ctx   = InferenceFactory.build(cfg, _tiny_model(), ("bg", "water", "sand", "veg"))
        assert "postprocessor" in ctx
        # PostprocessorPipeline with 0 processors (all disabled).
        assert len(ctx["postprocessor"]._processors) == 0

    def test_build_with_postprocess_false_empty_pipeline(self):
        torch = pytest.importorskip("torch")
        cfg   = _cfg(postprocess=False)
        ctx   = InferenceFactory.build(cfg, _tiny_model(), ("bg",))
        assert len(ctx["postprocessor"]._processors) == 0

    def test_build_with_fill_holes_adds_processor(self):
        torch = pytest.importorskip("torch")
        cfg   = _cfg(postprocess=True, fill_holes=True)
        ctx   = InferenceFactory.build(cfg, _tiny_model(), ("bg", "water", "sand", "veg"))
        names = [p.name for p in ctx["postprocessor"]._processors]
        assert "hole_filler" in names

    def test_build_with_min_object_size_adds_processor(self):
        torch = pytest.importorskip("torch")
        cfg   = _cfg(postprocess=True, min_object_size=64)
        ctx   = InferenceFactory.build(cfg, _tiny_model(), ("bg", "water", "sand", "veg"))
        names = [p.name for p in ctx["postprocessor"]._processors]
        assert "small_object_remover" in names

    def test_build_with_morph_open_adds_processor(self):
        torch = pytest.importorskip("torch")
        cfg   = _cfg(postprocess=True, morph_open_size=3)
        ctx   = InferenceFactory.build(cfg, _tiny_model(), ("bg", "water", "sand", "veg"))
        names = [p.name for p in ctx["postprocessor"]._processors]
        assert "morph_open" in names

    def test_build_with_morph_close_adds_processor(self):
        torch = pytest.importorskip("torch")
        cfg   = _cfg(postprocess=True, morph_close_size=3)
        ctx   = InferenceFactory.build(cfg, _tiny_model(), ("bg", "water", "sand", "veg"))
        names = [p.name for p in ctx["postprocessor"]._processors]
        assert "morph_close" in names

    def test_build_with_deterministic_true(self):
        torch = pytest.importorskip("torch")
        cfg   = _cfg(deterministic=True)
        try:
            ctx = InferenceFactory.build(cfg, _tiny_model(), ("bg",))
            assert "predictor" in ctx
        except Exception as exc:
            # deterministic algorithms may not be supported for all ops.
            assert "deterministic" in str(exc).lower() or "algorithm" in str(exc).lower()
        finally:
            try:
                torch.use_deterministic_algorithms(False)
            except Exception:
                pass

    def test_build_returns_all_required_keys(self):
        torch = pytest.importorskip("torch")
        cfg   = _cfg()
        ctx   = InferenceFactory.build(cfg, _tiny_model(), ("bg", "water", "sand", "veg"))
        for key in ("device", "predictor", "postprocessor", "exporter", "model"):
            assert key in ctx


# ==============================================================================
# build_dataloader()
# ==============================================================================

class TestBuildDataloader:
    def test_returns_dataloader_with_correct_batch_size(self):
        torch = pytest.importorskip("torch")
        from torch.utils.data import TensorDataset
        images = torch.randn(8, 4, 8, 8)
        masks  = torch.zeros(8, 8, 8, dtype=torch.long)
        ds     = TensorDataset(images, masks)
        cfg    = _cfg(batch_size=4)
        device = torch.device("cpu")
        loader = InferenceFactory.build_dataloader(cfg, ds, device)
        assert loader.batch_size == 4

    def test_pin_memory_false_on_cpu(self):
        torch = pytest.importorskip("torch")
        from torch.utils.data import TensorDataset
        images = torch.randn(4, 4, 8, 8)
        masks  = torch.zeros(4, 8, 8, dtype=torch.long)
        ds     = TensorDataset(images, masks)
        cfg    = _cfg(batch_size=4, pin_memory=True, device="cpu")
        device = torch.device("cpu")
        loader = InferenceFactory.build_dataloader(cfg, ds, device)
        # pin_memory should be False because device is cpu.
        assert loader.pin_memory is False


# ==============================================================================
# _SimpleBatchLoader
# ==============================================================================

class TestSimpleBatchLoader:
    def _make_dataset(self, n: int = 6):
        torch = pytest.importorskip("torch")
        from torch.utils.data import TensorDataset
        images = torch.randn(n, 4, 8, 8)
        masks  = torch.zeros(n, 8, 8, dtype=torch.long)
        return TensorDataset(images, masks)

    def test_len_correct(self):
        ds     = self._make_dataset(6)
        loader = _SimpleBatchLoader(ds, batch_size=4)
        assert len(loader) == 2   # ceil(6 / 4) = 2

    def test_len_exact_division(self):
        ds     = self._make_dataset(8)
        loader = _SimpleBatchLoader(ds, batch_size=4)
        assert len(loader) == 2

    def test_iter_yields_correct_number_of_batches(self):
        ds      = self._make_dataset(6)
        loader  = _SimpleBatchLoader(ds, batch_size=4)
        batches = list(loader)
        assert len(batches) == 2

    def test_iter_batch_shapes(self):
        torch  = pytest.importorskip("torch")
        ds     = self._make_dataset(6)
        loader = _SimpleBatchLoader(ds, batch_size=4)
        for i, batch in enumerate(loader):
            images, masks, meta = batch
            expected_bs = 4 if i == 0 else 2
            assert images.shape[0] == expected_bs

    def test_iter_yields_none_masks(self):
        ds      = self._make_dataset(4)
        loader  = _SimpleBatchLoader(ds, batch_size=4)
        batch   = next(iter(loader))
        assert batch[1] is None

    def test_iter_yields_empty_meta_list(self):
        ds     = self._make_dataset(4)
        loader = _SimpleBatchLoader(ds, batch_size=4)
        batch  = next(iter(loader))
        meta   = batch[2]
        assert isinstance(meta, list)
        assert all(m == {} for m in meta)

    def test_batch_size_1_works(self):
        ds      = self._make_dataset(3)
        loader  = _SimpleBatchLoader(ds, batch_size=1)
        batches = list(loader)
        assert len(batches) == 3

    def test_batch_size_zero_becomes_one(self):
        """Constructor clamps batch_size to max(1, batch_size)."""
        ds     = self._make_dataset(3)
        loader = _SimpleBatchLoader(ds, batch_size=0)
        assert loader._batch_size == 1
