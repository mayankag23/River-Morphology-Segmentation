"""Tests for seed.py, history.py, metrics.py"""
from __future__ import annotations
import time
import pytest
from src.training.engine.seed import SeedManager
from src.training.engine.history import TrainingHistory
from src.training.engine.contracts import EpochResult
from src.training.engine.metrics import MetricsAccumulator


# ==============================================================================
# SeedManager
# ==============================================================================

class TestSeedManager:
    def test_seed_runs_without_error(self):
        SeedManager.seed(42)

    def test_seed_numpy_is_deterministic(self):
        import numpy as np
        SeedManager.seed(0)
        a = np.random.rand(5)
        SeedManager.seed(0)
        b = np.random.rand(5)
        assert (a == b).all()

    def test_seed_python_random_is_deterministic(self):
        import random
        SeedManager.seed(7)
        a = random.random()
        SeedManager.seed(7)
        b = random.random()
        assert a == b

    def test_seed_torch_is_deterministic(self):
        torch = pytest.importorskip("torch")
        SeedManager.seed(13)
        a = torch.rand(3)
        SeedManager.seed(13)
        b = torch.rand(3)
        assert torch.allclose(a, b)

    def test_get_rng_state_returns_dict(self):
        SeedManager.seed(1)
        state = SeedManager.get_rng_state()
        assert "python" in state
        assert "numpy"  in state

    def test_restore_rng_state_restores_numpy(self):
        import numpy as np
        SeedManager.seed(99)
        state = SeedManager.get_rng_state()
        a = np.random.rand(5)
        SeedManager.restore_rng_state(state)
        b = np.random.rand(5)
        assert (a == b).all()

    def test_seed_worker_runs(self):
        SeedManager.seed(0)
        SeedManager.seed_worker(3)   # should not raise


# ==============================================================================
# TrainingHistory
# ==============================================================================

def _epoch(n: int, tl: float = 0.5, vl: float = 0.6) -> EpochResult:
    return EpochResult(epoch=n, train_loss=tl, val_loss=vl,
                       lr=1e-4, epoch_time=1.0, is_best=False)


class TestTrainingHistory:
    def test_empty_initially(self):
        assert len(TrainingHistory()) == 0

    def test_append_and_len(self):
        h = TrainingHistory()
        h.append(_epoch(1))
        h.append(_epoch(2))
        assert len(h) == 2

    def test_train_losses_property(self):
        h = TrainingHistory()
        h.append(_epoch(1, tl=0.3))
        h.append(_epoch(2, tl=0.2))
        assert h.train_losses == pytest.approx([0.3, 0.2])

    def test_best_epoch_returns_lowest_val_loss(self):
        h = TrainingHistory()
        h.append(_epoch(1, vl=0.8))
        h.append(_epoch(2, vl=0.3))
        h.append(_epoch(3, vl=0.5))
        assert h.best_epoch == 2

    def test_to_tuple_is_immutable(self):
        h = TrainingHistory()
        h.append(_epoch(1))
        t = h.to_tuple()
        assert isinstance(t, tuple)

    def test_last_returns_most_recent(self):
        h = TrainingHistory()
        h.append(_epoch(1))
        h.append(_epoch(2))
        assert h.last().epoch == 2

    def test_last_returns_none_when_empty(self):
        assert TrainingHistory().last() is None

    def test_iter(self):
        h = TrainingHistory()
        h.append(_epoch(1)); h.append(_epoch(2))
        epochs = [r.epoch for r in h]
        assert epochs == [1, 2]


# ==============================================================================
# MetricsAccumulator
# ==============================================================================

class TestMetricsAccumulator:
    def test_start_resets_state(self):
        acc = MetricsAccumulator()
        acc.start()
        assert acc.num_samples == 0

    def test_update_accumulates(self):
        acc = MetricsAccumulator()
        acc.start()
        acc.update(0.5, 4)
        acc.update(0.3, 4)
        assert acc.num_samples == 8

    def test_compute_returns_mean_loss(self):
        acc = MetricsAccumulator()
        acc.start()
        acc.update(1.0, 2)   # total loss = 2.0 for 2 samples
        acc.update(0.5, 2)   # total loss = 1.0 for 2 samples
        result = acc.compute()
        assert result["loss"] == pytest.approx(0.75)   # (2.0+1.0)/4

    def test_compute_records_epoch_time(self):
        acc = MetricsAccumulator()
        acc.start()
        time.sleep(0.01)
        result = acc.compute()
        assert result["epoch_time"] >= 0.005

    def test_zero_samples_returns_zero_loss(self):
        acc = MetricsAccumulator()
        acc.start()
        result = acc.compute()
        assert result["loss"] == 0.0

    def test_get_lr_from_real_optimizer(self):
        torch = pytest.importorskip("torch")
        import torch.nn as nn
        import torch.optim as optim
        model = nn.Linear(2, 2)
        opt   = optim.Adam(model.parameters(), lr=0.01)
        lr    = MetricsAccumulator.get_lr(opt)
        assert lr == pytest.approx(0.01)

    def test_get_lr_from_mock_returns_zero(self):
        assert MetricsAccumulator.get_lr(object()) == 0.0
