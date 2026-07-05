"""Tests for optimizer.py and scheduler.py"""
from __future__ import annotations
import pytest
torch = pytest.importorskip("torch")
import torch.nn as nn
from src.training.engine.contracts import OptimizerConfig, SchedulerConfig
from src.training.engine.optimizer import OptimizerFactory, OptimizerRegistry
from src.training.engine.scheduler import SchedulerFactory, SchedulerRegistry


def _model():
    return nn.Linear(4, 4)

def _opt_cfg(**kw):
    defaults = dict(name="adamw", lr=1e-3, weight_decay=1e-4, momentum=0.9,
                    betas=(0.9, 0.999), eps=1e-8)
    defaults.update(kw)
    return OptimizerConfig(**defaults)

def _sched_cfg(**kw):
    defaults = dict(name="cosine", t_max=10, eta_min=1e-6, step_size=5,
                    gamma=0.1, patience=3, min_lr=1e-7, mode="min", enabled=True)
    defaults.update(kw)
    return SchedulerConfig(**defaults)


class TestOptimizerFactory:
    def test_builds_adamw(self):
        import torch.optim as optim
        opt = OptimizerFactory.build(_model(), _opt_cfg(name="adamw"))
        assert isinstance(opt, optim.AdamW)

    def test_builds_adam(self):
        import torch.optim as optim
        opt = OptimizerFactory.build(_model(), _opt_cfg(name="adam"))
        assert isinstance(opt, optim.Adam)

    def test_builds_sgd(self):
        import torch.optim as optim
        opt = OptimizerFactory.build(_model(), _opt_cfg(name="sgd"))
        assert isinstance(opt, optim.SGD)

    def test_unknown_name_raises(self):
        with pytest.raises(KeyError, match="not registered"):
            OptimizerFactory.build(_model(), _opt_cfg(name="rmsprop"))

    def test_zero_lr_raises(self):
        with pytest.raises(ValueError, match="lr must be > 0"):
            OptimizerFactory.build(_model(), _opt_cfg(lr=0.0))

    def test_lr_is_set(self):
        opt = OptimizerFactory.build(_model(), _opt_cfg(lr=0.002))
        assert opt.param_groups[0]["lr"] == pytest.approx(0.002)

    def test_register_external_optimizer(self):
        import torch.optim as optim
        @OptimizerRegistry.register("_test_rmsprop")
        def _builder(params, cfg):
            return optim.RMSprop(params, lr=cfg.lr)
        opt = OptimizerFactory.build(_model(), _opt_cfg(name="_test_rmsprop"))
        assert isinstance(opt, optim.RMSprop)
        # cleanup
        del OptimizerRegistry._builders["_test_rmsprop"]

    def test_registered_names(self):
        names = OptimizerRegistry.registered_names()
        assert "adam" in names and "adamw" in names and "sgd" in names


class TestSchedulerFactory:
    def _optimizer(self, lr=1e-3):
        import torch.optim as optim
        return optim.Adam(_model().parameters(), lr=lr)

    def test_builds_cosine(self):
        from torch.optim.lr_scheduler import CosineAnnealingLR
        sched = SchedulerFactory.build(self._optimizer(), _sched_cfg(name="cosine"))
        assert isinstance(sched, CosineAnnealingLR)

    def test_builds_step(self):
        from torch.optim.lr_scheduler import StepLR
        sched = SchedulerFactory.build(self._optimizer(), _sched_cfg(name="step"))
        assert isinstance(sched, StepLR)

    def test_builds_plateau(self):
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        sched = SchedulerFactory.build(self._optimizer(), _sched_cfg(name="plateau"))
        assert isinstance(sched, ReduceLROnPlateau)

    def test_disabled_returns_none(self):
        sched = SchedulerFactory.build(self._optimizer(), _sched_cfg(enabled=False))
        assert sched is None

    def test_step_cosine_no_error(self):
        opt   = self._optimizer()
        sched = SchedulerFactory.build(opt, _sched_cfg(name="cosine", t_max=5))
        cfg   = _sched_cfg(name="cosine")
        SchedulerFactory.step(sched, cfg, val_loss=0.5)   # should not raise

    def test_step_plateau_passes_val_loss(self):
        opt   = self._optimizer(lr=0.1)
        sched = SchedulerFactory.build(opt, _sched_cfg(name="plateau", patience=1))
        cfg   = _sched_cfg(name="plateau")
        for _ in range(3):
            SchedulerFactory.step(sched, cfg, val_loss=1.0)
        # LR should have decayed after patience exceeded
        assert opt.param_groups[0]["lr"] < 0.1

    def test_step_none_scheduler_no_error(self):
        SchedulerFactory.step(None, _sched_cfg(), val_loss=0.3)

    def test_registered_names(self):
        names = SchedulerRegistry.registered_names()
        assert "cosine" in names and "step" in names and "plateau" in names

    def test_unknown_name_raises(self):
        with pytest.raises(KeyError):
            SchedulerFactory.build(self._optimizer(), _sched_cfg(name="warmup"))
