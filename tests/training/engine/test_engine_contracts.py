"""Tests for src/training/engine/contracts.py"""
from __future__ import annotations
import pytest
from src.training.engine.contracts import (
    CheckpointConfig, EpochResult, LossConfig, OptimizerConfig,
    SchedulerConfig, TrainingConfig, TrainingResult,
)


class TestOptimizerConfig:
    def test_frozen(self):
        cfg = OptimizerConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.lr = 1.0  # type: ignore[misc]

    def test_defaults(self):
        cfg = OptimizerConfig()
        assert cfg.name == "adamw"
        assert cfg.lr   == pytest.approx(1e-4)

    def test_from_config_reads_values(self):
        class _Opt:
            name = "sgd"; lr = 0.01; weight_decay = 1e-3
            momentum = 0.9; betas = (0.9, 0.999); eps = 1e-8
        class _Cfg:
            class training:
                optimizer = _Opt()
        cfg = OptimizerConfig.from_config(_Cfg())
        assert cfg.name == "sgd"
        assert cfg.lr   == pytest.approx(0.01)

    def test_from_config_no_section_returns_defaults(self):
        class _Cfg: pass
        assert OptimizerConfig.from_config(_Cfg()) == OptimizerConfig()


class TestSchedulerConfig:
    def test_frozen(self):
        cfg = SchedulerConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.t_max = 50  # type: ignore[misc]

    def test_defaults(self):
        cfg = SchedulerConfig()
        assert cfg.name == "cosine"; assert cfg.enabled is True

    def test_from_config_disabled(self):
        class _Sched:
            name="cosine"; t_max=10; eta_min=1e-7; step_size=5
            gamma=0.1; patience=3; min_lr=1e-8; mode="min"; enabled=False
        class _Cfg:
            class training:
                scheduler = _Sched()
        assert SchedulerConfig.from_config(_Cfg()).enabled is False


class TestLossConfig:
    def test_frozen(self):
        cfg = LossConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.name = "dice"  # type: ignore[misc]

    def test_defaults(self):
        cfg = LossConfig()
        assert cfg.name == "cross_entropy"; assert cfg.ignore_index == 255

    def test_from_config_reads_name(self):
        class _Loss:
            name="dice"; ignore_index=255; label_smoothing=0.0
            dice_smooth=1.0; focal_alpha=1.0; focal_gamma=2.0
            ce_weight=0.5; dice_weight=0.5
        class _Cfg:
            class training:
                loss = _Loss()
        assert LossConfig.from_config(_Cfg()).name == "dice"


class TestTrainingConfig:
    def test_frozen(self):
        cfg = TrainingConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.epochs = 999  # type: ignore[misc]

    def test_defaults(self):
        cfg = TrainingConfig()
        assert cfg.epochs == 100; assert cfg.seed == 42; assert cfg.device == "cpu"

    def test_from_config_reads_epochs(self):
        class _Train:
            epochs=5; seed=7; device="cpu"; mixed_precision=False
            grad_clip_value=0.0; accumulation_steps=1; num_workers=0
            batch_size=4; val_batch_size=0; early_stopping_patience=0
            deterministic=False; cudnn_benchmark=False
            class optimizer:
                name="adamw"; lr=1e-4; weight_decay=1e-4
                momentum=0.9; betas=(0.9,0.999); eps=1e-8
            class scheduler:
                name="cosine"; t_max=10; eta_min=1e-6; step_size=5
                gamma=0.1; patience=5; min_lr=1e-7; mode="min"; enabled=True
            class loss:
                name="cross_entropy"; ignore_index=255; label_smoothing=0.0
                dice_smooth=1.0; focal_alpha=1.0; focal_gamma=2.0
                ce_weight=0.5; dice_weight=0.5
            class checkpoint:
                checkpoint_dir="ckpts"; save_best=True; save_latest=True
                resume_from=None; checkpoint_version="1.0"; metric="val_loss"; mode="min"
        class _Cfg:
            training = _Train()
        cfg = TrainingConfig.from_config(_Cfg())
        assert cfg.epochs == 5; assert cfg.seed == 7


class TestEpochResult:
    def test_frozen(self):
        r = EpochResult(epoch=1, train_loss=0.5, val_loss=0.6, lr=1e-4, epoch_time=10.0, is_best=True)
        with pytest.raises((AttributeError, TypeError)):
            r.epoch = 2  # type: ignore[misc]

    def test_fields(self):
        r = EpochResult(epoch=3, train_loss=0.3, val_loss=0.4, lr=2e-4, epoch_time=5.0, is_best=False)
        assert r.epoch == 3; assert r.is_best is False


class TestTrainingResult:
    def _make(self):
        import types
        return TrainingResult(
            model=types.SimpleNamespace(), history=(),
            best_epoch=1, best_metric=0.5,
            best_checkpoint=None, latest_checkpoint=None,
            total_epochs=5, architecture="unetplusplus",
            num_parameters=100_000, seed=42,
            training_config=TrainingConfig(),
            operations_log=("a", "b"), stopped_early=False,
        )

    def test_frozen(self):
        r = self._make()
        with pytest.raises((AttributeError, TypeError)):
            r.total_epochs = 99  # type: ignore[misc]

    def test_summary_lines(self):
        lines = self._make().summary_lines()
        assert len(lines) > 0
        assert all(isinstance(l, str) for l in lines)
        assert all(ord(c) < 128 for l in lines for c in l)
