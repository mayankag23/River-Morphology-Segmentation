"""Tests for validator.py and trainer.py"""
from __future__ import annotations
import types
import pytest
from src.training.engine.contracts import (
    CheckpointConfig, LossConfig, OptimizerConfig,
    SchedulerConfig, TrainingConfig,
)
from src.training.engine.validator import TrainingValidator, TrainingValidationResult


# ==============================================================================
# TrainingValidator
# ==============================================================================

def _cfg(**kw):
    base = dict(epochs=5, seed=42, device="cpu", mixed_precision=False,
                grad_clip_value=0.0, accumulation_steps=1, num_workers=0,
                batch_size=4, val_batch_size=0, early_stopping_patience=0,
                deterministic=False, cudnn_benchmark=False,
                optimizer=OptimizerConfig(), scheduler=SchedulerConfig(),
                loss=LossConfig(), checkpoint=CheckpointConfig())
    base.update(kw)
    return TrainingConfig(**base)


def _model_result(in_channels=4, num_classes=4, trainable=1000):
    return types.SimpleNamespace(
        model=object(), in_channels=in_channels,
        num_classes=num_classes, num_trainable=trainable,
    )


def _data_result(num_train=10, num_classes=4, num_bands=4, valid=True):
    return types.SimpleNamespace(
        num_train_samples=num_train, num_classes=num_classes,
        num_bands=num_bands, is_valid=valid, validation_issues=[],
        train_dataset=[], validation_dataset=[], test_dataset=[],
    )


class TestTrainingValidator:
    def test_valid_inputs_pass(self):
        v = TrainingValidator()
        r = v.validate(_cfg(), _model_result(), _data_result())
        assert r.is_valid

    def test_zero_epochs_detected(self):
        v = TrainingValidator()
        r = v.validate(_cfg(epochs=0), _model_result(), _data_result())
        assert not r.is_valid
        assert any("epochs" in i for i in r.issues)

    def test_zero_lr_detected(self):
        v = TrainingValidator()
        r = v.validate(
            _cfg(optimizer=OptimizerConfig(lr=0.0)),
            _model_result(), _data_result(),
        )
        assert not r.is_valid

    def test_empty_train_dataset_detected(self):
        v = TrainingValidator()
        r = v.validate(_cfg(), _model_result(), _data_result(num_train=0))
        assert not r.is_valid

    def test_class_count_mismatch_detected(self):
        v = TrainingValidator()
        r = v.validate(
            _cfg(),
            _model_result(num_classes=4),
            _data_result(num_classes=3),
        )
        assert not r.is_valid
        assert any("num_classes" in i for i in r.issues)

    def test_channel_mismatch_detected(self):
        v = TrainingValidator()
        r = v.validate(
            _cfg(),
            _model_result(in_channels=12),
            _data_result(num_bands=6),
        )
        assert not r.is_valid
        assert any("in_channels" in i or "num_bands" in i for i in r.issues)

    def test_none_model_result_detected(self):
        v = TrainingValidator()
        r = v.validate(_cfg(), None, _data_result())
        assert not r.is_valid

    def test_none_data_result_detected(self):
        v = TrainingValidator()
        r = v.validate(_cfg(), _model_result(), None)
        assert not r.is_valid

    def test_missing_resume_checkpoint_detected(self, tmp_path):
        ckpt_cfg = CheckpointConfig(resume_from=str(tmp_path / "nonexistent.pt"))
        v = TrainingValidator()
        r = v.validate(_cfg(checkpoint=ckpt_cfg), _model_result(), _data_result())
        assert not r.is_valid
        assert any("resume_from" in i or "not found" in i for i in r.issues)

    def test_validation_result_is_valid_when_no_issues(self):
        result = TrainingValidationResult([])
        assert result.is_valid

    def test_validation_result_is_invalid_with_issues(self):
        result = TrainingValidationResult(["problem1"])
        assert not result.is_valid

    def test_issues_are_copy(self):
        result = TrainingValidationResult(["a"])
        result.issues.append("b")
        assert len(result.issues) == 1


# ==============================================================================
# Trainer (integration smoke tests using tiny model + synthetic DataLoader)
# ==============================================================================

class TestTrainerSmoke:
    """Trainer smoke tests that run a real forward/backward pass."""

    @pytest.fixture
    def tiny_setup(self):
        torch = pytest.importorskip("torch")
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        from src.training.engine.callbacks import CallbackList
        from src.training.engine.contracts import (
            LossConfig, SchedulerConfig, TrainingConfig, OptimizerConfig,
            CheckpointConfig,
        )
        from src.training.engine.losses import CrossEntropyLoss
        from src.training.engine.trainer import Trainer

        C, H, W, N_CLS = 2, 8, 8, 3
        images = torch.randn(8, C, H, W)
        masks  = torch.randint(0, N_CLS, (8, H, W), dtype=torch.long)
        ds     = TensorDataset(images, masks)
        loader = DataLoader(ds, batch_size=4)

        class _TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(C, N_CLS, 1)
            def forward(self, x): return self.conv(x)

        model   = _TinyModel()
        opt     = optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = CrossEntropyLoss(LossConfig(ignore_index=255))
        cfg     = TrainingConfig(
            epochs=2, seed=0, device="cpu", mixed_precision=False,
            grad_clip_value=0.0, accumulation_steps=1, num_workers=0,
            batch_size=4, val_batch_size=0, early_stopping_patience=0,
            optimizer=OptimizerConfig(), scheduler=SchedulerConfig(enabled=False),
            loss=LossConfig(), checkpoint=CheckpointConfig(), deterministic=False,
        )
        return Trainer(
            config=cfg, model=model, optimizer=opt, scheduler=None,
            loss_fn=loss_fn, callbacks=CallbackList(), device=torch.device("cpu"),
            scaler=None,
        ), loader

    def test_trainer_returns_epoch_results(self, tiny_setup):
        trainer, loader = tiny_setup
        results = trainer.run(loader, None, start_epoch=1, context={})
        assert len(results) == 2

    def test_train_loss_is_positive(self, tiny_setup):
        trainer, loader = tiny_setup
        results = trainer.run(loader, None, start_epoch=1, context={})
        assert all(r.train_loss >= 0 for r in results)

    def test_early_stopping_signal_halts_loop(self, tiny_setup):
        trainer, loader = tiny_setup
        ctx = {"stop_training": False}
        # Inject a callback that sets stop_training after epoch 1.
        from src.training.engine.callbacks import Callback
        class _Stop(Callback):
            def on_epoch_end(self, epoch, result, context):
                context["stop_training"] = True
        trainer._callbacks.append(_Stop())
        results = trainer.run(loader, None, start_epoch=1, context=ctx)
        assert len(results) == 1   # stopped after first epoch

    def test_gradient_clipping_does_not_crash(self, tiny_setup):
        torch = pytest.importorskip("torch")
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        from src.training.engine.callbacks import CallbackList
        from src.training.engine.contracts import (
            LossConfig, SchedulerConfig, TrainingConfig, OptimizerConfig,
            CheckpointConfig,
        )
        from src.training.engine.losses import CrossEntropyLoss
        from src.training.engine.trainer import Trainer

        C, H, W, N_CLS = 2, 8, 8, 3
        images = torch.randn(4, C, H, W)
        masks  = torch.randint(0, N_CLS, (4, H, W), dtype=torch.long)
        ds     = TensorDataset(images, masks)
        loader = DataLoader(ds, batch_size=4)

        class _Model(nn.Module):
            def __init__(self): super().__init__(); self.conv = nn.Conv2d(C, N_CLS, 1)
            def forward(self, x): return self.conv(x)

        model = _Model()
        cfg   = TrainingConfig(
            epochs=1, seed=0, device="cpu", mixed_precision=False,
            grad_clip_value=1.0, accumulation_steps=1, num_workers=0,
            batch_size=4, val_batch_size=0, early_stopping_patience=0,
            optimizer=OptimizerConfig(), scheduler=SchedulerConfig(enabled=False),
            loss=LossConfig(), checkpoint=CheckpointConfig(), deterministic=False,
        )
        trainer = Trainer(
            config=cfg, model=model,
            optimizer=optim.Adam(model.parameters(), lr=1e-3),
            scheduler=None,
            loss_fn=CrossEntropyLoss(LossConfig()),
            callbacks=CallbackList(), device=torch.device("cpu"), scaler=None,
        )
        results = trainer.run(loader, None, start_epoch=1, context={})
        assert len(results) == 1

    def test_gradient_accumulation(self, tiny_setup):
        torch = pytest.importorskip("torch")
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        from src.training.engine.callbacks import CallbackList
        from src.training.engine.contracts import (
            LossConfig, SchedulerConfig, TrainingConfig, OptimizerConfig,
            CheckpointConfig,
        )
        from src.training.engine.losses import CrossEntropyLoss
        from src.training.engine.trainer import Trainer

        C, H, W, N_CLS = 2, 8, 8, 3
        images = torch.randn(8, C, H, W)
        masks  = torch.randint(0, N_CLS, (8, H, W), dtype=torch.long)
        loader = DataLoader(TensorDataset(images, masks), batch_size=2)

        class _Model(nn.Module):
            def __init__(self): super().__init__(); self.conv = nn.Conv2d(C, N_CLS, 1)
            def forward(self, x): return self.conv(x)

        model = _Model()
        cfg   = TrainingConfig(
            epochs=1, seed=0, device="cpu", mixed_precision=False,
            grad_clip_value=0.0, accumulation_steps=4, num_workers=0,
            batch_size=2, val_batch_size=0, early_stopping_patience=0,
            optimizer=OptimizerConfig(), scheduler=SchedulerConfig(enabled=False),
            loss=LossConfig(), checkpoint=CheckpointConfig(), deterministic=False,
        )
        trainer = Trainer(
            config=cfg, model=model,
            optimizer=optim.Adam(model.parameters(), lr=1e-3),
            scheduler=None,
            loss_fn=CrossEntropyLoss(LossConfig()),
            callbacks=CallbackList(), device=torch.device("cpu"), scaler=None,
        )
        results = trainer.run(loader, None, start_epoch=1, context={})
        assert len(results) == 1
        assert results[0].train_loss >= 0
