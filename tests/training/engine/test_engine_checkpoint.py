"""Tests for src/training/engine/checkpoint.py"""
from __future__ import annotations
import pytest
from pathlib import Path
torch = pytest.importorskip("torch")
import torch.nn as nn
from src.training.engine.contracts import CheckpointConfig, EpochResult
from src.training.engine.checkpoint import CheckpointManager


def _cfg(tmp_path, **kw):
    defaults = dict(checkpoint_dir=str(tmp_path), save_best=True, save_latest=True,
                    resume_from=None, checkpoint_version="1.0",
                    metric="val_loss", mode="min")
    defaults.update(kw)
    return CheckpointConfig(**defaults)


def _result(epoch=1, train_loss=0.5, val_loss=0.6):
    return EpochResult(epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                       lr=1e-4, epoch_time=2.0, is_best=True)


def _context(tmp_path):
    model = nn.Linear(2, 2)
    import torch.optim as optim
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    return {"model": model, "optimizer": optimizer,
            "scheduler": None, "scaler": None, "model_result": None}


class TestCheckpointManager:
    def test_save_latest_creates_file(self, tmp_path):
        mgr = CheckpointManager(_cfg(tmp_path))
        ctx = _context(tmp_path)
        mgr.save(ctx, epoch=1, result=_result(), is_best=False)
        assert (tmp_path / "checkpoint_latest.pt").exists()

    def test_save_best_creates_file(self, tmp_path):
        mgr = CheckpointManager(_cfg(tmp_path))
        ctx = _context(tmp_path)
        mgr.save(ctx, epoch=1, result=_result(), is_best=True)
        assert (tmp_path / "checkpoint_best.pt").exists()

    def test_save_best_false_no_best_file(self, tmp_path):
        mgr = CheckpointManager(_cfg(tmp_path, save_best=False))
        ctx = _context(tmp_path)
        mgr.save(ctx, epoch=1, result=_result(), is_best=True)
        assert not (tmp_path / "checkpoint_best.pt").exists()

    def test_best_path_property_updated(self, tmp_path):
        mgr = CheckpointManager(_cfg(tmp_path))
        assert mgr.best_path is None
        mgr.save(_context(tmp_path), epoch=1, result=_result(), is_best=True)
        assert mgr.best_path is not None

    def test_load_returns_dict(self, tmp_path):
        mgr = CheckpointManager(_cfg(tmp_path))
        mgr.save(_context(tmp_path), epoch=3, result=_result(epoch=3), is_best=True)
        payload = mgr.load(tmp_path / "checkpoint_best.pt")
        assert payload["epoch"] == 3
        assert payload["version"] == "1.0"

    def test_load_missing_file_raises(self, tmp_path):
        mgr = CheckpointManager(_cfg(tmp_path))
        with pytest.raises(FileNotFoundError):
            mgr.load(tmp_path / "does_not_exist.pt")

    def test_restore_returns_epoch(self, tmp_path):
        mgr = CheckpointManager(_cfg(tmp_path))
        ctx = _context(tmp_path)
        mgr.save(ctx, epoch=5, result=_result(epoch=5), is_best=True)
        ctx2 = _context(tmp_path)
        epoch = mgr.restore(tmp_path / "checkpoint_best.pt", ctx2)
        assert epoch == 5

    def test_restore_loads_model_weights(self, tmp_path):
        mgr = CheckpointManager(_cfg(tmp_path))
        ctx = _context(tmp_path)
        # Set a distinctive weight value.
        ctx["model"].weight.data.fill_(3.14)
        mgr.save(ctx, epoch=1, result=_result(), is_best=True)
        # Fresh model with zeros.
        ctx2 = _context(tmp_path)
        ctx2["model"].weight.data.zero_()
        mgr.restore(tmp_path / "checkpoint_best.pt", ctx2)
        assert torch.allclose(ctx2["model"].weight.data,
                              torch.full_like(ctx2["model"].weight.data, 3.14))

    def test_checkpoint_contains_rng_state(self, tmp_path):
        mgr = CheckpointManager(_cfg(tmp_path))
        mgr.save(_context(tmp_path), epoch=1, result=_result(), is_best=True)
        payload = mgr.load(tmp_path / "checkpoint_best.pt")
        assert "rng_state" in payload
