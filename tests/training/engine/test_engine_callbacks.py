"""Tests for src/training/engine/callbacks.py"""
from __future__ import annotations
import pytest
from unittest.mock import MagicMock
from src.training.engine.callbacks import (
    Callback, CallbackList, CheckpointCallback,
    EarlyStoppingCallback, LoggingCallback,
)
from src.training.engine.contracts import EpochResult


def _result(epoch=1, train_loss=0.5, val_loss=0.6, is_best=False):
    return EpochResult(epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                       lr=1e-4, epoch_time=1.0, is_best=is_best)


class TestCallbackList:
    def test_len(self):
        cbs = CallbackList([MagicMock(spec=Callback), MagicMock(spec=Callback)])
        assert len(cbs) == 2

    def test_hooks_dispatched_in_order(self):
        log = []
        class _A(Callback):
            def on_epoch_begin(self, epoch, ctx): log.append("A")
        class _B(Callback):
            def on_epoch_begin(self, epoch, ctx): log.append("B")
        cb = CallbackList([_A(), _B()])
        cb.on_epoch_begin(1, {})
        assert log == ["A", "B"]

    def test_append(self):
        cbs = CallbackList()
        cbs.append(MagicMock(spec=Callback))
        assert len(cbs) == 1

    def test_all_hooks_called(self):
        mock = MagicMock(spec=Callback)
        cbs  = CallbackList([mock])
        ctx  = {}
        r    = _result()
        cbs.on_train_begin(ctx)
        cbs.on_epoch_begin(1, ctx)
        cbs.on_batch_begin(0, ctx)
        cbs.on_batch_end(0, 0.5, ctx)
        cbs.on_epoch_end(1, r, ctx)
        cbs.on_train_end(ctx)
        mock.on_train_begin.assert_called_once()
        mock.on_epoch_begin.assert_called_once()
        mock.on_batch_begin.assert_called_once()
        mock.on_batch_end.assert_called_once()
        mock.on_epoch_end.assert_called_once()
        mock.on_train_end.assert_called_once()


class TestEarlyStoppingCallback:
    def test_not_triggered_initially(self):
        cb = EarlyStoppingCallback(patience=3)
        assert cb.triggered is False

    def test_triggered_after_patience_exhausted(self):
        cb  = EarlyStoppingCallback(patience=2, mode="min", metric="val_loss")
        ctx = {}
        cb.on_epoch_end(1, _result(val_loss=0.8), ctx)
        cb.on_epoch_end(2, _result(val_loss=0.9), ctx)  # worse
        cb.on_epoch_end(3, _result(val_loss=0.95), ctx) # worse again
        assert cb.triggered is True
        assert ctx.get("stop_training") is True

    def test_not_triggered_on_improvement(self):
        cb  = EarlyStoppingCallback(patience=2, mode="min", metric="val_loss")
        ctx = {}
        cb.on_epoch_end(1, _result(val_loss=0.8), ctx)
        cb.on_epoch_end(2, _result(val_loss=0.6), ctx)  # improvement
        cb.on_epoch_end(3, _result(val_loss=0.7), ctx)  # slightly worse
        assert cb.triggered is False

    def test_patience_zero_never_triggers(self):
        cb  = EarlyStoppingCallback(patience=0)
        ctx = {}
        for i in range(20):
            cb.on_epoch_end(i+1, _result(val_loss=1.0), ctx)
        assert cb.triggered is False

    def test_mode_max(self):
        cb  = EarlyStoppingCallback(patience=2, mode="max", metric="val_loss")
        ctx = {}
        cb.on_epoch_end(1, _result(val_loss=0.9), ctx)
        cb.on_epoch_end(2, _result(val_loss=0.8), ctx)  # worse in max mode
        cb.on_epoch_end(3, _result(val_loss=0.7), ctx)  # worse again
        assert cb.triggered is True

    def test_wait_counter_resets_on_improvement(self):
        cb  = EarlyStoppingCallback(patience=5, mode="min")
        ctx = {}
        cb.on_epoch_end(1, _result(val_loss=0.9), ctx)
        cb.on_epoch_end(2, _result(val_loss=1.0), ctx)  # no improvement
        assert cb.wait == 1
        cb.on_epoch_end(3, _result(val_loss=0.5), ctx)  # improvement
        assert cb.wait == 0


class TestCheckpointCallback:
    def test_save_called_on_improvement(self):
        manager = MagicMock()
        cb      = CheckpointCallback(manager, mode="min", metric="val_loss")
        ctx     = {}
        cb.on_epoch_end(1, _result(val_loss=0.5), ctx)
        manager.save.assert_called_once()
        _, kwargs = manager.save.call_args
        assert kwargs.get("is_best") is True or manager.save.call_args[1].get("is_best") or manager.save.call_args[0][3]

    def test_save_called_every_epoch(self):
        manager = MagicMock()
        cb      = CheckpointCallback(manager, mode="min")
        ctx     = {}
        for i in range(3):
            cb.on_epoch_end(i+1, _result(val_loss=0.5+i*0.1), ctx)
        assert manager.save.call_count == 3


class TestLoggingCallback:
    def test_calls_log_epoch(self):
        mock_logger = MagicMock()
        cb  = LoggingCallback(log_every=1)
        ctx = {"training_logger": mock_logger}
        cb.on_epoch_end(1, _result(), ctx)
        mock_logger.log_epoch.assert_called_once()

    def test_skips_non_multiple_epochs(self):
        mock_logger = MagicMock()
        cb  = LoggingCallback(log_every=2)
        ctx = {"training_logger": mock_logger}
        cb.on_epoch_end(1, _result(epoch=1), ctx)  # skip (1 % 2 != 0)
        cb.on_epoch_end(2, _result(epoch=2), ctx)  # log
        assert mock_logger.log_epoch.call_count == 1

    def test_works_without_training_logger(self):
        cb = LoggingCallback()
        cb.on_epoch_end(1, _result(), {})   # should not raise
