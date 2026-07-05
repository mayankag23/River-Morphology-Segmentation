"""Tests for src/training/engine/losses.py"""
from __future__ import annotations
import pytest
torch = pytest.importorskip("torch")
import torch
import torch.nn as nn
from src.training.engine.contracts import LossConfig
from src.training.engine.losses import (
    CombinedLoss, CrossEntropyLoss, DiceLoss, FocalLoss, LossRegistry,
)


def _logits(B=2, C=4, H=8, W=8):
    return torch.randn(B, C, H, W)

def _masks(B=2, H=8, W=8, num_classes=4):
    return torch.randint(0, num_classes, (B, H, W), dtype=torch.long)

def _cfg(**kw):
    defaults = dict(name="cross_entropy", ignore_index=255, label_smoothing=0.0,
                    dice_smooth=1.0, focal_alpha=1.0, focal_gamma=2.0,
                    ce_weight=0.5, dice_weight=0.5)
    defaults.update(kw)
    return LossConfig(**defaults)


class TestLossRegistry:
    def test_registered_names_include_builtins(self):
        names = LossRegistry.registered_names()
        assert "cross_entropy" in names
        assert "dice"          in names
        assert "focal"         in names
        assert "combined"      in names

    def test_build_cross_entropy(self):
        loss = LossRegistry.build(_cfg(name="cross_entropy"))
        assert isinstance(loss, CrossEntropyLoss)

    def test_build_unknown_raises(self):
        with pytest.raises(KeyError, match="not registered"):
            LossRegistry.build(_cfg(name="nonexistent_loss"))

    def test_register_external(self):
        @LossRegistry.register
        class _MyLoss(nn.Module):
            loss_name = "_test_custom_loss_xyz"
            def __init__(self, cfg): super().__init__()
            def forward(self, l, t): return l.sum() * 0
        assert "_test_custom_loss_xyz" in LossRegistry.registered_names()
        # cleanup
        del LossRegistry._registered["_test_custom_loss_xyz"]


class TestCrossEntropyLoss:
    def test_output_is_scalar(self):
        loss = CrossEntropyLoss(_cfg())
        out  = loss(_logits(), _masks())
        assert out.shape == ()

    def test_output_is_non_negative(self):
        loss = CrossEntropyLoss(_cfg())
        assert loss(_logits(), _masks()).item() >= 0.0

    def test_ignore_index_pixels_excluded(self):
        loss = CrossEntropyLoss(_cfg(ignore_index=255))
        masks = _masks()
        masks[:, :, :] = 255   # all nodata
        out = loss(_logits(), masks)
        # With all-nodata mask, CE with ignore_index should produce 0 or nan-safe value
        assert not torch.isnan(out)

    def test_set_weight(self):
        loss = CrossEntropyLoss(_cfg())
        w    = torch.ones(4)
        loss.set_weight(w)
        out = loss(_logits(), _masks())
        assert out.shape == ()

    def test_gradient_flows(self):
        loss   = CrossEntropyLoss(_cfg())
        logits = _logits().requires_grad_(True)
        loss(logits, _masks()).backward()
        assert logits.grad is not None


class TestDiceLoss:
    def test_output_is_scalar(self):
        loss = DiceLoss(_cfg())
        assert loss(_logits(), _masks()).shape == ()

    def test_output_in_0_1(self):
        loss = DiceLoss(_cfg())
        val  = loss(_logits(), _masks()).item()
        assert 0.0 <= val <= 1.0 + 1e-6

    def test_perfect_prediction_low_loss(self):
        """One-hot logits should yield near-zero Dice loss."""
        loss   = DiceLoss(_cfg(dice_smooth=1e-6))
        logits = torch.zeros(1, 4, 4, 4)
        masks  = torch.zeros(1, 4, 4, dtype=torch.long)
        logits[0, 0] = 100.0  # strong prediction for class 0
        val = loss(logits, masks).item()
        assert val < 0.1

    def test_gradient_flows(self):
        loss   = DiceLoss(_cfg())
        logits = _logits().requires_grad_(True)
        loss(logits, _masks()).backward()
        assert logits.grad is not None


class TestFocalLoss:
    def test_output_is_scalar(self):
        loss = FocalLoss(_cfg())
        assert loss(_logits(), _masks()).shape == ()

    def test_output_non_negative(self):
        loss = FocalLoss(_cfg())
        assert loss(_logits(), _masks()).item() >= 0.0

    def test_all_ignore_index_returns_zero(self):
        loss  = FocalLoss(_cfg(ignore_index=255))
        masks = torch.full((2, 8, 8), 255, dtype=torch.long)
        val   = loss(_logits(), masks).item()
        assert val == pytest.approx(0.0, abs=1e-5)

    def test_gradient_flows(self):
        loss   = FocalLoss(_cfg())
        logits = _logits().requires_grad_(True)
        loss(logits, _masks()).backward()
        assert logits.grad is not None


class TestCombinedLoss:
    def test_output_is_scalar(self):
        loss = CombinedLoss(_cfg())
        assert loss(_logits(), _masks()).shape == ()

    def test_equal_weights_between_ce_and_dice(self):
        ce_cfg   = _cfg(name="cross_entropy")
        dice_cfg = _cfg(name="dice")
        comb_cfg = _cfg(name="combined", ce_weight=0.5, dice_weight=0.5)
        logits   = _logits(); masks = _masks()
        torch.manual_seed(0)
        ce   = CrossEntropyLoss(ce_cfg)(logits, masks)
        dice = DiceLoss(dice_cfg)(logits, masks)
        comb = CombinedLoss(comb_cfg)(logits, masks)
        expected = 0.5 * ce + 0.5 * dice
        assert comb.item() == pytest.approx(expected.item(), rel=1e-4)

    def test_gradient_flows(self):
        loss   = CombinedLoss(_cfg())
        logits = _logits().requires_grad_(True)
        loss(logits, _masks()).backward()
        assert logits.grad is not None
