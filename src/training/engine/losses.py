"""
Loss functions for the Training Engine Framework (Module 14).

All losses accept (logits, targets) with:
    logits:  (B, C, H, W) float32 — raw model output (no softmax applied).
    targets: (B, H, W)    int64   — class indices.

No activation is applied inside losses; the model outputs raw logits and
each loss function handles its own internal softmax/log_softmax as needed.

Metrics are NEVER computed inside loss functions.

Registered losses
-----------------
    cross_entropy    CrossEntropyLoss   — torch.nn.CrossEntropyLoss wrapper
    dice             DiceLoss           — soft Dice loss
    focal            FocalLoss          — focal loss for class imbalance
    combined         CombinedLoss       — weighted CE + Dice

Future losses can be added via @LossRegistry.register without modifying
this file.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.engine.contracts import LossConfig

__all__ = [
    "LossRegistry",
    "CrossEntropyLoss",
    "DiceLoss",
    "FocalLoss",
    "CombinedLoss",
]

_LOGGER: logging.Logger = logging.getLogger(__name__)


# ==============================================================================
# LossRegistry
# ==============================================================================

class LossRegistry:
    """
    Registry of named loss functions.

    Keys are lowercase strings matching LossConfig.name.
    """

    _registered: dict[str, type] = {}

    @classmethod
    def register(cls, loss_class: type) -> type:
        """Class decorator. Registers by loss_name class attribute."""
        name = getattr(loss_class, "loss_name", "")
        if not name:
            raise ValueError(
                f"LossRegistry.register: {loss_class.__name__} must define "
                f"a non-empty string 'loss_name'."
            )
        cls._registered[name] = loss_class
        return loss_class

    @classmethod
    def build(cls, config: LossConfig) -> nn.Module:
        """
        Instantiate the loss function specified in config.

        Args:
            config: LossConfig with name and all hyperparameters.

        Returns:
            Instantiated nn.Module loss function.

        Raises:
            KeyError: config.name is not registered.
        """
        name = config.name.lower().strip()
        if name not in cls._registered:
            available = sorted(cls._registered.keys())
            raise KeyError(
                f"LossRegistry: '{name}' is not registered. "
                f"Available: {available}"
            )
        return cls._registered[name](config)

    @classmethod
    def registered_names(cls) -> tuple[str, ...]:
        return tuple(sorted(cls._registered.keys()))

    @classmethod
    def clear(cls) -> None:
        """For test isolation only."""
        cls._registered.clear()


# ==============================================================================
# CrossEntropyLoss
# ==============================================================================

@LossRegistry.register
class CrossEntropyLoss(nn.Module):
    """
    Wrapper around torch.nn.CrossEntropyLoss.

    Supports label smoothing and per-class weights (from ClassWeights).
    The ignore_index pixels (nodata) are excluded from loss computation.

    Args:
        config: LossConfig supplying ignore_index and label_smoothing.
    """

    loss_name: str = "cross_entropy"

    def __init__(self, config: LossConfig) -> None:
        super().__init__()
        self._loss = nn.CrossEntropyLoss(
            ignore_index    = config.ignore_index,
            label_smoothing = config.label_smoothing,
            reduction       = "mean",
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, C, H, W) float32 raw logits.
            targets: (B, H, W)    int64   class indices.
        Returns:
            Scalar loss tensor.
        """
        # return self._loss(logits, targets)
        loss = self._loss(logits, targets)
 
        if torch.isnan(loss):
            return torch.zeros(
            (),
            device=logits.device,
            dtype=logits.dtype,
            requires_grad=True,
        )
        return loss

    def set_weight(self, weight: torch.Tensor) -> None:
        """Inject per-class weight tensor (from ClassWeights.as_tensor())."""
        self._loss.weight = weight


# ==============================================================================
# DiceLoss
# ==============================================================================

@LossRegistry.register
class DiceLoss(nn.Module):
    """
    Soft multi-class Dice loss.

    Computes Dice coefficient per class over the batch and returns
    1 - mean_Dice. Pixels matching ignore_index are masked out before
    the Dice computation.

    Args:
        config: LossConfig supplying ignore_index and dice_smooth.
    """

    loss_name: str = "dice"

    def __init__(self, config: LossConfig) -> None:
        super().__init__()
        self._ignore_index = config.ignore_index
        self._smooth       = config.dice_smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, C, H, W) float32.
            targets: (B, H, W)    int64.
        Returns:
            Scalar Dice loss in [0, 1].
        """
        num_classes = logits.shape[1]
        probs       = F.softmax(logits, dim=1)   # (B, C, H, W)

        # One-hot encode targets, masking the ignore_index pixels.
        valid_mask  = (targets != self._ignore_index).unsqueeze(1).float()  # (B,1,H,W)
        targets_clamped = targets.clone()
        targets_clamped[targets == self._ignore_index] = 0
        one_hot = F.one_hot(targets_clamped, num_classes).permute(0, 3, 1, 2).float()

        probs    = probs    * valid_mask
        one_hot  = one_hot * valid_mask

        dims     = (0, 2, 3)   # sum over batch and spatial
        inter    = (probs * one_hot).sum(dims)
        union    = probs.sum(dims) + one_hot.sum(dims)
        dice_cls = (2.0 * inter + self._smooth) / (union + self._smooth)
        return 1.0 - dice_cls.mean()


# ==============================================================================
# FocalLoss
# ==============================================================================

@LossRegistry.register
class FocalLoss(nn.Module):
    """
    Multi-class focal loss (Lin et al., 2017).

    Focuses on hard/misclassified examples by down-weighting easy ones.

    Args:
        config: LossConfig supplying ignore_index, focal_alpha, focal_gamma.
    """

    loss_name: str = "focal"

    def __init__(self, config: LossConfig) -> None:
        super().__init__()
        self._ignore_index = config.ignore_index
        self._alpha        = config.focal_alpha
        self._gamma        = config.focal_gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, C, H, W) float32.
            targets: (B, H, W)    int64.
        Returns:
            Scalar focal loss.
        """
        # Compute per-pixel CE loss without reduction.
        log_prob = F.log_softmax(logits, dim=1)
        ce_loss  = F.nll_loss(
            log_prob, targets,
            ignore_index = self._ignore_index,
            reduction    = "none",
        )   # (B, H, W)

        # p_t = exp(-ce_loss) -- probability of the correct class.
        pt      = torch.exp(-ce_loss)
        focal   = self._alpha * ((1.0 - pt) ** self._gamma) * ce_loss

        # Mask out ignore pixels (ce_loss is 0 at those positions from nll_loss).
        valid   = (targets != self._ignore_index)
        if valid.sum() == 0:
            return focal.sum() * 0.0
        return focal[valid].mean()


# ==============================================================================
# CombinedLoss
# ==============================================================================

@LossRegistry.register
class CombinedLoss(nn.Module):
    """
    Weighted combination of CrossEntropyLoss and DiceLoss.

    combined = ce_weight * CE + dice_weight * Dice

    Args:
        config: LossConfig supplying ce_weight and dice_weight in addition
                to all CE and Dice hyperparameters.
    """

    loss_name: str = "combined"

    def __init__(self, config: LossConfig) -> None:
        super().__init__()
        self._ce_loss   = CrossEntropyLoss(config)
        self._dice_loss = DiceLoss(config)
        self._ce_w      = config.ce_weight
        self._dice_w    = config.dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, C, H, W) float32.
            targets: (B, H, W)    int64.
        Returns:
            Weighted scalar loss.
        """
        return (
            self._ce_w   * self._ce_loss(logits, targets)
            + self._dice_w * self._dice_loss(logits, targets)
        )

    def set_weight(self, weight: torch.Tensor) -> None:
        """Pass per-class weights to the CE component."""
        self._ce_loss.set_weight(weight)
