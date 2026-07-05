"""
Abstract base class for all segmentation models (Module 13).

SegmentationModel defines the stable interface that ModelFactory, the
Training Engine (Module 14), and the Inference Pipeline (Module 16)
depend on. Concrete models register themselves in ModelRegistry and
are never instantiated outside ModelFactory.

Interface invariants
--------------------
1. forward(x) returns logits ONLY. No softmax, sigmoid, or argmax.
2. forward() accepts (B, C, H, W) float32 tensors with arbitrary C, H, W.
3. Concrete models must implement forward() and may override the optional
   hooks (initialize_weights, model_name) but must not break the base contract.
4. All models are torch.nn.Module subclasses, which gives them
   .parameters(), .state_dict(), .to(device), and script/trace compatibility.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Any

_LOGGER: logging.Logger = logging.getLogger(__name__)

__all__ = ["SegmentationModel"]


class SegmentationModel:
    """
    Abstract base for all river morphology segmentation models.

    This class intentionally does NOT directly inherit from torch.nn.Module
    at the class definition level to avoid importing torch at module import
    time. At runtime, every concrete subclass inherits from both
    SegmentationModel and torch.nn.Module (see unetplusplus.py for the
    concrete pattern).

    Subclasses must implement:
        forward(x):    The prediction pass. Returns logits (B, num_classes, H, W).

    Subclasses should implement:
        model_name:    Class-level str constant used by ModelRegistry.

    Subclasses may override:
        initialize_weights():  Called at the end of __init__ when init_seed is set.
    """

    #: Registered name. Concrete models override this at class level.
    model_name: str = ""

    @abstractmethod
    def forward(self, x: Any) -> Any:
        """
        Compute segmentation logits for a batch of images.

        Args:
            x: float32 torch.Tensor of shape (B, C, H, W).
               C must match the in_channels this model was constructed with.
               H and W can be any positive integers (not restricted to powers
               of 2, though encoder depth imposes a minimum spatial size).

        Returns:
            When deep_supervision=False:
                float32 torch.Tensor of shape (B, num_classes, H, W).
                Values are raw logits; no activation is applied.

            When deep_supervision=True:
                float32 torch.Tensor of shape (B, num_classes, H, W).
                This is the average of all auxiliary outputs, all upsampled
                to the input resolution.  The Training Engine receives this
                single tensor; it does not need to know about auxiliary heads.
        """

    def initialize_weights(self) -> None:
        """
        Initialize model weights.

        Default implementation applies Kaiming Normal initialization to all
        Conv2d layers and constant initialization to all BatchNorm2d layers.
        Subclasses may override this method for custom initialization schemes.

        This method is called automatically by ModelFactory after construction
        when ModelConfig.init_seed is not None.
        """
        try:
            import torch.nn as nn
        except ImportError:
            return
        
        import torch
        cfg = getattr(self, "_config", None)
        if cfg is not None and getattr(cfg, "init_seed", None) is not None:
            torch.manual_seed(cfg.init_seed)

        for module in self.modules():  # type: ignore[attr-defined]
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def count_parameters(self) -> tuple[int, int]:
        """
        Count total and trainable parameters.

        Returns:
            (total, trainable) as (int, int).
        """
        total     = sum(p.numel() for p in self.parameters())  # type: ignore[attr-defined]
        trainable = sum(
            p.numel()
            for p in self.parameters()  # type: ignore[attr-defined]
            if p.requires_grad
        )
        return total, trainable
