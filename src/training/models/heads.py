"""
Segmentation output heads for Module 13.

SegmentationHead converts decoder feature maps into per-class logit maps
using a 1x1 convolution. No activation is applied; the Training Engine
(Module 14) applies softmax / sigmoid as appropriate for the loss function.

A single SegmentationHead is the final layer of every model.
When deep supervision is enabled, UNetPlusPlusDecoder creates one
SegmentationHead per decoder scale; these auxiliary heads produce
intermediate logit maps that are upsampled and averaged in the model's
forward() method.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["SegmentationHead"]


class SegmentationHead(nn.Module):
    """
    Final 1x1 convolution that maps decoder features to class logits.

    Input:  (B, in_channels, H, W) float32 feature map from decoder.
    Output: (B, num_classes,  H, W) float32 logit map.

    No softmax, sigmoid, or argmax is ever applied here.

    Args:
        in_channels: Number of input feature channels from the decoder.
        num_classes: Number of output segmentation classes.
        kernel_size: Convolution kernel size (default 1 for 1x1 projection).
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        kernel_size: int = 1,
    ) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError(
                f"SegmentationHead: in_channels must be >= 1, got {in_channels}."
            )
        if num_classes < 1:
            raise ValueError(
                f"SegmentationHead: num_classes must be >= 1, got {num_classes}."
            )
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            num_classes,
            kernel_size = kernel_size,
            padding     = padding,
            bias        = True,   # bias is meaningful when there is no subsequent norm
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project feature map to logits.

        Args:
            x: (B, in_channels, H, W) float32 tensor.

        Returns:
            (B, num_classes, H, W) float32 logit tensor.
        """
        return self.conv(x)
