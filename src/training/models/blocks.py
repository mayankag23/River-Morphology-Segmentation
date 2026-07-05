"""
Shared convolutional building blocks for segmentation models (Module 13).

All blocks are configurable through constructor arguments that originate
from EncoderConfig / DecoderConfig (never hardcoded inside the block).

Supported normalizations: "batch", "instance", "none"
Supported activations:    "relu", "leaky_relu", "gelu"

Design rules
------------
- torch is imported at the top of this module (it is a required dependency
  for any module that instantiates blocks).
- Every block preserves spatial dimensions unless explicitly stated (no
  built-in downsampling; pooling is the encoder's responsibility).
- All blocks work on arbitrary channel counts (never assume RGB=3).
- All blocks accept (B, C, H, W) tensors.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "ConvBnAct",
    "DoubleConv",
    "ResidualBlock",
    "build_norm",
    "build_act",
]


# ==============================================================================
# Norm and activation factories
# ==============================================================================

def build_norm(norm_type: str, num_channels: int) -> nn.Module:
    """
    Build a normalisation layer by name.

    Args:
        norm_type:    "batch", "instance", or "none".
        num_channels: Channel count for the layer.

    Returns:
        An nn.Module or nn.Identity when norm_type is "none".

    Raises:
        ValueError: norm_type is not one of the supported values.
    """
    t = norm_type.lower().strip()
    if t == "batch":
        return nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3)
    if t == "instance":
        return nn.InstanceNorm2d(num_channels, affine=True)
    if t == "none":
        return nn.Identity()
    raise ValueError(
        f"build_norm: unsupported norm_type='{norm_type}'. "
        f"Choose from: 'batch', 'instance', 'none'."
    )


def build_act(act_type: str) -> nn.Module:
    """
    Build an activation function by name.

    Args:
        act_type: "relu", "leaky_relu", or "gelu".

    Returns:
        An nn.Module activation function.

    Raises:
        ValueError: act_type is not one of the supported values.
    """
    t = act_type.lower().strip()
    if t == "relu":
        return nn.ReLU(inplace=True)
    if t == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01, inplace=True)
    if t == "gelu":
        return nn.GELU()
    raise ValueError(
        f"build_act: unsupported act_type='{act_type}'. "
        f"Choose from: 'relu', 'leaky_relu', 'gelu'."
    )


# ==============================================================================
# ConvBnAct
# ==============================================================================

class ConvBnAct(nn.Module):
    """
    Conv2d -> Norm -> Activation.

    The most fundamental unit of the encoder/decoder: a 3x3 convolution
    with padding=1 (same spatial size), followed by normalisation and
    activation.

    Args:
        in_channels:  Input channel count.
        out_channels: Output channel count.
        kernel_size:  Convolution kernel size (default 3).
        padding:      Padding (default 1 for 3x3 same-conv).
        norm_type:    "batch", "instance", or "none".
        act_type:     "relu", "leaky_relu", or "gelu".
        bias:         Include bias in Conv2d. Default False (norm handles bias).
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kernel_size:  int  = 3,
        padding:      int  = 1,
        norm_type:    str  = "batch",
        act_type:     str  = "relu",
        bias:         bool = False,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size, padding=padding, bias=bias,
        )
        self.norm = build_norm(norm_type, out_channels)
        self.act  = build_act(act_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


# ==============================================================================
# DoubleConv
# ==============================================================================

class DoubleConv(nn.Module):
    """
    Two sequential ConvBnAct blocks: the standard UNet encoder/decoder unit.

    Input -> ConvBnAct -> ConvBnAct -> Output
    Spatial dimensions are preserved (same-padding 3x3 convolutions).

    Args:
        in_channels:  Input channel count.
        out_channels: Output channel count (applied to both conv layers).
        norm_type:    "batch", "instance", or "none".
        act_type:     "relu", "leaky_relu", or "gelu".
        dropout_rate: Spatial dropout rate after the second conv [0.0, 1.0].
                      0.0 disables dropout.
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        norm_type:    str   = "batch",
        act_type:     str   = "relu",
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv1   = ConvBnAct(in_channels,  out_channels, norm_type=norm_type, act_type=act_type)
        self.conv2   = ConvBnAct(out_channels, out_channels, norm_type=norm_type, act_type=act_type)
        self.dropout = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.conv2(self.conv1(x)))


# ==============================================================================
# ResidualBlock
# ==============================================================================

class ResidualBlock(nn.Module):
    """
    Residual block: ConvBnAct -> ConvBnAct + identity shortcut.

    When in_channels != out_channels, a 1x1 convolution adapts the shortcut.
    Spatial dimensions are preserved.

    Used as an optional drop-in replacement for DoubleConv when stronger
    gradient flow is needed in deep networks.

    Args:
        in_channels:  Input channel count.
        out_channels: Output channel count.
        norm_type:    Normalisation layer type.
        act_type:     Activation function type.
        dropout_rate: Spatial dropout rate [0.0, 1.0].
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        norm_type:    str   = "batch",
        act_type:     str   = "relu",
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv1   = ConvBnAct(in_channels,  out_channels, norm_type=norm_type, act_type=act_type)
        self.conv2   = ConvBnAct(out_channels, out_channels, norm_type=norm_type, act_type=act_type)
        self.dropout = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0.0 else nn.Identity()
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                build_norm(norm_type, out_channels),
            )
            if in_channels != out_channels
            else nn.Identity()
        )
        self.final_act = build_act(act_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out      = self.dropout(self.conv2(self.conv1(x)))
        return self.final_act(out + residual)
