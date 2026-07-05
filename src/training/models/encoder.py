"""
Encoder for UNet-family segmentation models (Module 13).

UNetEncoder implements the contracting path of UNet / UNet++:
    - N stages, each comprising a DoubleConv block followed by pooling.
    - Stage features are stored as skip connections for the decoder.
    - The final (bottleneck) stage applies DoubleConv without pooling.

The number of stages and filter counts are fully determined by
EncoderConfig.filters. A 5-element tuple produces a 4-stage encoder
plus a bottleneck, matching the original UNet paper.

Minimum spatial constraint:
    For a depth-D encoder, each spatial dimension must be >= 2^D.
    For the default 5-filter config (depth=4), H and W must be >= 16.
    The minimum is enforced at forward() time with a clear error message.

No pretrained weights are loaded here (multispectral != ImageNet RGB).
The pretrained stub in ModelConfig is reserved for Module 16.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.training.models.blocks import DoubleConv, build_norm, build_act

__all__ = ["UNetEncoder", "EncoderOutput"]


class EncoderOutput:
    """
    Structured output of UNetEncoder.forward().

    Attributes:
        features:    List of feature maps at each encoder stage, ordered
                     from shallowest (index 0) to deepest (index -1 = bottleneck).
                     Each tensor has shape (B, C_i, H_i, W_i) where C_i is the
                     filter count at stage i and H_i, W_i are halved by pooling.
        bottleneck:  The deepest feature map (alias for features[-1]).
    """

    __slots__ = ("features",)

    def __init__(self, features: list[torch.Tensor]) -> None:
        self.features = features

    @property
    def bottleneck(self) -> torch.Tensor:
        """The deepest feature map (output of the final DoubleConv)."""
        return self.features[-1]

    @property
    def skip_connections(self) -> list[torch.Tensor]:
        """
        All feature maps except the bottleneck, in encoder order.

        These are the skip connections consumed by the decoder. Index 0 is
        the shallowest (highest resolution); index -1 is one level above
        the bottleneck.
        """
        return self.features[:-1]


class UNetEncoder(nn.Module):
    """
    Configurable UNet contracting-path encoder.

    Args:
        in_channels:    Number of input spectral channels (from ModelConfig).
        filters:        Ordered filter counts per stage. Length == num_stages.
                        The last entry is the bottleneck channel count.
        norm_type:      Normalisation layer: "batch", "instance", or "none".
        act_type:       Activation: "relu", "leaky_relu", or "gelu".
        pool_type:      Downsampling: "max" or "avg".
        dropout_rate:   Spatial dropout after each DoubleConv [0.0, 1.0].
    """

    def __init__(
        self,
        in_channels:  int,
        filters:      tuple[int, ...],
        norm_type:    str   = "batch",
        act_type:     str   = "relu",
        pool_type:    str   = "max",
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        if len(filters) < 2:
            raise ValueError(
                f"UNetEncoder: filters must have at least 2 entries "
                f"(got {len(filters)}). The last entry is the bottleneck."
            )

        self._filters     = filters
        self._depth       = len(filters) - 1   # number of pooling stages
        self._pool_type   = pool_type.lower().strip()

        # Build encoder stages (all except the last = bottleneck).
        stages: list[nn.Module] = []
        prev_ch = in_channels
        for ch in filters[:-1]:
            stages.append(
                DoubleConv(
                    in_channels  = prev_ch,
                    out_channels = ch,
                    norm_type    = norm_type,
                    act_type     = act_type,
                    dropout_rate = dropout_rate,
                )
            )
            prev_ch = ch
        self.stages = nn.ModuleList(stages)

        # Bottleneck (no pooling after this).
        self.bottleneck = DoubleConv(
            in_channels  = prev_ch,
            out_channels = filters[-1],
            norm_type    = norm_type,
            act_type     = act_type,
            dropout_rate = dropout_rate,
        )

        # Pooling layer (reused at every stage).
        if self._pool_type == "max":
            self._pool: nn.Module = nn.MaxPool2d(kernel_size=2, stride=2)
        elif self._pool_type == "avg":
            self._pool = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            raise ValueError(
                f"UNetEncoder: unsupported pool_type='{pool_type}'. "
                f"Choose 'max' or 'avg'."
            )

    @property
    def out_channels(self) -> tuple[int, ...]:
        """Filter count at each stage (including bottleneck), shallowest first."""
        return tuple(self._filters)

    @property
    def depth(self) -> int:
        """Number of downsampling stages (= len(filters) - 1)."""
        return self._depth

    def forward(self, x: torch.Tensor) -> EncoderOutput:
        """
        Run the contracting path.

        Args:
            x: (B, in_channels, H, W) float32 tensor.

        Returns:
            EncoderOutput with features at each stage + bottleneck.

        Raises:
            ValueError: Input spatial dimensions are too small for encoder depth.
        """
        min_size = 2 ** self._depth
        h, w     = x.shape[2], x.shape[3]
        if h < min_size or w < min_size:
            raise ValueError(
                f"UNetEncoder: input spatial size ({h}, {w}) is too small "
                f"for encoder depth {self._depth}. "
                f"Minimum required: ({min_size}, {min_size})."
            )

        features: list[torch.Tensor] = []
        current = x
        for stage in self.stages:
            current = stage(current)
            features.append(current)
            current = self._pool(current)

        bottleneck = self.bottleneck(current)
        features.append(bottleneck)

        return EncoderOutput(features)
