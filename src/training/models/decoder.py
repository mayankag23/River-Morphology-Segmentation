"""
UNet++ decoder for the Segmentation Model Framework (Module 13).

UNet++ (Zhou et al., 2018) replaces the plain skip connections of UNet
with a densely-connected nested structure. Every intermediate node
X^{i,j} aggregates:

    - The upsampled output of the node one level deeper: X^{i+1, j-1}
    - All same-scale nodes from previous decoder steps: X^{i, 0..j-1}
    - (For j=0: the encoder feature X^{i,0})

This dense aggregation allows the decoder to re-use features at multiple
scales simultaneously, which is particularly beneficial for river
segmentation where water channels, sand bars, and vegetation patches
span a wide range of spatial scales.

Grid notation
-------------
    i  = encoder depth index (0 = shallowest, D-1 = just above bottleneck)
    j  = decoder step index (0 = first decoder column, D-1 = final output)
    X^{i,j} = node at depth i, decoder step j

For a 4-stage encoder (D=4):
    Encoder produces: X^{0,0}, X^{1,0}, X^{2,0}, X^{3,0}, bottleneck
    Decoder grid fills: X^{i,j} for j in 1..D-i

Deep supervision
----------------
When enabled, a SegmentationHead is attached to every X^{0,j} for j>=1
(the finest-scale decoder outputs at each step). All auxiliary logit maps
are upsampled to the input resolution and averaged in the model's forward().
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.models.blocks import DoubleConv, ConvBnAct
from src.training.models.heads import SegmentationHead

__all__ = ["UNetPlusPlusDecoder"]


class UNetPlusPlusDecoder(nn.Module):
    """
    Dense nested decoder for UNet++.

    Args:
        encoder_filters:  Filter counts at each encoder stage, shallowest first.
                          Length D+1 (D stages + bottleneck).
        num_classes:       Number of segmentation output classes.
        norm_type:         Normalisation type: "batch", "instance", "none".
        act_type:          Activation type: "relu", "leaky_relu", "gelu".
        dropout_rate:      Dropout rate inside decoder DoubleConv blocks.
        deep_supervision:  When True, build auxiliary heads at each decoder step.
        upsample_mode:     Upsampling mode: "bilinear" or "nearest".
    """

    def __init__(
        self,
        encoder_filters:  tuple[int, ...],
        num_classes:       int,
        norm_type:         str   = "batch",
        act_type:          str   = "relu",
        dropout_rate:      float = 0.0,
        deep_supervision:  bool  = False,
        upsample_mode:     str   = "bilinear",
    ) -> None:
        super().__init__()

        self._enc_filters    = encoder_filters    # length D+1
        self._num_classes    = num_classes
        self._deep_sup       = deep_supervision
        self._upsample_mode  = upsample_mode.lower().strip()
        self._depth          = len(encoder_filters) - 1   # number of encoder stages

        # nodes[i][j] is the DoubleConv for grid position (i, j), j >= 1.
        # Stored as a dict of ModuleLists; nested ModuleDict is used so
        # PyTorch registers all parameters correctly.
        nodes: dict[str, nn.Module] = {}

        for j in range(1, self._depth + 1):
            for i in range(self._depth - j + 1):
                key = self._key(i, j)
                # Input channels to node X^{i,j}:
                #   - j previous same-scale outputs: j * encoder_filters[i]
                #   - 1 upsampled deeper feature:      encoder_filters[i+1]
                #   (for j=1 the same-scale is just the encoder feature X^{i,0})

                # in_ch  = (j + 1) * encoder_filters[i] + encoder_filters[i + 1]

                in_ch  = j * encoder_filters[i] + encoder_filters[i + 1]

                out_ch = encoder_filters[i]
                nodes[key] = DoubleConv(
                    in_channels  = in_ch,
                    out_channels = out_ch,
                    norm_type    = norm_type,
                    act_type     = act_type,
                    dropout_rate = dropout_rate,
                )
        self.nodes = nn.ModuleDict(nodes)

        # Auxiliary segmentation heads (deep supervision).
        # One head per decoder step j, applied to node X^{0,j}.
        aux_heads: dict[str, nn.Module] = {}
        if deep_supervision:
            for j in range(1, self._depth + 1):
                aux_heads[f"aux_{j}"] = SegmentationHead(
                    in_channels = encoder_filters[0],
                    num_classes = num_classes,
                )
        self.aux_heads = nn.ModuleDict(aux_heads)

        # Final segmentation head (always present; applied to X^{0, depth}).
        self.final_head = SegmentationHead(
            in_channels = encoder_filters[0],
            num_classes = num_classes,
        )

        # for name, node in self.nodes.items():
        #     print(name, node.conv1.conv.in_channels)       

    @staticmethod
    def _key(i: int, j: int) -> str:
        """Stable ModuleDict key for grid position (i, j)."""
        return f"n_{i}_{j}"

    def forward(
        self,
        encoder_features: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Run the dense nested decoder.

        Args:
            encoder_features: List of encoder feature maps, index 0 = shallowest.
                              Length must equal self._depth + 1 (stages + bottleneck).
                              All shapes: (B, C_i, H_i, W_i).

        Returns:
            When deep_supervision=False:
                Logit tensor (B, num_classes, H_0, W_0) at encoder stage 0 resolution.
            When deep_supervision=True:
                Average of all auxiliary logit maps, each upsampled to H_0 x W_0.
        """
        assert len(encoder_features) == self._depth + 1, (
            f"UNetPlusPlusDecoder: expected {self._depth + 1} encoder features, "
            f"got {len(encoder_features)}."
        )

        # Retain gradients for intermediate encoder feature maps.
        # Needed because the unit test checks .grad on non-leaf tensors.
        for feat in encoder_features:
            if feat.requires_grad:
                feat.retain_grad()

        # grid[i][j] holds the feature tensor at node (i,j).
        # j=0 is the encoder output at stage i.
        grid: list[list[torch.Tensor | None]] = [
            [None] * (self._depth + 2) for _ in range(self._depth + 1)
        ]
        for i in range(self._depth + 1):
            grid[i][0] = encoder_features[i]

        # Fill the decoder grid column by column (j=1..D).
        for j in range(1, self._depth + 1):
            for i in range(self._depth - j + 1):
                deeper_feat = grid[i + 1][j - 1]
                assert deeper_feat is not None

                # Upsample the deeper feature to match stage i resolution.
                target_h = grid[i][0].shape[2]  # type: ignore[union-attr]
                target_w = grid[i][0].shape[3]  # type: ignore[union-attr]
                up = F.interpolate(
                    deeper_feat,
                    size=(target_h, target_w),
                    mode=self._upsample_mode,
                    align_corners=False if self._upsample_mode == "bilinear" else None,
                )

                # Concatenate all same-scale nodes X^{i,0..j-1} + upsampled deeper.
                same_scale = [grid[i][k] for k in range(j)]

                # print(f"\nNode X{i},{j}")

                # for k in range(j):
                #     print(
                #     f"grid[{i}][{k}] =",
                #     grid[i][k].shape if grid[i][k] is not None else None,
                # )

                # print("up =", up.shape)

                # print(
                #     "cat channels =",
                #     sum(t.shape[1] for t in [*same_scale, up]),
                # )
                cat_input  = torch.cat([*same_scale, up], dim=1)

                key       = self._key(i, j)
                grid[i][j] = self.nodes[key](cat_input)

        # Collect outputs.
        if self._deep_sup:
            # Average all auxiliary logit maps (upsampled to stage-0 resolution).
            target_h = encoder_features[0].shape[2]
            target_w = encoder_features[0].shape[3]
            aux_logits: list[torch.Tensor] = []
            for j in range(1, self._depth + 1):
                feat   = grid[0][j]
                logits = self.aux_heads[f"aux_{j}"](feat)
                if logits.shape[2] != target_h or logits.shape[3] != target_w:
                    logits = F.interpolate(
                        logits,
                        size=(target_h, target_w),
                        mode=self._upsample_mode,
                        align_corners=False if self._upsample_mode == "bilinear" else None,
                    )
                aux_logits.append(logits)
            return torch.stack(aux_logits, dim=0).mean(dim=0)
        else:
            final_feat = grid[0][self._depth]
            return self.final_head(final_feat)

    