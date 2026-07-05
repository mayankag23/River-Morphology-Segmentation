"""
Tests for src/training/models/blocks.py

Run:
    pytest tests/training/models/test_model_blocks.py -v \
        --cov=src/training/models/blocks --cov-report=term-missing
"""

from __future__ import annotations

import pytest
import torch

from src.training.models.blocks import (
    ConvBnAct,
    DoubleConv,
    ResidualBlock,
    build_act,
    build_norm,
)


class TestBuildNorm:
    def test_batch_norm(self) -> None:
        import torch.nn as nn
        layer = build_norm("batch", 32)
        assert isinstance(layer, nn.BatchNorm2d)

    def test_instance_norm(self) -> None:
        import torch.nn as nn
        layer = build_norm("instance", 16)
        assert isinstance(layer, nn.InstanceNorm2d)

    def test_none_returns_identity(self) -> None:
        import torch.nn as nn
        layer = build_norm("none", 8)
        assert isinstance(layer, nn.Identity)

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="norm_type"):
            build_norm("groupnorm", 32)


class TestBuildAct:
    def test_relu(self) -> None:
        import torch.nn as nn
        act = build_act("relu")
        assert isinstance(act, nn.ReLU)

    def test_leaky_relu(self) -> None:
        import torch.nn as nn
        act = build_act("leaky_relu")
        assert isinstance(act, nn.LeakyReLU)

    def test_gelu(self) -> None:
        import torch.nn as nn
        act = build_act("gelu")
        assert isinstance(act, nn.GELU)

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="act_type"):
            build_act("selu")


class TestConvBnAct:
    def _x(self, c: int = 4, h: int = 8, w: int = 8) -> torch.Tensor:
        return torch.randn(2, c, h, w)

    def test_output_shape_preserved(self) -> None:
        layer = ConvBnAct(4, 16)
        out   = layer(self._x())
        assert out.shape == (2, 16, 8, 8)

    def test_arbitrary_channels(self) -> None:
        layer = ConvBnAct(11, 33, norm_type="batch", act_type="relu")
        out   = layer(self._x(c=11))
        assert out.shape == (2, 33, 8, 8)

    def test_instance_norm_gelu(self) -> None:
        layer = ConvBnAct(4, 8, norm_type="instance", act_type="gelu")
        out   = layer(self._x())
        assert out.shape == (2, 8, 8, 8)

    def test_output_dtype_float32(self) -> None:
        layer = ConvBnAct(4, 8)
        assert layer(self._x()).dtype == torch.float32

    def test_no_norm(self) -> None:
        layer = ConvBnAct(4, 8, norm_type="none")
        out   = layer(self._x())
        assert out.shape == (2, 8, 8, 8)


class TestDoubleConv:
    def _x(self, c: int = 4) -> torch.Tensor:
        return torch.randn(2, c, 16, 16)

    def test_output_shape(self) -> None:
        block = DoubleConv(4, 8)
        assert block(self._x()).shape == (2, 8, 16, 16)

    def test_multi_band_input(self) -> None:
        block = DoubleConv(12, 32)
        assert block(self._x(c=12)).shape == (2, 32, 16, 16)

    def test_dropout_enabled(self) -> None:
        block = DoubleConv(4, 8, dropout_rate=0.5)
        block.train()
        out = block(self._x())
        assert out.shape == (2, 8, 16, 16)

    def test_dropout_zero_is_identity(self) -> None:
        import torch.nn as nn
        block = DoubleConv(4, 8, dropout_rate=0.0)
        assert isinstance(block.dropout, nn.Identity)

    def test_spatial_dims_preserved(self) -> None:
        for h, w in [(64, 64), (33, 57), (128, 96)]:
            x   = torch.randn(1, 4, h, w)
            out = DoubleConv(4, 8)(x)
            assert out.shape[2] == h and out.shape[3] == w


class TestResidualBlock:
    def _x(self, c: int = 4) -> torch.Tensor:
        return torch.randn(2, c, 16, 16)

    def test_same_channel_shape(self) -> None:
        block = ResidualBlock(8, 8)
        assert block(self._x(c=8)).shape == (2, 8, 16, 16)

    def test_channel_expansion(self) -> None:
        block = ResidualBlock(4, 16)
        assert block(self._x(c=4)).shape == (2, 16, 16, 16)

    def test_shortcut_is_conv_when_channels_differ(self) -> None:
        import torch.nn as nn
        block = ResidualBlock(4, 16)
        assert isinstance(block.shortcut, nn.Sequential)

    def test_shortcut_is_identity_when_same_channels(self) -> None:
        import torch.nn as nn
        block = ResidualBlock(8, 8)
        assert isinstance(block.shortcut, nn.Identity)

    def test_gradient_flows_through_residual(self) -> None:
        block = ResidualBlock(4, 4)
        x     = self._x(c=4).requires_grad_(True)
        loss  = block(x).sum()
        loss.backward()
        assert x.grad is not None
