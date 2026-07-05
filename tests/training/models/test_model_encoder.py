"""
Tests for src/training/models/encoder.py

Run:
    pytest tests/training/models/test_model_encoder.py -v \
        --cov=src/training/models/encoder --cov-report=term-missing
"""

from __future__ import annotations

import pytest
import torch

from src.training.models.encoder import EncoderOutput, UNetEncoder


def _enc(
    in_ch:    int             = 4,
    filters:  tuple[int, ...] = (8, 16, 32),
    **kwargs,
) -> UNetEncoder:
    return UNetEncoder(in_channels=in_ch, filters=filters, **kwargs)


class TestUNetEncoder:
    def test_output_has_correct_number_of_features(self) -> None:
        enc = _enc(filters=(8, 16, 32))
        out = enc(torch.randn(2, 4, 32, 32))
        assert len(out.features) == 3   # 2 stages + 1 bottleneck

    def test_bottleneck_alias(self) -> None:
        enc = _enc(filters=(8, 16, 32))
        out = enc(torch.randn(2, 4, 32, 32))
        assert out.bottleneck is out.features[-1]

    def test_skip_connections_excludes_bottleneck(self) -> None:
        enc = _enc(filters=(8, 16, 32))
        out = enc(torch.randn(2, 4, 32, 32))
        assert len(out.skip_connections) == len(out.features) - 1

    def test_feature_channel_counts(self) -> None:
        enc     = _enc(filters=(8, 16, 32))
        out     = enc(torch.randn(2, 4, 32, 32))
        filters = (8, 16, 32)
        for feat, ch in zip(out.features, filters):
            assert feat.shape[1] == ch

    def test_spatial_halving_at_each_stage(self) -> None:
        enc = _enc(in_ch=4, filters=(8, 16, 32))
        out = enc(torch.randn(1, 4, 64, 64))
        # Stage 0: (64,64) -> (64,64) [pre-pool skip]; then pooled to (32,32)
        # Stage 1: (32,32) -> (32,32); then pooled to (16,16)
        # Bottleneck: (16,16) -> (16,16)
        assert out.features[0].shape[2] == 64
        assert out.features[1].shape[2] == 32
        assert out.features[2].shape[2] == 16

    def test_arbitrary_input_channels(self) -> None:
        enc = _enc(in_ch=12, filters=(16, 32))
        out = enc(torch.randn(2, 12, 32, 32))
        assert out.features[0].shape[1] == 16

    def test_too_small_input_raises(self) -> None:
        enc = _enc(filters=(8, 16, 32, 64, 128))  # depth=4 needs H,W >= 16
        with pytest.raises(ValueError, match="too small"):
            enc(torch.randn(1, 4, 8, 8))

    def test_avg_pool(self) -> None:
        enc = _enc(filters=(8, 16), pool_type="avg")
        out = enc(torch.randn(1, 4, 16, 16))
        assert out.features[0].shape[2] == 16  # before pooling

    def test_invalid_pool_type_raises(self) -> None:
        with pytest.raises(ValueError, match="pool_type"):
            _enc(filters=(8, 16), pool_type="strided")

    def test_out_channels_property(self) -> None:
        enc = _enc(filters=(8, 16, 32))
        assert enc.out_channels == (8, 16, 32)

    def test_depth_property(self) -> None:
        enc = _enc(filters=(8, 16, 32))
        assert enc.depth == 2

    def test_gradient_flows_through_encoder(self) -> None:
        enc = _enc(filters=(8, 16, 32))
        x   = torch.randn(1, 4, 32, 32, requires_grad=True)
        out = enc(x)
        out.bottleneck.sum().backward()
        assert x.grad is not None

    def test_single_filter_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            _enc(filters=(8,))

    def test_5_stage_default_depth(self) -> None:
        enc = _enc(filters=(8, 16, 32, 64, 128), in_ch=4)
        out = enc(torch.randn(1, 4, 64, 64))
        assert len(out.features) == 5
