"""
Tests for src/training/models/decoder.py

Run:
    pytest tests/training/models/test_model_decoder.py -v \
        --cov=src/training/models/decoder --cov-report=term-missing
"""

from __future__ import annotations

import pytest
import torch

from src.training.models.decoder import UNetPlusPlusDecoder
from src.training.models.encoder import UNetEncoder


def _encoder_features(
    filters:   tuple[int, ...] = (8, 16, 32),
    in_ch:     int             = 4,
    h:         int             = 32,
    w:         int             = 32,
) -> list[torch.Tensor]:
    enc  = UNetEncoder(in_channels=in_ch, filters=filters)
    x    = torch.randn(2, in_ch, h, w)
    return enc(x).features


class TestUNetPlusPlusDecoder:
    def test_output_shape_no_deep_sup(self) -> None:
        filters  = (8, 16, 32)
        feats    = _encoder_features(filters)
        decoder  = UNetPlusPlusDecoder(
            encoder_filters=filters, num_classes=4,
            deep_supervision=False,
        )
        out = decoder(feats)
        # Output resolution must match the shallowest encoder feature map
        assert out.shape == (2, 4, feats[0].shape[2], feats[0].shape[3])

    def test_output_shape_deep_supervision(self) -> None:
        filters  = (8, 16, 32)
        feats    = _encoder_features(filters)
        decoder  = UNetPlusPlusDecoder(
            encoder_filters=filters, num_classes=4,
            deep_supervision=True,
        )
        out = decoder(feats)
        assert out.shape == (2, 4, feats[0].shape[2], feats[0].shape[3])

    def test_num_classes_in_output(self) -> None:
        for nc in [1, 2, 4, 8]:
            feats   = _encoder_features()
            decoder = UNetPlusPlusDecoder(
                encoder_filters=(8, 16, 32), num_classes=nc,
            )
            assert decoder(feats).shape[1] == nc

    def test_output_dtype_float32(self) -> None:
        feats   = _encoder_features()
        decoder = UNetPlusPlusDecoder(encoder_filters=(8, 16, 32), num_classes=4)
        assert decoder(feats).dtype == torch.float32

    def test_nearest_upsample(self) -> None:
        feats   = _encoder_features()
        decoder = UNetPlusPlusDecoder(
            encoder_filters=(8, 16, 32), num_classes=4,
            upsample_mode="nearest",
        )
        out = decoder(feats)
        assert out.shape[1] == 4

    def test_gradient_flows_through_decoder(self) -> None:
        feats = [f.requires_grad_(True) for f in _encoder_features()]
        decoder = UNetPlusPlusDecoder(encoder_filters=(8, 16, 32), num_classes=4)
        decoder(feats).sum().backward()
        assert all(f.grad is not None for f in feats)

    def test_5_stage_encoder_depth(self) -> None:
        filters = (8, 16, 32, 64, 128)
        feats   = _encoder_features(filters=filters, h=64, w=64)
        decoder = UNetPlusPlusDecoder(encoder_filters=filters, num_classes=4)
        out     = decoder(feats)
        assert out.shape == (2, 4, feats[0].shape[2], feats[0].shape[3])

    def test_wrong_feature_count_raises(self) -> None:
        decoder = UNetPlusPlusDecoder(encoder_filters=(8, 16, 32), num_classes=4)
        # Only 2 features provided; need 3.
        feats = [torch.randn(2, 8, 32, 32), torch.randn(2, 16, 16, 16)]
        with pytest.raises(AssertionError):
            decoder(feats)
