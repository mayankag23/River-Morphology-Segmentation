"""
Tests for src/training/models/heads.py

Run:
    pytest tests/training/models/test_model_heads.py -v \
        --cov=src/training/models/heads --cov-report=term-missing
"""

from __future__ import annotations

import pytest
import torch

from src.training.models.heads import SegmentationHead


class TestSegmentationHead:
    def test_output_shape(self) -> None:
        head = SegmentationHead(in_channels=64, num_classes=4)
        x    = torch.randn(2, 64, 16, 16)
        out  = head(x)
        assert out.shape == (2, 4, 16, 16)

    def test_single_class_binary(self) -> None:
        head = SegmentationHead(in_channels=32, num_classes=1)
        out  = head(torch.randn(3, 32, 8, 8))
        assert out.shape == (3, 1, 8, 8)

    def test_output_dtype_float32(self) -> None:
        head = SegmentationHead(in_channels=16, num_classes=4)
        assert head(torch.randn(1, 16, 8, 8)).dtype == torch.float32

    def test_no_activation_applied(self) -> None:
        """Logit range must not be bounded by sigmoid/softmax."""
        head   = SegmentationHead(in_channels=8, num_classes=4)
        x      = torch.randn(1, 8, 4, 4) * 100.0
        logits = head(x)
        # Raw large activations: logits should have magnitude > 1
        assert logits.abs().max().item() > 1.0

    def test_spatial_dims_preserved(self) -> None:
        for h, w in [(64, 64), (33, 57), (1, 1)]:
            head = SegmentationHead(8, 4)
            out  = head(torch.randn(1, 8, h, w))
            assert out.shape[2] == h and out.shape[3] == w

    def test_invalid_in_channels_raises(self) -> None:
        with pytest.raises(ValueError, match="in_channels"):
            SegmentationHead(in_channels=0, num_classes=4)

    def test_invalid_num_classes_raises(self) -> None:
        with pytest.raises(ValueError, match="num_classes"):
            SegmentationHead(in_channels=8, num_classes=0)

    def test_gradient_flows_through_head(self) -> None:
        head = SegmentationHead(8, 4)
        x    = torch.randn(1, 8, 4, 4, requires_grad=True)
        loss = head(x).sum()
        loss.backward()
        assert x.grad is not None
