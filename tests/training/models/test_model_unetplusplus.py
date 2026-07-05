"""
Tests for src/training/models/unetplusplus.py

Covers:
    - Forward pass shape correctness
    - Arbitrary input channels (multispectral, not RGB)
    - Arbitrary spatial sizes (non-power-of-2, large, small)
    - Configurable class counts
    - Deep supervision mode
    - Normalization type variants
    - Activation type variants
    - Dropout variants
    - Gradient flow (backprop)
    - No activation on output (raw logits)
    - Deterministic initialization with seed
    - model_name registration key

Run:
    pytest tests/training/models/test_model_unetplusplus.py -v \
        --cov=src/training/models/unetplusplus --cov-report=term-missing
"""

from __future__ import annotations

import pytest
import torch

from src.training.models.contracts import DecoderConfig, EncoderConfig, ModelConfig
from src.training.models.unetplusplus import UNetPlusPlus


def _cfg(
    in_ch:     int             = 4,
    n_cls:     int             = 4,
    filters:   tuple[int, ...] = (8, 16, 32),
    deep_sup:  bool            = False,
    norm_type: str             = "batch",
    act_type:  str             = "relu",
    dropout:   float           = 0.0,
    seed:      int | None      = None,
) -> ModelConfig:
    return ModelConfig(
        architecture = "unetplusplus",
        in_channels  = in_ch,
        num_classes  = n_cls,
        encoder      = EncoderConfig(
            filters      = filters,
            norm_type    = norm_type,
            act_type     = act_type,
            dropout_rate = dropout,
        ),
        decoder      = DecoderConfig(
            deep_supervision = deep_sup,
            norm_type        = norm_type,
            act_type         = act_type,
        ),
        init_seed = seed,
    )


def _model(cfg: ModelConfig | None = None, **kwargs) -> UNetPlusPlus:
    if cfg is None:
        cfg = _cfg(**kwargs)
    return UNetPlusPlus(cfg)


class TestUNetPlusPlusForward:
    """Basic forward pass shape invariants."""

    def test_output_shape_matches_input_spatial(self) -> None:
        model = _model()
        x     = torch.randn(2, 4, 64, 64)
        out   = model(x)
        assert out.shape == (2, 4, 64, 64)

    def test_batch_size_preserved(self) -> None:
        model = _model()
        for bs in [1, 3, 8]:
            out = model(torch.randn(bs, 4, 32, 32))
            assert out.shape[0] == bs

    def test_num_classes_in_output(self) -> None:
        for n in [1, 2, 4, 8]:
            model = _model(n_cls=n, filters=(8, 16, 32))
            out   = model(torch.randn(1, 4, 32, 32))
            assert out.shape[1] == n

    def test_output_dtype_float32(self) -> None:
        model = _model()
        assert model(torch.randn(1, 4, 32, 32)).dtype == torch.float32

    def test_no_nan_in_output(self) -> None:
        model = _model()
        out   = model(torch.randn(1, 4, 32, 32))
        assert not torch.isnan(out).any()

    def test_no_inf_in_output(self) -> None:
        model = _model()
        out   = model(torch.randn(1, 4, 32, 32))
        assert not torch.isinf(out).any()


class TestUNetPlusPlusChannels:
    """Multispectral / arbitrary channel count tests."""

    def test_single_channel_input(self) -> None:
        model = _model(in_ch=1)
        out   = model(torch.randn(1, 1, 32, 32))
        assert out.shape[1] == 4

    def test_12_channel_multispectral(self) -> None:
        model = _model(in_ch=12)
        out   = model(torch.randn(1, 12, 32, 32))
        assert out.shape == (1, 4, 32, 32)

    def test_arbitrary_channel_counts(self) -> None:
        for c in [3, 6, 11, 20]:
            model = _model(in_ch=c, filters=(8, 16, 32))
            out   = model(torch.randn(1, c, 32, 32))
            assert out.shape[0] == 1


class TestUNetPlusPlusSpatialSizes:
    """Arbitrary and non-power-of-2 spatial sizes."""

    def test_square_power_of_2(self) -> None:
        model = _model(filters=(8, 16, 32))   # depth=2, min size=4
        for s in [16, 32, 64, 128, 256]:
            out = model(torch.randn(1, 4, s, s))
            assert out.shape[2] == s and out.shape[3] == s

    def test_non_power_of_2_spatial(self) -> None:
        model = _model(filters=(8, 16, 32))   # depth=2
        for h, w in [(33, 33), (48, 64), (100, 150)]:
            out = model(torch.randn(1, 4, h, w))
            assert out.shape[2] == h and out.shape[3] == w

    def test_rectangular_input(self) -> None:
        model = _model(filters=(8, 16, 32))
        out   = model(torch.randn(1, 4, 32, 64))
        assert out.shape == (1, 4, 32, 64)

    def test_minimum_spatial_size(self) -> None:
        """depth=2 -> minimum H,W = 4."""
        model = _model(filters=(8, 16, 32))
        model.eval()
        out   = model(torch.randn(1, 4, 4, 4))
        assert out.shape[2] == 4

    def test_too_small_spatial_raises(self) -> None:
        model = _model(filters=(8, 16, 32, 64, 128))   # depth=4, min=16
        with pytest.raises(ValueError, match="too small"):
            model(torch.randn(1, 4, 8, 8))


class TestUNetPlusPlusDeepSupervision:
    """Deep supervision mode."""

    def test_deep_supervision_output_same_shape(self) -> None:
        model = _model(deep_sup=True, filters=(8, 16, 32))
        x     = torch.randn(2, 4, 32, 32)
        out   = model(x)
        assert out.shape == (2, 4, 32, 32)

    def test_deep_supervision_property(self) -> None:
        model = _model(deep_sup=True)
        assert model.deep_supervision is True

    def test_no_deep_supervision_property(self) -> None:
        model = _model(deep_sup=False)
        assert model.deep_supervision is False

    def test_deep_supervision_gradient_flows(self) -> None:
        model = _model(deep_sup=True, filters=(8, 16, 32))
        x     = torch.randn(1, 4, 32, 32, requires_grad=True)
        model(x).sum().backward()
        assert x.grad is not None


class TestUNetPlusPlusNormAct:
    """Normalisation and activation variants."""

    def test_instance_norm(self) -> None:
        model = _model(norm_type="instance", filters=(8, 16, 32))
        out   = model(torch.randn(1, 4, 32, 32))
        assert out.shape == (1, 4, 32, 32)

    def test_no_norm(self) -> None:
        model = _model(norm_type="none", filters=(8, 16, 32))
        out   = model(torch.randn(1, 4, 32, 32))
        assert out.shape == (1, 4, 32, 32)

    def test_leaky_relu(self) -> None:
        model = _model(act_type="leaky_relu", filters=(8, 16, 32))
        out   = model(torch.randn(1, 4, 32, 32))
        assert out.shape == (1, 4, 32, 32)

    def test_gelu(self) -> None:
        model = _model(act_type="gelu", filters=(8, 16, 32))
        out   = model(torch.randn(1, 4, 32, 32))
        assert out.shape == (1, 4, 32, 32)


class TestUNetPlusPlusLogits:
    """Verify raw logits are returned (no activation applied)."""

    def test_logits_not_bounded_to_0_1(self) -> None:
        """Softmax would bound outputs to [0,1]; raw logits should exceed that."""
        model = _model()
        model.eval()
        with torch.no_grad():
            # Large random input to get extreme values.
            x      = torch.randn(1, 4, 32, 32) * 10.0
            logits = model(x)
        # At least some values should be outside [0, 1].
        outside = ((logits > 1.0) | (logits < 0.0)).any()
        assert outside.item(), "Expected raw logits outside [0,1]"

    def test_output_sums_not_one(self) -> None:
        """Softmax would make class dim sum to 1; logits should not."""
        model  = _model()
        model.eval()
        with torch.no_grad():
            logits = model(torch.randn(1, 4, 8, 8))
        class_sums = logits.sum(dim=1)   # (1, H, W)
        all_one = torch.allclose(class_sums, torch.ones_like(class_sums))
        assert not all_one, "Logits must not sum to 1 (no softmax in model)"


class TestUNetPlusPlusDeterminism:
    """Deterministic initialization via init_seed."""

    def test_same_seed_same_weights(self) -> None:
        cfg1 = _cfg(seed=99, filters=(8, 16, 32))
        cfg2 = _cfg(seed=99, filters=(8, 16, 32))
        m1   = UNetPlusPlus(cfg1)
        m1.initialize_weights()
        m2   = UNetPlusPlus(cfg2)
        m2.initialize_weights()
        for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
            assert torch.allclose(p1, p2), f"Parameter {n1} differs between identical seeds"

    def test_different_seeds_different_weights(self) -> None:
        cfg1 = _cfg(seed=1, filters=(8, 16, 32))
        cfg2 = _cfg(seed=2, filters=(8, 16, 32))
        m1   = UNetPlusPlus(cfg1)
        m1.initialize_weights()
        m2   = UNetPlusPlus(cfg2)
        m2.initialize_weights()
        any_different = any(
            not torch.allclose(p1, p2)
            for p1, p2 in zip(m1.parameters(), m2.parameters())
        )
        assert any_different, "Different seeds should produce different weights"

    def test_deterministic_forward_in_eval_mode(self) -> None:
        cfg   = _cfg(seed=7, filters=(8, 16, 32))
        model = UNetPlusPlus(cfg)
        model.initialize_weights()
        model.eval()
        x     = torch.randn(1, 4, 32, 32)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2)


class TestUNetPlusPlusGradients:
    """Gradient flow through the full model."""

    def test_gradients_flow_to_input(self) -> None:
        model = _model(filters=(8, 16, 32))
        x     = torch.randn(1, 4, 32, 32, requires_grad=True)
        model(x).sum().backward()
        assert x.grad is not None

    def test_all_parameters_receive_gradient(self) -> None:
        model = _model(filters=(8, 16, 32))
        x     = torch.randn(1, 4, 32, 32)
        model(x).sum().backward()
        params_without_grad = [
            n for n, p in model.named_parameters() if p.grad is None
        ]
        assert params_without_grad == [], (
            f"Parameters without grad: {params_without_grad}"
        )


class TestUNetPlusPlusRegistration:
    def test_model_name_attribute(self) -> None:
        assert UNetPlusPlus.model_name == "unetplusplus"

    def test_registered_in_registry(self) -> None:
        from src.training.models.registry import ModelRegistry
        assert ModelRegistry.is_registered("unetplusplus")

    def test_registry_returns_unetplusplus_class(self) -> None:
        from src.training.models.registry import ModelRegistry
        cls = ModelRegistry.get("unetplusplus")
        assert cls is UNetPlusPlus
