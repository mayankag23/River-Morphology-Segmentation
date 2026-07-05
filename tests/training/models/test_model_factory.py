"""
Tests for src/training/models/factory.py

Run:
    pytest tests/training/models/test_model_factory.py -v \
        --cov=src/training/models/factory --cov-report=term-missing
"""

from __future__ import annotations

import pytest
import torch

from src.training.models.contracts import DecoderConfig, EncoderConfig, ModelConfig
from src.training.models.factory import ModelFactory
from src.training.models.registry import ModelRegistry


def _small_cfg(
    in_ch:    int  = 4,
    n_cls:    int  = 4,
    deep_sup: bool = False,
    seed:     int | None = None,
) -> ModelConfig:
    return ModelConfig(
        in_channels  = in_ch,
        num_classes  = n_cls,
        encoder      = EncoderConfig(filters=(8, 16, 32)),
        decoder      = DecoderConfig(deep_supervision=deep_sup),
        init_seed    = seed,
    )


class TestModelFactory:
    def test_returns_model_result(self) -> None:
        from src.training.models.contracts import ModelResult
        result = ModelFactory.build_from_model_config(_small_cfg())
        assert isinstance(result, ModelResult)

    def test_result_is_frozen(self) -> None:
        result = ModelFactory.build_from_model_config(_small_cfg())
        with pytest.raises((AttributeError, TypeError)):
            result.num_classes = 8  # type: ignore[misc]

    def test_model_can_forward(self) -> None:
        result = ModelFactory.build_from_model_config(_small_cfg())
        result.model.eval()
        with torch.no_grad():
            out = result.model(torch.randn(1, 4, 32, 32))
        assert out.shape == (1, 4, 32, 32)

    def test_architecture_field(self) -> None:
        result = ModelFactory.build_from_model_config(_small_cfg())
        assert result.architecture == "unetplusplus"

    def test_in_channels_field(self) -> None:
        result = ModelFactory.build_from_model_config(_small_cfg(in_ch=12))
        assert result.in_channels == 12

    def test_num_classes_field(self) -> None:
        result = ModelFactory.build_from_model_config(_small_cfg(n_cls=3))
        assert result.num_classes == 3

    def test_num_parameters_positive(self) -> None:
        result = ModelFactory.build_from_model_config(_small_cfg())
        assert result.num_parameters > 0
        assert result.num_trainable > 0

    def test_num_parameters_equals_sum(self) -> None:
        result = ModelFactory.build_from_model_config(_small_cfg())
        expected = sum(p.numel() for p in result.model.parameters())
        assert result.num_parameters == expected

    def test_deep_supervision_field(self) -> None:
        result = ModelFactory.build_from_model_config(_small_cfg(deep_sup=True))
        assert result.deep_supervision is True

    def test_operations_log_non_empty(self) -> None:
        result = ModelFactory.build_from_model_config(_small_cfg())
        assert len(result.operations_log) > 0

    def test_summary_lines_non_empty(self) -> None:
        result = ModelFactory.build_from_model_config(_small_cfg())
        lines  = result.summary_lines()
        assert len(lines) > 0
        assert all(isinstance(l, str) for l in lines)

    def test_unknown_architecture_raises(self) -> None:
        cfg = ModelConfig(
            architecture = "does_not_exist_xyz",
            in_channels  = 4,
            num_classes  = 4,
            encoder      = EncoderConfig(filters=(8, 16)),
        )
        with pytest.raises(KeyError, match="not registered"):
            ModelFactory.build_from_model_config(cfg)

    def test_invalid_in_channels_raises(self) -> None:
        cfg = ModelConfig(
            in_channels = 0,
            num_classes = 4,
            encoder     = EncoderConfig(filters=(8, 16)),
        )
        with pytest.raises(ValueError, match="in_channels"):
            ModelFactory.build_from_model_config(cfg)

    def test_invalid_num_classes_raises(self) -> None:
        cfg = ModelConfig(
            in_channels = 4,
            num_classes = 0,
            encoder     = EncoderConfig(filters=(8, 16)),
        )
        with pytest.raises(ValueError, match="num_classes"):
            ModelFactory.build_from_model_config(cfg)

    def test_invalid_encoder_filters_raises(self) -> None:
        cfg = ModelConfig(
            in_channels = 4,
            num_classes = 4,
            encoder     = EncoderConfig(filters=(0, 16, 32)),
        )
        with pytest.raises(ValueError, match="filter"):
            ModelFactory.build_from_model_config(cfg)

    def test_invalid_dropout_raises(self) -> None:
        cfg = ModelConfig(
            in_channels = 4,
            num_classes = 4,
            encoder     = EncoderConfig(filters=(8, 16), dropout_rate=1.5),
        )
        with pytest.raises(ValueError, match="dropout_rate"):
            ModelFactory.build_from_model_config(cfg)

    def test_deterministic_with_seed(self) -> None:
        r1 = ModelFactory.build_from_model_config(_small_cfg(seed=123))
        r2 = ModelFactory.build_from_model_config(_small_cfg(seed=123))
        for p1, p2 in zip(r1.model.parameters(), r2.model.parameters()):
            assert torch.allclose(p1, p2)

    def test_build_from_config_uses_model_config_from_config(self) -> None:
        """ModelFactory.build(config) must call ModelConfig.from_config(config)."""
        class _EncCfg:
            filters      = [8, 16, 32]
            dropout_rate = 0.0
            norm_type    = "batch"
            act_type     = "relu"
            pool_type    = "max"

        class _DecCfg:
            dropout_rate     = 0.0
            norm_type        = "batch"
            act_type         = "relu"
            deep_supervision = False
            upsample_mode    = "bilinear"

        class _ModelCfg:
            architecture = "unetplusplus"
            in_channels  = 6
            num_classes  = 3
            init_seed    = None
            pretrained   = False
            encoder      = _EncCfg()
            decoder      = _DecCfg()

        class _Cfg:
            model = _ModelCfg()

        result = ModelFactory.build(_Cfg())
        assert result.in_channels == 6
        assert result.num_classes  == 3
