"""
Tests for src/training/models/contracts.py

Run:
    pytest tests/training/models/test_model_contracts.py -v \
        --cov=src/training/models/contracts --cov-report=term-missing
"""

from __future__ import annotations

import pytest

from src.training.models.contracts import (
    DecoderConfig,
    EncoderConfig,
    ModelConfig,
    ModelResult,
)


class TestEncoderConfig:
    def test_frozen(self) -> None:
        cfg = EncoderConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.filters = (64,)  # type: ignore[misc]

    def test_defaults(self) -> None:
        cfg = EncoderConfig()
        assert cfg.filters      == (32, 64, 128, 256, 512)
        assert cfg.dropout_rate == 0.0
        assert cfg.norm_type    == "batch"
        assert cfg.act_type     == "relu"
        assert cfg.pool_type    == "max"

    def test_from_config_no_section_returns_defaults(self) -> None:
        class _Cfg:
            pass
        cfg = EncoderConfig.from_config(_Cfg())
        assert cfg == EncoderConfig()

    def test_from_config_reads_values(self) -> None:
        class _Enc:
            filters      = [16, 32, 64]
            dropout_rate = 0.2
            norm_type    = "instance"
            act_type     = "leaky_relu"
            pool_type    = "avg"

        class _Cfg:
            class model:
                encoder = _Enc()

        cfg = EncoderConfig.from_config(_Cfg())
        assert cfg.filters      == (16, 32, 64)
        assert cfg.dropout_rate == pytest.approx(0.2)
        assert cfg.norm_type    == "instance"
        assert cfg.act_type     == "leaky_relu"
        assert cfg.pool_type    == "avg"


class TestDecoderConfig:
    def test_frozen(self) -> None:
        cfg = DecoderConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.dropout_rate = 0.5  # type: ignore[misc]

    def test_defaults(self) -> None:
        cfg = DecoderConfig()
        assert cfg.dropout_rate    == 0.0
        assert cfg.deep_supervision is False
        assert cfg.upsample_mode   == "bilinear"

    def test_from_config_reads_deep_supervision(self) -> None:
        class _Dec:
            dropout_rate     = 0.1
            norm_type        = "batch"
            act_type         = "relu"
            deep_supervision = True
            upsample_mode    = "nearest"

        class _Cfg:
            class model:
                decoder = _Dec()

        cfg = DecoderConfig.from_config(_Cfg())
        assert cfg.deep_supervision is True
        assert cfg.upsample_mode    == "nearest"


class TestModelConfig:
    def test_frozen(self) -> None:
        cfg = ModelConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.num_classes = 8  # type: ignore[misc]

    def test_defaults(self) -> None:
        cfg = ModelConfig()
        assert cfg.architecture == "unetplusplus"
        assert cfg.in_channels  == 12
        assert cfg.num_classes  == 4
        assert cfg.init_seed    is None
        assert cfg.pretrained   is False

    def test_from_config_reads_all_fields(self) -> None:
        class _Model:
            architecture = "unetplusplus"
            in_channels  = 6
            num_classes  = 3
            init_seed    = 42
            pretrained   = False
            class encoder:
                filters      = [8, 16]
                dropout_rate = 0.0
                norm_type    = "batch"
                act_type     = "relu"
                pool_type    = "max"
            class decoder:
                dropout_rate     = 0.0
                norm_type        = "batch"
                act_type         = "relu"
                deep_supervision = False
                upsample_mode    = "bilinear"

        class _Cfg:
            model = _Model()

        cfg = ModelConfig.from_config(_Cfg())
        assert cfg.in_channels  == 6
        assert cfg.num_classes  == 3
        assert cfg.init_seed    == 42

    def test_from_config_no_model_section_returns_defaults(self) -> None:
        class _Cfg:
            pass
        cfg = ModelConfig.from_config(_Cfg())
        assert cfg == ModelConfig()


class TestModelResult:
    def _make(self) -> ModelResult:
        import types
        mock_model = types.SimpleNamespace()
        return ModelResult(
            model            = mock_model,
            architecture     = "unetplusplus",
            in_channels      = 12,
            num_classes      = 4,
            num_parameters   = 1_000_000,
            num_trainable    = 1_000_000,
            deep_supervision = False,
            config           = ModelConfig(),
            operations_log   = ("step1", "step2"),
        )

    def test_frozen(self) -> None:
        r = self._make()
        with pytest.raises((AttributeError, TypeError)):
            r.num_classes = 8  # type: ignore[misc]

    def test_summary_lines_non_empty(self) -> None:
        lines = self._make().summary_lines()
        assert len(lines) > 0
        assert all(isinstance(l, str) for l in lines)

    def test_summary_lines_ascii_only(self) -> None:
        for line in self._make().summary_lines():
            assert all(ord(c) < 128 for c in line), f"Non-ASCII in: {line!r}"
