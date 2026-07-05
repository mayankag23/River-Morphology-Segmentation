"""
UNet++ segmentation model for the River Morphology Segmentation project
(Module 13).

UNet++ (Zhou et al., 2018) is a nested U-Net architecture with dense skip
connections between encoder and decoder nodes. It consistently outperforms
the original UNet on medical and remote sensing segmentation benchmarks by
aggregating features at multiple scales simultaneously.

Registration
------------
UNetPlusPlus is registered with ModelRegistry at module import time via
the @ModelRegistry.register decorator, with the key "unetplusplus".

Config key: config.model.architecture = "unetplusplus"

Usage via ModelFactory (the recommended path)
----------------------------------------------
    from src.training.models.factory import ModelFactory
    result = ModelFactory.build(config)
    logits = result.model(image_batch)   # (B, num_classes, H, W)

Direct instantiation (for tests and debugging)
----------------------------------------------
    from src.training.models.unetplusplus import UNetPlusPlus
    from src.training.models.contracts import ModelConfig, EncoderConfig, DecoderConfig

    cfg = ModelConfig(
        in_channels = 12,
        num_classes = 4,
        encoder     = EncoderConfig(filters=(32, 64, 128, 256, 512)),
        decoder     = DecoderConfig(deep_supervision=True),
    )
    model = UNetPlusPlus(cfg)

Forward pass
------------
    x      = torch.randn(2, 12, 256, 256)
    logits = model(x)   # (2, 4, 256, 256) -- raw logits, no activation
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from src.training.models.base import SegmentationModel
from src.training.models.contracts import ModelConfig
from src.training.models.decoder import UNetPlusPlusDecoder
from src.training.models.encoder import UNetEncoder
from src.training.models.registry import ModelRegistry

__all__ = ["UNetPlusPlus"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


@ModelRegistry.register
class UNetPlusPlus(SegmentationModel, nn.Module):
    """
    UNet++ segmentation model.

    Supports:
        - Arbitrary input channel count (multispectral, not just RGB).
        - Configurable encoder depth and filter counts.
        - BatchNorm / InstanceNorm / no normalisation.
        - ReLU / LeakyReLU / GELU activations.
        - MaxPool / AvgPool downsampling.
        - Optional spatial dropout in encoder and decoder.
        - Deep supervision with auxiliary heads at each decoder step.
        - Bilinear or nearest-neighbor upsampling in the decoder.
        - Deterministic weight initialization via ModelConfig.init_seed.

    Args:
        config: Fully specified ModelConfig. Use ModelConfig.from_config(config)
                to build from a project Config object.
    """

    model_name: str = "unetplusplus"

    def __init__(self, config: ModelConfig) -> None:
        # Both parents must be __init__-ed; nn.Module goes first.
        nn.Module.__init__(self)

        self._config          = config
        self._in_channels     = config.in_channels
        self._num_classes     = config.num_classes
        self._deep_supervision = config.decoder.deep_supervision

        _LOGGER.debug(
            "UNetPlusPlus: in_channels=%d, num_classes=%d, filters=%s, "
            "deep_supervision=%s",
            config.in_channels,
            config.num_classes,
            config.encoder.filters,
            config.decoder.deep_supervision,
        )

        # Encoder.
        self.encoder = UNetEncoder(
            in_channels  = config.in_channels,
            filters      = config.encoder.filters,
            norm_type    = config.encoder.norm_type,
            act_type     = config.encoder.act_type,
            pool_type    = config.encoder.pool_type,
            dropout_rate = config.encoder.dropout_rate,
        )

        # Decoder (dense nested skip connections + optional deep supervision).
        self.decoder = UNetPlusPlusDecoder(
            encoder_filters = config.encoder.filters,
            num_classes      = config.num_classes,
            norm_type        = config.decoder.norm_type,
            act_type         = config.decoder.act_type,
            dropout_rate     = config.decoder.dropout_rate,
            deep_supervision = config.decoder.deep_supervision,
            upsample_mode    = config.decoder.upsample_mode,
        )

    @property
    def config(self) -> ModelConfig:
        """The ModelConfig used to construct this model."""
        return self._config

    @property
    def deep_supervision(self) -> bool:
        """True when the decoder emits auxiliary outputs."""
        return self._deep_supervision

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute segmentation logits.

        Args:
            x: float32 tensor (B, in_channels, H, W).
               H and W must satisfy H >= 2^depth and W >= 2^depth.

        Returns:
            float32 logit tensor (B, num_classes, H, W).
            Spatial resolution matches the input exactly (bilinear upsample).
            No softmax / sigmoid / argmax is applied.
        """
        encoder_out = self.encoder(x)
        logits      = self.decoder(encoder_out.features)
        return logits
