"""
src/training/models -- Segmentation Model Framework (Module 13).

Provides a registry-based, factory-driven interface for building
semantic segmentation models from project Config objects.

Input:   project Config (or ModelConfig directly)
Output:  ModelResult (immutable, wraps the instantiated torch.nn.Module)

Implemented models
------------------
    unetplusplus    UNetPlusPlus  (registered at import time)

Adding a new model
------------------
    1. Subclass SegmentationModel + nn.Module.
    2. Set model_name = "my_model".
    3. Decorate with @ModelRegistry.register.
    4. Import the module here to trigger registration.
    5. Set config.model.architecture = "my_model".

Usage
-----
    from src.training.models import ModelFactory

    result = ModelFactory.build(config)         # config-driven
    logits = result.model(image_batch)          # (B, num_classes, H, W) logits

    # Or with an explicit ModelConfig:
    from src.training.models import ModelConfig, EncoderConfig, DecoderConfig
    cfg    = ModelConfig(in_channels=12, num_classes=4)
    result = ModelFactory.build_from_model_config(cfg)
"""

# Contracts
from src.training.models.contracts import (
    EncoderConfig,
    DecoderConfig,
    ModelConfig,
    ModelResult,
)

# Abstract base
from src.training.models.base import SegmentationModel

# Registry
from src.training.models.registry import ModelRegistry

# Factory (also triggers registration of built-in models via its import)
from src.training.models.factory import ModelFactory

# Shared blocks (useful for custom model authors)
from src.training.models.blocks import (
    ConvBnAct,
    DoubleConv,
    ResidualBlock,
    build_norm,
    build_act,
)

# Sub-components (exposed for introspection and custom assembly)
from src.training.models.encoder import UNetEncoder, EncoderOutput
from src.training.models.decoder import UNetPlusPlusDecoder
from src.training.models.heads import SegmentationHead

# Concrete models
from src.training.models.unetplusplus import UNetPlusPlus

__all__ = [
    # Contracts
    "EncoderConfig",
    "DecoderConfig",
    "ModelConfig",
    "ModelResult",
    # Abstract base
    "SegmentationModel",
    # Registry + Factory (primary public interface)
    "ModelRegistry",
    "ModelFactory",
    # Blocks
    "ConvBnAct",
    "DoubleConv",
    "ResidualBlock",
    "build_norm",
    "build_act",
    # Sub-components
    "UNetEncoder",
    "EncoderOutput",
    "UNetPlusPlusDecoder",
    "SegmentationHead",
    # Concrete models
    "UNetPlusPlus",
]
