"""
Model factory for the Segmentation Model Framework (Module 13).

ModelFactory is the single public entry point for model construction.
It queries ModelRegistry, instantiates the requested architecture,
applies deterministic weight initialization, counts parameters, and
returns an immutable ModelResult.

Module 14 (Training Engine) calls ModelFactory.build(config) and receives
a ModelResult; it never instantiates models directly. This decouples the
training engine from any concrete model class.

Usage
-----
    from src.training.models.factory import ModelFactory

    result = ModelFactory.build(config)
    logits = result.model(image_batch)
"""

from __future__ import annotations

import logging
from typing import Any

from src.training.models.contracts import ModelConfig, ModelResult
from src.training.models.registry import ModelRegistry

# Trigger registration of all built-in models.
import src.training.models.unetplusplus  # noqa: F401

__all__ = ["ModelFactory"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


class ModelFactory:
    """
    Builds SegmentationModel instances from config via ModelRegistry.

    This class has no instance state; all methods are class methods.
    It follows the Factory pattern: callers describe what they want
    through ModelConfig; ModelFactory decides how to build it.
    """

    @classmethod
    def build(
        cls,
        config:       Any,
        model_config: ModelConfig | None = None,
    ) -> ModelResult:
        """
        Build a model and return an immutable ModelResult.

        Args:
            config:        Fully initialized project Config object. Used to
                           construct ModelConfig when model_config is None.
            model_config:  Optional pre-built ModelConfig. When provided,
                           overrides config. Useful in tests and scripts that
                           want direct control.

        Returns:
            Frozen ModelResult with the instantiated model and metadata.

        Raises:
            KeyError:   The requested architecture is not registered.
            ValueError: ModelConfig contains invalid values.
        """
        operations: list[str] = []

        # Step 1: Resolve ModelConfig.
        if model_config is None:
            model_config = ModelConfig.from_config(config)
        operations.append(
            f"config: architecture={model_config.architecture}, "
            f"in_channels={model_config.in_channels}, "
            f"num_classes={model_config.num_classes}"
        )

        # Step 2: Validate.
        cls._validate(model_config)
        operations.append("validation: passed")

        # Step 3: Retrieve model class from registry.
        model_cls = ModelRegistry.get(model_config.architecture)
        operations.append(f"registry: resolved '{model_config.architecture}' -> {model_cls.__name__}")

        # Step 4: Seed RNG for deterministic initialization.
        if model_config.init_seed is not None:
            try:
                import torch
                torch.manual_seed(model_config.init_seed)
            except ImportError:
                pass
            operations.append(f"seed: {model_config.init_seed}")

        # Step 5: Instantiate model.
        model = model_cls(model_config)
        operations.append(f"instantiate: {model_cls.__name__}")

        # Step 6: Initialize weights when seed is pinned.
        if model_config.init_seed is not None:
            model.initialize_weights()
            operations.append("initialize_weights: done")

        # Step 7: Count parameters.
        total, trainable = model.count_parameters()
        operations.append(
            f"parameters: total={total:,}, trainable={trainable:,}"
        )

        _LOGGER.info(
            "ModelFactory: built %s with %d trainable parameters.",
            model_cls.__name__, trainable,
        )

        result = ModelResult(
            model            = model,
            architecture     = model_config.architecture,
            in_channels      = model_config.in_channels,
            num_classes      = model_config.num_classes,
            num_parameters   = total,
            num_trainable    = trainable,
            deep_supervision = model_config.decoder.deep_supervision,
            config           = model_config,
            operations_log   = tuple(operations),
        )

        for line in result.summary_lines():
            _LOGGER.info(line)

        return result

    @classmethod
    def build_from_model_config(cls, model_config: ModelConfig) -> ModelResult:
        """
        Convenience method when a ModelConfig is already available.

        Equivalent to ModelFactory.build(config=None, model_config=model_config).

        Args:
            model_config: Fully specified ModelConfig.

        Returns:
            Frozen ModelResult.
        """
        return cls.build(config=None, model_config=model_config)

    @staticmethod
    def _validate(cfg: ModelConfig) -> None:
        """
        Validate ModelConfig field values.

        Raises:
            ValueError: Any field is out of range or semantically invalid.
        """
        if cfg.in_channels < 1:
            raise ValueError(
                f"ModelConfig.in_channels must be >= 1, got {cfg.in_channels}."
            )
        if cfg.num_classes < 1:
            raise ValueError(
                f"ModelConfig.num_classes must be >= 1, got {cfg.num_classes}."
            )
        if not cfg.encoder.filters:
            raise ValueError(
                "ModelConfig.encoder.filters must be a non-empty tuple."
            )
        if any(f < 1 for f in cfg.encoder.filters):
            raise ValueError(
                f"All encoder filter counts must be >= 1, got {cfg.encoder.filters}."
            )
        if not (0.0 <= cfg.encoder.dropout_rate <= 1.0):
            raise ValueError(
                f"ModelConfig.encoder.dropout_rate must be in [0, 1], "
                f"got {cfg.encoder.dropout_rate}."
            )
        if not (0.0 <= cfg.decoder.dropout_rate <= 1.0):
            raise ValueError(
                f"ModelConfig.decoder.dropout_rate must be in [0, 1], "
                f"got {cfg.decoder.dropout_rate}."
            )
