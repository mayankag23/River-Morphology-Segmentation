"""
Optimizer factory for the Training Engine Framework (Module 14).

OptimizerFactory builds torch optimizers from OptimizerConfig without any
if/else chains in the Training Engine. New optimizers are added by registering
them with OptimizerRegistry.

Supported out of the box:
    adam    torch.optim.Adam
    adamw   torch.optim.AdamW  (recommended default)
    sgd     torch.optim.SGD
"""

from __future__ import annotations

import logging
from typing import Any

from src.training.engine.contracts import OptimizerConfig

__all__ = ["OptimizerFactory", "OptimizerRegistry"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


class OptimizerRegistry:
    """
    Registry mapping optimizer names to builder callables.

    A builder callable has signature:
        builder(params, config: OptimizerConfig) -> torch.optim.Optimizer
    """

    _builders: dict[str, Any] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator that registers a builder function under name."""
        def decorator(fn: Any) -> Any:
            cls._builders[name.lower()] = fn
            return fn
        return decorator

    @classmethod
    def get(cls, name: str) -> Any:
        n = name.lower().strip()
        if n not in cls._builders:
            raise KeyError(
                f"OptimizerRegistry: '{name}' is not registered. "
                f"Available: {sorted(cls._builders)}"
            )
        return cls._builders[n]

    @classmethod
    def registered_names(cls) -> tuple[str, ...]:
        return tuple(sorted(cls._builders.keys()))


@OptimizerRegistry.register("adam")
def _build_adam(params: Any, cfg: OptimizerConfig) -> Any:
    import torch.optim as optim
    return optim.Adam(
        params,
        lr           = cfg.lr,
        betas        = cfg.betas,
        eps          = cfg.eps,
        weight_decay = cfg.weight_decay,
    )


@OptimizerRegistry.register("adamw")
def _build_adamw(params: Any, cfg: OptimizerConfig) -> Any:
    import torch.optim as optim
    return optim.AdamW(
        params,
        lr           = cfg.lr,
        betas        = cfg.betas,
        eps          = cfg.eps,
        weight_decay = cfg.weight_decay,
    )


@OptimizerRegistry.register("sgd")
def _build_sgd(params: Any, cfg: OptimizerConfig) -> Any:
    import torch.optim as optim
    return optim.SGD(
        params,
        lr           = cfg.lr,
        momentum     = cfg.momentum,
        weight_decay = cfg.weight_decay,
        nesterov     = cfg.momentum > 0.0,
    )


class OptimizerFactory:
    """
    Builds torch optimizers from OptimizerConfig via OptimizerRegistry.
    """

    @classmethod
    def build(cls, model: Any, config: OptimizerConfig) -> Any:
        """
        Build an optimizer for model parameters.

        Args:
            model:  torch.nn.Module whose parameters() will be optimized.
            config: OptimizerConfig specifying optimizer type and hyperparameters.

        Returns:
            torch.optim.Optimizer instance.

        Raises:
            KeyError: config.name is not registered.
            ValueError: config.lr <= 0.
        """
        if config.lr <= 0:
            raise ValueError(
                f"OptimizerConfig.lr must be > 0, got {config.lr}."
            )
        builder = OptimizerRegistry.get(config.name)
        optimizer = builder(model.parameters(), config)
        _LOGGER.debug(
            "OptimizerFactory: built %s (lr=%s, wd=%s).",
            config.name, config.lr, config.weight_decay,
        )
        return optimizer
