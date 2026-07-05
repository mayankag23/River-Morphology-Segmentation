"""
Learning-rate scheduler factory for the Training Engine Framework (Module 14).

SchedulerFactory builds torch LR schedulers from SchedulerConfig.
New schedulers are registered via SchedulerRegistry without modifying
existing code.

Supported out of the box:
    cosine    CosineAnnealingLR
    step      StepLR
    plateau   ReduceLROnPlateau
"""

from __future__ import annotations

import logging
from typing import Any

from src.training.engine.contracts import SchedulerConfig

__all__ = ["SchedulerFactory", "SchedulerRegistry"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


class SchedulerRegistry:
    """
    Registry mapping scheduler names to builder callables.

    Builder signature:
        builder(optimizer, config: SchedulerConfig) -> LRScheduler
    """

    _builders: dict[str, Any] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(fn: Any) -> Any:
            cls._builders[name.lower()] = fn
            return fn
        return decorator

    @classmethod
    def get(cls, name: str) -> Any:
        n = name.lower().strip()
        if n not in cls._builders:
            raise KeyError(
                f"SchedulerRegistry: '{name}' is not registered. "
                f"Available: {sorted(cls._builders)}"
            )
        return cls._builders[n]

    @classmethod
    def registered_names(cls) -> tuple[str, ...]:
        return tuple(sorted(cls._builders.keys()))


@SchedulerRegistry.register("cosine")
def _build_cosine(optimizer: Any, cfg: SchedulerConfig) -> Any:
    from torch.optim.lr_scheduler import CosineAnnealingLR
    return CosineAnnealingLR(optimizer, T_max=cfg.t_max, eta_min=cfg.eta_min)


@SchedulerRegistry.register("step")
def _build_step(optimizer: Any, cfg: SchedulerConfig) -> Any:
    from torch.optim.lr_scheduler import StepLR
    return StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)


@SchedulerRegistry.register("plateau")
def _build_plateau(optimizer: Any, cfg: SchedulerConfig) -> Any:
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    return ReduceLROnPlateau(
        optimizer,
        mode     = cfg.mode,
        factor   = cfg.gamma,
        patience = cfg.patience,
        min_lr   = cfg.min_lr,
    )


class SchedulerFactory:
    """
    Builds LR schedulers from SchedulerConfig via SchedulerRegistry.
    """

    @classmethod
    def build(cls, optimizer: Any, config: SchedulerConfig) -> Any | None:
        """
        Build an LR scheduler for an optimizer.

        Args:
            optimizer: torch.optim.Optimizer to attach the scheduler to.
            config:    SchedulerConfig specifying the scheduler type.

        Returns:
            LR scheduler instance, or None when config.enabled is False.

        Raises:
            KeyError: config.name is not registered.
        """
        if not config.enabled:
            _LOGGER.debug("SchedulerFactory: scheduler disabled.")
            return None

        builder   = SchedulerRegistry.get(config.name)
        scheduler = builder(optimizer, config)
        _LOGGER.debug("SchedulerFactory: built %s.", config.name)
        return scheduler

    @staticmethod
    def step(
        scheduler: Any | None,
        config:    SchedulerConfig,
        val_loss:  float = 0.0,
    ) -> None:
        """
        Call scheduler.step() with the correct arguments.

        ReduceLROnPlateau.step() requires the monitored metric;
        all others take no argument.

        Args:
            scheduler: Scheduler or None.
            config:    SchedulerConfig (used to detect plateau type).
            val_loss:  Metric value passed to ReduceLROnPlateau.
        """
        if scheduler is None:
            return
        if config.name.lower() == "plateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()
