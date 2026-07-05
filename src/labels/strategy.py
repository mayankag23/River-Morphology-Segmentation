"""
Label generation strategy interface and registry for Module 9.

LabelGenerationStrategy is the abstract interface that all label generation
implementations must satisfy. PseudoLabelGenerator (in generator.py) is
the first concrete implementation and is registered under "spectral_rules".

Future implementations (SamStrategy, MlStrategy, HybridStrategy) can be
added without modifying this file or LabelManager:
    (a) Subclass LabelGenerationStrategy and decorate with
        @LabelStrategyRegistry.register, or
    (b) Call LabelStrategyRegistry.register_external(MyStrategy) at runtime.

LabelManager selects the active strategy via
    config.labels.generation.strategy_type
defaulting to "spectral_rules" when the key is absent.

source_type = "pseudo_label" is a stable public identifier for the category
of label origin (automatic, as opposed to manually digitized). The specific
implementation within that category is recorded separately as
generation_strategy in LabelManifestEntry.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from src.labels.contracts import ClassificationContext, PseudoLabelResult

__all__ = ["LabelGenerationStrategy", "LabelStrategyRegistry"]

_LOGGER: logging.Logger = logging.getLogger(__name__)

_DEFAULT_STRATEGY_TYPE: str = "spectral_rules"


# ==============================================================================
# LabelGenerationStrategy (abstract interface)
# ==============================================================================

class LabelGenerationStrategy(ABC):
    """
    Abstract interface for all label generation implementations.

    Every concrete implementation:
        - Reads one feature-patch GeoTIFF from disk.
        - Produces a uint8 single-band mask GeoTIFF at output_mask_path.
        - Returns an immutable PseudoLabelResult describing the outcome.
        - Never calls Earth Engine.
        - Never reads or writes anything outside patch_path / output_mask_path.

    The strategy_type property returns the stable string identifier that is
    recorded in LabelManifestEntry.generation_strategy.
    """

    @property
    @abstractmethod
    def strategy_type(self) -> str:
        """
        Stable string identifier for this strategy implementation.

        Examples: "spectral_rules", "sam", "ml", "hybrid".
        Recorded in LabelManifestEntry.generation_strategy.
        """

    @abstractmethod
    def generate(
        self,
        patch_path:       Path,
        patch_id:          str,
        output_mask_path:  Path,
        context:           ClassificationContext | None = None,
    ) -> PseudoLabelResult:
        """
        Generate a pseudo-label mask for one patch.

        Args:
            patch_path:        Path to the source feature-stack GeoTIFF
                               (written by Module 7/8).
            patch_id:           Unique patch identifier.
            output_mask_path:   Absolute path at which to write the uint8
                               single-band mask GeoTIFF.
            context:            Optional ClassificationContext carrying temporal,
                               sensor, and geographic metadata. Implementations
                               may use or ignore any subset of fields.

        Returns:
            Immutable PseudoLabelResult.

        Raises:
            OSError: Patch cannot be read or mask cannot be written.
        """


# ==============================================================================
# LabelStrategyRegistry
# ==============================================================================

class LabelStrategyRegistry:
    """
    Central registry for LabelGenerationStrategy plugins.

    LabelManager.create() queries this registry rather than hardcoding
    which strategies exist. Any strategy class decorated with
    @LabelStrategyRegistry.register is automatically discoverable.

    Registration is idempotent: registering the same class twice under the
    same strategy_type replaces the previous entry with no error.
    """

    _registered: dict[str, type[LabelGenerationStrategy]] = {}

    @classmethod
    def register(
        cls, strategy_class: type[LabelGenerationStrategy]
    ) -> type[LabelGenerationStrategy]:
        """
        Class decorator. Registers strategy_class by its strategy_type.

        The strategy_type is determined by instantiating the class with a
        sentinel config and calling strategy_type, which requires concrete
        classes to define _STRATEGY_TYPE as a class-level constant.

        Usage:
            @LabelStrategyRegistry.register
            class PseudoLabelGenerator(LabelGenerationStrategy): ...
        """
        name = strategy_class._STRATEGY_TYPE  # type: ignore[attr-defined]
        cls._registered[name] = strategy_class
        _LOGGER.debug("LabelStrategyRegistry: registered '%s'", name)
        return strategy_class

    @classmethod
    def register_external(cls, strategy_class: type[LabelGenerationStrategy]) -> None:
        """
        Imperatively register an external strategy class.

        Allows runtime plugin injection without the decorator syntax, e.g.
        when the strategy is defined in a third-party package.
        """
        name = strategy_class._STRATEGY_TYPE  # type: ignore[attr-defined]
        cls._registered[name] = strategy_class
        _LOGGER.debug("LabelStrategyRegistry: registered external '%s'", name)

    @classmethod
    def create(
        cls,
        config:       Any,
        class_schema:  Any,
    ) -> LabelGenerationStrategy:
        """
        Instantiate the strategy selected by config.labels.generation.strategy_type.

        Falls back to "spectral_rules" when the key is absent.

        Args:
            config:       Fully initialized Config object.
            class_schema: ClassSchema for the active taxonomy.

        Returns:
            Configured LabelGenerationStrategy instance.

        Raises:
            ValueError: The configured strategy_type is not registered.
        """
        gen_cfg       = getattr(getattr(config, "labels", None), "generation", None)
        strategy_type = str(getattr(gen_cfg, "strategy_type", _DEFAULT_STRATEGY_TYPE))

        strategy_cls = cls._registered.get(strategy_type)
        if strategy_cls is None:
            registered = list(cls._registered.keys())
            raise ValueError(
                f"LabelStrategyRegistry: unknown strategy_type '{strategy_type}'. "
                f"Registered types: {registered}. "
                f"Use @LabelStrategyRegistry.register or "
                f"LabelStrategyRegistry.register_external() to add new strategies."
            )
        _LOGGER.debug(
            "LabelStrategyRegistry.create: selected strategy '%s'", strategy_type
        )
        return strategy_cls.from_config(config, class_schema)  # type: ignore[attr-defined]

    @classmethod
    def registered_types(cls) -> tuple[str, ...]:
        """Return names of all registered strategy types."""
        return tuple(cls._registered.keys())

    @classmethod
    def clear(cls) -> None:
        """
        Unregister all strategies.

        Intended for test isolation ONLY. Do NOT call in production code.
        """
        cls._registered.clear()