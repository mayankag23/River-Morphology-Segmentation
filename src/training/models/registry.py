"""
Model registry for the Segmentation Model Framework (Module 13).

ModelRegistry mirrors the plugin registry pattern established by
RuleRegistry (Module 9), LabelStrategyRegistry, and TransformRegistry
(Module 12). Every model class is registered by its model_name string.
ModelFactory queries this registry to instantiate models from config
without if/else chains.

Adding a new model requires only:
    1. Subclass SegmentationModel (and torch.nn.Module).
    2. Set model_name = "my_model" at class level.
    3. Decorate with @ModelRegistry.register.
    4. Set config.model.architecture = "my_model".

No existing code needs modification.

Built-in models are registered at import time via their module-level
decorator calls at the bottom of unetplusplus.py.  Importing
src.training.models triggers those registrations automatically.
"""

from __future__ import annotations

import logging
from typing import Any

from src.training.models.base import SegmentationModel

__all__ = ["ModelRegistry"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Central registry for SegmentationModel subclasses.

    Usage -- built-in models register themselves at import time:

        @ModelRegistry.register
        class UNetPlusPlus(SegmentationModel, nn.Module):
            model_name = "unetplusplus"
            ...

    Usage -- external plugin registration at runtime:

        ModelRegistry.register_external(MySARModel)

    Keys are the model_name class attribute strings.
    """

    _registered: dict[str, type] = {}

    @classmethod
    def register(cls, model_class: type) -> type:
        """
        Class decorator. Registers model_class by its model_name attribute.

        Args:
            model_class: A SegmentationModel subclass with model_name set.

        Returns:
            The model class unchanged (so the decorator is transparent).

        Raises:
            ValueError: model_name is empty or not a str.
        """
        name = getattr(model_class, "model_name", "")
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"ModelRegistry.register: {model_class.__name__} must define "
                f"a non-empty string class attribute 'model_name'."
            )
        cls._registered[name] = model_class
        _LOGGER.debug("ModelRegistry: registered '%s' -> %s", name, model_class.__name__)
        return model_class

    @classmethod
    def register_external(cls, model_class: type) -> None:
        """
        Imperatively register an external model class.

        Allows runtime plugin injection without the decorator syntax.
        Useful for third-party model packages that cannot decorate at
        class definition time.

        Args:
            model_class: A SegmentationModel subclass with model_name set.
        """
        name = getattr(model_class, "model_name", "")
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"ModelRegistry.register_external: {model_class.__name__} must "
                f"define a non-empty string 'model_name'."
            )
        cls._registered[name] = model_class
        _LOGGER.debug(
            "ModelRegistry: registered external '%s' -> %s",
            name, model_class.__name__,
        )

    @classmethod
    def get(cls, name: str) -> type:
        """
        Return the model class registered under name.

        Args:
            name: Architecture string (e.g. "unetplusplus").

        Returns:
            The registered model class.

        Raises:
            KeyError: name is not registered.
        """
        if name not in cls._registered:
            available = sorted(cls._registered.keys())
            raise KeyError(
                f"ModelRegistry: '{name}' is not registered. "
                f"Available architectures: {available}"
            )
        return cls._registered[name]

    @classmethod
    def registered_names(cls) -> tuple[str, ...]:
        """Return an alphabetically sorted tuple of all registered names."""
        return tuple(sorted(cls._registered.keys()))

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Return True if name is registered."""
        return name in cls._registered

    @classmethod
    def clear(cls) -> None:
        """
        Unregister all models.

        Intended for test isolation ONLY. Do NOT call in production code.
        """
        cls._registered.clear()
