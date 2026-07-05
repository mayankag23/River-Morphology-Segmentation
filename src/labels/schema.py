# no change
"""
Class schema management for the label pipeline (Module 9).

ClassDefinition and ClassSchema are immutable representations of the
segmentation class taxonomy defined in config.classes (Module 1). No class
IDs, names, or colors are hardcoded; ClassSchema is always constructed via
ClassSchema.from_config(). The architecture supports adding future classes
(WetSand, DrySand, Shadow, Cloud, FloatingVegetation, etc.) purely through
config.yaml, with zero code changes required anywhere in src/labels/.

Vegetation is an explicit class (not folded into background) because
sandbars in braided river systems vegetate over multiple seasons. Without
a distinct vegetation class, that transition cannot be distinguished from
genuine sand-to-water or water-to-sand changes, corrupting any time-series
analysis of channel migration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.core.exceptions import InvalidValueError, MissingFieldError

if TYPE_CHECKING:
    from src.core.config import Config

__all__ = ["ClassDefinition", "ClassSchema"]

_LOGGER: logging.Logger = logging.getLogger(__name__)
_DEFAULT_COLOR: tuple[int, int, int] = (128, 128, 128)


@dataclass(frozen=True)
class ClassDefinition:
    """
    Immutable descriptor for one segmentation class.

    Attributes:
        class_id: Integer label value used in mask pixel data.
        name:     Human-readable class name, e.g. "water".
        color:    RGB display color as a 3-tuple of ints in [0, 255].
    """

    class_id: int
    name:     str
    color:    tuple[int, int, int]


@dataclass(frozen=True)
class ClassSchema:
    """
    Immutable segmentation class taxonomy, sourced entirely from Config.

    Attributes:
        classes: Tuple of ClassDefinition, ordered by ascending class_id.
    """

    classes: tuple[ClassDefinition, ...]

    @classmethod
    def from_config(cls, config: Config) -> ClassSchema:
        """
        Build a ClassSchema from config.classes.

        Args:
            config: Fully initialized Config object.

        Returns:
            ClassSchema reflecting the configured taxonomy, sorted by
            ascending class_id.

        Raises:
            MissingFieldError: config.classes or config.classes.labels absent.
            InvalidValueError: A color is malformed, or class IDs collide.
        """
        classes_cfg = getattr(config, "classes", None)
        if classes_cfg is None:
            raise MissingFieldError(
                field="classes",
                context="config.classes is required to build a ClassSchema.",
            )

        labels_cfg = getattr(classes_cfg, "labels", None)
        colors_cfg = getattr(classes_cfg, "colors", None)
        if labels_cfg is None:
            raise MissingFieldError(
                field="classes.labels",
                context="config.classes.labels (name -> id) is required.",
            )

        definitions: list[ClassDefinition] = []
        for name in labels_cfg:
            class_id = int(getattr(labels_cfg, name))
            color_raw = (
                getattr(colors_cfg, name, list(_DEFAULT_COLOR))
                if colors_cfg is not None
                else list(_DEFAULT_COLOR)
            )
            color = tuple(int(c) for c in color_raw)
            if len(color) != 3:
                raise InvalidValueError(
                    field=f"classes.colors.{name}",
                    value=color_raw,
                    reason="must be a 3-element RGB list",
                )
            definitions.append(
                ClassDefinition(class_id=class_id, name=name, color=color)
            )

        definitions.sort(key=lambda d: d.class_id)

        ids = [d.class_id for d in definitions]
        if len(ids) != len(set(ids)):
            raise InvalidValueError(
                field="classes.labels", value=ids, reason="class IDs must be unique",
            )

        return cls(classes=tuple(definitions))

    @property
    def class_ids(self) -> tuple[int, ...]:
        """Ordered tuple of all valid class IDs."""
        return tuple(d.class_id for d in self.classes)

    @property
    def class_names(self) -> tuple[str, ...]:
        """Ordered tuple of all configured class names."""
        return tuple(d.name for d in self.classes)

    @property
    def num_classes(self) -> int:
        """Total number of defined classes."""
        return len(self.classes)

    def is_valid_class_id(self, class_id: int) -> bool:
        """Return True if class_id is part of this schema."""
        return class_id in self.class_ids

    def has_class_name(self, name: str) -> bool:
        """Return True if name is a configured class name in this schema."""
        return name in self.class_names

    def get_name(self, class_id: int) -> str:
        """
        Return the class name for a given class_id.

        Raises:
            InvalidValueError: class_id is not in this schema.
        """
        return self.get_definition(class_id).name

    def get_id_by_name(self, name: str) -> int:
        """
        Return the class_id for a given class name.

        Raises:
            InvalidValueError: name is not in this schema.
        """
        for definition in self.classes:
            if definition.name == name:
                return definition.class_id
        raise InvalidValueError(
            field="class name", value=name,
            reason=f"not a configured class name. Valid names: {list(self.class_names)}",
        )

    def get_definition(self, class_id: int) -> ClassDefinition:
        """
        Return the ClassDefinition for class_id.

        Raises:
            InvalidValueError: class_id is not in this schema.
        """
        for definition in self.classes:
            if definition.class_id == class_id:
                return definition
        raise InvalidValueError(
            field="class_id", value=class_id,
            reason=f"not a valid class ID. Valid IDs: {list(self.class_ids)}",
        )