"""
Color management for the Visualization Framework (Module 18).

ClassColorMap resolves per-class colors from VisualizationConfig.class_colors,
falling back to ColorRegistry defaults when a class is not explicitly configured.

All colors are stored as (R, G, B) tuples with components in [0.0, 1.0] so
they are directly usable as matplotlib color arguments.

Design rule: no color is hardcoded in this module. The DEFAULT_COLORS dict
defines the factory default palette, but it is always overridable via config.

Default palette (scientifically meaningful for river morphology):
    background  (0.20, 0.20, 0.20)   dark grey
    water       (0.13, 0.47, 0.71)   steel blue
    sand        (0.95, 0.85, 0.55)   sandy yellow
    vegetation  (0.20, 0.63, 0.17)   forest green
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.visualization.contracts import VisualizationConfig

__all__ = ["ClassColorMap", "ColorRegistry"]

_LOGGER: logging.Logger = logging.getLogger(__name__)

# Default color palette — overridden by config.visualization.class_colors.
DEFAULT_COLORS: dict[str, tuple[float, float, float]] = {
    "background":  (0.20, 0.20, 0.20),
    "water":       (0.13, 0.47, 0.71),
    "sand":        (0.95, 0.85, 0.55),
    "vegetation":  (0.20, 0.63, 0.17),
}

# Matplotlib-compatible cycle colors for unknown classes (index-based fallback).
_CYCLE_COLORS: list[tuple[float, float, float]] = [
    (0.84, 0.15, 0.16),   # red
    (0.58, 0.40, 0.74),   # purple
    (0.55, 0.34, 0.29),   # brown
    (0.89, 0.47, 0.76),   # pink
    (0.50, 0.50, 0.50),   # medium grey
    (0.74, 0.74, 0.13),   # yellow-green
]


class ColorRegistry:
    """
    Registry for custom color overrides.

    Allows runtime registration of class colors outside config, useful for
    tests and plugins.

    Usage:
        ColorRegistry.register("my_class", (0.5, 0.3, 0.1))
    """

    _overrides: dict[str, tuple[float, float, float]] = {}

    @classmethod
    def register(cls, class_name: str, color: tuple[float, float, float]) -> None:
        """Register a color override for class_name."""
        cls._overrides[class_name] = color

    @classmethod
    def get(cls, class_name: str) -> tuple[float, float, float] | None:
        """Return registered override or None."""
        return cls._overrides.get(class_name)

    @classmethod
    def clear(cls) -> None:
        """Clear all overrides. For test isolation ONLY."""
        cls._overrides.clear()


class ClassColorMap:
    """
    Resolves per-class colors from config and registry.

    Priority order:
        1. VisualizationConfig.class_colors (explicit config override)
        2. ColorRegistry._overrides (runtime override)
        3. DEFAULT_COLORS (built-in palette)
        4. _CYCLE_COLORS[class_id % len(_CYCLE_COLORS)] (unknown class fallback)

    Args:
        config:      VisualizationConfig.
        class_names: Ordered class names from RiverMorphologyResult.
    """

    def __init__(
        self,
        config:      VisualizationConfig,
        class_names: tuple[str, ...],
    ) -> None:
        self._config      = config
        self._class_names = class_names
        self._resolved:   dict[str, tuple[float, float, float]] = {}

        for idx, name in enumerate(class_names):
            color: tuple[float, float, float]
            if name in config.class_colors:
                raw = config.class_colors[name]
                color = (float(raw[0]), float(raw[1]), float(raw[2]))
            elif ColorRegistry.get(name) is not None:
                color = ColorRegistry.get(name)  # type: ignore[assignment]
            elif name in DEFAULT_COLORS:
                color = DEFAULT_COLORS[name]
            else:
                color = _CYCLE_COLORS[idx % len(_CYCLE_COLORS)]
                _LOGGER.debug(
                    "ClassColorMap: no color for '%s'; using cycle color %s.",
                    name, color,
                )
            self._resolved[name] = color

    def get(self, class_name: str) -> tuple[float, float, float]:
        """Return (R, G, B) color for class_name. Falls back to grey."""
        return self._resolved.get(class_name, (0.5, 0.5, 0.5))

    def as_list(self) -> list[tuple[float, float, float]]:
        """Return colors in class_names order."""
        return [self._resolved.get(n, (0.5, 0.5, 0.5)) for n in self._class_names]

    def to_rgba_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Convert a (H, W) integer class-ID mask to an (H, W, 4) RGBA float32 array.

        Args:
            mask: (H, W) uint8 or int array with class IDs.

        Returns:
            (H, W, 4) float32 RGBA array in [0, 1].
        """
        H, W    = mask.shape
        rgba    = np.zeros((H, W, 4), dtype=np.float32)
        rgba[:, :, 3] = 1.0   # alpha = 1 by default

        for idx, name in enumerate(self._class_names):
            r, g, b         = self.get(name)
            cls_mask        = (mask == idx)
            rgba[cls_mask, 0] = r
            rgba[cls_mask, 1] = g
            rgba[cls_mask, 2] = b

        return rgba

    def legend_handles(self) -> list:
        """
        Return a list of matplotlib Patch objects for a legend.

        Requires matplotlib.
        """
        try:
            from matplotlib.patches import Patch
            return [
                Patch(facecolor=self.get(name), label=name)
                for name in self._class_names
            ]
        except ImportError:
            return []