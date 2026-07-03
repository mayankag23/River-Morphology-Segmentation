"""
Class weight computation for the River Morphology training pipeline (Module 11).

ClassWeights provides a per-class weight vector for use in weighted cross-
entropy losses. Weights are derived from the class pixel distribution in the
training split (from SplitStatistics, Module 10). Computing weights from
the training split only (not the full dataset) prevents evaluation data
from influencing training-time class weighting.

torch is imported lazily.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from src.core.exceptions import InvalidValueError

__all__ = ["ClassWeightStrategy", "ClassWeights"]

_LOGGER: logging.Logger = logging.getLogger(__name__)

_EPS: float = 1e-8


class ClassWeightStrategy(str, Enum):
    """Supported class weight computation strategies."""

    NONE               = "none"
    INVERSE_FREQUENCY  = "inverse_frequency"
    MANUAL             = "manual"

    @classmethod
    def from_string(cls, value: str) -> ClassWeightStrategy:
        """Convert a config string to ClassWeightStrategy."""
        try:
            return cls(value.lower().strip())
        except ValueError:
            raise InvalidValueError(
                field="training.class_weights.strategy",
                value=value,
                reason=f"must be one of {[s.value for s in cls]}",
            )


@dataclass(frozen=True)
class ClassWeights:
    """
    Immutable per-class weight vector for loss function weighting.

    Attributes:
        strategy:     The strategy used to compute these weights.
        num_classes:   Number of classes.
        weights:       Per-class weights as a tuple, ordered by class_id.
        class_names:   Ordered class names matching weights.
    """

    strategy:     str
    num_classes:  int
    weights:      tuple[float, ...]
    class_names:  tuple[str, ...]

    def as_tensor(self) -> Any:
        """
        Return weights as a float32 torch.Tensor.

        Raises:
            ImportError: torch is not installed.
        """
        try:
            import torch
        except ImportError as exc:
            raise ImportError("torch is not installed.") from exc
        return torch.tensor(list(self.weights), dtype=torch.float32)

    @classmethod
    def from_config_and_statistics(
        cls,
        config:           Any,
        train_statistics:  Any,     # SplitStatistics from Module 10
        class_schema:      Any,     # ClassSchema from Module 9
    ) -> ClassWeights:
        """
        Compute ClassWeights from config and training split statistics.

        Args:
            config:            Fully initialized Config object.
            train_statistics:   SplitStatistics for the training split
                               (from DatasetStatisticsCalculator.compute).
            class_schema:       ClassSchema defining the class taxonomy.

        Returns:
            Immutable ClassWeights.
        """
        train_cfg    = getattr(config, "training", None)
        cw_cfg       = getattr(train_cfg, "class_weights", None)
        strategy_str = str(getattr(cw_cfg, "strategy", "inverse_frequency"))
        strategy     = ClassWeightStrategy.from_string(strategy_str)
        num_classes  = class_schema.num_classes
        class_names  = class_schema.class_names

        if strategy == ClassWeightStrategy.NONE:
            weights = tuple(1.0 for _ in range(num_classes))
            return cls(
                strategy=strategy.value,
                num_classes=num_classes,
                weights=weights,
                class_names=class_names,
            )

        if strategy == ClassWeightStrategy.MANUAL:
            raw = list(getattr(cw_cfg, "manual_weights", []))
            if len(raw) != num_classes:
                raise InvalidValueError(
                    field="training.class_weights.manual_weights",
                    value=len(raw),
                    reason=f"must have exactly {num_classes} elements",
                )
            weights = tuple(float(w) for w in raw)
            return cls(
                strategy=strategy.value,
                num_classes=num_classes,
                weights=weights,
                class_names=class_names,
            )

        # INVERSE_FREQUENCY: weight proportional to 1 / pixel_frequency
        pixel_counts = np.array([
            float(stat.pixel_count)
            for stat in sorted(
                train_statistics.class_statistics,
                key=lambda s: s.class_id,
            )
        ], dtype=np.float64)

        total = pixel_counts.sum()
        if total < _EPS:
            # No pixels seen in training statistics (stats computed without masks).
            # Fall back to uniform weights.
            _LOGGER.warning(
                "Total training pixel count is zero; "
                "using uniform class weights."
            )
            weights = tuple(1.0 for _ in range(num_classes))
        else:
            freq    = np.clip(pixel_counts / total, _EPS, 1.0)
            inv_freq = 1.0 / freq
            inv_freq = inv_freq / inv_freq.sum() * num_classes
            weights = tuple(float(w) for w in inv_freq)

        _LOGGER.info(
            "Class weights (%s): %s",
            strategy.value,
            {n: f"{w:.3f}" for n, w in zip(class_names, weights)},
        )
        return cls(
            strategy=strategy.value,
            num_classes=num_classes,
            weights=weights,
            class_names=class_names,
        )