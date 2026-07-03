"""
Custom samplers for the River Morphology training pipeline (Module 11).

TemporalSampler wraps PyTorch's WeightedRandomSampler to provide
season-balanced or year-balanced sampling within the training loop.

Season-balanced sampling is important for river morphology because:
    - Monsoon season provides most water pixels (class=1).
    - Pre-monsoon/post-monsoon provides most exposed sandbar pixels (class=2).
    - Over-sampling one season would bias the model toward that river state.
    - Equal-weight sampling by season corrects for imbalanced scene acquisition.

torch is imported lazily so this module is importable without torch installed.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

import numpy as np

from src.core.exceptions import InvalidValueError

__all__ = ["TemporalSampler", "SamplerStrategy"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


class SamplerStrategy(str, Enum):
    """Supported temporal sampling strategies."""

    NONE              = "none"
    TEMPORAL_BALANCED = "temporal_balanced"
    CLASS_BALANCED    = "class_balanced"

    @classmethod
    def from_string(cls, value: str) -> SamplerStrategy:
        """Convert a config string to SamplerStrategy."""
        try:
            return cls(value.lower().strip())
        except ValueError:
            raise InvalidValueError(
                field="training.sampler.strategy",
                value=value,
                reason=f"must be one of {[s.value for s in cls]}",
            )


class TemporalSampler:
    """
    Creates a WeightedRandomSampler for season-balanced or uniform sampling.

    Args:
        config: Fully initialized Config object. Reads from
                config.training.sampler.
    """

    def __init__(self, config: Any) -> None:
        self._config = config
        self._logger: logging.Logger = logging.getLogger(__name__)

        train_cfg    = getattr(config, "training", None)
        sampler_cfg  = getattr(train_cfg, "sampler", None)
        self._strategy    = SamplerStrategy.from_string(
            str(getattr(sampler_cfg, "strategy", "none"))
        )
        self._random_seed = int(getattr(sampler_cfg, "random_seed", 42))

    def build(
        self,
        train_entries: list[Any],    # list[DatasetManifestEntry]
        replacement:   bool = True,
    ) -> Any | None:
        """
        Build a WeightedRandomSampler for the training entries.

        Args:
            train_entries: DatasetManifestEntry records for the training split.
            replacement:   Whether to sample with replacement (typically True
                           for training DataLoaders to preserve epoch length).

        Returns:
            torch.utils.data.WeightedRandomSampler, or None if strategy is NONE.

        Raises:
            ImportError: torch is not installed.
        """
        if self._strategy == SamplerStrategy.NONE or not train_entries:
            return None

        try:
            import torch
            from torch.utils.data import WeightedRandomSampler
        except ImportError as exc:
            raise ImportError("torch is not installed.") from exc

        weights = self._compute_weights(train_entries)
        sampler = WeightedRandomSampler(
            weights     = torch.tensor(weights, dtype=torch.double),
            num_samples = len(train_entries),
            replacement = replacement,
        )
        self._logger.debug(
            "WeightedRandomSampler created: strategy=%s, n=%d",
            self._strategy.value, len(train_entries),
        )
        return sampler

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _compute_weights(self, entries: list[Any]) -> list[float]:
        """
        Compute per-sample sampling weights based on the configured strategy.

        TEMPORAL_BALANCED: Each season gets equal total weight. A season
            with fewer samples gets higher per-sample weight.
        CLASS_BALANCED: Weight by label_valid_pixel_ratio (rough proxy for
            class richness; higher ratio -> more reliable label -> lower
            weight to balance easy vs hard patches).
        """
        if self._strategy == SamplerStrategy.TEMPORAL_BALANCED:
            return self._temporal_weights(entries)
        return self._class_balanced_weights(entries)

    @staticmethod
    def _temporal_weights(entries: list[Any]) -> list[float]:
        """Equal total weight per season."""
        from collections import Counter
        season_counts = Counter(e.season for e in entries)
        n_seasons     = len(season_counts)
        weights: list[float] = []
        for entry in entries:
            count = season_counts[entry.season]
            # Each season gets weight 1/n_seasons total, so per-sample
            # weight = 1 / (n_seasons * count_in_this_season).
            weights.append(1.0 / (n_seasons * count))
        return weights

    @staticmethod
    def _class_balanced_weights(entries: list[Any]) -> list[float]:
        """
        Higher weight for patches with lower label_valid_pixel_ratio
        (partial-nodata patches are harder and under-represented).
        """
        ratios  = np.array(
            [float(e.label_valid_pixel_ratio) for e in entries], dtype=np.float64
        )
        # Invert: lower ratio gets higher weight. Clip to avoid division by zero.
        weights = 1.0 / np.clip(ratios, 0.01, 1.0)
        weights = weights / weights.sum()    # normalize to [0, 1]
        return weights.tolist()