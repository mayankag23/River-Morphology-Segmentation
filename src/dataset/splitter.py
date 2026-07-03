"""
Dataset splitting for the Dataset Assembly pipeline (Module 10).

DatasetSplitter assigns each DatasetSample to one of three mutually
exclusive splits: train, validation, test.

SCENE-LEVEL SPLITTING (core leakage prevention):
    All patches from the same scene always go to the SAME split. This
    guarantees:
        - Overlapping patches (stride < patch_size) never appear in
          different splits.
        - Temporal ordering is respected (temporal strategy).
        - Spatial independence is respected (spatial strategy).

Split strategies:
    RANDOM:   Scene groups shuffled with config.dataset.split.random_seed.
    TEMPORAL: Scene groups sorted by acquisition_date; earliest assigned
              to train, ensuring the model never sees future river states
              during training.
    SPATIAL:  Scene groups sorted by aoi_id. All scenes from one AOI go
              to one split, providing full geographic independence.
"""

from __future__ import annotations

import logging
import random as _random
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from src.core.exceptions import InvalidValueError
from src.dataset.manifest import DatasetSample

if TYPE_CHECKING:
    from src.core.config import Config

__all__ = ["SplitStrategy", "SplitResult", "DatasetSplitter"]

_LOGGER: logging.Logger = logging.getLogger(__name__)

_SPLIT_TRAIN:      str = "train"
_SPLIT_VALIDATION: str = "validation"
_SPLIT_TEST:       str = "test"


# ==============================================================================
# SplitStrategy
# ==============================================================================

class SplitStrategy(str, Enum):
    """Supported dataset splitting strategies."""

    RANDOM   = "random"
    TEMPORAL = "temporal"
    SPATIAL  = "spatial"

    @classmethod
    def from_string(cls, value: str) -> SplitStrategy:
        """
        Convert a config string to SplitStrategy.

        Raises:
            InvalidValueError: value is not a recognized strategy string.
        """
        try:
            return cls(value.lower().strip())
        except ValueError:
            raise InvalidValueError(
                field="dataset.split.strategy",
                value=value,
                reason=f"must be one of {[s.value for s in cls]}",
            )


# ==============================================================================
# SplitResult
# ==============================================================================

@dataclass(frozen=True)
class SplitResult:
    """
    Immutable split assignment result.

    Attributes:
        train_samples:      Samples assigned to the training split.
        validation_samples:  Samples assigned to the validation split.
        test_samples:        Samples assigned to the test split.
        strategy:            SplitStrategy used.
        scene_assignments:    Mapping (scene_id, split) as a tuple of pairs.
        train_ratio_actual:   Actual training ratio achieved.
        validation_ratio_actual: Actual validation ratio achieved.
        test_ratio_actual:   Actual test ratio achieved.
    """

    train_samples:           tuple[DatasetSample, ...]
    validation_samples:       tuple[DatasetSample, ...]
    test_samples:             tuple[DatasetSample, ...]
    strategy:                 str
    scene_assignments:         tuple[tuple[str, str], ...]
    train_ratio_actual:        float
    validation_ratio_actual:    float
    test_ratio_actual:          float

    @property
    def train_count(self) -> int:
        return len(self.train_samples)

    @property
    def validation_count(self) -> int:
        return len(self.validation_samples)

    @property
    def test_count(self) -> int:
        return len(self.test_samples)

    @property
    def total_count(self) -> int:
        return self.train_count + self.validation_count + self.test_count

    def get_scene_split(self, scene_id: str) -> str | None:
        """Return the split assigned to a scene_id, or None if not found."""
        for s, split in self.scene_assignments:
            if s == scene_id:
                return split
        return None


# ==============================================================================
# DatasetSplitter
# ==============================================================================

class DatasetSplitter:
    """
    Assigns DatasetSamples to train / validation / test splits at the
    SCENE level to prevent data leakage.

    Args:
        config: Fully initialized Config object. Reads split parameters
                from config.dataset.split.
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._logger: logging.Logger = logging.getLogger(__name__)

        dataset_cfg = getattr(config, "dataset", None)
        split_cfg   = getattr(dataset_cfg, "split", None)

        self._strategy   = SplitStrategy.from_string(
            str(getattr(split_cfg, "strategy", "temporal"))
        )
        self._train_ratio = float(getattr(split_cfg, "train_ratio", 0.70))
        self._val_ratio   = float(getattr(split_cfg, "val_ratio",   0.15))
        self._test_ratio  = float(getattr(split_cfg, "test_ratio",  0.15))
        self._random_seed = int(getattr(split_cfg,   "random_seed", 42))

        self._validate_ratios()

    def split(
        self,
        samples: list[DatasetSample],
        strategy: SplitStrategy | None = None,
    ) -> SplitResult:
        """
        Assign each sample to a split.

        Args:
            samples:  Non-empty list of DatasetSample objects.
            strategy: Override the configured strategy for this call.

        Returns:
            SplitResult with all three split sample lists.

        Raises:
            InvalidValueError: samples is empty.
        """
        if not samples:
            raise InvalidValueError(
                field="samples", value=0,
                reason="must contain at least one sample to split",
            )

        resolved_strategy = strategy or self._strategy

        scene_groups = self._group_by_scene(samples)
        scene_to_split = self._assign_scenes(
            scene_groups, resolved_strategy
        )

        train:      list[DatasetSample] = []
        validation: list[DatasetSample] = []
        test:       list[DatasetSample] = []

        for sample in samples:
            assigned = scene_to_split.get(sample.scene_id, _SPLIT_TRAIN)
            if assigned == _SPLIT_TRAIN:
                train.append(sample)
            elif assigned == _SPLIT_VALIDATION:
                validation.append(sample)
            else:
                test.append(sample)

        total = len(samples)

        self._logger.info(
            "Split complete: strategy=%s, train=%d, val=%d, test=%d",
            resolved_strategy.value, len(train), len(validation), len(test),
        )

        return SplitResult(
            train_samples=tuple(train),
            validation_samples=tuple(validation),
            test_samples=tuple(test),
            strategy=resolved_strategy.value,
            scene_assignments=tuple(scene_to_split.items()),
            train_ratio_actual=len(train) / total if total else 0.0,
            validation_ratio_actual=len(validation) / total if total else 0.0,
            test_ratio_actual=len(test) / total if total else 0.0,
        )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _validate_ratios(self) -> None:
        """Raise if the configured split ratios do not sum to approximately 1."""
        total = self._train_ratio + self._val_ratio + self._test_ratio
        if not (0.99 <= total <= 1.01):
            raise InvalidValueError(
                field="dataset.split.[train_ratio + val_ratio + test_ratio]",
                value=total,
                reason="must sum to 1.0 (within 1% tolerance)",
            )

    @staticmethod
    def _group_by_scene(
        samples: list[DatasetSample],
    ) -> dict[str, list[DatasetSample]]:
        """Group samples by scene_id."""
        groups: dict[str, list[DatasetSample]] = {}
        for sample in samples:
            groups.setdefault(sample.scene_id, []).append(sample)
        return groups

    def _assign_scenes(
        self,
        scene_groups: dict[str, list[DatasetSample]],
        strategy:     SplitStrategy,
    ) -> dict[str, str]:
        """
        Assign each scene_id to a split name.

        Returns:
            Dict mapping scene_id -> split name.
        """
        if strategy == SplitStrategy.TEMPORAL:
            return self._temporal_assignment(scene_groups)
        if strategy == SplitStrategy.SPATIAL:
            return self._spatial_assignment(scene_groups)
        return self._random_assignment(scene_groups)

    def _temporal_assignment(
        self,
        scene_groups: dict[str, list[DatasetSample]],
    ) -> dict[str, str]:
        """
        Sort scenes by earliest acquisition_date, assign chronologically.

        Earlier scenes -> train (model learns from past).
        Later scenes   -> validation/test (model evaluated on future river
        states, simulating real deployment conditions).
        """
        def scene_min_date(scene_id: str) -> str:
            return min(s.acquisition_date for s in scene_groups[scene_id])

        sorted_scenes = sorted(scene_groups.keys(), key=scene_min_date)
        return self._proportional_assignment(sorted_scenes)

    def _spatial_assignment(
        self,
        scene_groups: dict[str, list[DatasetSample]],
    ) -> dict[str, str]:
        """
        Group scenes by aoi_id; assign all scenes from one AOI to one split.

        All samples from the same geographic area go to the same split,
        ensuring spatial independence between train and evaluation sets.
        """
        aoi_to_scenes: dict[str, list[str]] = {}
        for scene_id, samples in scene_groups.items():
            aoi_id = samples[0].aoi_id if samples else ""
            aoi_to_scenes.setdefault(aoi_id, []).append(scene_id)

        sorted_aois = sorted(aoi_to_scenes.keys())
        rng = _random.Random(self._random_seed)
        rng.shuffle(sorted_aois)

        n_aois      = len(sorted_aois)
        n_train     = max(1, int(n_aois * self._train_ratio))
        n_val       = max(1, int(n_aois * self._val_ratio))

        assignments: dict[str, str] = {}
        for i, aoi_id in enumerate(sorted_aois):
            if i < n_train:
                split = _SPLIT_TRAIN
            elif i < n_train + n_val:
                split = _SPLIT_VALIDATION
            else:
                split = _SPLIT_TEST
            for scene_id in aoi_to_scenes[aoi_id]:
                assignments[scene_id] = split

        return assignments

    def _random_assignment(
        self,
        scene_groups: dict[str, list[DatasetSample]],
    ) -> dict[str, str]:
        """Shuffle scenes with configured seed, assign proportionally."""
        scene_ids = sorted(scene_groups.keys())
        rng = _random.Random(self._random_seed)
        rng.shuffle(scene_ids)
        return self._proportional_assignment(scene_ids)

    def _proportional_assignment(
        self,
        ordered_scenes: list[str],
    ) -> dict[str, str]:
        """
        Assign the first train_ratio% to train, next val_ratio% to val,
        remainder to test. Ensures at least 1 scene per non-empty split.
        """
        n       = len(ordered_scenes)
        n_train = max(1, int(n * self._train_ratio))
        n_val   = max(1, int(n * self._val_ratio)) if n > 1 else 0

        # Adjust to leave at least 1 for test when n >= 3
        if n >= 3 and n_train + n_val >= n:
            n_val = max(0, n - n_train - 1)

        assignments: dict[str, str] = {}
        for i, scene_id in enumerate(ordered_scenes):
            if i < n_train:
                assignments[scene_id] = _SPLIT_TRAIN
            elif i < n_train + n_val:
                assignments[scene_id] = _SPLIT_VALIDATION
            else:
                assignments[scene_id] = _SPLIT_TEST

        return assignments