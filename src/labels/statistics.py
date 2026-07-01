"""
Label statistics computation for the label management pipeline (Module 9).

LabelStatisticsCalculator aggregates per-class pixel counts, validity
counts, temporal (seasonal/yearly) distributions, and configurable
class-pair ratios across all processed labels, producing an immutable
LabelStatistics summary. Pixel counts are accumulated ONLY for labels
that pass validation, so rejected or corrupted masks never skew class
distribution statistics.

Ratio definitions (e.g. water:sand) and bare-sediment fraction class
groups are read entirely from config.labels; no class name is hardcoded
in this module.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.labels.schema import ClassSchema
from src.labels.temporal import TemporalMetadata
from src.labels.validator import LabelValidationResult

__all__ = [
    "ClassPixelStatistics",
    "ClassRatio",
    "SeasonCount",
    "YearCount",
    "LabelStatistics",
    "LabelStatisticsCalculator",
]

_LOGGER: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClassPixelStatistics:
    """
    Immutable per-class pixel aggregate.

    Attributes:
        class_id:     Integer class label.
        class_name:    Human-readable class name.
        pixel_count:    Total pixel count across all valid labels.
        percentage:     Percentage of all valid (non-nodata) pixels
                       belonging to this class, in [0.0, 100.0].
    """

    class_id:     int
    class_name:    str
    pixel_count:    int
    percentage:     float


@dataclass(frozen=True)
class ClassRatio:
    """
    Immutable ratio between two class pixel counts.

    Attributes:
        name:                Ratio identifier, e.g. "water_sand_ratio".
        numerator_class:      Name of the numerator class.
        denominator_class:     Name of the denominator class.
        value:                 numerator_pixel_count / denominator_pixel_count,
                              or None if the denominator class is absent
                              from the schema or has zero pixels.
    """

    name:                str
    numerator_class:      str
    denominator_class:     str
    value:                 float | None


@dataclass(frozen=True)
class SeasonCount:
    """Immutable count of labels belonging to one season."""

    season: str
    count:   int


@dataclass(frozen=True)
class YearCount:
    """Immutable count of labels belonging to one calendar year."""

    year:  int
    count:  int


@dataclass(frozen=True)
class LabelStatistics:
    """
    Immutable summary of label dataset composition.

    Attributes:
        total_labels:             Total number of labels processed (valid + rejected).
        valid_labels:               Number of labels that passed validation.
        rejected_labels:             Number of labels that failed validation.
        class_pixel_stats:            Per-class pixel counts/percentages,
                                     ordered by class_id, across all valid labels.
        class_imbalance_ratio:         Ratio of the most-frequent to
                                     least-frequent nonzero class pixel_count.
                                     1.0 when balanced or only one class present.
        class_ratios:                  Configurable class-pair ratios
                                     (e.g. water:sand), from config.labels.ratios.
        bare_sediment_fraction:         numerator pixel sum / denominator pixel
                                     sum, per config.labels.bare_sediment_*.
                                     None if denominator sum is zero.
        seasonal_distribution:          Label counts grouped by season,
                                     sorted by season name.
        yearly_distribution:             Label counts grouped by year,
                                     sorted ascending.
    """

    total_labels:             int
    valid_labels:               int
    rejected_labels:             int
    class_pixel_stats:            tuple[ClassPixelStatistics, ...]
    class_imbalance_ratio:         float
    class_ratios:                  tuple[ClassRatio, ...]
    bare_sediment_fraction:         float | None
    seasonal_distribution:          tuple[SeasonCount, ...]
    yearly_distribution:             tuple[YearCount, ...]

    def summary_lines(self) -> list[str]:
        """Return ASCII-formatted summary lines."""
        lines = [
            f"  Total labels:    {self.total_labels}",
            f"  Valid labels:     {self.valid_labels}",
            f"  Rejected labels:   {self.rejected_labels}",
            f"  Class imbalance:    {self.class_imbalance_ratio:.2f}x",
        ]
        for stat in self.class_pixel_stats:
            lines.append(
                f"    {stat.class_name} (id={stat.class_id}): "
                f"{stat.pixel_count} px ({stat.percentage:.1f}%)"
            )
        for ratio in self.class_ratios:
            val = f"{ratio.value:.3f}" if ratio.value is not None else "N/A"
            lines.append(f"  {ratio.name}: {val}")
        bs = (
            f"{self.bare_sediment_fraction:.3f}"
            if self.bare_sediment_fraction is not None else "N/A"
        )
        lines.append(f"  bare_sediment_fraction: {bs}")
        lines.append(
            f"  Seasons: {[(s.season, s.count) for s in self.seasonal_distribution]}"
        )
        lines.append(
            f"  Years:   {[(y.year, y.count) for y in self.yearly_distribution]}"
        )
        return lines


class LabelStatisticsCalculator:
    """
    Accumulates per-label class pixel counts and temporal metadata, and
    produces a LabelStatistics summary.

    Args:
        class_schema:                ClassSchema defining the class taxonomy.
        nodata_value:                 Integer sentinel for invalid/unlabeled
                                     mask pixels.
        ratio_definitions:             Mapping of ratio name -> (numerator
                                     class name, denominator class name).
                                     Sourced from config.labels.ratios.
        bare_sediment_numerator:        Class names summed for the bare
                                     sediment fraction numerator. Sourced
                                     from config.labels.bare_sediment_numerator.
        bare_sediment_denominator:      Class names summed for the bare
                                     sediment fraction denominator. Sourced
                                     from config.labels.bare_sediment_denominator.
    """

    def __init__(
        self,
        class_schema:                ClassSchema,
        nodata_value:                 int,
        ratio_definitions:             dict[str, tuple[str, str]] | None = None,
        bare_sediment_numerator:        tuple[str, ...] = (),
        bare_sediment_denominator:      tuple[str, ...] = (),
    ) -> None:
        self._schema = class_schema
        self._nodata_value = int(nodata_value)
        self._ratio_definitions = ratio_definitions or {}
        self._bare_sediment_numerator = bare_sediment_numerator
        self._bare_sediment_denominator = bare_sediment_denominator

        self._pixel_counts: dict[int, int] = {cid: 0 for cid in class_schema.class_ids}
        self._season_counts: Counter = Counter()
        self._year_counts: Counter = Counter()
        self._total_labels = 0
        self._valid_labels = 0
        self._rejected_labels = 0
        self._logger: logging.Logger = logging.getLogger(__name__)

    @classmethod
    def from_config(cls, class_schema: ClassSchema, config: Any) -> LabelStatisticsCalculator:
        """
        Build a LabelStatisticsCalculator from config.labels.

        Args:
            class_schema: ClassSchema for the active taxonomy.
            config:        Fully initialized Config object.

        Returns:
            LabelStatisticsCalculator with ratio and bare-sediment
            definitions sourced from config.labels.
        """
        labels_cfg = getattr(config, "labels", None)
        nodata_value = int(getattr(labels_cfg, "nodata_value", 255))

        ratios_cfg = getattr(labels_cfg, "ratios", None)
        ratio_definitions: dict[str, tuple[str, str]] = {}
        if ratios_cfg is not None:
            for ratio_name in ratios_cfg:
                pair = list(getattr(ratios_cfg, ratio_name))
                if len(pair) == 2:
                    ratio_definitions[ratio_name] = (str(pair[0]), str(pair[1]))

        numerator = tuple(
            str(n) for n in getattr(labels_cfg, "bare_sediment_numerator", [])
        )
        denominator = tuple(
            str(n) for n in getattr(labels_cfg, "bare_sediment_denominator", [])
        )

        return cls(
            class_schema=class_schema,
            nodata_value=nodata_value,
            ratio_definitions=ratio_definitions,
            bare_sediment_numerator=numerator,
            bare_sediment_denominator=denominator,
        )

    def accumulate(
        self,
        mask_data:           Any | None,
        validation_result:     LabelValidationResult,
        temporal_metadata:       TemporalMetadata,
    ) -> None:
        """
        Add one label's data to the running aggregate.

        Pixel counts and temporal distribution are accumulated ONLY for
        valid labels with readable mask data, so rejected/missing/corrupted
        masks do not skew class distribution statistics.

        Args:
            mask_data:            Mask pixel array, shape (height, width),
                                  or None if the mask could not be read.
            validation_result:      LabelValidationResult for this mask.
            temporal_metadata:       TemporalMetadata for this label.
        """
        self._total_labels += 1
        if validation_result.is_valid and mask_data is not None:
            self._valid_labels += 1
            unique, counts = np.unique(mask_data, return_counts=True)
            for value, count in zip(unique, counts):
                cid = int(value)
                if cid in self._pixel_counts:
                    self._pixel_counts[cid] += int(count)
            self._season_counts[temporal_metadata.season] += 1
            self._year_counts[temporal_metadata.year] += 1
        else:
            self._rejected_labels += 1

    def compute(self) -> LabelStatistics:
        """
        Finalize and return the aggregated LabelStatistics.

        Returns:
            Immutable LabelStatistics summary.
        """
        total_valid_pixels = sum(self._pixel_counts.values())

        class_stats: list[ClassPixelStatistics] = []
        for definition in self._schema.classes:
            count = self._pixel_counts.get(definition.class_id, 0)
            pct = (
                (count / total_valid_pixels * 100.0)
                if total_valid_pixels > 0 else 0.0
            )
            class_stats.append(ClassPixelStatistics(
                class_id=definition.class_id,
                class_name=definition.name,
                pixel_count=count,
                percentage=pct,
            ))

        nonzero_counts = [c for c in self._pixel_counts.values() if c > 0]
        imbalance = (
            max(nonzero_counts) / min(nonzero_counts)
            if len(nonzero_counts) >= 2 else 1.0
        )

        class_ratios = tuple(
            self._compute_ratio(name, numerator, denominator)
            for name, (numerator, denominator) in self._ratio_definitions.items()
        )

        bare_sediment = self._compute_bare_sediment_fraction()

        seasons = tuple(
            SeasonCount(season=s, count=c)
            for s, c in sorted(self._season_counts.items())
        )
        years = tuple(
            YearCount(year=y, count=c)
            for y, c in sorted(self._year_counts.items())
        )

        return LabelStatistics(
            total_labels=self._total_labels,
            valid_labels=self._valid_labels,
            rejected_labels=self._rejected_labels,
            class_pixel_stats=tuple(class_stats),
            class_imbalance_ratio=imbalance,
            class_ratios=class_ratios,
            bare_sediment_fraction=bare_sediment,
            seasonal_distribution=seasons,
            yearly_distribution=years,
        )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _pixel_count_for_name(self, class_name: str) -> int | None:
        """Return the pixel count for a class name, or None if unknown."""
        if not self._schema.has_class_name(class_name):
            return None
        class_id = self._schema.get_id_by_name(class_name)
        return self._pixel_counts.get(class_id, 0)

    def _compute_ratio(
        self, name: str, numerator_class: str, denominator_class: str,
    ) -> ClassRatio:
        """Compute one ClassRatio; value is None if classes/pixels are unavailable."""
        num = self._pixel_count_for_name(numerator_class)
        den = self._pixel_count_for_name(denominator_class)
        value = (num / den) if (num is not None and den) else None
        return ClassRatio(
            name=name, numerator_class=numerator_class,
            denominator_class=denominator_class, value=value,
        )

    def _compute_bare_sediment_fraction(self) -> float | None:
        """Compute the bare-sediment fraction from configured class groups."""
        if not self._bare_sediment_numerator or not self._bare_sediment_denominator:
            return None

        numerator_sum = 0
        for name in self._bare_sediment_numerator:
            count = self._pixel_count_for_name(name)
            if count is not None:
                numerator_sum += count

        denominator_sum = 0
        for name in self._bare_sediment_denominator:
            count = self._pixel_count_for_name(name)
            if count is not None:
                denominator_sum += count

        if denominator_sum == 0:
            return None
        return numerator_sum / denominator_sum