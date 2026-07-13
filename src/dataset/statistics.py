"""
Dataset statistics computation for the Dataset Assembly pipeline (Module 10).

DatasetStatisticsCalculator computes per-split class distribution by reading
mask files with rasterio. Statistics are computed independently per split
to support imbalance analysis in training vs. validation vs. test.

Reuses SeasonCount and YearCount from src.labels.statistics (Module 9) and
ClassSchema from src.labels.schema (Module 9) to avoid duplicating frozen
dataclasses and class taxonomy logic.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from src.labels.schema import ClassSchema
from src.labels.statistics import SeasonCount, YearCount

if TYPE_CHECKING:
    from src.core.config import Config

from src.dataset.manifest import DatasetSample

__all__ = ["ClassStatistics", "SplitStatistics", "DatasetStatisticsCalculator"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClassStatistics:
    """
    Immutable per-class pixel aggregate for one split.

    Attributes:
        class_id:       Integer class label.
        class_name:      Human-readable class name.
        pixel_count:     Total pixel count across all valid samples.
        sample_count:    Number of samples containing this class.
        percentage:      Percentage of all valid pixels. [0, 100]
    """

    class_id:     int
    class_name:    str
    pixel_count:    int
    sample_count:   int
    percentage:     float


@dataclass(frozen=True)
class SplitStatistics:
    """
    Immutable statistics for one dataset split.

    Attributes:
        split_name:             "train", "validation", "test", or "overall".
        sample_count:            Number of samples in this split.
        class_statistics:         Per-class pixel distributions.
        class_imbalance_ratio:     Most-frequent / least-frequent nonzero class.
                                  1.0 when balanced or only one class present.
        water_sand_ratio:          water_pixels / sand_pixels, or None.
        vegetation_sand_ratio:      vegetation_pixels / sand_pixels, or None.
        bare_sediment_fraction:      sand_pixels / (water+sand+vegetation), or None.
        seasonal_distribution:        Sample counts grouped by season.
        yearly_distribution:           Sample counts grouped by calendar year.
        total_valid_pixels:            Sum of all valid (non-nodata) pixels.
    """

    split_name:            str
    sample_count:           int
    class_statistics:        tuple[ClassStatistics, ...]
    class_imbalance_ratio:    float
    water_sand_ratio:         float | None
    vegetation_sand_ratio:     float | None
    bare_sediment_fraction:     float | None
    seasonal_distribution:       tuple[SeasonCount, ...]
    yearly_distribution:          tuple[YearCount, ...]
    total_valid_pixels:           int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable plain dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SplitStatistics":
        """Reconstruct split statistics from persisted JSON data."""
        return cls(
            split_name=str(data["split_name"]),
            sample_count=int(data["sample_count"]),
            class_statistics=tuple(
                ClassStatistics(
                    class_id=int(item["class_id"]),
                    class_name=str(item["class_name"]),
                    pixel_count=int(item["pixel_count"]),
                    sample_count=int(item["sample_count"]),
                    percentage=float(item["percentage"]),
                )
                for item in data["class_statistics"]
            ),
            class_imbalance_ratio=float(data["class_imbalance_ratio"]),
            water_sand_ratio=(
                None
                if data.get("water_sand_ratio") is None
                else float(data["water_sand_ratio"])
            ),
            vegetation_sand_ratio=(
                None
                if data.get("vegetation_sand_ratio") is None
                else float(data["vegetation_sand_ratio"])
            ),
            bare_sediment_fraction=(
                None
                if data.get("bare_sediment_fraction") is None
                else float(data["bare_sediment_fraction"])
            ),
            seasonal_distribution=tuple(
                SeasonCount(
                    season=str(item["season"]),
                    count=int(item["count"]),
                )
                for item in data["seasonal_distribution"]
            ),
            yearly_distribution=tuple(
                YearCount(
                    year=int(item["year"]),
                    count=int(item["count"]),
                )
                for item in data["yearly_distribution"]
            ),
            total_valid_pixels=int(data["total_valid_pixels"]),
        )


class DatasetStatisticsCalculator:
    """
    Computes SplitStatistics for a list of DatasetSample objects.

    Args:
        class_schema:                ClassSchema defining the class taxonomy.
        nodata_value:                 Integer sentinel for nodata pixels in masks.
        ratio_definitions:             Mapping of ratio_name -> (num_class, den_class).
        bare_sediment_numerator:        Class names summed for numerator.
        bare_sediment_denominator:      Class names summed for denominator.
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
        self._bare_num = bare_sediment_numerator
        self._bare_den = bare_sediment_denominator
        self._logger: logging.Logger = logging.getLogger(__name__)

    @classmethod
    def from_config(
        cls,
        class_schema: ClassSchema,
        config: Any,
    ) -> DatasetStatisticsCalculator:
        """
        Build a DatasetStatisticsCalculator from config.labels.

        Args:
            class_schema: ClassSchema for the active taxonomy.
            config:        Fully initialized Config object.
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

    def compute(
        self,
        samples: list[DatasetSample],
        split_name: str,
        read_masks: bool = True,
    ) -> SplitStatistics:
        """
        Compute statistics for one split.

        Args:
            samples:     Samples belonging to this split.
            split_name:  "train", "validation", "test", or "overall".
            read_masks:  When True, read mask files for per-class pixel counts.
                         When False, pixel counts are zero (metadata-only mode).

        Returns:
            Immutable SplitStatistics.
        """
        pixel_counts: dict[int, int] = {cid: 0 for cid in self._schema.class_ids}
        class_sample_counts: dict[int, int] = {cid: 0 for cid in self._schema.class_ids}
        season_counter: Counter = Counter()
        year_counter:   Counter = Counter()

        for sample in samples:
            season_counter[sample.season] += 1
            year_counter[sample.year]     += 1

            if read_masks:
                mask_array = self._read_mask(Path(sample.mask_path))
                if mask_array is not None:
                    unique, counts = np.unique(mask_array, return_counts=True)
                    seen_classes: set[int] = set()
                    for val, count in zip(unique, counts):
                        cid = int(val)
                        if cid in pixel_counts:
                            pixel_counts[cid] += int(count)
                            seen_classes.add(cid)
                    for cid in seen_classes:
                        class_sample_counts[cid] += 1

        total_valid_pixels = sum(pixel_counts.values())

        class_stats: list[ClassStatistics] = []
        for definition in self._schema.classes:
            count = pixel_counts.get(definition.class_id, 0)
            pct   = (count / total_valid_pixels * 100.0) if total_valid_pixels > 0 else 0.0
            class_stats.append(ClassStatistics(
                class_id=definition.class_id,
                class_name=definition.name,
                pixel_count=count,
                sample_count=class_sample_counts.get(definition.class_id, 0),
                percentage=pct,
            ))

        nonzero = [c for c in pixel_counts.values() if c > 0]
        imbalance = max(nonzero) / min(nonzero) if len(nonzero) >= 2 else 1.0

        seasons = tuple(SeasonCount(s, c) for s, c in sorted(season_counter.items()))
        years   = tuple(YearCount(y, c)   for y, c in sorted(year_counter.items()))

        water_sand_ratio       = self._compute_ratio("water", "sand", pixel_counts)
        vegetation_sand_ratio  = self._compute_ratio("vegetation", "sand", pixel_counts)
        bare_sediment_fraction = self._compute_bare_sediment(pixel_counts)

        return SplitStatistics(
            split_name=split_name,
            sample_count=len(samples),
            class_statistics=tuple(class_stats),
            class_imbalance_ratio=imbalance,
            water_sand_ratio=water_sand_ratio,
            vegetation_sand_ratio=vegetation_sand_ratio,
            bare_sediment_fraction=bare_sediment_fraction,
            seasonal_distribution=seasons,
            yearly_distribution=years,
            total_valid_pixels=total_valid_pixels,
        )

    def save_statistics(
        self,
        statistics_by_split: dict[str, SplitStatistics],
        output_dir: Path,
    ) -> Path:
        """
        Write all split statistics to statistics.json.

        Args:
            statistics_by_split: Mapping of split_name -> SplitStatistics.
            output_dir:          Directory to write statistics.json.

        Returns:
            Absolute path to the written file.
        """
        output_dir = Path(output_dir).resolve()
        path       = output_dir / "statistics.json"
        payload    = {
            split: stats.to_dict()
            for split, stats in statistics_by_split.items()
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=True)
        self._logger.info("Statistics JSON written: %s", path.name)
        return path

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _read_mask(self, mask_path: Path) -> Any | None:
        """Read band 1 of a mask GeoTIFF. Returns None on any failure."""
        try:
            import rasterio
            with rasterio.open(mask_path) as ds:
                return ds.read(1)
        except Exception as exc:
            self._logger.debug("Could not read mask %s: %s", mask_path, exc)
            return None

    def _compute_ratio(
        self,
        numerator_name: str,
        denominator_name: str,
        pixel_counts: dict[int, int],
    ) -> float | None:
        if not self._schema.has_class_name(numerator_name):
            return None
        if not self._schema.has_class_name(denominator_name):
            return None
        num = pixel_counts.get(self._schema.get_id_by_name(numerator_name), 0)
        den = pixel_counts.get(self._schema.get_id_by_name(denominator_name), 0)
        return (num / den) if den > 0 else None

    def _compute_bare_sediment(self, pixel_counts: dict[int, int]) -> float | None:
        if not self._bare_num or not self._bare_den:
            return None
        num_sum = sum(
            pixel_counts.get(self._schema.get_id_by_name(n), 0)
            for n in self._bare_num
            if self._schema.has_class_name(n)
        )
        den_sum = sum(
            pixel_counts.get(self._schema.get_id_by_name(n), 0)
            for n in self._bare_den
            if self._schema.has_class_name(n)
        )
        return (num_sum / den_sum) if den_sum > 0 else None