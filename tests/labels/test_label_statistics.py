"""
Unit tests for src/labels/statistics.py.

Run:
    pytest tests/labels/test_label_statistics.py -v \
        --cov=src/labels/statistics --cov-report=term-missing
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.labels.schema import ClassDefinition, ClassSchema
from src.labels.statistics import LabelStatisticsCalculator
from src.labels.temporal import TemporalMetadata
from src.labels.validator import LabelValidationResult
from tests.conftest import make_valid_config, write_config


def _schema() -> ClassSchema:
    return ClassSchema(classes=(
        ClassDefinition(0, "background", (128, 128, 128)),
        ClassDefinition(1, "water", (0, 119, 190)),
        ClassDefinition(2, "sand", (255, 200, 87)),
        ClassDefinition(3, "vegetation", (34, 139, 34)),
    ))


def _temporal(season: str = "monsoon", year: int = 2023) -> TemporalMetadata:
    return TemporalMetadata(
        scene_id="s", patch_id="p", acquisition_date="2023-07-15",
        year=year, month=7, season=season, hydrological_year=year,
        sensor="L8", river_name=None, reach_id=None, basin_id=None,
        aoi_id="a", label_version="1.0.0", annotator="x", confidence=1.0,
        confidence_source="automatic", processing_history=(),
    )


def _valid_result() -> LabelValidationResult:
    return LabelValidationResult(
        patch_id="p", is_valid=True, issues=(), num_classes_present=2,
        is_single_class=False, valid_pixel_ratio=1.0, crs_match=True,
        transform_match=True, dimension_match=True, mask_exists=True,
    )


def _invalid_result() -> LabelValidationResult:
    return LabelValidationResult(
        patch_id="p", is_valid=False, issues=("bad",), num_classes_present=0,
        is_single_class=False, valid_pixel_ratio=0.0, crs_match=False,
        transform_match=False, dimension_match=False, mask_exists=False,
    )


def _config(tmp_path: Path, ratios: dict | None = None, bare_num=None, bare_denom=None):
    from src.core.config import Config
    data = make_valid_config()
    data["labels"] = {
        "nodata_value": 255,
        "ratios": ratios or {"water_sand_ratio": ["water", "sand"]},
        "bare_sediment_numerator": bare_num if bare_num is not None else ["sand"],
        "bare_sediment_denominator": bare_denom if bare_denom is not None else ["water", "sand", "vegetation"],
    }
    return Config(config_path=write_config(tmp_path, data))


class TestLabelStatisticsCalculatorAccumulate:
    def test_valid_label_counts_pixels(self) -> None:
        calc = LabelStatisticsCalculator(_schema(), nodata_value=255)
        mask = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        calc.accumulate(mask, _valid_result(), _temporal())
        stats = calc.compute()
        assert stats.valid_labels == 1

    def test_rejected_excluded_from_pixel_counts(self) -> None:
        calc = LabelStatisticsCalculator(_schema(), nodata_value=255)
        calc.accumulate(None, _invalid_result(), _temporal())
        stats = calc.compute()
        assert stats.rejected_labels == 1
        for cs in stats.class_pixel_stats:
            assert cs.pixel_count == 0

    def test_percentages_sum_to_100(self) -> None:
        calc = LabelStatisticsCalculator(_schema(), nodata_value=255)
        mask = np.array([0, 0, 1, 1, 2, 3], dtype=np.uint8).reshape(2, 3)
        calc.accumulate(mask, _valid_result(), _temporal())
        stats = calc.compute()
        assert sum(cs.percentage for cs in stats.class_pixel_stats) == pytest.approx(100.0)

    def test_imbalance_ratio(self) -> None:
        calc = LabelStatisticsCalculator(_schema(), nodata_value=255)
        mask = np.array([1] * 9 + [2], dtype=np.uint8).reshape(2, 5)
        calc.accumulate(mask, _valid_result(), _temporal())
        stats = calc.compute()
        assert stats.class_imbalance_ratio == pytest.approx(9.0)

    def test_imbalance_one_when_no_labels(self) -> None:
        calc = LabelStatisticsCalculator(_schema(), nodata_value=255)
        stats = calc.compute()
        assert stats.class_imbalance_ratio == pytest.approx(1.0)

    def test_seasonal_distribution(self) -> None:
        calc = LabelStatisticsCalculator(_schema(), nodata_value=255)
        mask = np.zeros((2, 2), dtype=np.uint8)
        calc.accumulate(mask, _valid_result(), _temporal(season="monsoon"))
        calc.accumulate(mask, _valid_result(), _temporal(season="winter"))
        calc.accumulate(mask, _valid_result(), _temporal(season="monsoon"))
        stats = calc.compute()
        seasons = {s.season: s.count for s in stats.seasonal_distribution}
        assert seasons["monsoon"] == 2 and seasons["winter"] == 1

    def test_yearly_distribution(self) -> None:
        calc = LabelStatisticsCalculator(_schema(), nodata_value=255)
        mask = np.zeros((2, 2), dtype=np.uint8)
        calc.accumulate(mask, _valid_result(), _temporal(year=2022))
        calc.accumulate(mask, _valid_result(), _temporal(year=2023))
        stats = calc.compute()
        years = {y.year: y.count for y in stats.yearly_distribution}
        assert years[2022] == 1 and years[2023] == 1

    def test_total_labels_counts_all(self) -> None:
        calc = LabelStatisticsCalculator(_schema(), nodata_value=255)
        mask = np.zeros((2, 2), dtype=np.uint8)
        calc.accumulate(mask, _valid_result(), _temporal())
        calc.accumulate(None, _invalid_result(), _temporal())
        stats = calc.compute()
        assert stats.total_labels == 2


class TestRatiosAndBareSediment:
    def test_ratio_computed(self) -> None:
        calc = LabelStatisticsCalculator(
            _schema(), nodata_value=255,
            ratio_definitions={"water_sand_ratio": ("water", "sand")},
        )
        mask = np.array([1, 1, 1, 1, 2], dtype=np.uint8).reshape(1, 5)
        calc.accumulate(mask, _valid_result(), _temporal())
        stats = calc.compute()
        ratio = next(r for r in stats.class_ratios if r.name == "water_sand_ratio")
        assert ratio.value == pytest.approx(4.0)

    def test_ratio_none_when_denominator_zero(self) -> None:
        calc = LabelStatisticsCalculator(
            _schema(), nodata_value=255,
            ratio_definitions={"water_sand_ratio": ("water", "sand")},
        )
        mask = np.array([1, 1], dtype=np.uint8).reshape(1, 2)
        calc.accumulate(mask, _valid_result(), _temporal())
        stats = calc.compute()
        ratio = next(r for r in stats.class_ratios if r.name == "water_sand_ratio")
        assert ratio.value is None

    def test_ratio_none_for_unknown_class_name(self) -> None:
        calc = LabelStatisticsCalculator(
            _schema(), nodata_value=255,
            ratio_definitions={"bad_ratio": ("unknown_class", "sand")},
        )
        stats = calc.compute()
        ratio = next(r for r in stats.class_ratios if r.name == "bad_ratio")
        assert ratio.value is None

    def test_bare_sediment_fraction(self) -> None:
        calc = LabelStatisticsCalculator(
            _schema(), nodata_value=255,
            bare_sediment_numerator=("sand",),
            bare_sediment_denominator=("water", "sand", "vegetation"),
        )
        mask = np.array([1, 2, 2, 3], dtype=np.uint8).reshape(1, 4)
        calc.accumulate(mask, _valid_result(), _temporal())
        stats = calc.compute()
        assert stats.bare_sediment_fraction == pytest.approx(2 / 4)

    def test_bare_sediment_none_when_no_config(self) -> None:
        calc = LabelStatisticsCalculator(_schema(), nodata_value=255)
        mask = np.array([1, 2], dtype=np.uint8).reshape(1, 2)
        calc.accumulate(mask, _valid_result(), _temporal())
        stats = calc.compute()
        assert stats.bare_sediment_fraction is None

    def test_from_config(self, tmp_path: Path) -> None:
        config = _config(tmp_path)
        calc = LabelStatisticsCalculator.from_config(_schema(), config)
        mask = np.array([1, 1, 2], dtype=np.uint8).reshape(1, 3)
        calc.accumulate(mask, _valid_result(), _temporal())
        stats = calc.compute()
        ratio = next(r for r in stats.class_ratios if r.name == "water_sand_ratio")
        assert ratio.value == pytest.approx(2.0)


class TestSummaryLines:
    def test_ascii_only(self) -> None:
        calc = LabelStatisticsCalculator(
            _schema(), nodata_value=255,
            ratio_definitions={"water_sand_ratio": ("water", "sand")},
            bare_sediment_numerator=("sand",),
            bare_sediment_denominator=("water", "sand"),
        )
        mask = np.array([1, 2], dtype=np.uint8).reshape(1, 2)
        calc.accumulate(mask, _valid_result(), _temporal())
        stats = calc.compute()
        for line in stats.summary_lines():
            assert all(ord(c) < 128 for c in line)