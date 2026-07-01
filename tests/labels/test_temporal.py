"""
Unit tests for src/labels/temporal.py.

Run:
    pytest tests/labels/test_temporal.py -v \
        --cov=src/labels/temporal --cov-report=term-missing
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from src.core.exceptions import InvalidValueError, MissingFieldError
from src.labels.temporal import (
    HydrologicalYearResolver,
    SeasonResolver,
    TemporalMetadata,
    TemporalMetadataBuilder,
    validate_temporal_consistency,
)
from tests.conftest import make_valid_config, write_config

_SEASONS = {
    "pre_monsoon":  (3, 4, 5),
    "monsoon":      (6, 7, 8, 9),
    "post_monsoon": (10, 11),
    "winter":       (12, 1, 2),
}


def _config(tmp_path: Path, hydro_start_month: int = 6, labels_overrides: dict | None = None):
    from src.core.config import Config
    data = make_valid_config()
    data["temporal"] = {
        "seasons": {k: list(v) for k, v in _SEASONS.items()},
        "hydrological_year_start_month": hydro_start_month,
    }
    data["labels"] = {
        "nodata_value": 255, "mask_filename_pattern": "{patch_id}_mask.tif",
        "default_label_version": "1.0.0", "default_annotator": "auto_generated",
        "default_confidence": 1.0, "default_confidence_source": "automatic",
        "min_distinct_classes": 1, "reject_single_class_masks": False,
        "max_nodata_ratio": 0.5, "output_formats": ["csv", "json"],
        "ratios": {}, "bare_sediment_numerator": [], "bare_sediment_denominator": [],
    }
    if labels_overrides:
        data["labels"].update(labels_overrides)
    return Config(config_path=write_config(tmp_path, data))


@pytest.fixture
def season_resolver() -> SeasonResolver:
    return SeasonResolver(_SEASONS)


@pytest.fixture
def hydro_resolver() -> HydrologicalYearResolver:
    return HydrologicalYearResolver(6)


@pytest.fixture
def config(tmp_path: Path):
    return _config(tmp_path)


@pytest.fixture
def builder(season_resolver, hydro_resolver, config) -> TemporalMetadataBuilder:
    return TemporalMetadataBuilder(season_resolver, hydro_resolver, config)


# ==============================================================================
# SeasonResolver tests
# ==============================================================================

class TestSeasonResolver:
    def test_known_seasons(self, season_resolver: SeasonResolver) -> None:
        assert set(season_resolver.known_seasons) == set(_SEASONS.keys())

    def test_month_out_of_range_raises(self) -> None:
        with pytest.raises(InvalidValueError, match="range \\[1, 12\\]"):
            SeasonResolver({"bad": (13,)})

    def test_overlapping_month_raises(self) -> None:
        with pytest.raises(InvalidValueError, match="already assigned"):
            SeasonResolver({"a": (6,), "b": (6,)})

    def test_from_config(self, config) -> None:
        resolver = SeasonResolver.from_config(config)
        assert "monsoon" in resolver.known_seasons

    def test_from_config_missing_raises(self, tmp_path: Path) -> None:
        from src.core.config import Config
        cfg = Config(config_path=write_config(tmp_path, make_valid_config()))
        with pytest.raises(MissingFieldError, match="temporal.seasons"):
            SeasonResolver.from_config(cfg)

    @pytest.mark.parametrize("month,expected", [
        (3, "pre_monsoon"), (7, "monsoon"), (10, "post_monsoon"), (1, "winter"),
    ])
    def test_resolve(self, season_resolver, month, expected) -> None:
        assert season_resolver.resolve(month) == expected

    def test_resolve_unknown_returns_unknown(self) -> None:
        partial = SeasonResolver({"summer": (6,)})
        assert partial.resolve(1) == "unknown"

    def test_resolve_invalid_month_raises(self, season_resolver) -> None:
        with pytest.raises(InvalidValueError):
            season_resolver.resolve(13)


# ==============================================================================
# HydrologicalYearResolver tests
# ==============================================================================

class TestHydrologicalYearResolver:
    def test_construction_validates_start_month(self) -> None:
        with pytest.raises(InvalidValueError, match="range \\[1, 12\\]"):
            HydrologicalYearResolver(13)

    def test_from_config(self, config) -> None:
        resolver = HydrologicalYearResolver.from_config(config)
        assert resolver.start_month == 6

    def test_from_config_default_one(self, tmp_path: Path) -> None:
        from src.core.config import Config
        data = make_valid_config()
        data["temporal"] = {"seasons": {k: list(v) for k, v in _SEASONS.items()}}
        cfg = Config(config_path=write_config(tmp_path, data))
        resolver = HydrologicalYearResolver.from_config(cfg)
        assert resolver.start_month == 1

    def test_month_after_start_same_year(self, hydro_resolver) -> None:
        assert hydro_resolver.resolve(2023, 7) == 2023

    def test_month_before_start_previous_year(self, hydro_resolver) -> None:
        assert hydro_resolver.resolve(2023, 3) == 2022

    def test_month_equal_start(self, hydro_resolver) -> None:
        assert hydro_resolver.resolve(2023, 6) == 2023

    def test_default_start_month_equals_calendar_year(self) -> None:
        resolver = HydrologicalYearResolver(1)
        assert resolver.resolve(2023, 1) == 2023
        assert resolver.resolve(2023, 12) == 2023

    def test_invalid_month_raises(self, hydro_resolver) -> None:
        with pytest.raises(InvalidValueError):
            hydro_resolver.resolve(2023, 0)


# ==============================================================================
# TemporalMetadataBuilder tests
# ==============================================================================

class TestTemporalMetadataBuilder:
    def test_basic_build(self, builder: TemporalMetadataBuilder) -> None:
        meta = builder.build(
            scene_id="scene001", patch_id="scene001_r000_c000",
            scene_start_date="2023-07-01", scene_end_date="2023-07-31",
            sensors=("L8", "L9"), aoi_id="aoi_1",
        )
        assert isinstance(meta, TemporalMetadata)
        assert meta.sensor == "L8,L9"

    def test_midpoint_computation(self, builder) -> None:
        meta = builder.build(
            scene_id="s", patch_id="p", scene_start_date="2023-07-01",
            scene_end_date="2023-07-31", sensors=("L8",), aoi_id="a",
        )
        assert meta.acquisition_date == "2023-07-16"

    def test_season_resolved(self, builder) -> None:
        meta = builder.build(
            scene_id="s", patch_id="p", scene_start_date="2023-07-01",
            scene_end_date="2023-07-31", sensors=("L8",), aoi_id="a",
        )
        assert meta.season == "monsoon"

    def test_hydrological_year_after_start_month(self, builder) -> None:
        meta = builder.build(
            scene_id="s", patch_id="p", scene_start_date="2023-07-01",
            scene_end_date="2023-07-31", sensors=("L8",), aoi_id="a",
        )
        assert meta.hydrological_year == 2023

    def test_hydrological_year_before_start_month(self, builder) -> None:
        meta = builder.build(
            scene_id="s", patch_id="p", scene_start_date="2023-03-01",
            scene_end_date="2023-03-15", sensors=("L8",), aoi_id="a",
        )
        assert meta.hydrological_year == 2022

    def test_defaults_from_config(self, builder) -> None:
        meta = builder.build(
            scene_id="s", patch_id="p", scene_start_date="2023-01-01",
            scene_end_date="2023-01-02", sensors=("L8",), aoi_id="a",
        )
        assert meta.label_version == "1.0.0"
        assert meta.annotator == "auto_generated"
        assert meta.confidence == pytest.approx(1.0)
        assert meta.confidence_source == "automatic"

    def test_overrides_applied(self, builder) -> None:
        meta = builder.build(
            scene_id="s", patch_id="p", scene_start_date="2023-01-01",
            scene_end_date="2023-01-02", sensors=("L8",), aoi_id="a",
            label_version="2.0.0", annotator="jane", confidence=0.8,
            confidence_source="manual", river_name="Kosi",
            reach_id="reach-12", basin_id="basin-3",
        )
        assert meta.label_version == "2.0.0"
        assert meta.annotator == "jane"
        assert meta.confidence == pytest.approx(0.8)
        assert meta.confidence_source == "manual"
        assert meta.river_name == "Kosi"
        assert meta.reach_id == "reach-12"
        assert meta.basin_id == "basin-3"

    def test_optional_fields_none_by_default(self, builder) -> None:
        meta = builder.build(
            scene_id="s", patch_id="p", scene_start_date="2023-01-01",
            scene_end_date="2023-01-02", sensors=("L8",), aoi_id="a",
        )
        assert meta.river_name is None
        assert meta.reach_id is None
        assert meta.basin_id is None

    def test_confidence_out_of_range_raises(self, builder) -> None:
        with pytest.raises(InvalidValueError, match="confidence"):
            builder.build(
                scene_id="s", patch_id="p", scene_start_date="2023-01-01",
                scene_end_date="2023-01-02", sensors=("L8",), aoi_id="a",
                confidence=1.5,
            )

    def test_malformed_dates_raise(self, builder) -> None:
        with pytest.raises(InvalidValueError, match="YYYY-MM-DD"):
            builder.build(
                scene_id="s", patch_id="p", scene_start_date="01-01-2023",
                scene_end_date="2023-01-02", sensors=("L8",), aoi_id="a",
            )

    def test_start_after_end_raises(self, builder) -> None:
        with pytest.raises(InvalidValueError, match="must not be after"):
            builder.build(
                scene_id="s", patch_id="p", scene_start_date="2023-12-01",
                scene_end_date="2023-01-01", sensors=("L8",), aoi_id="a",
            )

    def test_processing_history_preserved(self, builder) -> None:
        meta = builder.build(
            scene_id="s", patch_id="p", scene_start_date="2023-01-01",
            scene_end_date="2023-01-02", sensors=("L8",), aoi_id="a",
            processing_history=("discovered", "validated"),
        )
        assert meta.processing_history == ("discovered", "validated")


# ==============================================================================
# validate_temporal_consistency tests
# ==============================================================================

class TestValidateTemporalConsistency:
    def _meta(self, **overrides) -> TemporalMetadata:
        base = dict(
            scene_id="s", patch_id="p", acquisition_date="2023-07-15",
            year=2023, month=7, season="monsoon", hydrological_year=2023,
            sensor="L8", river_name=None, reach_id=None, basin_id=None,
            aoi_id="a", label_version="1.0.0", annotator="x", confidence=1.0,
            confidence_source="automatic", processing_history=(),
        )
        base.update(overrides)
        return TemporalMetadata(**base)

    def test_consistent_passes(self, season_resolver, hydro_resolver) -> None:
        ok, issues = validate_temporal_consistency(self._meta(), season_resolver, hydro_resolver)
        assert ok is True
        assert issues == ()

    def test_mismatched_season_fails(self, season_resolver, hydro_resolver) -> None:
        meta = self._meta(season="winter")
        ok, issues = validate_temporal_consistency(meta, season_resolver, hydro_resolver)
        assert ok is False
        assert any("season" in i for i in issues)

    def test_mismatched_hydrological_year_fails(self, season_resolver, hydro_resolver) -> None:
        meta = self._meta(hydrological_year=1999)
        ok, issues = validate_temporal_consistency(meta, season_resolver, hydro_resolver)
        assert ok is False
        assert any("hydrological_year" in i for i in issues)

    def test_invalid_month_fails(self, season_resolver, hydro_resolver) -> None:
        meta = self._meta(month=13)
        ok, issues = validate_temporal_consistency(meta, season_resolver, hydro_resolver)
        assert ok is False

    def test_implausible_year_fails(self, season_resolver, hydro_resolver) -> None:
        meta = self._meta(year=1900)
        ok, issues = validate_temporal_consistency(meta, season_resolver, hydro_resolver)
        assert ok is False

    def test_future_year_fails(self, season_resolver, hydro_resolver) -> None:
        meta = self._meta(year=datetime.now().year + 5)
        ok, issues = validate_temporal_consistency(meta, season_resolver, hydro_resolver)
        assert ok is False

    def test_invalid_confidence_fails(self, season_resolver, hydro_resolver) -> None:
        meta = self._meta(confidence=1.5)
        ok, issues = validate_temporal_consistency(meta, season_resolver, hydro_resolver)
        assert ok is False