# no change
"""
Temporal metadata management for the label pipeline (Module 9).

River morphology changes continuously: sandbars become vegetated, channels
migrate, and water extent varies with monsoon and flood-recession cycles.
TemporalMetadata preserves this context for every label so that downstream
modules never treat labels as temporally static snapshots, and so that
multiple independent observations of the same AOI at different dates can
coexist without one overwriting another.

Season boundaries and the hydrological-year start month are entirely
configuration-driven via config.temporal. No season name, month range, or
hydrological year boundary is hardcoded anywhere in this module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import TYPE_CHECKING

from src.core.exceptions import InvalidValueError, MissingFieldError

if TYPE_CHECKING:
    from src.core.config import Config

__all__ = [
    "SeasonResolver",
    "HydrologicalYearResolver",
    "TemporalMetadata",
    "TemporalMetadataBuilder",
    "validate_temporal_consistency",
]

_LOGGER: logging.Logger = logging.getLogger(__name__)
_DATE_FORMAT: str = "%Y-%m-%d"
_UNKNOWN_SEASON: str = "unknown"
_MIN_PLAUSIBLE_YEAR: int = 1972  # Landsat program start


# ==============================================================================
# SeasonResolver
# ==============================================================================

class SeasonResolver:
    """
    Resolves a calendar month to a season name using config-driven boundaries.

    Args:
        season_months: Mapping of season name -> tuple of month integers
                       in [1, 12].

    Raises:
        InvalidValueError: A month value is outside [1, 12], or a month is
                           assigned to more than one season.
    """

    def __init__(self, season_months: dict[str, tuple[int, ...]]) -> None:
        seen: dict[int, str] = {}
        for season, months in season_months.items():
            for month in months:
                if not (1 <= month <= 12):
                    raise InvalidValueError(
                        field=f"temporal.seasons.{season}",
                        value=month,
                        reason="month values must be in range [1, 12]",
                    )
                if month in seen:
                    raise InvalidValueError(
                        field=f"temporal.seasons.{season}",
                        value=month,
                        reason=(
                            f"month {month} is already assigned to season "
                            f"'{seen[month]}'; each month must map to exactly "
                            "one season"
                        ),
                    )
                seen[month] = season

        self._month_to_season: dict[int, str] = seen
        self._logger: logging.Logger = logging.getLogger(__name__)

    @classmethod
    def from_config(cls, config: Config) -> SeasonResolver:
        """
        Build a SeasonResolver from config.temporal.seasons.

        Raises:
            MissingFieldError: config.temporal.seasons is absent.
        """
        temporal_cfg = getattr(config, "temporal", None)
        seasons_cfg = getattr(temporal_cfg, "seasons", None) if temporal_cfg else None
        if seasons_cfg is None:
            raise MissingFieldError(
                field="temporal.seasons",
                context=(
                    "config.temporal.seasons must define a mapping of season "
                    "name to a list of month integers, e.g.:\n"
                    "  temporal:\n"
                    "    seasons:\n"
                    "      monsoon: [6, 7, 8, 9]"
                ),
            )
        season_months: dict[str, tuple[int, ...]] = {
            name: tuple(int(m) for m in getattr(seasons_cfg, name))
            for name in seasons_cfg
        }
        return cls(season_months)

    def resolve(self, month: int) -> str:
        """
        Return the configured season name for a calendar month.

        Returns "unknown" if the month is not covered by any configured
        season.

        Raises:
            InvalidValueError: month is outside [1, 12].
        """
        if not (1 <= month <= 12):
            raise InvalidValueError(
                field="month", value=month, reason="must be in range [1, 12]",
            )
        return self._month_to_season.get(month, _UNKNOWN_SEASON)

    @property
    def known_seasons(self) -> tuple[str, ...]:
        """Tuple of all distinct configured season names, sorted."""
        return tuple(sorted(set(self._month_to_season.values())))


# ==============================================================================
# HydrologicalYearResolver
# ==============================================================================

class HydrologicalYearResolver:
    """
    Resolves a (year, month) pair to a hydrological year using a
    configurable start month.

    A hydrological year groups months belonging to one continuous flow
    regime (e.g. June of year Y through May of year Y+1), rather than the
    calendar year, which can split a single monsoon-to-recession cycle
    across two calendar years.

    Args:
        start_month: Calendar month [1, 12] that begins each hydrological
                     year. A value of 1 makes hydrological_year identical
                     to the calendar year (the backward-compatible default).
                     Sourced from
                     config.temporal.hydrological_year_start_month.

    Raises:
        InvalidValueError: start_month is outside [1, 12].
    """

    def __init__(self, start_month: int) -> None:
        if not (1 <= start_month <= 12):
            raise InvalidValueError(
                field="temporal.hydrological_year_start_month",
                value=start_month,
                reason="must be in range [1, 12]",
            )
        self._start_month = start_month
        self._logger: logging.Logger = logging.getLogger(__name__)

    @classmethod
    def from_config(cls, config: Config) -> HydrologicalYearResolver:
        """Build a HydrologicalYearResolver from config.temporal.hydrological_year_start_month."""
        temporal_cfg = getattr(config, "temporal", None)
        start_month = int(
            getattr(temporal_cfg, "hydrological_year_start_month", 1)
        )
        return cls(start_month)

    @property
    def start_month(self) -> int:
        """The configured hydrological year start month."""
        return self._start_month

    def resolve(self, year: int, month: int) -> int:
        """
        Return the hydrological year for a given calendar (year, month).

        Months before start_month belong to the hydrological year that
        began the previous calendar year (year - 1).

        Raises:
            InvalidValueError: month is outside [1, 12].
        """
        if not (1 <= month <= 12):
            raise InvalidValueError(
                field="month", value=month, reason="must be in range [1, 12]",
            )
        return year if month >= self._start_month else year - 1


# ==============================================================================
# TemporalMetadata
# ==============================================================================

@dataclass(frozen=True)
class TemporalMetadata:
    """
    Immutable temporal and provenance record for one label.

    Each observation of an AOI is independent: this record never implies
    that the same geography is correctly labeled at any other date.

    Attributes:
        scene_id:              Source scene identifier.
        patch_id:                Source patch identifier.
        acquisition_date:          Representative date (YYYY-MM-DD) for the
                                  underlying imagery, computed as the
                                  midpoint of the scene's collection date range.
        year:                      Calendar year of acquisition_date.
        month:                     Calendar month of acquisition_date [1, 12].
        season:                    Season name resolved via SeasonResolver.
        hydrological_year:           Hydrological year resolved via
                                  HydrologicalYearResolver.
        sensor:                     Comma-separated sensor names, e.g. "L8,L9".
        river_name:                  Optional river name. None if unspecified.
        reach_id:                    Optional river reach identifier. None if
                                  unspecified.
        basin_id:                    Optional drainage basin identifier. None
                                  if unspecified.
        aoi_id:                      Identifier for the configured area of interest.
        label_version:                Version string for this label's annotation.
        annotator:                    Identifier of the annotation source
                                  (e.g. "auto_generated", a person's name).
        confidence:                    Annotation confidence in [0.0, 1.0].
        confidence_source:             Origin of the confidence value, e.g.
                                  "manual", "automatic", "reviewed".
        processing_history:             Ordered tuple of processing steps
                                  applied to this label.
    """

    scene_id:              str
    patch_id:                str
    acquisition_date:          str
    year:                      int
    month:                     int
    season:                    str
    hydrological_year:           int
    sensor:                     str
    river_name:                  str | None
    reach_id:                    str | None
    basin_id:                    str | None
    aoi_id:                      str
    label_version:                str
    annotator:                    str
    confidence:                    float
    confidence_source:             str
    processing_history:             tuple[str, ...]


# ==============================================================================
# TemporalMetadataBuilder
# ==============================================================================

class TemporalMetadataBuilder:
    """
    Builds TemporalMetadata records using config-driven defaults.

    Args:
        season_resolver:        SeasonResolver for month -> season assignment.
        hydro_year_resolver:      HydrologicalYearResolver for hydrological
                                year assignment.
        config:                   Fully initialized Config; used for label
                                defaults (label_version, annotator, confidence,
                                confidence_source) read from config.labels.
    """

    def __init__(
        self,
        season_resolver:      SeasonResolver,
        hydro_year_resolver:    HydrologicalYearResolver,
        config:                 Config,
    ) -> None:
        self._season_resolver = season_resolver
        self._hydro_year_resolver = hydro_year_resolver

        labels_cfg = getattr(config, "labels", None)
        self._default_label_version = str(
            getattr(labels_cfg, "default_label_version", "1.0.0")
        )
        self._default_annotator = str(
            getattr(labels_cfg, "default_annotator", "auto_generated")
        )
        self._default_confidence = float(
            getattr(labels_cfg, "default_confidence", 1.0)
        )
        self._default_confidence_source = str(
            getattr(labels_cfg, "default_confidence_source", "automatic")
        )
        self._logger: logging.Logger = logging.getLogger(__name__)

    def build(
        self,
        scene_id:             str,
        patch_id:               str,
        scene_start_date:        str,
        scene_end_date:           str,
        sensors:                  tuple[str, ...],
        aoi_id:                   str,
        river_name:               str | None = None,
        reach_id:                 str | None = None,
        basin_id:                 str | None = None,
        label_version:             str | None = None,
        annotator:                 str | None = None,
        confidence:                 float | None = None,
        confidence_source:          str | None = None,
        processing_history:          tuple[str, ...] = (),
    ) -> TemporalMetadata:
        """
        Build a TemporalMetadata record for one patch.

        acquisition_date is computed as the midpoint between
        scene_start_date and scene_end_date, representing the temporal
        center of the composite collection period.

        Args:
            scene_id:            Source scene identifier.
            patch_id:             Source patch identifier.
            scene_start_date:      Scene collection date range start
                                  (YYYY-MM-DD).
            scene_end_date:        Scene collection date range end
                                  (YYYY-MM-DD).
            sensors:                Sensor names used to build the scene.
            aoi_id:                 AOI identifier.
            river_name:             Optional river name.
            reach_id:               Optional river reach identifier.
            basin_id:               Optional drainage basin identifier.
            label_version:          Override label version. Defaults to
                                   config.labels.default_label_version.
            annotator:              Override annotator. Defaults to
                                   config.labels.default_annotator.
            confidence:             Override confidence in [0.0, 1.0].
                                   Defaults to config.labels.default_confidence.
            confidence_source:       Override confidence source. Defaults to
                                   config.labels.default_confidence_source.
            processing_history:      Ordered processing steps applied.

        Returns:
            Populated, immutable TemporalMetadata.

        Raises:
            InvalidValueError: dates are malformed/out of order, or
                               confidence is outside [0.0, 1.0].
        """
        midpoint = self._compute_midpoint(scene_start_date, scene_end_date)
        resolved_confidence = (
            confidence if confidence is not None else self._default_confidence
        )
        if not (0.0 <= resolved_confidence <= 1.0):
            raise InvalidValueError(
                field="confidence",
                value=resolved_confidence,
                reason="must be in range [0.0, 1.0]",
            )

        return TemporalMetadata(
            scene_id=scene_id,
            patch_id=patch_id,
            acquisition_date=midpoint.strftime(_DATE_FORMAT),
            year=midpoint.year,
            month=midpoint.month,
            season=self._season_resolver.resolve(midpoint.month),
            hydrological_year=self._hydro_year_resolver.resolve(
                midpoint.year, midpoint.month
            ),
            sensor=",".join(sensors),
            river_name=river_name,
            reach_id=reach_id,
            basin_id=basin_id,
            aoi_id=aoi_id,
            label_version=label_version or self._default_label_version,
            annotator=annotator or self._default_annotator,
            confidence=resolved_confidence,
            confidence_source=confidence_source or self._default_confidence_source,
            processing_history=tuple(processing_history),
        )

    @staticmethod
    def _compute_midpoint(start_date: str, end_date: str) -> date:
        """Return the midpoint date between start_date and end_date."""
        try:
            start = datetime.strptime(start_date, _DATE_FORMAT).date()
            end = datetime.strptime(end_date, _DATE_FORMAT).date()
        except ValueError as exc:
            raise InvalidValueError(
                field="scene date range",
                value=f"{start_date} to {end_date}",
                reason=f"dates must be in YYYY-MM-DD format: {exc}",
            ) from exc

        if start > end:
            raise InvalidValueError(
                field="scene date range",
                value=f"{start_date} to {end_date}",
                reason="start_date must not be after end_date",
            )

        return start + (end - start) // 2


# ==============================================================================
# Temporal consistency validation
# ==============================================================================

def validate_temporal_consistency(
    metadata:               TemporalMetadata,
    season_resolver:          SeasonResolver,
    hydro_year_resolver:        HydrologicalYearResolver,
) -> tuple[bool, tuple[str, ...]]:
    """
    Check internal consistency of a TemporalMetadata record.

    Verifies:
        - month is in [1, 12]
        - the recorded season matches season_resolver.resolve(month)
        - the recorded hydrological_year matches
          hydro_year_resolver.resolve(year, month)
        - year is plausible (Landsat program start through next calendar year)
        - confidence is in [0.0, 1.0]

    Args:
        metadata:              TemporalMetadata to check.
        season_resolver:         SeasonResolver used to re-derive the expected season.
        hydro_year_resolver:       HydrologicalYearResolver used to re-derive the
                                  expected hydrological year.

    Returns:
        Tuple of (is_consistent, issues). issues is empty when is_consistent
        is True.
    """
    issues: list[str] = []

    if not (1 <= metadata.month <= 12):
        issues.append(f"month {metadata.month} is outside range [1, 12]")
    else:
        expected_season = season_resolver.resolve(metadata.month)
        if metadata.season != expected_season:
            issues.append(
                f"season '{metadata.season}' does not match expected "
                f"'{expected_season}' for month {metadata.month}"
            )

        expected_hydro_year = hydro_year_resolver.resolve(metadata.year, metadata.month)
        if metadata.hydrological_year != expected_hydro_year:
            issues.append(
                f"hydrological_year {metadata.hydrological_year} does not match "
                f"expected {expected_hydro_year} for year={metadata.year}, "
                f"month={metadata.month}"
            )

    current_year = datetime.now().year
    if metadata.year < _MIN_PLAUSIBLE_YEAR or metadata.year > current_year + 1:
        issues.append(
            f"year {metadata.year} is outside plausible range "
            f"[{_MIN_PLAUSIBLE_YEAR}, {current_year + 1}]"
        )

    if not (0.0 <= metadata.confidence <= 1.0):
        issues.append(
            f"confidence {metadata.confidence} is outside range [0.0, 1.0]"
        )

    return (len(issues) == 0, tuple(issues))