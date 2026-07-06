"""
Temporal and seasonal analytics for Module 17.

TemporalAnalyzer:
    Computes TemporalChange between consecutive (by acquisition_date) samples.
    Change is signed: positive = increase, negative = decrease.

SeasonalAggregator:
    Groups SampleAnalysis objects by season and computes mean/std area fractions.
    Season assignment uses the 'season' field from SamplePrediction when available.
    When config.seasons defines month-to-season mappings, those take precedence.

Scientific assumptions
-----------------------
- "Consecutive" means chronologically adjacent by acquisition_date (YYYY-MM-DD).
  When multiple samples share the same date, they are averaged before differencing.
- Percentage change = 100 * (to - from) / from. Returns 0.0 when from == 0
  to avoid division by zero.
- Seasonal aggregation groups by the season name string. The default season names
  ("monsoon", "pre-monsoon", "post-monsoon", "winter") come from InferenceResult
  metadata; custom seasons from config.analytics.seasons override if provided.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np

from src.morphology.contracts import (
    AnalyticsConfig,
    ClassMorphologyMetrics,
    SampleAnalysis,
    SeasonalSummary,
    TemporalChange,
)

__all__ = ["TemporalAnalyzer", "SeasonalAggregator"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


# ==============================================================================
# TemporalAnalyzer
# ==============================================================================

class TemporalAnalyzer:
    """
    Computes TemporalChange between chronologically consecutive sample analyses.

    Args:
        config:      AnalyticsConfig.
        class_names: Ordered class names.
    """

    def __init__(
        self,
        config:      AnalyticsConfig,
        class_names: tuple[str, ...],
    ) -> None:
        self._config      = config
        self._class_names = class_names

    def compute(
        self,
        analyses: list[SampleAnalysis],
    ) -> list[TemporalChange]:
        """
        Compute TemporalChange for all consecutive pairs (sorted by date).

        Args:
            analyses: List of SampleAnalysis objects (any order).

        Returns:
            List of TemporalChange, one per consecutive pair per class.
            Empty when fewer than 2 distinct dates exist.
        """
        if len(analyses) < 2:
            return []

        # Sort by acquisition_date then average metrics per date.
        by_date: dict[str, list[SampleAnalysis]] = defaultdict(list)
        for a in analyses:
            by_date[a.acquisition_date].append(a)

        sorted_dates = sorted(d for d in by_date if d)
        if len(sorted_dates) < 2:
            return []

        # Build mean ClassMorphologyMetrics per date.
        date_metrics: dict[str, dict[str, ClassMorphologyMetrics]] = {}
        for date in sorted_dates:
            date_metrics[date] = self._mean_metrics(by_date[date])

        # Compute changes for consecutive pairs.
        changes: list[TemporalChange] = []
        for i in range(len(sorted_dates) - 1):
            d_from = sorted_dates[i]
            d_to   = sorted_dates[i + 1]
            for class_name in self._class_names:
                m_from = date_metrics[d_from].get(class_name)
                m_to   = date_metrics[d_to].get(class_name)
                if m_from is None or m_to is None:
                    continue
                delta      = m_to.pixel_count - m_from.pixel_count
                frac_delta = m_to.area_fraction - m_from.area_fraction
                pct        = (
                    100.0 * delta / m_from.pixel_count
                    if m_from.pixel_count > 0 else 0.0
                )
                changes.append(TemporalChange(
                    class_name          = class_name,
                    date_from           = d_from,
                    date_to             = d_to,
                    pixel_count_from    = m_from.pixel_count,
                    pixel_count_to      = m_to.pixel_count,
                    pixel_delta         = delta,
                    area_fraction_from  = m_from.area_fraction,
                    area_fraction_to    = m_to.area_fraction,
                    fraction_delta      = frac_delta,
                    pct_change          = float(pct),
                ))

        return changes

    def _mean_metrics(
        self,
        analyses: list[SampleAnalysis],
    ) -> dict[str, ClassMorphologyMetrics]:
        """Average ClassMorphologyMetrics across multiple analyses on the same date."""
        if len(analyses) == 1:
            return dict(analyses[0].class_metrics)

        result: dict[str, ClassMorphologyMetrics] = {}
        for class_name in self._class_names:
            vals = [a.class_metrics[class_name] for a in analyses
                    if class_name in a.class_metrics]
            if not vals:
                continue
            result[class_name] = ClassMorphologyMetrics(
                class_name      = class_name,
                class_id        = vals[0].class_id,
                pixel_count     = int(round(np.mean([v.pixel_count for v in vals]))),
                area_fraction   = float(np.mean([v.area_fraction  for v in vals])),
                total_fraction  = float(np.mean([v.total_fraction for v in vals])),
                area_m2         = float(np.mean([v.area_m2        for v in vals])),
                mean_confidence = float(np.mean([v.mean_confidence for v in vals])),
                low_conf_pixels = int(round(np.mean([v.low_conf_pixels for v in vals]))),
            )
        return result


# ==============================================================================
# SeasonalAggregator
# ==============================================================================

class SeasonalAggregator:
    """
    Groups SampleAnalysis objects by season and computes summary statistics.

    Season assignment:
        1. If config.seasons has month-to-season mappings, use those.
        2. Otherwise use the 'season' field from SampleAnalysis directly.
        3. Samples with empty season fields are grouped under "unknown".

    Args:
        config:      AnalyticsConfig.
        class_names: Ordered class names.
    """

    def __init__(
        self,
        config:      AnalyticsConfig,
        class_names: tuple[str, ...],
    ) -> None:
        self._config      = config
        self._class_names = class_names
        # Build month -> season lookup from config.
        self._month_to_season: dict[int, str] = {}
        for season_name, months in config.seasons.items():
            for m in months:
                self._month_to_season[int(m)] = season_name

    def compute(
        self,
        analyses: list[SampleAnalysis],
    ) -> dict[str, SeasonalSummary]:
        """
        Build SeasonalSummary for each season present in the analyses.

        Args:
            analyses: List of SampleAnalysis objects.

        Returns:
            Dict season_name -> SeasonalSummary.
        """
        if not analyses:
            return {}

        # Group by resolved season.
        groups: dict[str, list[SampleAnalysis]] = defaultdict(list)
        for analysis in analyses:
            season = self._resolve_season(analysis)
            groups[season].append(analysis)

        result: dict[str, SeasonalSummary] = {}
        for season, group in groups.items():
            result[season] = self._build_summary(season, group)

        return result

    def _resolve_season(self, analysis: SampleAnalysis) -> str:
        """Resolve season name for one analysis using month mapping or field."""
        if self._month_to_season and analysis.hydrological_year > 0:
            # Try to extract month from acquisition_date (YYYY-MM-DD).
            try:
                month = int(analysis.acquisition_date.split("-")[1])
                if month in self._month_to_season:
                    return self._month_to_season[month]
            except (IndexError, ValueError):
                pass
        return analysis.season if analysis.season else "unknown"

    def _build_summary(
        self,
        season:   str,
        analyses: list[SampleAnalysis],
    ) -> SeasonalSummary:
        """Build SeasonalSummary from a list of analyses in the same season."""
        frac_by_class: dict[str, list[float]] = {n: [] for n in self._class_names}
        for analysis in analyses:
            for class_name, cm in analysis.class_metrics.items():
                if class_name in frac_by_class:
                    frac_by_class[class_name].append(cm.area_fraction)

        mean_fracs: dict[str, float] = {}
        std_fracs:  dict[str, float] = {}
        for class_name, vals in frac_by_class.items():
            if vals:
                mean_fracs[class_name] = float(np.mean(vals))
                std_fracs[class_name]  = float(np.std(vals))
            else:
                mean_fracs[class_name] = 0.0
                std_fracs[class_name]  = 0.0

        return SeasonalSummary(
            season               = season,
            num_samples          = len(analyses),
            mean_class_fractions = mean_fracs,
            std_class_fractions  = std_fracs,
            sample_ids           = tuple(a.sample_id for a in analyses),
        )
