"""
Dataset quality analysis for the Dataset Assembly pipeline (Module 10).

DatasetQualityAnalyzer synthesizes findings from DatasetValidator,
DataLeakageDetector, and SplitStatistics into a human-readable quality
report and a machine-readable quality score. The report identifies
whether the dataset is suitable for training and provides actionable
recommendations.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.dataset.leakage import LeakageDetectionResult
from src.dataset.statistics import SplitStatistics
from src.dataset.validator import DatasetValidationResult

__all__ = ["QualityIssue", "QualityReport", "DatasetQualityAnalyzer"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QualityIssue:
    """
    Immutable descriptor for one dataset quality problem.

    Attributes:
        severity:     "ERROR", "WARN", or "INFO".
        category:      Issue category, e.g. "class_balance", "split_size".
        description:   Human-readable description.
    """

    severity:    str
    category:     str
    description:  str


@dataclass(frozen=True)
class QualityReport:
    """
    Immutable quality assessment of the assembled dataset.

    Attributes:
        overall_quality_score:     Float in [0.0, 1.0]. Heuristic score
                                   combining valid sample fraction, class
                                   balance, leakage absence, and split coverage.
        is_suitable_for_training:   True if no ERROR-severity issues exist.
        total_samples:              Total samples before exclusions.
        valid_samples:               Samples passing all QC checks.
        excluded_samples:             Samples excluded from training.
        has_leakage:                 True if DataLeakageDetector found violations.
        issues:                      Ordered tuple of QualityIssue records.
        recommendations:              Actionable recommendations for improvement.
    """

    overall_quality_score:    float
    is_suitable_for_training:  bool
    total_samples:             int
    valid_samples:              int
    excluded_samples:           int
    has_leakage:               bool
    issues:                    tuple[QualityIssue, ...]
    recommendations:            tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable plain dict."""
        return asdict(self)

    def summary_lines(self) -> list[str]:
        """Return ASCII-formatted summary lines."""
        status = "[PASS]" if self.is_suitable_for_training else "[FAIL]"
        return [
            f"  {status} overall_quality_score: {self.overall_quality_score:.2f}",
            f"         valid_samples:        {self.valid_samples}/{self.total_samples}",
            f"         has_leakage:          {self.has_leakage}",
            f"         issues:               {len(self.issues)} total",
        ]


class DatasetQualityAnalyzer:
    """
    Generates a QualityReport from validation, leakage, and statistics results.

    Args:
        min_samples_per_split: Minimum samples required in each non-empty split.
    """

    def __init__(self, min_samples_per_split: int = 5) -> None:
        self._min_samples_per_split = int(min_samples_per_split)
        self._logger: logging.Logger = logging.getLogger(__name__)

    def analyze(
        self,
        validation_result:    DatasetValidationResult,
        leakage_result:        LeakageDetectionResult,
        train_statistics:      SplitStatistics,
        validation_statistics:  SplitStatistics,
        test_statistics:        SplitStatistics,
    ) -> QualityReport:
        """
        Analyze dataset quality and return a QualityReport.

        Args:
            validation_result:     From DatasetValidator.validate().
            leakage_result:         From DataLeakageDetector.detect().
            train_statistics:       From DatasetStatisticsCalculator.compute("train").
            validation_statistics:   From DatasetStatisticsCalculator.compute("validation").
            test_statistics:         From DatasetStatisticsCalculator.compute("test").

        Returns:
            Immutable QualityReport.
        """
        issues: list[QualityIssue] = []
        recommendations: list[str] = []

        # Validation errors
        for vi in validation_result.issues:
            severity = "ERROR" if not validation_result.is_valid else "WARN"
            issues.append(QualityIssue(
                severity=severity,
                category="validation",
                description=vi.description,
            ))

        # CRS consistency
        if not validation_result.crs_is_consistent:
            issues.append(QualityIssue(
                severity="ERROR",
                category="crs",
                description="Multiple CRS values detected across the dataset",
            ))
            recommendations.append("Ensure all exported scenes use the same CRS.")

        # Leakage
        if leakage_result.has_leakage:
            issues.append(QualityIssue(
                severity="ERROR",
                category="leakage",
                description=(
                    f"Data leakage: {len(leakage_result.patch_violations)} patch "
                    f"and {len(leakage_result.scene_violations)} scene violations"
                ),
            ))
            recommendations.append(
                "Use scene-level splitting to prevent train/val/test leakage."
            )

        # Minimum samples per split
        for split_stats, split_name in (
            (train_statistics, "train"),
            (validation_statistics, "validation"),
            (test_statistics, "test"),
        ):
            if 0 < split_stats.sample_count < self._min_samples_per_split:
                issues.append(QualityIssue(
                    severity="WARN",
                    category="split_size",
                    description=(
                        f"{split_name} split has only {split_stats.sample_count} "
                        f"samples (minimum recommended: {self._min_samples_per_split})"
                    ),
                ))
                recommendations.append(
                    f"Increase the {split_name} split size or add more scenes."
                )

        # Class imbalance
        for split_stats, split_name in (
            (train_statistics, "train"),
            (validation_statistics, "validation"),
        ):
            if split_stats.class_imbalance_ratio > 10.0:
                issues.append(QualityIssue(
                    severity="WARN",
                    category="class_balance",
                    description=(
                        f"{split_name} split has severe class imbalance: "
                        f"{split_stats.class_imbalance_ratio:.1f}x"
                    ),
                ))
                recommendations.append(
                    f"Consider class-weighted loss or oversampling to address "
                    f"imbalance in the {split_name} split."
                )

        # Compute quality score (heuristic)
        quality_score = self._compute_quality_score(
            validation_result, leakage_result, train_statistics, issues
        )
        has_errors = any(i.severity == "ERROR" for i in issues)

        self._logger.info(
            "Quality analysis complete. score=%.2f, suitable=%s, issues=%d",
            quality_score, not has_errors, len(issues),
        )

        return QualityReport(
            overall_quality_score=quality_score,
            is_suitable_for_training=not has_errors,
            total_samples=validation_result.total_samples,
            valid_samples=validation_result.valid_samples,
            excluded_samples=validation_result.invalid_samples,
            has_leakage=leakage_result.has_leakage,
            issues=tuple(issues),
            recommendations=tuple(sorted(set(recommendations))),
        )

    def save_report(self, report: QualityReport, output_dir: Path) -> Path:
        """
        Write QualityReport to quality_report.json in output_dir.

        Returns:
            Absolute path to the written file.
        """
        output_dir = Path(output_dir).resolve()
        path       = output_dir / "quality_report.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(report.to_dict(), fh, indent=2, ensure_ascii=True)
        self._logger.info("Quality report written: %s", path.name)
        return path

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_quality_score(
        validation_result: DatasetValidationResult,
        leakage_result:    LeakageDetectionResult,
        train_statistics:  SplitStatistics,
        issues:            list[QualityIssue],
    ) -> float:
        """
        Heuristic quality score in [0.0, 1.0].

        Components:
            0.4 -- valid sample fraction
            0.3 -- no leakage (0.0 if has_leakage, 0.3 otherwise)
            0.2 -- class balance (decays with imbalance ratio)
            0.1 -- no errors
        """
        total = validation_result.total_samples
        valid_fraction = (
            validation_result.valid_samples / total if total > 0 else 0.0
        )
        leakage_score  = 0.0 if leakage_result.has_leakage else 0.3
        imbalance      = train_statistics.class_imbalance_ratio
        balance_score  = 0.2 * max(0.0, 1.0 - (imbalance - 1.0) / 20.0)
        error_count    = sum(1 for i in issues if i.severity == "ERROR")
        error_score    = 0.1 if error_count == 0 else 0.0

        return round(
            min(1.0, 0.4 * valid_fraction + leakage_score + balance_score + error_score),
            4,
        )