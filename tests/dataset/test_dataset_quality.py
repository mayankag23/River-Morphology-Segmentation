"""
Unit tests for src/dataset/quality.py.

Run:
    pytest tests/dataset/test_dataset_quality.py -v \
        --cov=src/dataset/quality --cov-report=term-missing
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.dataset.leakage import LeakageDetectionResult
from src.dataset.quality import DatasetQualityAnalyzer, QualityReport
from src.dataset.statistics import SplitStatistics
from src.dataset.validator import DatasetValidationResult


def _validation_result(is_valid: bool = True, n_valid: int = 10) -> DatasetValidationResult:
    return DatasetValidationResult(
        is_valid=is_valid, total_samples=n_valid, valid_samples=n_valid,
        invalid_samples=0, issues=(), duplicate_sample_ids=(),
        missing_patch_files=(), missing_mask_files=(),
        crs_values_found=("EPSG:4326",), crs_is_consistent=True,
        below_min_pixel_ratio_count=0, min_total_samples_met=True,
    )


def _leakage_result(has_leakage: bool = False) -> LeakageDetectionResult:
    return LeakageDetectionResult(
        has_leakage=has_leakage, total_samples_checked=10,
        total_scenes_checked=3, patch_violations=(),
        scene_violations=(), violation_records=(),
    )


def _split_stats(
    split_name: str = "train",
    n: int = 7,
    imbalance: float = 1.0,
) -> SplitStatistics:
    return SplitStatistics(
        split_name=split_name, sample_count=n, class_statistics=(),
        class_imbalance_ratio=imbalance, water_sand_ratio=None,
        vegetation_sand_ratio=None, bare_sediment_fraction=None,
        seasonal_distribution=(), yearly_distribution=(), total_valid_pixels=n * 256 * 256,
    )


class TestDatasetQualityAnalyzer:
    def test_good_dataset_passes(self) -> None:
        analyzer = DatasetQualityAnalyzer(min_samples_per_split=3)
        report   = analyzer.analyze(
            _validation_result(), _leakage_result(),
            _split_stats("train", 7), _split_stats("validation", 2),
            _split_stats("test", 1),
        )
        assert report.is_suitable_for_training is True
        assert report.overall_quality_score > 0.5

    def test_leakage_produces_error(self) -> None:
        analyzer = DatasetQualityAnalyzer()
        report   = analyzer.analyze(
            _validation_result(), _leakage_result(has_leakage=True),
            _split_stats(), _split_stats("validation"), _split_stats("test"),
        )
        assert report.has_leakage is True
        assert report.is_suitable_for_training is False
        assert any(i.severity == "ERROR" for i in report.issues)

    def test_validation_failure_produces_error(self) -> None:
        from src.dataset.validator import ValidationIssue
        bad_validation = DatasetValidationResult(
            is_valid=False, total_samples=5, valid_samples=0, invalid_samples=5,
            issues=(ValidationIssue("", "inconsistent_crs", "CRS mismatch"),),
            duplicate_sample_ids=(), missing_patch_files=(), missing_mask_files=(),
            crs_values_found=("EPSG:4326", "EPSG:32644"), crs_is_consistent=False,
            below_min_pixel_ratio_count=0, min_total_samples_met=False,
        )
        analyzer = DatasetQualityAnalyzer()
        report   = analyzer.analyze(
            bad_validation, _leakage_result(),
            _split_stats(), _split_stats("validation"), _split_stats("test"),
        )
        assert report.is_suitable_for_training is False

    def test_high_imbalance_produces_warning(self) -> None:
        analyzer = DatasetQualityAnalyzer()
        report   = analyzer.analyze(
            _validation_result(), _leakage_result(),
            _split_stats("train", imbalance=50.0),
            _split_stats("validation"), _split_stats("test"),
        )
        assert any(i.severity == "WARN" and "balance" in i.category for i in report.issues)

    def test_small_split_produces_warning(self) -> None:
        analyzer = DatasetQualityAnalyzer(min_samples_per_split=5)
        report   = analyzer.analyze(
            _validation_result(), _leakage_result(),
            _split_stats("train", n=10), _split_stats("validation", n=2),
            _split_stats("test", n=1),
        )
        assert any("split_size" in i.category for i in report.issues)

    def test_quality_score_in_0_1(self) -> None:
        analyzer = DatasetQualityAnalyzer()
        report   = analyzer.analyze(
            _validation_result(), _leakage_result(),
            _split_stats(), _split_stats("validation"), _split_stats("test"),
        )
        assert 0.0 <= report.overall_quality_score <= 1.0

    def test_report_is_frozen(self) -> None:
        analyzer = DatasetQualityAnalyzer()
        report   = analyzer.analyze(
            _validation_result(), _leakage_result(),
            _split_stats(), _split_stats("validation"), _split_stats("test"),
        )
        with pytest.raises((AttributeError, TypeError)):
            report.has_leakage = True  # type: ignore[misc]

    def test_save_report_creates_json(self, tmp_path: Path) -> None:
        analyzer = DatasetQualityAnalyzer()
        report   = analyzer.analyze(
            _validation_result(), _leakage_result(),
            _split_stats(), _split_stats("validation"), _split_stats("test"),
        )
        path = analyzer.save_report(report, tmp_path)
        assert path.exists()

    def test_summary_lines_ascii(self) -> None:
        analyzer = DatasetQualityAnalyzer()
        report   = analyzer.analyze(
            _validation_result(), _leakage_result(),
            _split_stats(), _split_stats("validation"), _split_stats("test"),
        )
        for line in report.summary_lines():
            assert all(ord(c) < 128 for c in line)