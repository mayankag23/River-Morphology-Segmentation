"""
Unit tests for src/dataset/leakage.py.

Run:
    pytest tests/dataset/test_data_leakage.py -v \
        --cov=src/dataset/leakage --cov-report=term-missing
"""

from __future__ import annotations

import pytest

from src.dataset.leakage import DataLeakageDetector
from src.dataset.manifest import DatasetSample


def _sample(patch_id: str, scene_id: str = "scene001") -> DatasetSample:
    return DatasetSample(
        sample_id=patch_id, patch_id=patch_id, scene_id=scene_id,
        patch_path=f"/data/{patch_id}.tif", mask_path=f"/data/{patch_id}_mask.tif",
        crs="EPSG:4326", width=8, height=8, num_bands=4, row_index=0, col_index=0,
        patch_valid_pixel_ratio=1.0, label_valid_pixel_ratio=1.0,
        num_classes_present=2, acquisition_date="2023-07-15",
        year=2023, month=7, season="monsoon", hydrological_year=2023,
        sensor="L8", river_name="", reach_id="", basin_id="",
        aoi_id="aoi_1", label_version="1.0.0", annotator="x",
        confidence=1.0, confidence_source="automatic",
    )


@pytest.fixture
def detector() -> DataLeakageDetector:
    return DataLeakageDetector()


class TestDataLeakageDetector:
    def test_no_leakage_passes(self, detector: DataLeakageDetector) -> None:
        train = [_sample("t1", "scene1"), _sample("t2", "scene1")]
        val   = [_sample("v1", "scene2")]
        test  = [_sample("e1", "scene3")]
        result = detector.detect(train, val, test)
        assert result.has_leakage is False
        assert len(result.patch_violations) == 0
        assert len(result.scene_violations) == 0

    def test_patch_in_two_splits_detected(self, detector: DataLeakageDetector) -> None:
        train = [_sample("shared", "scene1")]
        val   = [_sample("shared", "scene1")]
        test  = [_sample("unique", "scene2")]
        result = detector.detect(train, val, test)
        assert result.has_leakage is True
        assert "shared" in result.patch_violations

    def test_scene_in_two_splits_detected(self, detector: DataLeakageDetector) -> None:
        train = [_sample("t1", "scene1")]
        val   = [_sample("v1", "scene1")]  # same scene as train
        test  = [_sample("e1", "scene2")]
        result = detector.detect(train, val, test)
        assert result.has_leakage is True
        assert "scene1" in result.scene_violations

    def test_total_samples_counted(self, detector: DataLeakageDetector) -> None:
        train = [_sample("t1"), _sample("t2")]
        val   = [_sample("v1")]
        test  = [_sample("e1")]
        result = detector.detect(train, val, test)
        assert result.total_samples_checked == 4

    def test_result_is_frozen(self, detector: DataLeakageDetector) -> None:
        result = detector.detect([], [], [])
        with pytest.raises((AttributeError, TypeError)):
            result.has_leakage = True  # type: ignore[misc]

    def test_empty_splits_no_leakage(self, detector: DataLeakageDetector) -> None:
        result = detector.detect([], [], [])
        assert result.has_leakage is False
        assert result.total_samples_checked == 0

    def test_no_leakage_message_logged(
        self, detector: DataLeakageDetector, caplog
    ) -> None:
        import logging
        with caplog.at_level(logging.INFO, logger="src.dataset.leakage"):
            detector.detect(
                [_sample("t1", "sc1")],
                [_sample("v1", "sc2")],
                [_sample("e1", "sc3")],
            )
        assert any("Leakage check passed" in r.message for r in caplog.records)

    def test_multiple_violations_all_recorded(self, detector: DataLeakageDetector) -> None:
        train = [_sample("a", "scene1"), _sample("b", "scene2")]
        val   = [_sample("a", "scene1"), _sample("b", "scene2")]  # both leak
        test  = [_sample("c", "scene3")]
        result = detector.detect(train, val, test)
        assert result.has_leakage is True
        assert len(result.patch_violations) >= 2