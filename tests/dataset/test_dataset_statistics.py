"""
Unit tests for src/dataset/statistics.py.

Run:
    pytest tests/dataset/test_dataset_statistics.py -v \
        --cov=src/dataset/statistics --cov-report=term-missing
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.dataset.manifest import DatasetSample
from src.dataset.statistics import ClassStatistics, DatasetStatisticsCalculator, SplitStatistics
from src.labels.schema import ClassDefinition, ClassSchema


def _schema() -> ClassSchema:
    return ClassSchema(classes=(
        ClassDefinition(0, "background", (128, 128, 128)),
        ClassDefinition(1, "water",      (0, 119, 190)),
        ClassDefinition(2, "sand",       (255, 200, 87)),
        ClassDefinition(3, "vegetation", (34, 139, 34)),
    ))


def _sample(
    patch_id: str = "p1",
    mask_path: str = "/data/mask.tif",
    season: str = "monsoon",
    year: int = 2023,
) -> DatasetSample:
    return DatasetSample(
        sample_id=patch_id, patch_id=patch_id, scene_id="scene001",
        patch_path=f"/data/{patch_id}.tif", mask_path=mask_path,
        crs="EPSG:4326", width=4, height=4, num_bands=4, row_index=0, col_index=0,
        patch_valid_pixel_ratio=1.0, label_valid_pixel_ratio=1.0,
        num_classes_present=3, acquisition_date="2023-07-15",
        year=year, month=7, season=season, hydrological_year=year,
        sensor="L8", river_name="", reach_id="", basin_id="",
        aoi_id="aoi_1", label_version="1.0.0", annotator="x",
        confidence=1.0, confidence_source="automatic",
    )


class TestSplitStatistics:
    def test_frozen(self) -> None:
        stats = SplitStatistics(
            split_name="train", sample_count=0, class_statistics=(),
            class_imbalance_ratio=1.0, water_sand_ratio=None,
            vegetation_sand_ratio=None, bare_sediment_fraction=None,
            seasonal_distribution=(), yearly_distribution=(), total_valid_pixels=0,
        )
        with pytest.raises((AttributeError, TypeError)):
            stats.sample_count = 99  # type: ignore[misc]

    def test_to_dict_json_serializable(self) -> None:
        import json
        stats = SplitStatistics(
            split_name="train", sample_count=5, class_statistics=(),
            class_imbalance_ratio=1.0, water_sand_ratio=None,
            vegetation_sand_ratio=None, bare_sediment_fraction=None,
            seasonal_distribution=(), yearly_distribution=(), total_valid_pixels=100,
        )
        raw = json.dumps(stats.to_dict(), ensure_ascii=True)
        assert len(raw) > 0


class TestDatasetStatisticsCalculator:
    def test_compute_without_masks(self) -> None:
        """Metadata-only mode (read_masks=False): pixel counts are zero."""
        calc = DatasetStatisticsCalculator(
            class_schema=_schema(), nodata_value=255,
        )
        samples = [_sample("p1", season="monsoon"), _sample("p2", season="winter")]
        stats   = calc.compute(samples, "train", read_masks=False)
        assert stats.sample_count == 2
        assert stats.total_valid_pixels == 0
        seasons = {s.season: s.count for s in stats.seasonal_distribution}
        assert seasons["monsoon"] == 1
        assert seasons["winter"]  == 1

    def test_yearly_distribution(self) -> None:
        calc    = DatasetStatisticsCalculator(_schema(), nodata_value=255)
        samples = [_sample("p1", year=2022), _sample("p2", year=2023), _sample("p3", year=2022)]
        stats   = calc.compute(samples, "train", read_masks=False)
        years   = {y.year: y.count for y in stats.yearly_distribution}
        assert years[2022] == 2
        assert years[2023] == 1

    def test_imbalance_one_when_no_pixels(self) -> None:
        calc  = DatasetStatisticsCalculator(_schema(), nodata_value=255)
        stats = calc.compute([], "train", read_masks=False)
        assert stats.class_imbalance_ratio == pytest.approx(1.0)

    def test_water_sand_ratio_computed(self, tmp_path: Path) -> None:
        """Write a real mask file and verify ratio computation."""
        import rasterio
        from affine import Affine
        from rasterio.crs import CRS

        mask_path = tmp_path / "mask.tif"
        values = np.array([[1, 1, 1, 2]], dtype=np.uint8)
        profile = {
            "driver": "GTiff", "dtype": "uint8", "width": 4, "height": 1,
            "count": 1, "crs": CRS.from_string("EPSG:4326"),
            "transform": Affine(0.001, 0, 87, 0, -0.001, 26.5),
        }
        with rasterio.open(mask_path, "w", **profile) as ds:
            ds.write(values, 1)

        calc = DatasetStatisticsCalculator(
            _schema(), nodata_value=255,
            ratio_definitions={"water_sand_ratio": ("water", "sand")},
        )
        sample = _sample("p1", mask_path=str(mask_path))
        stats  = calc.compute([sample], "train", read_masks=True)
        assert stats.water_sand_ratio == pytest.approx(3.0)  # 3 water / 1 sand

    def test_save_statistics_creates_json(self, tmp_path: Path) -> None:
        calc  = DatasetStatisticsCalculator(_schema(), nodata_value=255)
        stats = calc.compute([], "train", read_masks=False)
        path  = calc.save_statistics({"train": stats}, tmp_path)
        assert path.exists()

    def test_summary_lines_ascii(self) -> None:
        calc  = DatasetStatisticsCalculator(_schema(), nodata_value=255)
        stats = calc.compute([_sample("p1")], "train", read_masks=False)
        for line in stats.class_statistics:
            assert all(ord(c) < 128 for c in line.class_name)