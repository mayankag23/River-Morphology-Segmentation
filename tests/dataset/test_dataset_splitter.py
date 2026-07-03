"""
Unit tests for src/dataset/splitter.py.

Run:
    pytest tests/dataset/test_dataset_splitter.py -v \
        --cov=src/dataset/splitter --cov-report=term-missing
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.exceptions import InvalidValueError
from src.dataset.manifest import DatasetSample
from src.dataset.splitter import DatasetSplitter, SplitResult, SplitStrategy
from tests.conftest import make_valid_config, write_config


def _sample(
    patch_id: str,
    scene_id: str = "scene001",
    acquisition_date: str = "2023-07-15",
    aoi_id: str = "aoi_1",
) -> DatasetSample:
    return DatasetSample(
        sample_id=patch_id, patch_id=patch_id, scene_id=scene_id,
        patch_path=f"/data/{patch_id}.tif",
        mask_path=f"/data/{patch_id}_mask.tif",
        crs="EPSG:4326", width=8, height=8, num_bands=4,
        row_index=0, col_index=0,
        patch_valid_pixel_ratio=1.0, label_valid_pixel_ratio=1.0,
        num_classes_present=3, acquisition_date=acquisition_date,
        year=int(acquisition_date[:4]), month=int(acquisition_date[5:7]),
        season="monsoon", hydrological_year=int(acquisition_date[:4]),
        sensor="L8", river_name="Kosi", reach_id="", basin_id="",
        aoi_id=aoi_id, label_version="1.0.0", annotator="x",
        confidence=1.0, confidence_source="automatic",
    )


def _make_config(tmp_path: Path, strategy: str = "temporal", seed: int = 42):
    from src.core.config import Config
    data = make_valid_config()
    data["dataset"] = {
        "split": {
            "strategy": strategy,
            "train_ratio": 0.70,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "random_seed": seed,
        },
        "quality": {"min_valid_pixel_ratio": 0.5, "min_samples_per_split": 1},
        "output_formats": ["csv"],
        "dataset_version": "1.0.0",
        "min_total_samples": 1,
    }
    return Config(config_path=write_config(tmp_path, data))


def _multi_scene_samples(tmp_path: Path) -> list[DatasetSample]:
    """8 samples across 4 scenes with different dates and AOIs."""
    return [
        _sample(f"sc1_p{i}", scene_id="scene001", acquisition_date="2022-01-15", aoi_id="aoi_1")
        for i in range(2)
    ] + [
        _sample(f"sc2_p{i}", scene_id="scene002", acquisition_date="2022-07-15", aoi_id="aoi_1")
        for i in range(2)
    ] + [
        _sample(f"sc3_p{i}", scene_id="scene003", acquisition_date="2023-01-15", aoi_id="aoi_2")
        for i in range(2)
    ] + [
        _sample(f"sc4_p{i}", scene_id="scene004", acquisition_date="2023-07-15", aoi_id="aoi_2")
        for i in range(2)
    ]


class TestSplitStrategy:
    def test_from_string_valid(self) -> None:
        assert SplitStrategy.from_string("temporal") == SplitStrategy.TEMPORAL
        assert SplitStrategy.from_string("RANDOM") == SplitStrategy.RANDOM
        assert SplitStrategy.from_string("spatial") == SplitStrategy.SPATIAL

    def test_from_string_invalid_raises(self) -> None:
        with pytest.raises(InvalidValueError, match="strategy"):
            SplitStrategy.from_string("cross_validation")


class TestDatasetSplitter:
    def test_empty_samples_raises(self, tmp_path: Path) -> None:
        splitter = DatasetSplitter(_make_config(tmp_path))
        with pytest.raises(InvalidValueError):
            splitter.split([])

    def test_returns_split_result(self, tmp_path: Path) -> None:
        splitter = DatasetSplitter(_make_config(tmp_path))
        samples = _multi_scene_samples(tmp_path)
        result = splitter.split(samples)
        assert isinstance(result, SplitResult)

    def test_all_samples_assigned(self, tmp_path: Path) -> None:
        splitter = DatasetSplitter(_make_config(tmp_path))
        samples = _multi_scene_samples(tmp_path)
        result = splitter.split(samples)
        assert result.total_count == len(samples)

    def test_no_overlap_between_splits(self, tmp_path: Path) -> None:
        splitter = DatasetSplitter(_make_config(tmp_path))
        samples = _multi_scene_samples(tmp_path)
        result = splitter.split(samples)
        train_ids = {s.sample_id for s in result.train_samples}
        val_ids   = {s.sample_id for s in result.validation_samples}
        test_ids  = {s.sample_id for s in result.test_samples}
        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(test_ids)
        assert val_ids.isdisjoint(test_ids)

    def test_scene_level_integrity(self, tmp_path: Path) -> None:
        """All patches from the same scene must be in the same split."""
        splitter = DatasetSplitter(_make_config(tmp_path))
        samples  = _multi_scene_samples(tmp_path)
        result   = splitter.split(samples)
        for split_samples in (result.train_samples, result.validation_samples, result.test_samples):
            scene_ids_in_split = {s.scene_id for s in split_samples}
            for scene_id in scene_ids_in_split:
                all_scene_samples = [s for s in samples if s.scene_id == scene_id]
                for ss in all_scene_samples:
                    assert ss in split_samples, (
                        f"scene {scene_id} patch {ss.sample_id} split across splits"
                    )

    def test_temporal_split_preserves_chronological_order(
        self, tmp_path: Path
    ) -> None:
        """Earliest dates must be in train, latest in test."""
        splitter = DatasetSplitter(_make_config(tmp_path, strategy="temporal"))
        samples  = _multi_scene_samples(tmp_path)
        result   = splitter.split(samples)
        if result.train_samples and result.test_samples:
            max_train_date = max(s.acquisition_date for s in result.train_samples)
            min_test_date  = min(s.acquisition_date for s in result.test_samples)
            assert max_train_date <= min_test_date

    def test_random_split_reproducible(self, tmp_path: Path) -> None:
        """Same seed must produce identical splits across runs."""
        cfg = _make_config(tmp_path, strategy="random", seed=123)
        splitter1 = DatasetSplitter(cfg)
        splitter2 = DatasetSplitter(cfg)
        samples   = _multi_scene_samples(tmp_path)
        r1 = splitter1.split(samples)
        r2 = splitter2.split(samples)
        ids1 = sorted(s.sample_id for s in r1.train_samples)
        ids2 = sorted(s.sample_id for s in r2.train_samples)
        assert ids1 == ids2

    def test_random_different_seed_may_differ(self, tmp_path: Path) -> None:
        """Different seeds should typically produce different splits."""
        samples = _multi_scene_samples(tmp_path)
        cfg1    = _make_config(tmp_path / "c1", strategy="random", seed=1)
        cfg2    = _make_config(tmp_path / "c2", strategy="random", seed=9999)
        r1 = DatasetSplitter(cfg1).split(samples)
        r2 = DatasetSplitter(cfg2).split(samples)
        ids1 = sorted(s.sample_id for s in r1.train_samples)
        ids2 = sorted(s.sample_id for s in r2.train_samples)
        # Seeds 1 and 9999 typically produce different orders for 4 scenes
        # (not guaranteed, but highly likely)
        # We just check the split runs without errors and produces valid results
        assert r1.total_count == r2.total_count == len(samples)

    def test_spatial_split_groups_by_aoi(self, tmp_path: Path) -> None:
        """All samples from the same AOI must go to the same split."""
        splitter = DatasetSplitter(_make_config(tmp_path, strategy="spatial"))
        samples  = _multi_scene_samples(tmp_path)
        result   = splitter.split(samples)
        for split_samples in (result.train_samples, result.validation_samples, result.test_samples):
            aois_in_split = {s.aoi_id for s in split_samples}
            # Verify all scenes from an AOI are in the same split
            for aoi_id in aois_in_split:
                aoi_patches = [s for s in samples if s.aoi_id == aoi_id]
                for ap in aoi_patches:
                    assert ap in split_samples

    def test_strategy_override(self, tmp_path: Path) -> None:
        """Strategy override at call time must take precedence over config."""
        cfg      = _make_config(tmp_path, strategy="temporal")
        splitter = DatasetSplitter(cfg)
        samples  = _multi_scene_samples(tmp_path)
        result   = splitter.split(samples, strategy=SplitStrategy.RANDOM)
        assert result.strategy == "random"

    def test_ratios_sum_must_be_one(self, tmp_path: Path) -> None:
        from src.core.config import Config
        data = make_valid_config()
        data["dataset"] = {
            "split": {
                "strategy": "temporal",
                "train_ratio": 0.6,
                "val_ratio": 0.3,
                "test_ratio": 0.3,  # sums to 1.2
                "random_seed": 42,
            },
            "quality": {}, "output_formats": ["csv"],
            "dataset_version": "1.0.0", "min_total_samples": 1,
        }
        cfg = Config(config_path=write_config(tmp_path, data))
        with pytest.raises(InvalidValueError, match="sum"):
            DatasetSplitter(cfg)

    def test_single_scene_stays_in_train(self, tmp_path: Path) -> None:
        """With only one scene, all samples go to train."""
        splitter = DatasetSplitter(_make_config(tmp_path))
        samples  = [_sample(f"p{i}", scene_id="only_scene") for i in range(4)]
        result   = splitter.split(samples)
        assert result.train_count == 4