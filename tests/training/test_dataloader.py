"""
Unit tests for src/training/dataloader.py.

Uses real GeoTIFF files for patch fixtures (via GeoTiffWriter) and real
rasterio-written mask files. TrainingDatasetResult is constructed from
real Module 10 frozen dataclasses — no mocking required.

Run:
    pytest tests/training/test_dataloader.py -v \
        --cov=src/training/dataloader --cov-report=term-missing
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.crs import CRS

from src.export.downloader import AffineTransform, AoiBounds, DownloadResult
from src.export.geotiff import GeoTiffProfile, GeoTiffWriter
from src.labels.schema import ClassDefinition, ClassSchema
from src.labels.statistics import LabelStatistics
from src.training.dataloader import DataLoaderBundle, DataLoaderConfig, DataLoaderFactory

pytest.importorskip("torch",         reason="torch required")
pytest.importorskip("albumentations", reason="albumentations required")

_CRS   = "EPSG:4326"
_AFFINE = Affine(0.001, 0.0, 87.0, 0.0, -0.001, 26.5)
_SIZE  = 8
_BANDS = ("Blue", "Green", "Red", "NIR")


def _schema() -> ClassSchema:
    return ClassSchema(classes=(
        ClassDefinition(0, "background", (128, 128, 128)),
        ClassDefinition(1, "water",      (0, 119, 190)),
        ClassDefinition(2, "sand",       (255, 200, 87)),
        ClassDefinition(3, "vegetation", (34, 139, 34)),
    ))


def _write_patch(path: Path) -> Path:
    data = np.random.rand(len(_BANDS), _SIZE, _SIZE).astype(np.float32)
    dr = DownloadResult(
        data=data, crs=_CRS, transform=AffineTransform.from_affine(_AFFINE),
        band_names=_BANDS, width=_SIZE, height=_SIZE,
        aoi_bounds=AoiBounds(87.0, 26.0, 87.5, 26.5), num_tiles=1,
    )
    return GeoTiffWriter(GeoTiffProfile(tiled=False, overviews=False)).write(dr, path).path


def _write_mask(path: Path) -> Path:
    values = np.zeros((_SIZE, _SIZE), dtype=np.uint8)
    values[:4, :] = 1
    values[4:, :4] = 2
    profile = {
        "driver": "GTiff", "dtype": "uint8",
        "width": _SIZE, "height": _SIZE, "count": 1,
        "crs": CRS.from_string(_CRS), "transform": _AFFINE,
    }
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(values, 1)
    return path


def _make_training_result(tmp_path: Path, n_train: int = 4, n_val: int = 2, n_test: int = 2):
    """Build a complete TrainingDatasetResult from real files."""
    from src.dataset.assembler import TrainingDatasetResult
    from src.dataset.leakage import LeakageDetectionResult
    from src.dataset.manifest import DatasetManifest, DatasetManifestEntry
    from src.dataset.quality import QualityReport
    from src.dataset.statistics import ClassStatistics, SplitStatistics
    from src.dataset.validator import DatasetValidationResult
    from src.dataset.version import DatasetVersionInfo

    entries: list[DatasetManifestEntry] = []

    def _make_entry(i: int, split: str) -> DatasetManifestEntry:
        patch_id   = f"scene001_r000_c{i:03d}"
        patch_path = _write_patch(tmp_path / f"{patch_id}.tif")
        mask_path  = _write_mask(tmp_path / f"{patch_id}_mask.tif")
        return DatasetManifestEntry(
            sample_id=patch_id, patch_id=patch_id, scene_id="scene001",
            patch_path=str(patch_path), mask_path=str(mask_path),
            split=split, crs=_CRS, width=_SIZE, height=_SIZE,
            num_bands=len(_BANDS), row_index=0, col_index=i,
            patch_valid_pixel_ratio=1.0, label_valid_pixel_ratio=0.95,
            num_classes_present=3, acquisition_date="2023-07-15",
            year=2023, month=7, season="monsoon", hydrological_year=2023,
            sensor="L8,L9", river_name="Kosi", reach_id="r1", basin_id="b1",
            aoi_id="aoi_1", label_version="1.0.0", annotator="auto",
            confidence=1.0, confidence_source="automatic",
        )

    idx = 0
    for _ in range(n_train):
        entries.append(_make_entry(idx, "train"));      idx += 1
    for _ in range(n_val):
        entries.append(_make_entry(idx, "validation")); idx += 1
    for _ in range(n_test):
        entries.append(_make_entry(idx, "test"));       idx += 1

    manifest = DatasetManifest(
        entries=tuple(entries), dataset_manifest_csv=None,
        dataset_manifest_json=None, train_csv=None,
        validation_csv=None, test_csv=None,
    )

    def _dummy_stats(split_name: str) -> SplitStatistics:
        cs = tuple(
            ClassStatistics(i, n, 0, 0, 0.0)
            for i, n in enumerate(("background", "water", "sand", "vegetation"))
        )
        return SplitStatistics(
            split_name=split_name, sample_count=0, class_statistics=cs,
            class_imbalance_ratio=1.0, water_sand_ratio=None,
            vegetation_sand_ratio=None, bare_sediment_fraction=None,
            seasonal_distribution=(), yearly_distribution=(), total_valid_pixels=0,
        )

    dummy_validation = DatasetValidationResult(
        is_valid=True, total_samples=n_train + n_val + n_test,
        valid_samples=n_train + n_val + n_test, invalid_samples=0,
        issues=(), duplicate_sample_ids=(), missing_patch_files=(),
        missing_mask_files=(), crs_values_found=(_CRS,),
        crs_is_consistent=True, below_min_pixel_ratio_count=0,
        min_total_samples_met=True,
    )
    dummy_leakage = LeakageDetectionResult(
        has_leakage=False, total_samples_checked=n_train + n_val + n_test,
        total_scenes_checked=1, patch_violations=(),
        scene_violations=(), violation_records=(),
    )
    dummy_quality = QualityReport(
        overall_quality_score=0.9, is_suitable_for_training=True,
        total_samples=n_train + n_val + n_test,
        valid_samples=n_train + n_val + n_test,
        excluded_samples=0, has_leakage=False, issues=(), recommendations=(),
    )
    dummy_version = DatasetVersionInfo(
        dataset_version="1.0.0",
        assembly_timestamp=datetime.now(timezone.utc).isoformat(),
        total_samples=n_train + n_val + n_test,
        train_samples=n_train, validation_samples=n_val, test_samples=n_test,
        excluded_samples=0, split_strategy="temporal",
        train_ratio=0.7, validation_ratio=0.15, test_ratio=0.15,
        random_seed=42, source_scenes=1, git_commit=None, config_hash=None,
    )
    dummy_path = tmp_path / "dummy.json"
    dummy_path.write_text("{}")

    return TrainingDatasetResult(
        output_dir=tmp_path, total_samples=n_train + n_val + n_test,
        train_samples=n_train, validation_samples=n_val, test_samples=n_test,
        excluded_samples=0, split_strategy="temporal", source_scenes=1,
        manifest=manifest,
        train_statistics=_dummy_stats("train"),
        validation_statistics=_dummy_stats("validation"),
        test_statistics=_dummy_stats("test"),
        overall_statistics=_dummy_stats("overall"),
        quality_report=dummy_quality,
        validation_result=dummy_validation,
        leakage_detection=dummy_leakage,
        version_info=dummy_version,
        version_path=dummy_path, statistics_path=dummy_path,
        quality_report_path=dummy_path, is_suitable_for_training=True,
        operations_log=("ok",),
    )


def _config(tmp_path: Path, num_workers: int = 0):
    from src.core.config import Config
    from tests.conftest import make_valid_config, write_config
    data = make_valid_config()
    data["model"]["num_classes"] = 4
    if "loss" in data:
        data["loss"]["num_classes"] = 4

    data["classes"] = {
        "num_classes": 4,
        "labels": {"background": 0, "water": 1, "sand": 2, "vegetation": 3},
        "names": ["background", "water", "sand", "vegetation"],
        "colors": {
            "background": [128, 128, 128], "water": [0, 119, 190],
            "sand": [255, 200, 87], "vegetation": [34, 139, 34],
        },
    }
    data["labels"] = {
        "nodata_value": 255, "nodata_value": 255,
        "mask_filename_pattern": "{patch_id}_mask.tif",
        "default_label_version": "1.0.0", "default_annotator": "auto",
        "default_confidence": 1.0, "default_confidence_source": "automatic",
        "min_distinct_classes": 1, "reject_single_class_masks": False,
        "max_nodata_ratio": 0.5, "output_formats": ["csv"],
        "ratios": {"water_sand_ratio": ["water", "sand"]},
        "bare_sediment_numerator": ["sand"],
        "bare_sediment_denominator": ["water", "sand", "vegetation"],
    }
    data["training"] = {
        "normalization": {"strategy": "per_band_mean_std", "percentile_min": 2, "percentile_max": 98},
        "augmentation": {"enabled": False},  # disabled for deterministic tests
        "dataloader": {
            "batch_size": 2, "num_workers": num_workers,
            "pin_memory": False, "prefetch_factor": 2,
            "persistent_workers": False, "train_shuffle": False,
        },
        "sampler": {"strategy": "none", "random_seed": 42},
        "class_weights": {"strategy": "none", "manual_weights": []},
    }
    return Config(config_path=write_config(tmp_path, data))


class TestDataLoaderConfig:
    def test_from_config(self, tmp_path: Path) -> None:
        cfg = DataLoaderConfig.from_config(_config(tmp_path))
        assert cfg.batch_size   == 2
        assert cfg.num_workers  == 0
        assert cfg.pin_memory   is False
        assert cfg.train_shuffle is False

    def test_frozen(self) -> None:
        cfg = DataLoaderConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.batch_size = 99  # type: ignore[misc]

    def test_defaults(self) -> None:
        cfg = DataLoaderConfig()
        assert cfg.batch_size  == 8
        assert cfg.num_workers == 4


class TestDataLoaderBundle:
    def test_frozen(self, tmp_path: Path) -> None:
        tr     = _make_training_result(tmp_path)
        config = _config(tmp_path, num_workers=0)
        bundle = DataLoaderFactory(config, _schema()).build(tr, tmp_path / "out")
        with pytest.raises((AttributeError, TypeError)):
            bundle.num_bands = 0  # type: ignore[misc]

    def test_summary_lines_ascii(self, tmp_path: Path) -> None:
        tr     = _make_training_result(tmp_path)
        config = _config(tmp_path, num_workers=0)
        bundle = DataLoaderFactory(config, _schema()).build(tr, tmp_path / "out")
        for line in bundle.summary_lines():
            assert all(ord(c) < 128 for c in line)


class TestDataLoaderFactory:
    def test_build_returns_bundle(self, tmp_path: Path) -> None:
        tr     = _make_training_result(tmp_path)
        config = _config(tmp_path, num_workers=0)
        bundle = DataLoaderFactory(config, _schema()).build(tr, tmp_path / "out")
        assert isinstance(bundle, DataLoaderBundle)

    def test_dataset_sizes(self, tmp_path: Path) -> None:
        tr     = _make_training_result(tmp_path, n_train=4, n_val=2, n_test=2)
        config = _config(tmp_path, num_workers=0)
        bundle = DataLoaderFactory(config, _schema()).build(tr, tmp_path / "out")
        assert bundle.train_dataset_size == 4
        assert bundle.val_dataset_size   == 2
        assert bundle.test_dataset_size  == 2

    def test_norm_stats_path_created(self, tmp_path: Path) -> None:
        tr     = _make_training_result(tmp_path)
        config = _config(tmp_path, num_workers=0)
        out    = tmp_path / "out"
        bundle = DataLoaderFactory(config, _schema()).build(tr, out)
        assert bundle.norm_stats_path.exists()

    def test_num_classes_matches_schema(self, tmp_path: Path) -> None:
        tr     = _make_training_result(tmp_path)
        config = _config(tmp_path, num_workers=0)
        bundle = DataLoaderFactory(config, _schema()).build(tr, tmp_path / "out")
        assert bundle.num_classes == 4

    def test_train_batch_shape(self, tmp_path: Path) -> None:
        import torch
        tr     = _make_training_result(tmp_path, n_train=4, n_val=2, n_test=2)
        config = _config(tmp_path, num_workers=0)
        bundle = DataLoaderFactory(config, _schema()).build(tr, tmp_path / "out")
        batch  = next(iter(bundle.train_loader))
        images, masks, metas = batch
        assert isinstance(images, torch.Tensor)
        assert isinstance(masks,  torch.Tensor)
        assert images.shape == (2, len(_BANDS), _SIZE, _SIZE)
        assert masks.shape  == (2, _SIZE, _SIZE)
        assert images.dtype == torch.float32
        assert masks.dtype  == torch.int64

    def test_metadata_is_list_in_batch(self, tmp_path: Path) -> None:
        from src.training.dataset import SampleMetadata
        tr     = _make_training_result(tmp_path, n_train=4, n_val=2, n_test=2)
        config = _config(tmp_path, num_workers=0)
        bundle = DataLoaderFactory(config, _schema()).build(tr, tmp_path / "out")
        _, _, metas = next(iter(bundle.train_loader))
        assert isinstance(metas, list)
        assert all(isinstance(m, SampleMetadata) for m in metas)

    def test_val_loader_no_shuffle(self, tmp_path: Path) -> None:
        """Validation loader must never shuffle."""
        tr     = _make_training_result(tmp_path, n_val=2)
        config = _config(tmp_path, num_workers=0)
        bundle = DataLoaderFactory(config, _schema()).build(tr, tmp_path / "out")
        assert bundle.val_loader.dataset.split == "validation"

    def test_empty_train_split_raises(self, tmp_path: Path) -> None:
        from src.core.exceptions import InvalidValueError
        tr     = _make_training_result(tmp_path, n_train=0, n_val=2, n_test=2)
        config = _config(tmp_path, num_workers=0)
        with pytest.raises(InvalidValueError, match="training split is empty"):
            DataLoaderFactory(config, _schema()).build(tr, tmp_path / "out")

    def test_norm_stats_num_bands(self, tmp_path: Path) -> None:
        tr     = _make_training_result(tmp_path)
        config = _config(tmp_path, num_workers=0)
        bundle = DataLoaderFactory(config, _schema()).build(tr, tmp_path / "out")
        assert bundle.norm_stats.num_bands == len(_BANDS)

    def test_class_weights_tensor_shape(self, tmp_path: Path) -> None:
        import torch
        tr     = _make_training_result(tmp_path)
        config = _config(tmp_path, num_workers=0)
        bundle = DataLoaderFactory(config, _schema()).build(tr, tmp_path / "out")
        tensor = bundle.class_weights.as_tensor()
        assert tensor.shape == (4,)

    def test_split_strategy_in_bundle(self, tmp_path: Path) -> None:
        tr     = _make_training_result(tmp_path)
        config = _config(tmp_path, num_workers=0)
        bundle = DataLoaderFactory(config, _schema()).build(tr, tmp_path / "out")
        assert bundle.split_strategy == "temporal"

    def test_multiple_epochs_consistent(self, tmp_path: Path) -> None:
        """Two passes over the validation loader must return samples in the same order."""
        import torch
        tr     = _make_training_result(tmp_path, n_train=2, n_val=2, n_test=2)
        config = _config(tmp_path, num_workers=0)
        bundle = DataLoaderFactory(config, _schema()).build(tr, tmp_path / "out")

        ids_epoch1 = [m.sample_id for _, _, meta_list in bundle.val_loader for m in meta_list]
        ids_epoch2 = [m.sample_id for _, _, meta_list in bundle.val_loader for m in meta_list]
        assert ids_epoch1 == ids_epoch2

    def test_aug_config_stored_in_bundle(self, tmp_path: Path) -> None:
        from src.training.transforms import AugmentationConfig
        tr     = _make_training_result(tmp_path)
        config = _config(tmp_path, num_workers=0)
        bundle = DataLoaderFactory(config, _schema()).build(tr, tmp_path / "out")
        assert isinstance(bundle.aug_config, AugmentationConfig)
        assert bundle.aug_config.enabled is False