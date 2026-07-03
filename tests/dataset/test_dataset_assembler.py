"""
Unit tests for src/dataset/assembler.py.

Uses real GeoTiffWriter (Module 7) for patch fixtures and direct rasterio
for mask fixtures. Real PatchDatasetResult and LabelDatasetResult instances
are constructed from frozen dataclasses (no mocking needed).

Run:
    pytest tests/dataset/test_dataset_assembler.py -v \
        --cov=src/dataset/assembler --cov-report=term-missing
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.crs import CRS

from src.core.exceptions import InvalidValueError
from src.dataset.assembler import DatasetAssembler, TrainingDatasetResult
from src.export.downloader import AffineTransform, AoiBounds, DownloadResult
from src.export.geotiff import GeoTiffProfile, GeoTiffWriter
from src.export.metadata import METADATA_SCHEMA_VERSION, SceneMetadata
from src.labels.manifest import LabelManifest, LabelManifestEntry
from src.labels.manager import LabelDatasetResult
from src.labels.schema import ClassDefinition, ClassSchema
from src.labels.statistics import LabelStatistics
from src.patches.generator import PatchDatasetResult
from src.patches.manifest import PatchManifest, PatchManifestEntry
from tests.conftest import make_valid_config, write_config

_CRS   = "EPSG:4326"
_AFFINE = Affine(0.001, 0.0, 87.0, 0.0, -0.001, 26.5)
_SIZE  = 8
_BANDS = ("Blue", "Green", "Red", "NIR")


def _config(tmp_path: Path, strategy: str = "temporal"):
    from src.core.config import Config
    data = make_valid_config()
    data["classes"] = {
        "num_classes": 4,
        "labels": {"background": 0, "water": 1, "sand": 2, "vegetation": 3},
        "names": ["background", "water", "sand", "vegetation"],
        "colors": {
            "background": [128, 128, 128], "water": [0, 119, 190],
            "sand": [255, 200, 87], "vegetation": [34, 139, 34],
        },
    }

    num_classes = data["classes"]["num_classes"]
    
    data["model"]["num_classes"] = num_classes
    if "loss" in data:
        data["loss"]["num_classes"] = num_classes

    data["temporal"] = {
        "seasons": {"monsoon": [6, 7, 8, 9], "winter": [12, 1, 2],
                    "pre_monsoon": [3, 4, 5], "post_monsoon": [10, 11]},
        "hydrological_year_start_month": 6,
    }
    data["labels"] = {
        "nodata_value": 255, "mask_filename_pattern": "{patch_id}_mask.tif",
        "default_label_version": "1.0.0", "default_annotator": "auto",
        "default_confidence": 1.0, "default_confidence_source": "automatic",
        "min_distinct_classes": 1, "reject_single_class_masks": False,
        "max_nodata_ratio": 0.5, "output_formats": ["csv"],
        "ratios": {"water_sand_ratio": ["water", "sand"]},
        "bare_sediment_numerator": ["sand"],
        "bare_sediment_denominator": ["water", "sand", "vegetation"],
    }
    data["dataset"] = {
        "split": {
            "strategy": strategy, "train_ratio": 0.7, "val_ratio": 0.15,
            "test_ratio": 0.15, "random_seed": 42,
        },
        "quality": {"min_valid_pixel_ratio": 0.5, "min_samples_per_split": 1},
        "output_formats": ["csv"], "dataset_version": "1.0.0",
        "min_total_samples": 1,
    }
    return Config(config_path=write_config(tmp_path, data))


def _write_patch(directory: Path, patch_id: str) -> Path:
    data = np.random.rand(len(_BANDS), _SIZE, _SIZE).astype(np.float32)
    dr = DownloadResult(
        data=data, crs=_CRS, transform=AffineTransform.from_affine(_AFFINE),
        band_names=_BANDS, width=_SIZE, height=_SIZE,
        aoi_bounds=AoiBounds(87.0, 26.0, 87.5, 26.5), num_tiles=1,
    )
    writer = GeoTiffWriter(GeoTiffProfile(tiled=False, overviews=False))
    return writer.write(dr, directory / f"{patch_id}.tif").path


def _write_mask(directory: Path, patch_id: str) -> Path:
    path   = directory / f"{patch_id}_mask.tif"
    values = np.zeros((_SIZE, _SIZE), dtype=np.uint8)
    values[:4, :] = 1
    values[4:, :4] = 2
    profile = {
        "driver": "GTiff", "dtype": "uint8", "width": _SIZE, "height": _SIZE,
        "count": 1, "crs": CRS.from_string(_CRS), "transform": _AFFINE,
    }
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(values, 1)
    return path


def _make_patch_result(
    patches_dir: Path, scene_id: str, n: int = 2,
    acquisition_date: str = "2023-07-15",
) -> PatchDatasetResult:
    entries: list[PatchManifestEntry] = []
    for i in range(n):
        patch_id = f"{scene_id}_r000_c{i:03d}"
        patch_path = _write_patch(patches_dir, patch_id)
        entries.append(PatchManifestEntry(
            patch_id=patch_id, scene_id=scene_id,
            source_image_path=str(patches_dir / "image.tif"),
            patch_path=str(patch_path), row_index=0, col_index=i,
            row_off=0, col_off=i * _SIZE,
            height=_SIZE, width=_SIZE, num_bands=len(_BANDS), crs=_CRS,
            valid_pixel_ratio=1.0, file_size_bytes=patch_path.stat().st_size,
            created_at=datetime.now(timezone.utc).isoformat(),
        ))
    manifest = PatchManifest(entries=tuple(entries), csv_path=None, json_path=None)
    return PatchDatasetResult(
        scene_id=scene_id, output_dir=patches_dir, scene_patches_dir=patches_dir,
        manifest=manifest, patches_generated=n, patches_skipped=0, total_windows=n,
        patch_size=_SIZE, stride=_SIZE, operations_log=("ok",),
    )


def _make_label_result(
    labels_dir: Path, scene_id: str, patch_ids: list[str],
    start_date: str = "2023-07-01", end_date: str = "2023-07-31",
) -> LabelDatasetResult:
    entries: list[LabelManifestEntry] = []
    for patch_id in patch_ids:
        mask_path = _write_mask(labels_dir, patch_id)
        entries.append(LabelManifestEntry(
            patch_id=patch_id, scene_id=scene_id,
            patch_path=f"/data/{patch_id}.tif",
            mask_path=str(mask_path), crs=_CRS, width=_SIZE, height=_SIZE,
            is_valid=True, validation_issues="", num_classes_present=3,
            valid_pixel_ratio=0.95, source_type="filesystem",
            acquisition_date="2023-07-15", year=2023, month=7, season="monsoon",
            hydrological_year=2023, sensor="L8,L9",
            river_name="", reach_id="", basin_id="", aoi_id="aoi_1",
            label_version="1.0.0", annotator="auto", confidence=1.0,
            confidence_source="automatic", processing_history="validated",
            created_at=datetime.now(timezone.utc).isoformat(),
        ))
    manifest = LabelManifest(entries=tuple(entries), csv_path=None, json_path=None)
    schema = ClassSchema(classes=(
        ClassDefinition(0, "background", (128, 128, 128)),
        ClassDefinition(1, "water", (0, 119, 190)),
        ClassDefinition(2, "sand", (255, 200, 87)),
        ClassDefinition(3, "vegetation", (34, 139, 34)),
    ))
    stats = LabelStatistics(
        total_labels=len(patch_ids), valid_labels=len(patch_ids), rejected_labels=0,
        class_pixel_stats=(), class_imbalance_ratio=1.0, class_ratios=(),
        bare_sediment_fraction=None, seasonal_distribution=(), yearly_distribution=(),
    )
    return LabelDatasetResult(
        scene_id=scene_id, output_dir=labels_dir, scene_labels_dir=labels_dir,
        manifest=manifest, statistics=stats, class_schema=schema,
        source_type="filesystem", labels_processed=len(patch_ids),
        labels_valid=len(patch_ids), labels_rejected=0, labels_missing=0,
        labels_duplicate=0, operations_log=("ok",),
    )


def _build_fixtures(
    tmp_path: Path,
    strategy: str = "temporal",
    n_scenes: int = 4,
    patches_per_scene: int = 2,
):
    patches_dir = tmp_path / "patches"
    labels_dir  = tmp_path / "labels"
    patches_dir.mkdir()
    labels_dir.mkdir()

    dates = ["2022-01-15", "2022-07-15", "2023-01-15", "2023-07-15"]
    patch_results: list[PatchDatasetResult] = []
    label_results: list[LabelDatasetResult] = []

    for i in range(n_scenes):
        scene_id = f"scene{i + 1:03d}"
        date     = dates[i % len(dates)]
        pr = _make_patch_result(patches_dir, scene_id, patches_per_scene, date)
        patch_results.append(pr)
        patch_ids = [e.patch_id for e in pr.manifest.entries]
        lr = _make_label_result(labels_dir, scene_id, patch_ids)
        label_results.append(lr)

    return patch_results, label_results


class TestDatasetAssemblerConstruction:
    def test_does_not_touch_filesystem(self, tmp_path: Path) -> None:
        DatasetAssembler(_config(tmp_path))
        assert not (tmp_path / "dataset_out").exists()


class TestDatasetAssemblerAssemble:
    def test_returns_training_dataset_result(self, tmp_path: Path) -> None:
        pr, lr = _build_fixtures(tmp_path)
        assembler = DatasetAssembler(_config(tmp_path))
        result    = assembler.assemble(pr, lr, tmp_path / "dataset_out", read_masks=False)
        assert isinstance(result, TrainingDatasetResult)

    def test_total_samples_correct(self, tmp_path: Path) -> None:
        pr, lr = _build_fixtures(tmp_path, n_scenes=4, patches_per_scene=2)
        assembler = DatasetAssembler(_config(tmp_path))
        result    = assembler.assemble(pr, lr, tmp_path / "out", read_masks=False)
        assert result.total_samples == 8

    def test_no_leakage(self, tmp_path: Path) -> None:
        pr, lr = _build_fixtures(tmp_path)
        assembler = DatasetAssembler(_config(tmp_path))
        result    = assembler.assemble(pr, lr, tmp_path / "out", read_masks=False)
        assert result.leakage_detection.has_leakage is False

    def test_manifest_files_created(self, tmp_path: Path) -> None:
        pr, lr    = _build_fixtures(tmp_path)
        assembler = DatasetAssembler(_config(tmp_path))
        out_dir   = tmp_path / "out"
        result    = assembler.assemble(pr, lr, out_dir, read_masks=False)
        assert result.manifest.dataset_manifest_csv.exists()

    def test_version_json_created(self, tmp_path: Path) -> None:
        pr, lr    = _build_fixtures(tmp_path)
        assembler = DatasetAssembler(_config(tmp_path))
        result    = assembler.assemble(pr, lr, tmp_path / "out", read_masks=False)
        assert result.version_path.exists()

    def test_quality_report_json_created(self, tmp_path: Path) -> None:
        pr, lr    = _build_fixtures(tmp_path)
        assembler = DatasetAssembler(_config(tmp_path))
        result    = assembler.assemble(pr, lr, tmp_path / "out", read_masks=False)
        assert result.quality_report_path.exists()

    def test_statistics_json_created(self, tmp_path: Path) -> None:
        pr, lr    = _build_fixtures(tmp_path)
        assembler = DatasetAssembler(_config(tmp_path))
        result    = assembler.assemble(pr, lr, tmp_path / "out", read_masks=False)
        assert result.statistics_path.exists()

    def test_splits_sum_to_total(self, tmp_path: Path) -> None:
        pr, lr    = _build_fixtures(tmp_path, n_scenes=4, patches_per_scene=2)
        assembler = DatasetAssembler(_config(tmp_path))
        result    = assembler.assemble(pr, lr, tmp_path / "out", read_masks=False)
        assert (
            result.train_samples + result.validation_samples + result.test_samples
        ) == result.total_samples

    def test_version_info_stored(self, tmp_path: Path) -> None:
        pr, lr    = _build_fixtures(tmp_path)
        assembler = DatasetAssembler(_config(tmp_path))
        result    = assembler.assemble(pr, lr, tmp_path / "out", read_masks=False)
        assert result.version_info.dataset_version == "1.0.0"

    def test_empty_valid_labels_raises(self, tmp_path: Path) -> None:
        pr, _ = _build_fixtures(tmp_path)
        # Build label results with no valid entries
        from src.labels.manifest import LabelManifest
        from src.labels.manager import LabelDatasetResult
        from src.labels.schema import ClassDefinition, ClassSchema
        from src.labels.statistics import LabelStatistics
        schema = ClassSchema(classes=(ClassDefinition(0, "background", (0, 0, 0)),))
        stats  = LabelStatistics(
            total_labels=0, valid_labels=0, rejected_labels=0,
            class_pixel_stats=(), class_imbalance_ratio=1.0, class_ratios=(),
            bare_sediment_fraction=None, seasonal_distribution=(), yearly_distribution=(),
        )
        empty_lr = [
            LabelDatasetResult(
                scene_id=p.scene_id, output_dir=tmp_path,
                scene_labels_dir=tmp_path,
                manifest=LabelManifest(entries=(), csv_path=None, json_path=None),
                statistics=stats, class_schema=schema, source_type="filesystem",
                labels_processed=0, labels_valid=0, labels_rejected=0,
                labels_missing=0, labels_duplicate=0, operations_log=(),
            )
            for p in pr
        ]
        assembler = DatasetAssembler(_config(tmp_path))
        with pytest.raises(InvalidValueError, match="No valid samples"):
            assembler.assemble(pr, empty_lr, tmp_path / "out", read_masks=False)

    def test_orphan_patches_excluded(self, tmp_path: Path) -> None:
        """Patches with no matching valid label are silently excluded."""
        pr, lr = _build_fixtures(tmp_path, n_scenes=2, patches_per_scene=2)
        # Remove all label entries from the first scene
        from src.labels.manifest import LabelManifest
        new_lr = [lr[0].__class__(
            scene_id=lr[0].scene_id, output_dir=lr[0].output_dir,
            scene_labels_dir=lr[0].scene_labels_dir,
            manifest=LabelManifest(entries=(), csv_path=None, json_path=None),
            statistics=lr[0].statistics, class_schema=lr[0].class_schema,
            source_type=lr[0].source_type, labels_processed=0, labels_valid=0,
            labels_rejected=0, labels_missing=0, labels_duplicate=0,
            operations_log=(),
        )] + lr[1:]
        assembler = DatasetAssembler(_config(tmp_path))
        result    = assembler.assemble(pr, new_lr, tmp_path / "out", read_masks=False)
        # Only scene 2's patches should be assembled
        assert result.total_samples == 2

    def test_summary_lines_ascii_only(self, tmp_path: Path) -> None:
        pr, lr    = _build_fixtures(tmp_path)
        assembler = DatasetAssembler(_config(tmp_path))
        result    = assembler.assemble(pr, lr, tmp_path / "out", read_masks=False)
        for line in result.summary_lines():
            assert all(ord(c) < 128 for c in line)

    def test_result_is_frozen(self, tmp_path: Path) -> None:
        pr, lr    = _build_fixtures(tmp_path)
        assembler = DatasetAssembler(_config(tmp_path))
        result    = assembler.assemble(pr, lr, tmp_path / "out", read_masks=False)
        with pytest.raises((AttributeError, TypeError)):
            result.total_samples = 0  # type: ignore[misc]