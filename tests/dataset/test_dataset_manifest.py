"""
Unit tests for src/dataset/manifest.py.

Run:
    pytest tests/dataset/test_dataset_manifest.py -v \
        --cov=src/dataset/manifest --cov-report=term-missing
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.core.exceptions import InvalidValueError
from src.dataset.manifest import (
    DATASET_MANIFEST_SCHEMA_VERSION,
    DatasetManifest,
    DatasetManifestEntry,
    DatasetManifestManager,
    DatasetSample,
)


def _sample(
    patch_id: str = "sc_r000_c000",
    scene_id: str = "scene001",
    season: str = "monsoon",
    year: int = 2023,
    aoi_id: str = "aoi_1",
) -> DatasetSample:
    return DatasetSample(
        sample_id=patch_id, patch_id=patch_id, scene_id=scene_id,
        patch_path=f"/data/patches/{patch_id}.tif",
        mask_path=f"/data/labels/{patch_id}_mask.tif",
        crs="EPSG:4326", width=256, height=256, num_bands=4,
        row_index=0, col_index=0,
        patch_valid_pixel_ratio=1.0, label_valid_pixel_ratio=0.95,
        num_classes_present=3, acquisition_date="2023-07-15",
        year=year, month=7, season=season, hydrological_year=year,
        sensor="L8,L9", river_name="Kosi", reach_id="r1", basin_id="b1",
        aoi_id=aoi_id, label_version="1.0.0", annotator="auto_generated",
        confidence=1.0, confidence_source="automatic",
    )


def _entry(patch_id: str = "sc_r000_c000", split: str = "train") -> DatasetManifestEntry:
    return DatasetManifestEntry.from_sample(_sample(patch_id=patch_id), split)


class TestDatasetSample:
    def test_frozen(self) -> None:
        s = _sample()
        with pytest.raises((AttributeError, TypeError)):
            s.sample_id = "other"  # type: ignore[misc]


class TestDatasetManifestEntry:
    def test_from_sample_sets_split(self) -> None:
        entry = _entry(split="train")
        assert entry.split == "train"

    def test_from_sample_invalid_split_raises(self) -> None:
        with pytest.raises(InvalidValueError, match="split"):
            DatasetManifestEntry.from_sample(_sample(), "invalid_split")

    def test_frozen(self) -> None:
        e = _entry()
        with pytest.raises((AttributeError, TypeError)):
            e.split = "test"  # type: ignore[misc]

    def test_to_dict_ascii_only(self) -> None:
        for key, value in _entry().to_dict().items():
            if isinstance(value, str):
                assert all(ord(c) < 128 for c in value)


class TestDatasetManifest:
    def test_entry_count(self) -> None:
        m = DatasetManifest(
            entries=(_entry("a"), _entry("b", "validation")),
            dataset_manifest_csv=None, dataset_manifest_json=None,
            train_csv=None, validation_csv=None, test_csv=None,
        )
        assert m.entry_count == 2

    def test_entries_for_split(self) -> None:
        m = DatasetManifest(
            entries=(_entry("a", "train"), _entry("b", "validation")),
            dataset_manifest_csv=None, dataset_manifest_json=None,
            train_csv=None, validation_csv=None, test_csv=None,
        )
        assert len(m.entries_for_split("train")) == 1
        assert len(m.entries_for_split("validation")) == 1
        assert len(m.entries_for_split("test")) == 0

    def test_from_csv_roundtrip(self, tmp_path: Path) -> None:
        mgr = DatasetManifestManager()
        mgr.add_entry(_entry("a", "train"))
        mgr.add_entry(_entry("b", "validation"))
        manifest = mgr.save(tmp_path, formats=["csv"])
        loaded = DatasetManifest.from_csv(manifest.dataset_manifest_csv)
        assert loaded.entry_count == 2
        assert loaded.entries[0].split == "train"

    def test_from_csv_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            DatasetManifest.from_csv(tmp_path / "nonexistent.csv")

    def test_frozen(self) -> None:
        m = DatasetManifest(
            entries=(), dataset_manifest_csv=None, dataset_manifest_json=None,
            train_csv=None, validation_csv=None, test_csv=None,
        )
        with pytest.raises((AttributeError, TypeError)):
            m.entries = ()  # type: ignore[misc]


class TestDatasetManifestManager:
    def test_add_entry(self) -> None:
        mgr = DatasetManifestManager()
        mgr.add_entry(_entry())
        assert mgr.entry_count == 1

    def test_add_non_entry_raises(self) -> None:
        with pytest.raises(InvalidValueError, match="DatasetManifestEntry"):
            DatasetManifestManager().add_entry("not_an_entry")  # type: ignore[arg-type]

    def test_save_creates_all_files(self, tmp_path: Path) -> None:
        mgr = DatasetManifestManager()
        mgr.add_entry(_entry("a", "train"))
        mgr.add_entry(_entry("b", "validation"))
        mgr.add_entry(_entry("c", "test"))
        manifest = mgr.save(tmp_path, formats=["csv", "json"])
        assert manifest.dataset_manifest_csv.exists()
        assert manifest.dataset_manifest_json.exists()
        assert manifest.train_csv.exists()
        assert manifest.validation_csv.exists()
        assert manifest.test_csv.exists()

    def test_csv_has_correct_row_count(self, tmp_path: Path) -> None:
        mgr = DatasetManifestManager()
        for i in range(5):
            mgr.add_entry(_entry(f"p{i}", "train"))
        manifest = mgr.save(tmp_path, formats=["csv"])
        with open(manifest.dataset_manifest_csv, newline="") as fh:
            assert len(list(csv.DictReader(fh))) == 5

    def test_json_version_present(self, tmp_path: Path) -> None:
        mgr = DatasetManifestManager()
        manifest = mgr.save(tmp_path, formats=["json"])
        with open(manifest.dataset_manifest_json) as fh:
            payload = json.load(fh)
        assert payload["version"] == DATASET_MANIFEST_SCHEMA_VERSION

    def test_float_int_preserved_csv_roundtrip(self, tmp_path: Path) -> None:
        mgr = DatasetManifestManager()
        mgr.add_entry(_entry("a", "train"))
        manifest = mgr.save(tmp_path, formats=["csv"])
        loaded = DatasetManifest.from_csv(manifest.dataset_manifest_csv)
        assert loaded.entries[0].confidence == pytest.approx(1.0)
        assert loaded.entries[0].width == 256
        assert loaded.entries[0].hydrological_year == 2023

    def test_output_ascii_only(self, tmp_path: Path) -> None:
        mgr = DatasetManifestManager()
        mgr.add_entry(_entry())
        manifest = mgr.save(tmp_path, formats=["csv"])
        content = manifest.dataset_manifest_csv.read_text(encoding="utf-8")
        assert all(ord(c) < 128 for c in content)