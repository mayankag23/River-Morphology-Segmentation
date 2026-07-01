"""
Unit tests for src/labels/manifest.py.

Run:
    pytest tests/labels/test_label_manifest.py -v \
        --cov=src/labels/manifest --cov-report=term-missing
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.core.exceptions import InvalidValueError
from src.labels.manifest import (
    LABEL_MANIFEST_SCHEMA_VERSION,
    LabelManifest,
    LabelManifestEntry,
    LabelManifestManager,
)


def _entry(patch_id: str = "scene001_r000_c000", is_valid: bool = True) -> LabelManifestEntry:
    return LabelManifestEntry(
        patch_id=patch_id, scene_id="scene001",
        patch_path=f"/data/scenes/scene001/patches/{patch_id}.tif",
        mask_path=f"/data/scenes/scene001/labels/{patch_id}_mask.tif",
        crs="EPSG:4326", width=256, height=256, is_valid=is_valid,
        validation_issues="", num_classes_present=3, valid_pixel_ratio=0.97,
        source_type="filesystem", acquisition_date="2023-07-15", year=2023,
        month=7, season="monsoon", hydrological_year=2023, sensor="L8,L9",
        river_name="Kosi", reach_id="reach-1", basin_id="basin-1",
        aoi_id="aoi_1", label_version="1.0.0", annotator="auto_generated",
        confidence=1.0, confidence_source="automatic",
        processing_history="discovered,validated,organized",
        created_at="2024-01-15T10:30:00+00:00",
    )


class TestLabelManifestEntry:
    def test_frozen(self) -> None:
        e = _entry()
        with pytest.raises((AttributeError, TypeError)):
            e.patch_id = "other"  # type: ignore[misc]

    def test_ascii_fields(self) -> None:
        for key, value in _entry().to_dict().items():
            if isinstance(value, str):
                assert all(ord(c) < 128 for c in value)


class TestLabelManifest:
    def test_frozen(self) -> None:
        m = LabelManifest(entries=(_entry(),), csv_path=None, json_path=None)
        with pytest.raises((AttributeError, TypeError)):
            m.entries = ()  # type: ignore[misc]

    def test_entry_count(self) -> None:
        m = LabelManifest(entries=(_entry("a"), _entry("b")), csv_path=None, json_path=None)
        assert m.entry_count == 2

    def test_from_csv_roundtrip(self, tmp_path: Path) -> None:
        mgr = LabelManifestManager()
        mgr.add_entry(_entry("p1"))
        mgr.add_entry(_entry("p2", is_valid=False))
        manifest = mgr.save(tmp_path, formats=["csv"])
        loaded = LabelManifest.from_csv(manifest.csv_path)
        assert loaded.entry_count == 2
        assert loaded.entries[1].is_valid is False
        assert loaded.entries[0].hydrological_year == 2023

    def test_from_json_roundtrip(self, tmp_path: Path) -> None:
        mgr = LabelManifestManager()
        mgr.add_entry(_entry("p1"))
        manifest = mgr.save(tmp_path, formats=["json"])
        loaded = LabelManifest.from_json(manifest.json_path)
        assert loaded.entries[0].patch_id == "p1"

    def test_from_csv_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            LabelManifest.from_csv(tmp_path / "nonexistent.csv")


class TestLabelManifestManager:
    def test_add_entry(self) -> None:
        mgr = LabelManifestManager()
        mgr.add_entry(_entry())
        assert mgr.entry_count == 1

    def test_entries_property(self) -> None:
        mgr = LabelManifestManager()
        mgr.add_entry(_entry())
        assert isinstance(mgr.entries, tuple)

    def test_add_non_entry_raises(self) -> None:
        with pytest.raises(InvalidValueError, match="LabelManifestEntry"):
            LabelManifestManager().add_entry("not_an_entry")  # type: ignore[arg-type]

    def test_save_both_formats(self, tmp_path: Path) -> None:
        mgr = LabelManifestManager()
        mgr.add_entry(_entry())
        manifest = mgr.save(tmp_path, formats=["csv", "json"])
        assert manifest.csv_path.exists() and manifest.json_path.exists()

    def test_csv_row_count(self, tmp_path: Path) -> None:
        mgr = LabelManifestManager()
        mgr.add_entry(_entry("a"))
        mgr.add_entry(_entry("b"))
        manifest = mgr.save(tmp_path, formats=["csv"])
        with open(manifest.csv_path, newline="") as fh:
            assert len(list(csv.DictReader(fh))) == 2

    def test_json_version_present(self, tmp_path: Path) -> None:
        mgr = LabelManifestManager()
        manifest = mgr.save(tmp_path, formats=["json"])
        with open(manifest.json_path) as fh:
            payload = json.load(fh)
        assert payload["version"] == LABEL_MANIFEST_SCHEMA_VERSION

    def test_load_existing_appends(self, tmp_path: Path) -> None:
        mgr1 = LabelManifestManager()
        mgr1.add_entry(_entry("a"))
        mgr1.save(tmp_path, formats=["csv"])
        mgr2 = LabelManifestManager()
        mgr2.load_existing(tmp_path)
        mgr2.add_entry(_entry("b"))
        manifest = mgr2.save(tmp_path, formats=["csv"])
        assert manifest.entry_count == 2

    def test_load_existing_no_file_returns_zero(self, tmp_path: Path) -> None:
        assert LabelManifestManager().load_existing(tmp_path) == 0

    def test_csv_ascii_only(self, tmp_path: Path) -> None:
        mgr = LabelManifestManager()
        mgr.add_entry(_entry())
        manifest = mgr.save(tmp_path, formats=["csv"])
        assert all(ord(c) < 128 for c in manifest.csv_path.read_text())