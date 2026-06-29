"""
Unit tests for src/export/manifest.py.

Pure Python tests -- no rasterio, no EE.

Run:
    pytest tests/export/test_dataset_manifest.py -v
    pytest tests/export/test_dataset_manifest.py -v \
        --cov=src/export/manifest --cov-report=term-missing
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.core.exceptions import InvalidValueError
from src.export.manifest import (
    DatasetManifest,
    DatasetManifestManager,
    ManifestEntry,
    MANIFEST_SCHEMA_VERSION,
)


# ==============================================================================
# Helpers
# ==============================================================================

def _entry(scene_id: str = "sc_001", is_valid: bool = True) -> ManifestEntry:
    return ManifestEntry(
        scene_id             = scene_id,
        image_path           = f"/data/scenes/{scene_id}/image.tif",
        metadata_path        = f"/data/scenes/{scene_id}/metadata.json",
        export_timestamp     = "2024-01-15T10:30:00+00:00",
        num_bands            = 14,
        width                = 1850,
        height               = 1850,
        crs                  = "EPSG:4326",
        start_date           = "2023-11-01",
        end_date             = "2024-02-28",
        sensors              = "L8,L9",
        composite_method     = "median",
        cloud_cover_limit    = 20.0,
        num_spectral_indices = 9,
        file_size_bytes      = 10_000_000,
        is_valid             = is_valid,
    )


# ==============================================================================
# ManifestEntry tests
# ==============================================================================

class TestManifestEntry:
    def test_frozen(self) -> None:
        e = _entry()
        with pytest.raises((AttributeError, TypeError)):
            e.scene_id = "other"  # type: ignore[misc]

    def test_to_dict_is_dict(self) -> None:
        assert isinstance(_entry().to_dict(), dict)

    def test_is_valid_false_preserved(self) -> None:
        assert _entry(is_valid=False).is_valid is False

    def test_all_string_fields_ascii(self) -> None:
        for key, value in _entry().to_dict().items():
            if isinstance(value, str):
                assert all(ord(c) < 128 for c in value)


# ==============================================================================
# DatasetManifest (frozen result) tests
# ==============================================================================

class TestDatasetManifest:
    def test_frozen(self, tmp_path: Path) -> None:
        m = DatasetManifest(
            entries=(_entry(),), csv_path=None, json_path=None
        )
        with pytest.raises((AttributeError, TypeError)):
            m.entries = ()  # type: ignore[misc]

    def test_entry_count_property(self, tmp_path: Path) -> None:
        m = DatasetManifest(
            entries=(_entry("a"), _entry("b")), csv_path=None, json_path=None
        )
        assert m.entry_count == 2

    def test_from_csv_roundtrip(self, tmp_path: Path) -> None:
        mgr = DatasetManifestManager()
        mgr.add_entry(_entry("sc_001"))
        mgr.add_entry(_entry("sc_002", is_valid=False))
        manifest = mgr.save(tmp_path, formats=["csv"])

        loaded = DatasetManifest.from_csv(manifest.csv_path)
        assert loaded.entry_count == 2
        assert loaded.entries[0].scene_id == "sc_001"
        assert loaded.entries[1].is_valid is False

    def test_from_json_roundtrip(self, tmp_path: Path) -> None:
        mgr = DatasetManifestManager()
        mgr.add_entry(_entry("sc_001"))
        manifest = mgr.save(tmp_path, formats=["json"])

        loaded = DatasetManifest.from_json(manifest.json_path)
        assert loaded.entries[0].scene_id == "sc_001"

    def test_from_csv_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            DatasetManifest.from_csv(tmp_path / "nonexistent.csv")

    def test_from_json_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            DatasetManifest.from_json(tmp_path / "nonexistent.json")


# ==============================================================================
# DatasetManifestManager tests
# ==============================================================================

class TestDatasetManifestManager:
    def test_initial_state_empty(self) -> None:
        assert len(DatasetManifestManager()._entries) == 0

    def test_add_entry_increments_entries(self) -> None:
        mgr = DatasetManifestManager()
        mgr.add_entry(_entry())
        assert len(mgr._entries) == 1

    def test_add_non_entry_raises(self) -> None:
        with pytest.raises(InvalidValueError, match="ManifestEntry"):
            DatasetManifestManager().add_entry("not_an_entry")  # type: ignore[arg-type]

    def test_save_csv_creates_file(self, tmp_path: Path) -> None:
        mgr = DatasetManifestManager()
        mgr.add_entry(_entry())
        manifest = mgr.save(tmp_path, formats=["csv"])
        assert manifest.csv_path is not None
        assert manifest.csv_path.exists()

    def test_save_json_creates_file(self, tmp_path: Path) -> None:
        mgr = DatasetManifestManager()
        mgr.add_entry(_entry())
        manifest = mgr.save(tmp_path, formats=["json"])
        assert manifest.json_path is not None
        assert manifest.json_path.exists()

    def test_save_both_formats(self, tmp_path: Path) -> None:
        mgr = DatasetManifestManager()
        mgr.add_entry(_entry())
        manifest = mgr.save(tmp_path, formats=["csv", "json"])
        assert manifest.csv_path is not None
        assert manifest.json_path is not None
        assert manifest.csv_path.exists()
        assert manifest.json_path.exists()

    def test_save_returns_frozen_manifest(self, tmp_path: Path) -> None:
        mgr = DatasetManifestManager()
        mgr.add_entry(_entry())
        manifest = mgr.save(tmp_path)
        assert isinstance(manifest, DatasetManifest)

    def test_csv_row_count(self, tmp_path: Path) -> None:
        mgr = DatasetManifestManager()
        mgr.add_entry(_entry("a"))
        mgr.add_entry(_entry("b"))
        manifest = mgr.save(tmp_path, formats=["csv"])
        with open(manifest.csv_path, newline="") as fh:
            assert len(list(csv.DictReader(fh))) == 2

    def test_json_entry_count(self, tmp_path: Path) -> None:
        mgr = DatasetManifestManager()
        mgr.add_entry(_entry("a"))
        manifest = mgr.save(tmp_path, formats=["json"])
        with open(manifest.json_path) as fh:
            payload = json.load(fh)
        assert payload["entry_count"] == 1

    def test_json_version_present(self, tmp_path: Path) -> None:
        mgr = DatasetManifestManager()
        manifest = mgr.save(tmp_path, formats=["json"])
        with open(manifest.json_path) as fh:
            assert "version" in json.load(fh)

    def test_float_preserved_in_csv_roundtrip(self, tmp_path: Path) -> None:
        mgr = DatasetManifestManager()
        mgr.add_entry(_entry())
        manifest = mgr.save(tmp_path, formats=["csv"])
        loaded   = DatasetManifest.from_csv(manifest.csv_path)
        assert loaded.entries[0].cloud_cover_limit == pytest.approx(20.0)

    def test_int_preserved_in_csv_roundtrip(self, tmp_path: Path) -> None:
        mgr = DatasetManifestManager()
        mgr.add_entry(_entry())
        manifest = mgr.save(tmp_path, formats=["csv"])
        loaded   = DatasetManifest.from_csv(manifest.csv_path)
        assert loaded.entries[0].file_size_bytes == 10_000_000

    def test_bool_false_preserved_in_csv_roundtrip(self, tmp_path: Path) -> None:
        mgr = DatasetManifestManager()
        mgr.add_entry(_entry(is_valid=False))
        manifest = mgr.save(tmp_path, formats=["csv"])
        loaded   = DatasetManifest.from_csv(manifest.csv_path)
        assert loaded.entries[0].is_valid is False

    def test_load_existing_appends(self, tmp_path: Path) -> None:
        mgr1 = DatasetManifestManager()
        mgr1.add_entry(_entry("sc_A"))
        mgr1.save(tmp_path, formats=["csv"])

        mgr2 = DatasetManifestManager()
        mgr2.load_existing(tmp_path)
        mgr2.add_entry(_entry("sc_B"))
        manifest = mgr2.save(tmp_path, formats=["csv"])

        assert manifest.entry_count == 2

    def test_load_existing_returns_count(self, tmp_path: Path) -> None:
        mgr1 = DatasetManifestManager()
        mgr1.add_entry(_entry("sc_X"))
        mgr1.save(tmp_path, formats=["csv"])

        mgr2 = DatasetManifestManager()
        n = mgr2.load_existing(tmp_path)
        assert n == 1

    def test_load_existing_no_file_returns_zero(self, tmp_path: Path) -> None:
        n = DatasetManifestManager().load_existing(tmp_path)
        assert n == 0

    def test_manifest_path_in_correct_location(self, tmp_path: Path) -> None:
        mgr = DatasetManifestManager()
        mgr.add_entry(_entry())
        manifest = mgr.save(tmp_path, formats=["csv"])
        assert manifest.csv_path.parent == tmp_path

    def test_csv_output_ascii_only(self, tmp_path: Path) -> None:
        mgr = DatasetManifestManager()
        mgr.add_entry(_entry())
        manifest = mgr.save(tmp_path, formats=["csv"])
        content  = manifest.csv_path.read_text(encoding="utf-8")
        assert all(ord(c) < 128 for c in content)

    def test_json_output_ascii_only(self, tmp_path: Path) -> None:
        mgr = DatasetManifestManager()
        mgr.add_entry(_entry())
        manifest = mgr.save(tmp_path, formats=["json"])
        content  = manifest.json_path.read_text(encoding="utf-8")
        assert all(ord(c) < 128 for c in content)