"""
Unit tests for src/patches/manifest.py.

Pure Python -- no rasterio, no EE.

Run:
    pytest tests/patches/test_patch_manifest.py -v
    pytest tests/patches/test_patch_manifest.py -v \
        --cov=src/patches/manifest --cov-report=term-missing
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.core.exceptions import InvalidValueError
from src.patches.manifest import (
    PATCH_MANIFEST_SCHEMA_VERSION,
    PatchManifest,
    PatchManifestEntry,
    PatchManifestManager,
)


def _entry(patch_id: str = "scene001_r000_c000") -> PatchManifestEntry:
    return PatchManifestEntry(
        patch_id=patch_id,
        scene_id="scene001",
        source_image_path="/data/scenes/scene001/image.tif",
        patch_path=f"/data/patches/scenes/scene001/patches/{patch_id}.tif",
        row_index=0, col_index=0, row_off=0, col_off=0,
        height=256, width=256, num_bands=14, crs="EPSG:4326",
        valid_pixel_ratio=0.95, file_size_bytes=500_000,
        created_at="2024-01-15T10:30:00+00:00",
    )


# ==============================================================================
# PatchManifestEntry tests
# ==============================================================================

class TestPatchManifestEntry:
    """Tests for the frozen PatchManifestEntry dataclass."""

    def test_frozen(self) -> None:
        e = _entry()
        with pytest.raises((AttributeError, TypeError)):
            e.patch_id = "other"  # type: ignore[misc]

    def test_to_dict(self) -> None:
        assert isinstance(_entry().to_dict(), dict)

    def test_ascii_fields(self) -> None:
        for key, value in _entry().to_dict().items():
            if isinstance(value, str):
                assert all(ord(c) < 128 for c in value), (
                    f"Non-ASCII in field '{key}': {value!r}"
                )


# ==============================================================================
# PatchManifest (frozen) tests
# ==============================================================================

class TestPatchManifest:
    """Tests for the frozen PatchManifest result type."""

    def test_frozen(self) -> None:
        m = PatchManifest(entries=(_entry(),), csv_path=None, json_path=None)
        with pytest.raises((AttributeError, TypeError)):
            m.entries = ()  # type: ignore[misc]

    def test_entry_count(self) -> None:
        m = PatchManifest(entries=(_entry("a"), _entry("b")), csv_path=None, json_path=None)
        assert m.entry_count == 2

    def test_from_csv_roundtrip(self, tmp_path: Path) -> None:
        mgr = PatchManifestManager()
        mgr.add_entry(_entry("p1"))
        mgr.add_entry(_entry("p2"))
        manifest = mgr.save(tmp_path, formats=["csv"])

        loaded = PatchManifest.from_csv(manifest.csv_path)
        assert loaded.entry_count == 2
        assert loaded.entries[0].patch_id == "p1"

    def test_from_json_roundtrip(self, tmp_path: Path) -> None:
        mgr = PatchManifestManager()
        mgr.add_entry(_entry("p1"))
        manifest = mgr.save(tmp_path, formats=["json"])

        loaded = PatchManifest.from_json(manifest.json_path)
        assert loaded.entries[0].patch_id == "p1"

    def test_from_csv_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            PatchManifest.from_csv(tmp_path / "nonexistent.csv")

    def test_from_json_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            PatchManifest.from_json(tmp_path / "nonexistent.json")


# ==============================================================================
# PatchManifestManager tests
# ==============================================================================

class TestPatchManifestManager:
    """Tests for PatchManifestManager."""

    def test_initial_empty(self) -> None:
        assert PatchManifestManager().entry_count == 0

    def test_add_entry(self) -> None:
        mgr = PatchManifestManager()
        mgr.add_entry(_entry())
        assert mgr.entry_count == 1

    def test_add_non_entry_raises(self) -> None:
        with pytest.raises(InvalidValueError, match="PatchManifestEntry"):
            PatchManifestManager().add_entry("not_an_entry")  # type: ignore[arg-type]

    def test_save_csv_creates_file(self, tmp_path: Path) -> None:
        mgr = PatchManifestManager()
        mgr.add_entry(_entry())
        manifest = mgr.save(tmp_path, formats=["csv"])
        assert manifest.csv_path.exists()

    def test_save_json_creates_file(self, tmp_path: Path) -> None:
        mgr = PatchManifestManager()
        mgr.add_entry(_entry())
        manifest = mgr.save(tmp_path, formats=["json"])
        assert manifest.json_path.exists()

    def test_save_both_formats(self, tmp_path: Path) -> None:
        mgr = PatchManifestManager()
        mgr.add_entry(_entry())
        manifest = mgr.save(tmp_path, formats=["csv", "json"])
        assert manifest.csv_path.exists()
        assert manifest.json_path.exists()

    def test_csv_row_count(self, tmp_path: Path) -> None:
        mgr = PatchManifestManager()
        mgr.add_entry(_entry("a"))
        mgr.add_entry(_entry("b"))
        manifest = mgr.save(tmp_path, formats=["csv"])
        with open(manifest.csv_path, newline="") as fh:
            assert len(list(csv.DictReader(fh))) == 2

    def test_json_version_present(self, tmp_path: Path) -> None:
        mgr = PatchManifestManager()
        manifest = mgr.save(tmp_path, formats=["json"])
        with open(manifest.json_path) as fh:
            payload = json.load(fh)
        assert payload["version"] == PATCH_MANIFEST_SCHEMA_VERSION

    def test_float_preserved_csv_roundtrip(self, tmp_path: Path) -> None:
        mgr = PatchManifestManager()
        mgr.add_entry(_entry())
        manifest = mgr.save(tmp_path, formats=["csv"])
        loaded   = PatchManifest.from_csv(manifest.csv_path)
        assert loaded.entries[0].valid_pixel_ratio == pytest.approx(0.95)

    def test_int_preserved_csv_roundtrip(self, tmp_path: Path) -> None:
        mgr = PatchManifestManager()
        mgr.add_entry(_entry())
        manifest = mgr.save(tmp_path, formats=["csv"])
        loaded   = PatchManifest.from_csv(manifest.csv_path)
        assert loaded.entries[0].file_size_bytes == 500_000
        assert loaded.entries[0].row_index       == 0

    def test_load_existing_appends(self, tmp_path: Path) -> None:
        mgr1 = PatchManifestManager()
        mgr1.add_entry(_entry("a"))
        mgr1.save(tmp_path, formats=["csv"])

        mgr2 = PatchManifestManager()
        mgr2.load_existing(tmp_path)
        mgr2.add_entry(_entry("b"))
        manifest = mgr2.save(tmp_path, formats=["csv"])

        assert manifest.entry_count == 2

    def test_load_existing_no_file_returns_zero(self, tmp_path: Path) -> None:
        assert PatchManifestManager().load_existing(tmp_path) == 0

    def test_load_existing_returns_count(self, tmp_path: Path) -> None:
        mgr1 = PatchManifestManager()
        mgr1.add_entry(_entry("x"))
        mgr1.save(tmp_path, formats=["csv"])

        mgr2 = PatchManifestManager()
        n = mgr2.load_existing(tmp_path)
        assert n == 1

    def test_manifest_path_at_root(self, tmp_path: Path) -> None:
        mgr = PatchManifestManager()
        mgr.add_entry(_entry())
        manifest = mgr.save(tmp_path, formats=["csv"])
        assert manifest.csv_path.parent == tmp_path

    def test_csv_ascii_only(self, tmp_path: Path) -> None:
        mgr = PatchManifestManager()
        mgr.add_entry(_entry())
        manifest = mgr.save(tmp_path, formats=["csv"])
        content  = manifest.csv_path.read_text(encoding="utf-8")
        assert all(ord(c) < 128 for c in content)

    def test_json_ascii_only(self, tmp_path: Path) -> None:
        mgr = PatchManifestManager()
        mgr.add_entry(_entry())
        manifest = mgr.save(tmp_path, formats=["json"])
        content  = manifest.json_path.read_text(encoding="utf-8")
        assert all(ord(c) < 128 for c in content)