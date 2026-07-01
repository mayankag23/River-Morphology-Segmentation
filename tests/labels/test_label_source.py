"""
Unit tests for src/labels/source.py.

Run:
    pytest tests/labels/test_label_source.py -v \
        --cov=src/labels/source --cov-report=term-missing
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.exceptions import InvalidValueError
from src.labels.source import FilesystemLabelSource, LabelSource, LabelSourceRecord
from tests.conftest import make_valid_config, write_config


def _config(tmp_path: Path, pattern: str = "{patch_id}_mask.tif"):
    from src.core.config import Config
    data = make_valid_config()
    data["labels"] = {"mask_filename_pattern": pattern}
    return Config(config_path=write_config(tmp_path, data))


class TestLabelSourceRecord:
    def test_frozen(self) -> None:
        r = LabelSourceRecord(patch_id="p", mask_path=None, found=False, source_type="filesystem")
        with pytest.raises((AttributeError, TypeError)):
            r.found = True  # type: ignore[misc]


class TestFilesystemLabelSourceConstruction:
    def test_valid_construction(self, tmp_path: Path) -> None:
        source = FilesystemLabelSource(tmp_path, "{patch_id}_mask.tif")
        assert source.source_type == "filesystem"
        assert source.labels_dir == tmp_path.resolve()

    def test_missing_placeholder_raises(self, tmp_path: Path) -> None:
        with pytest.raises(InvalidValueError, match="patch_id"):
            FilesystemLabelSource(tmp_path, "mask.tif")

    def test_from_config(self, tmp_path: Path) -> None:
        config = _config(tmp_path, pattern="{patch_id}_lbl.tif")
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        source = FilesystemLabelSource.from_config(config, labels_dir)
        assert source.source_type == "filesystem"

    def test_from_config_default_pattern(self, tmp_path: Path) -> None:
        from src.core.config import Config
        data = make_valid_config()
        config = Config(config_path=write_config(tmp_path, data))
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        source = FilesystemLabelSource.from_config(config, labels_dir)
        records = source.discover(("p1",))
        assert records[0].mask_path is None or records[0].mask_path.name == "p1_mask.tif"


class TestFilesystemLabelSourceDiscover:
    def test_discovers_existing_masks(self, tmp_path: Path) -> None:
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        (labels_dir / "p1_mask.tif").write_bytes(b"fake")
        source = FilesystemLabelSource(labels_dir, "{patch_id}_mask.tif")
        records = source.discover(("p1",))
        assert records[0].found is True
        assert records[0].mask_path == (labels_dir / "p1_mask.tif").resolve()

    def test_missing_mask_returns_not_found(self, tmp_path: Path) -> None:
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        source = FilesystemLabelSource(labels_dir, "{patch_id}_mask.tif")
        records = source.discover(("missing_patch",))
        assert records[0].found is False
        assert records[0].mask_path is None

    def test_preserves_order(self, tmp_path: Path) -> None:
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        (labels_dir / "a_mask.tif").write_bytes(b"x")
        (labels_dir / "c_mask.tif").write_bytes(b"x")
        source = FilesystemLabelSource(labels_dir, "{patch_id}_mask.tif")
        records = source.discover(("a", "b", "c"))
        assert [r.patch_id for r in records] == ["a", "b", "c"]
        assert [r.found for r in records] == [True, False, True]

    def test_source_type_in_records(self, tmp_path: Path) -> None:
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        source = FilesystemLabelSource(labels_dir, "{patch_id}_mask.tif")
        records = source.discover(("x",))
        assert records[0].source_type == "filesystem"

    def test_empty_patch_ids_returns_empty(self, tmp_path: Path) -> None:
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        source = FilesystemLabelSource(labels_dir, "{patch_id}_mask.tif")
        assert source.discover(()) == ()

    def test_custom_pattern(self, tmp_path: Path) -> None:
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        (labels_dir / "scene1_label.tif").write_bytes(b"x")
        source = FilesystemLabelSource(labels_dir, "{patch_id}_label.tif")
        records = source.discover(("scene1",))
        assert records[0].found is True


class TestLabelSourceIsAbstract:
    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            LabelSource()  # type: ignore[abstract]