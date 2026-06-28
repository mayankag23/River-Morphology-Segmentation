"""
Unit tests for src/core/directories.py.

Coverage:
    - DirectoryStatus (frozen dataclass)
    - ProjectStructureReport (properties and summary_lines)
    - DirectoryManager (create_all, verify_all, repair, check_write_permissions)

Run:
    pytest tests/test_directories.py -v
    pytest tests/test_directories.py -v --cov=src/core/directories --cov-report=term-missing
"""

from __future__ import annotations

import logging
import os
import stat
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from src.core.directories import (
    REQUIRED_DIRECTORY_KEYS,
    DirectoryManager,
    DirectoryStatus,
    ProjectStructureReport,
)
from src.core.exceptions import ConfigurationError
from tests.conftest import make_valid_config, write_config


# ==============================================================================
# Helpers
# ==============================================================================

def _make_status(
    key: str = "logs_dir",
    path: Path | None = None,
    exists: bool = True,
    is_writable: bool = True,
    was_created: bool = False,
) -> DirectoryStatus:
    """Build a DirectoryStatus with sensible defaults for testing."""
    return DirectoryStatus(
        key=key,
        path=path or Path("/tmp/test"),
        exists=exists,
        is_writable=is_writable,
        was_created=was_created,
    )


def _make_report(statuses: list[DirectoryStatus]) -> ProjectStructureReport:
    return ProjectStructureReport(statuses=statuses)


# ==============================================================================
# DirectoryStatus tests
# ==============================================================================

class TestDirectoryStatus:
    """Tests for the frozen DirectoryStatus dataclass."""

    def test_attributes_set_correctly(self, tmp_path: Path) -> None:
        status = DirectoryStatus(
            key="logs_dir",
            path=tmp_path,
            exists=True,
            is_writable=True,
            was_created=False,
        )
        assert status.key         == "logs_dir"
        assert status.path        == tmp_path
        assert status.exists      is True
        assert status.is_writable is True
        assert status.was_created is False

    def test_frozen_prevents_mutation(self) -> None:
        status = _make_status()
        with pytest.raises((AttributeError, TypeError)):
            status.exists = False  # type: ignore[misc]

    def test_not_writable_when_not_exists(self) -> None:
        status = _make_status(exists=False, is_writable=False)
        assert status.is_writable is False

    def test_was_created_false_by_default(self) -> None:
        status = _make_status(was_created=False)
        assert status.was_created is False

    def test_was_created_true(self) -> None:
        status = _make_status(was_created=True)
        assert status.was_created is True


# ==============================================================================
# ProjectStructureReport tests
# ==============================================================================

class TestProjectStructureReport:
    """Tests for ProjectStructureReport computed properties and summary output."""

    def test_all_exist_true_when_all_present(self) -> None:
        report = _make_report([
            _make_status("a", exists=True),
            _make_status("b", exists=True),
        ])
        assert report.all_exist is True

    def test_all_exist_false_when_any_missing(self) -> None:
        report = _make_report([
            _make_status("a", exists=True),
            _make_status("b", exists=False, is_writable=False),
        ])
        assert report.all_exist is False

    def test_all_exist_false_when_empty(self) -> None:
        report = _make_report([])
        assert report.all_exist is False

    def test_all_writable_true_when_all_exist_and_writable(self) -> None:
        report = _make_report([
            _make_status("a", exists=True, is_writable=True),
            _make_status("b", exists=True, is_writable=True),
        ])
        assert report.all_writable is True

    def test_all_writable_false_when_some_not_writable(self) -> None:
        report = _make_report([
            _make_status("a", exists=True,  is_writable=True),
            _make_status("b", exists=True,  is_writable=False),
        ])
        assert report.all_writable is False

    def test_all_writable_false_when_none_exist(self) -> None:
        report = _make_report([
            _make_status("a", exists=False, is_writable=False),
        ])
        assert report.all_writable is False

    def test_missing_returns_correct_subset(self) -> None:
        present = _make_status("a", exists=True)
        absent  = _make_status("b", exists=False, is_writable=False)
        report  = _make_report([present, absent])
        assert report.missing == [absent]

    def test_missing_empty_when_all_present(self) -> None:
        report = _make_report([_make_status("a", exists=True)])
        assert report.missing == []

    def test_non_writable_returns_correct_subset(self) -> None:
        ok     = _make_status("a", exists=True,  is_writable=True)
        bad    = _make_status("b", exists=True,  is_writable=False)
        report = _make_report([ok, bad])
        assert report.non_writable == [bad]

    def test_created_count_correct(self) -> None:
        report = _make_report([
            _make_status("a", was_created=True),
            _make_status("b", was_created=True),
            _make_status("c", was_created=False),
        ])
        assert report.created_count == 2

    def test_total_count_correct(self) -> None:
        report = _make_report([_make_status("a"), _make_status("b")])
        assert report.total_count == 2

    def test_total_count_zero_for_empty(self) -> None:
        report = _make_report([])
        assert report.total_count == 0

    def test_summary_lines_count_matches_statuses(self) -> None:
        report = _make_report([
            _make_status("a"),
            _make_status("b"),
            _make_status("c"),
        ])
        assert len(report.summary_lines()) == 3

    def test_summary_lines_ok_tag(self) -> None:
        report  = _make_report([_make_status("a", exists=True, is_writable=True)])
        lines   = report.summary_lines()
        assert "[OK]" in lines[0]

    def test_summary_lines_missing_tag(self) -> None:
        report = _make_report([
            _make_status("a", exists=False, is_writable=False)
        ])
        lines = report.summary_lines()
        assert "[MISSING]" in lines[0]

    def test_summary_lines_no_write_tag(self) -> None:
        report = _make_report([
            _make_status("a", exists=True, is_writable=False)
        ])
        lines = report.summary_lines()
        assert "[NO-WRITE]" in lines[0]

    def test_summary_lines_created_tag(self) -> None:
        report = _make_report([
            _make_status("a", exists=True, is_writable=True, was_created=True)
        ])
        lines = report.summary_lines()
        assert "[CREATED]" in lines[0]

    def test_summary_lines_ascii_only(self) -> None:
        report = _make_report([
            _make_status("a", exists=True),
            _make_status("b", exists=False, is_writable=False),
        ])
        for line in report.summary_lines():
            assert all(ord(c) < 128 for c in line), (
                f"Non-ASCII character found in summary line: {line!r}"
            )

    def test_summary_lines_contain_key(self) -> None:
        report = _make_report([_make_status("my_special_dir")])
        assert any("my_special_dir" in line for line in report.summary_lines())


# ==============================================================================
# REQUIRED_DIRECTORY_KEYS tests
# ==============================================================================

class TestRequiredDirectoryKeys:
    """Tests for the REQUIRED_DIRECTORY_KEYS constant."""

    def test_is_tuple(self) -> None:
        assert isinstance(REQUIRED_DIRECTORY_KEYS, tuple)

    def test_non_empty(self) -> None:
        assert len(REQUIRED_DIRECTORY_KEYS) > 0

    def test_contains_logs_dir(self) -> None:
        assert "logs_dir" in REQUIRED_DIRECTORY_KEYS

    def test_contains_checkpoints_dir(self) -> None:
        assert "checkpoints_dir" in REQUIRED_DIRECTORY_KEYS

    def test_parent_before_child(self) -> None:
        keys = list(REQUIRED_DIRECTORY_KEYS)
        pairs = [
            ("data_dir",       "raw_dir"),
            ("data_dir",       "patches_dir"),
            ("patches_dir",    "patches_images_dir"),
            ("outputs_dir",    "geotiffs_dir"),
            ("logs_dir",       "tensorboard_dir"),
        ]
        for parent, child in pairs:
            assert keys.index(parent) < keys.index(child), (
                f"Expected '{parent}' to appear before '{child}' "
                f"in REQUIRED_DIRECTORY_KEYS"
            )

    def test_no_duplicates(self) -> None:
        assert len(REQUIRED_DIRECTORY_KEYS) == len(set(REQUIRED_DIRECTORY_KEYS))


# ==============================================================================
# DirectoryManager tests
# ==============================================================================

class TestDirectoryManagerCreateAll:
    """Tests for DirectoryManager.create_all()."""

    def test_creates_all_directories(self, config, tmp_path: Path) -> None:
        manager = DirectoryManager(config)
        report  = manager.create_all()
        assert report.all_exist is True

    def test_all_created_directories_are_real(self, config, tmp_path: Path) -> None:
        manager = DirectoryManager(config)
        manager.create_all()
        for key in REQUIRED_DIRECTORY_KEYS:
            path = getattr(config.paths, key)
            assert path.is_dir(), f"Directory not created: {key} -> {path}"

    def test_create_all_is_idempotent(self, config, tmp_path: Path) -> None:
        manager = DirectoryManager(config)
        manager.create_all()
        report2 = manager.create_all()
        assert report2.all_exist is True

    def test_was_created_flag_on_first_call(self, config, tmp_path: Path) -> None:
        manager = DirectoryManager(config)
        report  = manager.create_all()
        assert report.created_count > 0

    def test_was_created_false_on_second_call(self, config, tmp_path: Path) -> None:
        manager = DirectoryManager(config)
        manager.create_all()
        report2 = manager.create_all()
        assert report2.created_count == 0

    def test_report_total_count_matches_keys(self, config, tmp_path: Path) -> None:
        manager = DirectoryManager(config)
        report  = manager.create_all()
        assert report.total_count == len(REQUIRED_DIRECTORY_KEYS)

    def test_all_directories_writable_after_create(
        self, config, tmp_path: Path
    ) -> None:
        manager = DirectoryManager(config)
        report  = manager.create_all()
        assert report.all_writable is True

    def test_invalid_key_raises_configuration_error(
        self, config, tmp_path: Path
    ) -> None:
        manager = DirectoryManager(config)
        with pytest.raises(ConfigurationError, match="config.paths.nonexistent_key"):
            manager._resolve_path("nonexistent_key")


class TestDirectoryManagerVerifyAll:
    """Tests for DirectoryManager.verify_all()."""

    def test_detects_all_existing_dirs(self, config, tmp_path: Path) -> None:
        manager = DirectoryManager(config)
        manager.create_all()
        report = manager.verify_all()
        assert report.all_exist is True

    def test_detects_missing_dirs(self, config, tmp_path: Path) -> None:
        manager = DirectoryManager(config)
        report  = manager.verify_all()
        assert len(report.missing) > 0

    def test_verify_does_not_create_dirs(self, config, tmp_path: Path) -> None:
        manager   = DirectoryManager(config)
        manager.verify_all()
        logs_path = config.paths.logs_dir
        assert not logs_path.exists(), (
            "verify_all() must not create any directories"
        )

    def test_was_created_always_false_in_verify(
        self, config, tmp_path: Path
    ) -> None:
        manager = DirectoryManager(config)
        manager.create_all()
        report = manager.verify_all()
        assert all(not s.was_created for s in report.statuses)

    def test_report_contains_entry_for_every_key(
        self, config, tmp_path: Path
    ) -> None:
        manager = DirectoryManager(config)
        report  = manager.verify_all()
        reported_keys = {s.key for s in report.statuses}
        for key in REQUIRED_DIRECTORY_KEYS:
            assert key in reported_keys


class TestDirectoryManagerRepair:
    """Tests for DirectoryManager.repair()."""

    def test_repair_creates_missing_dirs(self, config, tmp_path: Path) -> None:
        manager = DirectoryManager(config)
        report  = manager.repair()
        assert report.all_exist is True

    def test_repair_returns_all_exist_after_completion(
        self, config, tmp_path: Path
    ) -> None:
        manager = DirectoryManager(config)
        report  = manager.repair()
        assert report.all_exist is True

    def test_repair_does_not_affect_existing_dirs(
        self, config, tmp_path: Path
    ) -> None:
        manager = DirectoryManager(config)
        manager.create_all()
        sentinel = config.paths.logs_dir / "sentinel.txt"
        sentinel.write_text("sentinel")
        manager.repair()
        assert sentinel.exists(), (
            "repair() must not delete or recreate existing directories"
        )

    def test_repair_when_no_missing_returns_existing_report(
        self, config, tmp_path: Path
    ) -> None:
        manager = DirectoryManager(config)
        manager.create_all()
        report = manager.repair()
        assert report.all_exist is True
        assert report.created_count == 0

    def test_repair_only_missing_subset(self, config, tmp_path: Path) -> None:
        manager = DirectoryManager(config)
        manager.create_all()
        # Remove one directory to simulate a missing dir.
        import shutil
        shutil.rmtree(config.paths.logs_dir)
        assert not config.paths.logs_dir.exists()
        report = manager.repair()
        assert config.paths.logs_dir.exists()
        assert report.all_exist is True


class TestDirectoryManagerWritePermissions:
    """Tests for DirectoryManager.check_write_permissions()."""

    def test_empty_list_when_all_writable(self, config, tmp_path: Path) -> None:
        manager = DirectoryManager(config)
        manager.create_all()
        issues = manager.check_write_permissions()
        assert issues == []

    def test_returns_list_of_strings(self, config, tmp_path: Path) -> None:
        manager = DirectoryManager(config)
        manager.create_all()
        issues = manager.check_write_permissions()
        assert isinstance(issues, list)
        for item in issues:
            assert isinstance(item, str)

    def test_issues_contain_ascii_only(self, config, tmp_path: Path) -> None:
        manager = DirectoryManager(config)
        manager.create_all()
        for issue in manager.check_write_permissions():
            assert all(ord(c) < 128 for c in issue)

    @pytest.mark.skipif(
        os.getuid() == 0,
        reason="Root user bypasses permission checks",
    ) if hasattr(os, "getuid") else pytest.mark.skip(
        reason="os.getuid not available on this platform"
    )
    def test_detects_non_writable_directory(
        self, config, tmp_path: Path
    ) -> None:
        manager = DirectoryManager(config)
        manager.create_all()
        target = config.paths.logs_dir
        original_mode = target.stat().st_mode
        try:
            target.chmod(0o444)
            issues = manager.check_write_permissions()
            assert any("logs_dir" in issue for issue in issues)
        finally:
            target.chmod(original_mode)