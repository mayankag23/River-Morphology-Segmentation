"""
Directory management for the River Morphology Segmentation System.

Provides DirectoryManager, which creates, verifies, and repairs all required
project directories. All paths are derived exclusively from the Config object;
no paths are hardcoded in this module.

Typical usage:

    from src.core.config import Config
    from src.core.directories import DirectoryManager

    config = Config("config/config.yaml")
    manager = DirectoryManager(config)

    # First run: create all directories.
    report = manager.create_all()

    # Subsequent runs: verify and repair if needed.
    report = manager.verify_all()
    if not report.all_exist:
        report = manager.repair()

    # Check write access.
    issues = manager.check_write_permissions()
    if issues:
        for issue in issues:
            print(issue)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from src.core.config import Config
from src.core.exceptions import ConfigurationError

__all__ = [
    "DirectoryManager",
    "DirectoryStatus",
    "ProjectStructureReport",
    "REQUIRED_DIRECTORY_KEYS",
]

_LOGGER: logging.Logger = logging.getLogger(__name__)

# Ordered tuple of config.paths attribute names for all required directories.
# Ordering is significant: parent directories must appear before their children
# so that creation order is always correct (e.g. "data_dir" before "raw_dir").
REQUIRED_DIRECTORY_KEYS: Final[tuple[str, ...]] = (
    "data_dir",
    "raw_dir",
    "processed_dir",
    "patches_dir",
    "patches_images_dir",
    "patches_masks_dir",
    "splits_dir",
    "checkpoints_dir",
    "outputs_dir",
    "geotiffs_dir",
    "shapefiles_dir",
    "visualizations_dir",
    "logs_dir",
    "tensorboard_dir",
)


@dataclass(frozen=True)
class DirectoryStatus:
    """
    Immutable status snapshot for a single project directory.

    Attributes:
        key:         Config paths attribute name, e.g. "logs_dir".
        path:        Absolute resolved filesystem path.
        exists:      True if the path exists and is a directory on disk.
        is_writable: True if the current process has write permission.
                     Always False when exists is False.
        was_created: True if this particular operation created the directory.
                     Always False in results from verify_all().
    """

    key:         str
    path:        Path
    exists:      bool
    is_writable: bool
    was_created: bool


@dataclass
class ProjectStructureReport:
    """
    Aggregate status report for all required project directories.

    Produced by DirectoryManager.create_all(), verify_all(), or repair().
    All computed properties are derived from the statuses list.

    Attributes:
        statuses: Ordered list of DirectoryStatus, one per required directory.
    """

    statuses: list[DirectoryStatus] = field(default_factory=list)

    @property
    def all_exist(self) -> bool:
        """True if every required directory exists on disk."""
        return bool(self.statuses) and all(s.exists for s in self.statuses)

    @property
    def all_writable(self) -> bool:
        """
        True if every existing directory is writable by the current process.

        Directories that do not exist are excluded from this check. Combine
        with all_exist to confirm complete readiness.
        """
        existing = [s for s in self.statuses if s.exists]
        return bool(existing) and all(s.is_writable for s in existing)

    @property
    def missing(self) -> list[DirectoryStatus]:
        """DirectoryStatus entries for directories that do not exist."""
        return [s for s in self.statuses if not s.exists]

    @property
    def non_writable(self) -> list[DirectoryStatus]:
        """DirectoryStatus entries for directories that exist but are not writable."""
        return [s for s in self.statuses if s.exists and not s.is_writable]

    @property
    def created_count(self) -> int:
        """Number of directories created during this operation."""
        return sum(1 for s in self.statuses if s.was_created)

    @property
    def total_count(self) -> int:
        """Total number of directories tracked in this report."""
        return len(self.statuses)

    def summary_lines(self) -> list[str]:
        """
        Return one ASCII-formatted status line per tracked directory.

        Tags are fixed-width for aligned console output:
            [OK]       - exists and is writable
            [CREATED]  - was just created in this operation
            [MISSING]  - does not exist on disk
            [NO-WRITE] - exists but the process cannot write to it
        """
        lines: list[str] = []
        for status in self.statuses:
            if not status.exists:
                tag = "[MISSING] "
            elif not status.is_writable:
                tag = "[NO-WRITE]"
            elif status.was_created:
                tag = "[CREATED] "
            else:
                tag = "[OK]      "
            lines.append(f"  {tag}  {status.key}: {status.path}")
        return lines


class DirectoryManager:
    """
    Creates, verifies, and repairs all required project directories.

    All directory paths are read from config.paths; the canonical set of
    required directories is defined by REQUIRED_DIRECTORY_KEYS. Nothing is
    hardcoded. The Config object must have been initialized by Config.__init__
    so that all paths attributes are absolute Path objects.

    Args:
        config: Fully initialized Config object with resolved Path attributes.

    Raises:
        ConfigurationError: Raised by _resolve_path() if a key in
                            REQUIRED_DIRECTORY_KEYS is absent from config.paths,
                            which indicates a mismatch between the code and config.
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._logger: logging.Logger = logging.getLogger(__name__)

    def _resolve_path(self, key: str) -> Path:
        """
        Retrieve the resolved Path for a config.paths attribute name.

        Args:
            key: Attribute name in config.paths, e.g. "logs_dir".

        Returns:
            Absolute Path object.

        Raises:
            ConfigurationError: If the key is absent from config.paths or is
                                 not a Path object, indicating that the config.yaml
                                 paths section is missing the corresponding entry.
        """
        value = getattr(self._config.paths, key, None)
        if value is None or not isinstance(value, Path):
            raise ConfigurationError(
                f"config.paths.{key} is not defined or is not a Path object. "
                f"Ensure '{key}' is present under the 'paths' section in "
                f"config.yaml and that Config has been fully initialized."
            )
        return value

    def _check_writable(self, path: Path) -> bool:
        """
        Check if the current process has write permission for an existing path.

        Uses os.access() which respects effective UID/GID on POSIX systems
        and ACL-based permissions on Windows.

        Args:
            path: A path that exists on disk. Behaviour is undefined for
                  paths that do not exist.

        Returns:
            True if the process can write to path, False otherwise.
        """
        return os.access(path, os.W_OK)

    def _build_status(self, key: str, *, was_created: bool) -> DirectoryStatus:
        """
        Build an immutable DirectoryStatus snapshot for a single directory.

        Args:
            key:         Config paths attribute name.
            was_created: Whether this specific operation created the directory.

        Returns:
            Frozen DirectoryStatus instance reflecting current disk state.
        """
        path   = self._resolve_path(key)
        exists = path.is_dir()
        return DirectoryStatus(
            key=key,
            path=path,
            exists=exists,
            is_writable=self._check_writable(path) if exists else False,
            was_created=was_created,
        )

    def create_all(self) -> ProjectStructureReport:
        """
        Create all required project directories.

        Directories that already exist are left unchanged (exist_ok=True).
        Parent directories are created automatically (parents=True). The
        creation order follows REQUIRED_DIRECTORY_KEYS, ensuring parents are
        always created before their children.

        Returns:
            ProjectStructureReport with the state of each directory after all
            creation attempts. Inspect report.all_exist and report.all_writable
            to confirm that the project is ready.

        Raises:
            ConfigurationError: A required key is missing from config.paths.
            PermissionError:    The OS denied permission to create a directory.
            OSError:            Any other filesystem-level failure.
        """
        statuses: list[DirectoryStatus] = []

        for key in REQUIRED_DIRECTORY_KEYS:
            path           = self._resolve_path(key)
            existed_before = path.is_dir()

            try:
                path.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                self._logger.error(
                    "Permission denied creating [%s]: %s", key, path
                )
                raise
            except OSError as exc:
                self._logger.error(
                    "OS error creating [%s] %s: %s", key, path, exc
                )
                raise

            was_created = not existed_before and path.is_dir()
            if was_created:
                self._logger.debug("Created: [%s] %s", key, path)

            statuses.append(self._build_status(key, was_created=was_created))

        report = ProjectStructureReport(statuses=statuses)
        self._logger.info(
            "Directory setup: %d created, %d total, "
            "all_exist=%s, all_writable=%s",
            report.created_count,
            report.total_count,
            report.all_exist,
            report.all_writable,
        )
        return report

    def verify_all(self) -> ProjectStructureReport:
        """
        Check the current state of all required directories without modifying anything.

        Logs warnings for missing directories and non-writable directories.
        Use create_all() or repair() to correct issues found by this method.

        Returns:
            ProjectStructureReport reflecting the current filesystem state.

        Raises:
            ConfigurationError: A required key is missing from config.paths.
        """
        statuses = [
            self._build_status(key, was_created=False)
            for key in REQUIRED_DIRECTORY_KEYS
        ]
        report = ProjectStructureReport(statuses=statuses)

        if report.missing:
            self._logger.warning(
                "%d missing director%s: %s",
                len(report.missing),
                "y" if len(report.missing) == 1 else "ies",
                [s.key for s in report.missing],
            )

        if report.non_writable:
            self._logger.warning(
                "%d non-writable director%s: %s",
                len(report.non_writable),
                "y" if len(report.non_writable) == 1 else "ies",
                [s.key for s in report.non_writable],
            )

        return report

    def repair(self) -> ProjectStructureReport:
        """
        Create any missing directories without modifying existing ones.

        Calls verify_all() to identify gaps, then creates only the missing
        directories. Returns the result of a fresh verify_all() call after
        repair attempts, so the caller receives an up-to-date state report.

        Returns:
            ProjectStructureReport from a fresh verify_all() after repair.
            If no directories were missing, the initial report is returned.

        Raises:
            ConfigurationError: A required key is missing from config.paths.
            PermissionError:    The OS denied permission to create a directory.
            OSError:            Any other filesystem-level failure during repair.
        """
        initial = self.verify_all()

        if not initial.missing:
            self._logger.info("Repair not needed: all directories are present.")
            return initial

        n_missing = len(initial.missing)
        self._logger.info(
            "Repairing %d missing director%s.",
            n_missing,
            "y" if n_missing == 1 else "ies",
        )

        for status in initial.missing:
            try:
                status.path.mkdir(parents=True, exist_ok=True)
                self._logger.info("Repaired: [%s] %s", status.key, status.path)
            except PermissionError:
                self._logger.error(
                    "Permission denied repairing [%s]: %s",
                    status.key, status.path,
                )
                raise
            except OSError as exc:
                self._logger.error(
                    "Failed to repair [%s] %s: %s", status.key, status.path, exc
                )
                raise

        return self.verify_all()

    def check_write_permissions(self) -> list[str]:
        """
        Report write permission failures for all existing directories.

        Returns:
            List of ASCII problem description strings. An empty list means
            every existing directory is writable by the current process.
        """
        report = self.verify_all()
        return [
            f"No write permission: {s.key} -> {s.path}"
            for s in report.statuses
            if s.exists and not s.is_writable
        ]