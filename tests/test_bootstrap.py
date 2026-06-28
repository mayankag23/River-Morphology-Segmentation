"""
Unit tests for src/core/bootstrap.py.

Coverage:
    - _load_env_file (tested via bootstrap with mocking)
    - add_file_logging_handlers
    - print_startup_summary
    - bootstrap

Run:
    pytest tests/test_bootstrap.py -v
    pytest tests/test_bootstrap.py -v --cov=src/core/bootstrap --cov-report=term-missing
"""

from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.core.bootstrap import (
    _build_config_lines,
    _build_dir_lines,
    _build_env_lines,
    add_file_logging_handlers,
    bootstrap,
    print_startup_summary,
)
from src.core.config import Config
from src.core.directories import DirectoryManager, ProjectStructureReport
from src.core.environment import EnvironmentInfo
from src.core.exceptions import ConfigurationError, EnvironmentValidationError
from tests.conftest import make_valid_config, write_config


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def env_info_ok() -> EnvironmentInfo:
    """Return an EnvironmentInfo representing a fully working environment."""
    return EnvironmentInfo(
        python_version=(3, 11, 0),
        python_ok=True,
        torch_installed=True,
        torch_version="2.2.2",
        torch_ok=True,
        cuda_available=True,
        cuda_version="12.1",
        cuda_device_count=1,
        cuda_device_name="Tesla T4",
        rasterio_installed=True,
        rasterio_version="1.3.9",
        rasterio_ok=True,
        gdal_version="3.6.4",
        warnings=[],
        errors=[],
    )


@pytest.fixture
def env_info_no_cuda() -> EnvironmentInfo:
    """Return an EnvironmentInfo with CUDA unavailable."""
    return EnvironmentInfo(
        python_version=(3, 11, 0),
        python_ok=True,
        torch_installed=True,
        torch_version="2.2.2",
        torch_ok=True,
        cuda_available=False,
        cuda_version=None,
        cuda_device_count=0,
        cuda_device_name=None,
        rasterio_installed=True,
        rasterio_version="1.3.9",
        rasterio_ok=True,
        gdal_version="3.6.4",
        warnings=["CUDA not available"],
        errors=[],
    )


@pytest.fixture
def env_info_errors() -> EnvironmentInfo:
    """Return an EnvironmentInfo with critical errors."""
    return EnvironmentInfo(
        python_version=(3, 9, 0),
        python_ok=False,
        torch_installed=False,
        torch_version=None,
        torch_ok=False,
        cuda_available=False,
        cuda_version=None,
        cuda_device_count=0,
        cuda_device_name=None,
        rasterio_installed=False,
        rasterio_version=None,
        rasterio_ok=False,
        gdal_version=None,
        warnings=[],
        errors=["Python too old", "PyTorch not installed", "rasterio not installed"],
    )


@pytest.fixture
def src_logger_cleanup():
    """Remove RotatingFileHandlers from the 'src' logger after each test."""
    yield
    src_logger = logging.getLogger("src")
    to_remove = [
        h for h in list(src_logger.handlers)
        if isinstance(h, logging.handlers.RotatingFileHandler)
    ]
    for handler in to_remove:
        src_logger.removeHandler(handler)
        handler.close()


# ==============================================================================
# _build_env_lines tests
# ==============================================================================

class TestBuildEnvLines:
    """Tests for the _build_env_lines internal helper."""

    def test_returns_list_of_strings(self, env_info_ok: EnvironmentInfo) -> None:
        lines = _build_env_lines(env_info_ok)
        assert isinstance(lines, list)
        assert all(isinstance(line, str) for line in lines)

    def test_contains_python_version(self, env_info_ok: EnvironmentInfo) -> None:
        lines = _build_env_lines(env_info_ok)
        assert any("Python" in line for line in lines)
        assert any("3.11.0" in line for line in lines)

    def test_ok_tag_when_all_good(self, env_info_ok: EnvironmentInfo) -> None:
        lines = _build_env_lines(env_info_ok)
        assert any("[OK]" in line for line in lines)

    def test_fail_tag_when_python_old(
        self, env_info_errors: EnvironmentInfo
    ) -> None:
        lines = _build_env_lines(env_info_errors)
        assert any("[FAIL]" in line and "Python" in line for line in lines)

    def test_warn_tag_when_no_cuda(
        self, env_info_no_cuda: EnvironmentInfo
    ) -> None:
        lines = _build_env_lines(env_info_no_cuda)
        assert any("[WARN]" in line for line in lines)

    def test_ascii_only(self, env_info_ok: EnvironmentInfo) -> None:
        lines = _build_env_lines(env_info_ok)
        for line in lines:
            assert all(ord(c) < 128 for c in line), (
            f"Non-ASCII character in line: {line!r}"
        )

    def test_torch_not_installed(self, env_info_errors: EnvironmentInfo) -> None:
        lines = _build_env_lines(env_info_errors)
        assert any("NOT INSTALLED" in line and "PyTorch" in line for line in lines)
 
    def test_cuda_device_name_shown(self, env_info_ok: EnvironmentInfo) -> None:
        lines = _build_env_lines(env_info_ok)
        assert any("Tesla T4" in line for line in lines)

    # ==============================================================================
    # _build_config_lines tests
    # ==============================================================================

class TestBuildConfigLines:
    """Tests for the _build_config_lines internal helper."""

    def test_returns_list_of_strings(self, config: Config) -> None:
        lines = _build_config_lines(config)
        assert isinstance(lines, list)
        assert all(isinstance(line, str) for line in lines)

    def test_ascii_only(self, config: Config) -> None:
        lines = _build_config_lines(config)
        for line in lines:
            assert all(ord(c) < 128 for c in line), (
                f"Non-ASCII character in line: {line!r}"
            )

    def test_shows_architecture(self, config: Config) -> None:
        lines = _build_config_lines(config)
        assert any("UnetPlusPlus" in line for line in lines)

    def test_shows_aoi_info_when_null(self, config: Config) -> None:
        lines = _build_config_lines(config)
        assert any("AOI" in line for line in lines)
        assert any("not set" in line for line in lines)

    def test_shows_date_range_not_set(self, config: Config) -> None:
        lines = _build_config_lines(config)
        assert any(
            "Date range" in line and "not set" in line
            for line in lines
        )

    def test_shows_norm_stats_not_computed(self, config: Config) -> None:
        lines = _build_config_lines(config)
        assert any(
            "Normalization" in line and "not computed" in line
            for line in lines
        )

    def test_shows_norm_stats_ok_when_set(self, tmp_path: Path) -> None:
        from src.core.config import Config

        data = make_valid_config()
        data["preprocessing"]["channel_means"] = [0.05] * 11
        data["preprocessing"]["channel_stds"] = [0.02] * 11

        cfg = Config(config_path=write_config(tmp_path, data))
        lines = _build_config_lines(cfg)

        assert any(
            "Normalization" in line and "computed" in line
            for line in lines
        )


# ==============================================================================
# _build_dir_lines tests
# ==============================================================================

class TestBuildDirLines:
    """Tests for the _build_dir_lines internal helper."""

    def test_returns_list_of_strings(
        self, config: Config, tmp_path: Path
    ) -> None:
        manager = DirectoryManager(config)
        report = manager.verify_all()
        lines = _build_dir_lines(report)

        assert isinstance(lines, list)
        assert all(isinstance(line, str) for line in lines)

    def test_ascii_only(self, config: Config, tmp_path: Path) -> None:
        manager = DirectoryManager(config)
        report = manager.verify_all()
        lines = _build_dir_lines(report)

        for line in lines:
            assert all(ord(c) < 128 for c in line), (
                f"Non-ASCII character in line: {line!r}"
            )

    def test_shows_missing_count_when_dirs_absent(
        self, config: Config, tmp_path: Path
    ) -> None:
        manager = DirectoryManager(config)
        report = manager.verify_all()
        lines = _build_dir_lines(report)

        combined = " ".join(lines)
        assert "missing" in combined.lower()

    def test_shows_all_present_when_all_exist(
        self, config: Config, tmp_path: Path
    ) -> None:
        manager = DirectoryManager(config)
        manager.create_all()

        report = manager.verify_all()
        lines = _build_dir_lines(report)

        combined = " ".join(lines)
        assert (
            "present" in combined.lower()
            or "writable" in combined.lower()
        )


# ==============================================================================
# add_file_logging_handlers tests
# ==============================================================================

class TestAddFileLoggingHandlers:
    """Tests for add_file_logging_handlers."""

    def test_no_handlers_when_logs_dir_absent(
        self, config: Config, src_logger_cleanup: None
    ) -> None:
        src_logger = logging.getLogger("src")
        initial = len(src_logger.handlers)

        add_file_logging_handlers(config)

        # logs_dir does not exist -> no new handlers
        assert len(src_logger.handlers) == initial

    def test_adds_rotating_handlers_when_logs_dir_exists(
        self, config: Config, tmp_path: Path, src_logger_cleanup: None
    ) -> None:
        DirectoryManager(config).create_all()

        add_file_logging_handlers(config)

        src_logger = logging.getLogger("src")
        file_handlers = [
            h
            for h in src_logger.handlers
            if isinstance(
                h,
                logging.handlers.RotatingFileHandler,
            )
        ]

        assert len(file_handlers) >= 2

    def test_idempotent_when_called_twice(
        self, config: Config, tmp_path: Path, src_logger_cleanup: None
    ) -> None:
        DirectoryManager(config).create_all()

        add_file_logging_handlers(config)

        src_logger = logging.getLogger("src")

        count_after_1 = sum(
            1
            for h in src_logger.handlers
            if isinstance(
                h,
                logging.handlers.RotatingFileHandler,
            )
        )

        add_file_logging_handlers(config)

        count_after_2 = sum(
            1
            for h in src_logger.handlers
            if isinstance(
                h,
                logging.handlers.RotatingFileHandler,
            )
        )

        assert count_after_1 == count_after_2

    def test_file_handler_level_is_debug(
        self, config: Config, tmp_path: Path, src_logger_cleanup: None
    ) -> None:
        DirectoryManager(config).create_all()

        add_file_logging_handlers(config)

        src_logger = logging.getLogger("src")

        handlers = [
            h
            for h in src_logger.handlers
            if isinstance(
                h,
                logging.handlers.RotatingFileHandler,
            )
        ]

        levels = [h.level for h in handlers]
        assert logging.DEBUG in levels

    def test_error_handler_level_is_error(
        self, config: Config, tmp_path: Path, src_logger_cleanup: None
    ) -> None:
        DirectoryManager(config).create_all()

        add_file_logging_handlers(config)

        src_logger = logging.getLogger("src")

        handlers = [
            h
            for h in src_logger.handlers
            if isinstance(
                h,
                logging.handlers.RotatingFileHandler,
            )
        ]

        levels = [h.level for h in handlers]
        assert logging.ERROR in levels
