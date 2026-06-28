# ---

# ### File 7 — `tests/test_main.py`

# ```python
"""
Unit tests for main.py.

Coverage:
    - parse_args
    - _run_check_only
    - main

Run:
    pytest tests/test_main.py -v
    pytest tests/test_main.py -v --cov=main --cov-report=term-missing
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from main import _run_check_only, main, parse_args
from tests.conftest import make_valid_config, write_config


# ==============================================================================
# parse_args tests
# ==============================================================================

class TestParseArgs:
    """Tests for the argument parser."""

    def test_default_config_path(self) -> None:
        args = parse_args([])
        assert args.config == Path("config") / "config.yaml"

    def test_custom_config_path(self) -> None:
        args = parse_args(["--config", "custom/path/config.yaml"])
        assert args.config == Path("custom/path/config.yaml")

    def test_strict_env_default_false(self) -> None:
        args = parse_args([])
        assert args.strict_env is False

    def test_strict_env_flag(self) -> None:
        args = parse_args(["--strict-env"])
        assert args.strict_env is True

    def test_check_only_default_false(self) -> None:
        args = parse_args([])
        assert args.check_only is False

    def test_check_only_flag(self) -> None:
        args = parse_args(["--check-only"])
        assert args.check_only is True

    def test_no_summary_default_false(self) -> None:
        args = parse_args([])
        assert args.no_summary is False

    def test_no_summary_flag(self) -> None:
        args = parse_args(["--no-summary"])
        assert args.no_summary is True

    def test_env_file_default_none(self) -> None:
        args = parse_args([])
        assert args.env_file is None

    def test_env_file_custom(self) -> None:
        args = parse_args(["--env-file", ".env.test"])
        assert args.env_file == Path(".env.test")

    def test_all_args_combined(self) -> None:
        args = parse_args([
            "--config", "my/config.yaml",
            "--env-file", ".env",
            "--strict-env",
            "--check-only",
            "--no-summary",
        ])
        assert args.config    == Path("my/config.yaml")
        assert args.env_file  == Path(".env")
        assert args.strict_env is True
        assert args.check_only is True
        assert args.no_summary is True


# ==============================================================================
# main() exit code tests
# ==============================================================================

class TestMainExitCodes:
    """Tests for main() exit codes under various conditions."""

    def test_returns_zero_on_success(self, valid_config_file: Path) -> None:
        code = main([
            "--config", str(valid_config_file),
            "--no-summary",
        ])
        assert code == 0

    def test_returns_one_on_missing_config(self, tmp_path: Path) -> None:
        code = main([
            "--config", str(tmp_path / "config" / "no.yaml"),
            "--no-summary",
        ])
        assert code == 1

    def test_returns_two_on_strict_env_failure(
        self, valid_config_file: Path
    ) -> None:
        from src.core.exceptions import EnvironmentValidationError
        with patch(
            "main.bootstrap",
            side_effect=EnvironmentValidationError(
                check="test", details="critical failure"
            ),
        ):
            code = main([
                "--config", str(valid_config_file),
                "--no-summary",
                "--strict-env",
            ])
        assert code == 2

    def test_returns_one_on_configuration_error(
        self, tmp_path: Path
    ) -> None:
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        bad = config_dir / "config.yaml"
        bad.write_text("")  # Empty YAML -> ConfigurationError
        code = main(["--config", str(bad), "--no-summary"])
        assert code == 1

    def test_returns_one_on_permission_error(
        self, valid_config_file: Path
    ) -> None:
        with patch(
            "main.DirectoryManager.create_all",
            side_effect=PermissionError("Access denied"),
        ):
            code = main([
                "--config", str(valid_config_file),
                "--no-summary",
            ])
        assert code == 1

    def test_returns_one_on_unexpected_exception(
        self, valid_config_file: Path
    ) -> None:
        with patch("main.bootstrap", side_effect=RuntimeError("Unexpected!")):
            code = main([
                "--config", str(valid_config_file),
                "--no-summary",
            ])
        assert code == 1

    def test_no_summary_suppresses_stdout(
        self, valid_config_file: Path, capsys
    ) -> None:
        main(["--config", str(valid_config_file), "--no-summary"])
        captured = capsys.readouterr()
        assert captured.out == ""


# ==============================================================================
# check-only mode tests
# ==============================================================================

class TestCheckOnlyMode:
    """Tests for main() --check-only mode and _run_check_only()."""

    def test_check_only_returns_one_when_dirs_missing(
        self, valid_config_file: Path
    ) -> None:
        code = main([
            "--config", str(valid_config_file),
            "--check-only",
        ])
        # Directories have not been created -> missing -> exit 1
        assert code == 1

    def test_check_only_returns_zero_when_all_present(
        self, valid_config_file: Path, tmp_path: Path
    ) -> None:
        from src.core.config import Config
        from src.core.directories import DirectoryManager
        config = Config(valid_config_file)
        DirectoryManager(config).create_all()
        code = main([
            "--config", str(valid_config_file),
            "--check-only",
        ])
        assert code == 0

    def test_check_only_does_not_create_directories(
        self, valid_config_file: Path, tmp_path: Path
    ) -> None:
        main(["--config", str(valid_config_file), "--check-only"])
        assert not (tmp_path / "logs").exists()
        assert not (tmp_path / "data").exists()

    def test_run_check_only_missing_config(self, tmp_path: Path) -> None:
        code = _run_check_only(
            config_path=tmp_path / "config" / "no.yaml",
            env_file=None,
        )
        assert code == 1

    def test_run_check_only_returns_int(
        self, valid_config_file: Path
    ) -> None:
        code = _run_check_only(
            config_path=valid_config_file,
            env_file=None,
        )
        assert isinstance(code, int)


# ==============================================================================
# stdout / ASCII safety tests
# ==============================================================================

class TestOutputAsciiSafety:
    """Verify that all output from main.py is ASCII-only."""

    def test_main_output_ascii_only(
        self, valid_config_file: Path, capsys
    ) -> None:
        main(["--config", str(valid_config_file)])
        captured = capsys.readouterr()
        all_output = captured.out + captured.err
        assert all(ord(c) < 128 for c in all_output), (
            "main() must produce only ASCII characters in all output"
        )

    def test_check_only_output_ascii_only(
        self, valid_config_file: Path, capsys
    ) -> None:
        main(["--config", str(valid_config_file), "--check-only"])
        captured = capsys.readouterr()
        all_output = captured.out + captured.err
        assert all(ord(c) < 128 for c in all_output), (
            "--check-only output must be ASCII-only"
        )
# ```

# ---

## 3. Explanation

### Startup sequence
