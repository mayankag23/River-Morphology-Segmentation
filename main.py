#!/usr/bin/env python3
"""
River Morphology Segmentation System - Application Entry Point.

Runs the complete initialization sequence and exits with an appropriate
status code. Use this script for local development and CI verification.

For Google Colab usage, call bootstrap() and DirectoryManager directly from
notebook cells rather than running this script.

Exit codes:
    0 - Initialization successful; all directories exist and are writable.
    1 - General error (config invalid, filesystem failure, permissions).
    2 - Environment validation failure (strict mode only).

Usage:
    python main.py
    python main.py --config path/to/config.yaml
    python main.py --env-file path/to/.env --strict-env
    python main.py --check-only
    python main.py --no-summary
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.core.bootstrap import add_file_logging_handlers, bootstrap
from src.core.directories import DirectoryManager
from src.core.exceptions import (
    ConfigurationError,
    EnvironmentValidationError,
    RiverMorphologyError,
)

_LOGGER: logging.Logger = logging.getLogger(__name__)

_DEFAULT_CONFIG: Path   = Path("config") / "config.yaml"
_SEPARATOR_WIDTH: int   = 70


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse command-line arguments for the application entry point.

    Args:
        argv: Argument list to parse. None means read from sys.argv[1:].

    Returns:
        Parsed Namespace with attributes:
            config (Path), env_file (Path|None), strict_env (bool),
            check_only (bool), no_summary (bool).
    """
    parser = argparse.ArgumentParser(
        prog="river_morphology",
        description=(
            "River Morphology Segmentation System"
            " -- automated semantic segmentation from Landsat imagery."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py\n"
            "  python main.py --config config/config.yaml --strict-env\n"
            "  python main.py --check-only\n"
            "  python main.py --no-summary\n"
        ),
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        metavar="PATH",
        help=(
            "Path to config.yaml (default: config/config.yaml). "
            "Resolved relative to the current working directory."
        ),
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        metavar="PATH",
        dest="env_file",
        help=(
            "Path to .env file containing GEE_PROJECT_ID and optionally "
            "GEE_SERVICE_ACCOUNT_KEY. Auto-discovers .env in cwd if not set."
        ),
    )
    parser.add_argument(
        "--strict-env",
        action="store_true",
        default=False,
        dest="strict_env",
        help=(
            "Exit with code 2 if any critical dependency (Python, PyTorch, "
            "rasterio) is missing or incompatible. CUDA absence is always "
            "a warning, not a strict error."
        ),
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        default=False,
        dest="check_only",
        help=(
            "Run environment and directory checks only. "
            "No directories are created. "
            "Exit 0 if all checks pass, 1 if issues are found."
        ),
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        default=False,
        dest="no_summary",
        help=(
            "Suppress the startup banner and directory setup output. "
            "Errors are still printed to stderr."
        ),
    )

    return parser.parse_args(argv)


def _run_check_only(
    config_path: Path,
    env_file: Path | None,
) -> int:
    """
    Run pre-flight checks and report status without modifying the filesystem.

    Args:
        config_path: Path to config.yaml.
        env_file:    Optional explicit path to a .env file.

    Returns:
        0 if all checks pass, 1 if any issue is detected.
    """
    try:
        config = bootstrap(
            config_path=config_path,
            env_file=env_file,
            strict_env=False,
            print_summary=True,
        )
    except (FileNotFoundError, ConfigurationError, RiverMorphologyError) as exc:
        print(f"\n[ERROR] Check failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"\n[ERROR] Unexpected error during check: {exc}", file=sys.stderr)
        return 1

    manager = DirectoryManager(config)
    report  = manager.verify_all()
    issues  = manager.check_write_permissions()

    if issues:
        print("\n[WARN] Write permission issues:")
        for issue in issues:
            print(f"  - {issue}")

    if report.all_exist and not issues:
        print(
            f"\n[OK] All checks passed"
            f" ({report.total_count} directories verified)."
        )
        return 0

    n_miss = len(report.missing)
    n_bad  = len(report.non_writable)
    print(
        f"\n[WARN] Issues found: {n_miss} missing, {n_bad} not writable."
    )
    return 1


def main(argv: list[str] | None = None) -> int:
    """
    Run the full initialization sequence.

    Sequence:
        1. Parse command-line arguments.
        2. Bootstrap: validate environment, load config, verify dirs.
        3. Create all required directories via DirectoryManager.
        4. Activate file logging now that the logs directory exists.
        5. Report the final status and return the appropriate exit code.

    Args:
        argv: Argument list. None means use sys.argv[1:].

    Returns:
        Exit code: 0 for full success, 1 for general failure,
        2 for strict environment validation failure.
    """
    args = parse_args(argv)

    if args.check_only:
        return _run_check_only(args.config, args.env_file)

    # Step 1: Bootstrap.
    try:
        config = bootstrap(
            config_path=args.config,
            env_file=args.env_file,
            strict_env=args.strict_env,
            print_summary=not args.no_summary,
        )
    except EnvironmentValidationError as exc:
        print(
            f"\n[ERROR] Environment validation failed:\n{exc}",
            file=sys.stderr,
        )
        return 2
    except FileNotFoundError as exc:
        print(
            f"\n[ERROR] Configuration file not found:\n{exc}",
            file=sys.stderr,
        )
        return 1
    except ConfigurationError as exc:
        print(f"\n[ERROR] Configuration error:\n{exc}", file=sys.stderr)
        return 1
    except RiverMorphologyError as exc:
        print(f"\n[ERROR] Initialization error:\n{exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(
            f"\n[ERROR] Unexpected error during bootstrap:\n{exc}",
            file=sys.stderr,
        )
        _LOGGER.exception("Unexpected error during bootstrap")
        return 1

    # Step 2: Create all required directories.
    try:
        manager = DirectoryManager(config)
        report  = manager.create_all()
    except PermissionError as exc:
        print(
            f"\n[ERROR] Permission denied creating directories:\n{exc}",
            file=sys.stderr,
        )
        return 1
    except OSError as exc:
        print(
            f"\n[ERROR] Filesystem error creating directories:\n{exc}",
            file=sys.stderr,
        )
        return 1

    # Step 3: Activate file logging.
    add_file_logging_handlers(config)
    _LOGGER.info("File logging activated after directory creation.")

    # Step 4: Report directory setup results.
    if not args.no_summary:
        print(
            f"\n Directory Setup"
            f" ({report.created_count} created,"
            f" {report.total_count} total)"
        )
        print("-" * _SEPARATOR_WIDTH)
        for line in report.summary_lines():
            print(line)

    # Step 5: Check for post-creation permission issues.
    permission_issues = manager.check_write_permissions()
    if permission_issues:
        for issue in permission_issues:
            _LOGGER.warning("Permission issue: %s", issue)
            if not args.no_summary:
                print(f"  [WARN] {issue}")

    if not args.no_summary:
        status_tag = (
            "[OK]"
            if report.all_exist and report.all_writable
            else "[WARN]"
        )
        print(f"\n {status_tag} Project root: {config.project_root}")
        print("=" * _SEPARATOR_WIDTH)

    _LOGGER.info(
        "Initialization complete. Project root: %s", config.project_root
    )

    return 0 if (report.all_exist and report.all_writable) else 1


if __name__ == "__main__":
    sys.exit(main())