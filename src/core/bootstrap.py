"""
Bootstrap module for the River Morphology Segmentation System.

Orchestrates the complete pre-flight initialization sequence and returns a
validated Config object ready for use by all pipeline modules.

Startup sequence (performed by bootstrap()):
    1. Load environment variables from a .env file (non-fatal if absent).
    2. Validate the runtime environment: Python, PyTorch, CUDA, rasterio.
    3. Load and validate config.yaml through Config.__init__().
    4. Verify the project directory structure without creating anything.
    5. Print the ASCII startup banner to stdout (optional).
    6. Return the Config object.

After bootstrap(), the caller must:
    - Call DirectoryManager(config).create_all() to create any missing dirs.
    - Call add_file_logging_handlers(config) to activate file-based logging.

Typical Colab usage:

    from src.core.bootstrap import bootstrap, add_file_logging_handlers
    from src.core.directories import DirectoryManager

    config = bootstrap("config/config.yaml")
    DirectoryManager(config).create_all()
    add_file_logging_handlers(config)
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path
from typing import Final

from src.core.config import Config
from src.core.directories import DirectoryManager, ProjectStructureReport
from src.core.environment import EnvironmentInfo, validate_environment
from src.core.exceptions import (
    ConfigurationError,
    EnvironmentValidationError,
    RiverMorphologyError,
)

__all__ = [
    "bootstrap",
    "add_file_logging_handlers",
    "print_startup_summary",
]

_LOGGER: logging.Logger = logging.getLogger(__name__)

# Width of all ASCII separator lines in the startup banner.
_BANNER_WIDTH: Final[int] = 70


# ==============================================================================
# Internal helpers
# ==============================================================================

def _load_env_file(env_file: str | Path | None) -> bool:
    """
    Load environment variables from a .env file via python-dotenv.

    Existing shell environment variables are NOT overridden (override=False).
    This ensures that CI/CD pipeline variables always take precedence over
    local .env file values.

    Args:
        env_file: Explicit path to a .env file. If None, load_dotenv()
                  auto-discovers a .env file by walking up from cwd.

    Returns:
        True if a .env file was found and variables were loaded.
        False if the file was not found, python-dotenv is not installed,
        or no changes were produced.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        _LOGGER.warning(
            "python-dotenv is not installed. Environment variables must be "
            "set manually (e.g. export GEE_PROJECT_ID=your-id). "
            "Install with: pip install python-dotenv==1.0.1"
        )
        return False

    if env_file is not None:
        resolved = Path(env_file).resolve()
        if not resolved.exists():
            _LOGGER.warning(
                "Specified .env file not found: %s. Continuing without it.",
                resolved,
            )
            return False
        result: bool = load_dotenv(dotenv_path=resolved, override=False)
        if result:
            _LOGGER.debug("Loaded .env file: %s", resolved)
        return result

    result = load_dotenv(override=False)
    if result:
        _LOGGER.debug("Auto-discovered and loaded .env file from cwd.")
    else:
        _LOGGER.debug("No .env file found. Using existing shell environment variables.")
    return result


def _separator(char: str) -> str:
    """Return a horizontal separator line of _BANNER_WIDTH characters."""
    return char * _BANNER_WIDTH


def _build_env_lines(env: EnvironmentInfo) -> list[str]:
    """
    Build ASCII-formatted lines summarising environment validation results.

    Args:
        env: EnvironmentInfo produced by validate_environment().

    Returns:
        List of strings, one per validated component.
    """
    lines: list[str] = []

    py_str = ".".join(str(v) for v in env.python_version)
    tag    = "[OK]  " if env.python_ok else "[FAIL]"
    lines.append(f"  {tag}  Python {py_str} (required >= 3.11)")

    if env.torch_installed:
        tag = "[OK]  " if env.torch_ok else "[FAIL]"
        lines.append(
            f"  {tag}  PyTorch {env.torch_version} (required >= 2.0)"
        )
    else:
        # lines.append("  [FAIL]  PyTorch is not installed")
        lines.append("[FAIL] PyTorch NOT INSTALLED")

    if env.cuda_available:
        lines.append(
            f"  [OK]    CUDA {env.cuda_version} - "
            f"{env.cuda_device_count} device(s) - {env.cuda_device_name}"
        )
    else:
        lines.append(
            "  [WARN]  CUDA not available"
            " - training will run on CPU (very slow)"
        )

    if env.rasterio_installed:
        tag  = "[OK]  " if env.rasterio_ok else "[FAIL]"
        gdal = (
            f" with GDAL {env.gdal_version}"
            if env.gdal_version else ""
        )
        lines.append(
            f"  {tag}  rasterio {env.rasterio_version}{gdal}"
            f" (required >= 1.3)"
        )
    else:
        lines.append("  [FAIL]  rasterio is not installed")

    return lines


def _build_config_lines(config: Config) -> list[str]:
    """
    Build ASCII-formatted lines summarising the current configuration state.

    Args:
        config: Initialized Config object.

    Returns:
        List of strings describing key configuration values and readiness.
    """
    lines: list[str] = [
        f"  [OK]    Loaded: {config.config_path}",
        f"  [OK]    Model: "
        f"{config.model.architecture} / {config.model.encoder_name}",
    ]

    if config.has_aoi:
        aoi = config.aoi
        lines.append(
            f"  [OK]    AOI: lon=[{aoi.min_lon}, {aoi.max_lon}]"
            f" lat=[{aoi.min_lat}, {aoi.max_lat}]"
        )
    else:
        lines.append(
            "  [INFO]  AOI: not set"
            " - configure aoi.* in config.yaml before GEE download"
        )

    if config.has_date_range:
        lines.append(
            f"  [OK]    Date range: {config.date_range.start}"
            f" to {config.date_range.end}"
        )
    else:
        lines.append(
            "  [INFO]  Date range: not set"
            " - configure date_range.* in config.yaml before GEE download"
        )

    if config.has_normalization_stats:
        lines.append("  [OK]    Normalization stats: computed")
    else:
        lines.append(
            "  [INFO]  Normalization stats: not computed"
            " - run src/preprocessing/normalize.py after first data download"
        )

    return lines


def _build_dir_lines(report: ProjectStructureReport) -> list[str]:
    """
    Build ASCII-formatted lines summarising project directory verification.

    Args:
        report: ProjectStructureReport from DirectoryManager.verify_all().

    Returns:
        Per-directory status lines plus a footer summary.
    """
    lines: list[str] = report.summary_lines()
    lines.append("")

    if report.all_exist and report.all_writable:
        lines.append(
            f"  All {report.total_count} directories present and writable."
        )
    else:
        n_miss = len(report.missing)
        if n_miss:
            noun = "directory" if n_miss == 1 else "directories"
            lines.append(
                f"  {n_miss} {noun} missing."
                " Call DirectoryManager(config).create_all() to create them."
            )
        n_bad = len(report.non_writable)
        if n_bad:
            noun = "directory" if n_bad == 1 else "directories"
            lines.append(
                f"  {n_bad} {noun} not writable."
                " Check filesystem permissions."
            )

    return lines


# ==============================================================================
# Public API
# ==============================================================================

def add_file_logging_handlers(config: Config) -> None:
    """
    Attach rotating file handlers to the 'src' logger if not already present.

    Must be called AFTER DirectoryManager.create_all() because the logs
    directory must exist before RotatingFileHandler can be instantiated.
    Safe to call multiple times -- duplicate handlers are never added.

    Handler configuration matches logging.yaml:
        - file handler:       DEBUG+, 10 MB per file, 5 rotating backups
        - error_file handler: ERROR+, 5 MB per file, 3 rotating backups

    Args:
        config: Initialized Config object. config.paths.logs_dir must exist
                on disk before this function is called.
    """
    logs_dir: Path = config.paths.logs_dir

    if not logs_dir.is_dir():
        _LOGGER.debug(
            "Logs directory does not exist: %s. File handlers not added.",
            logs_dir,
        )
        return

    src_logger = logging.getLogger("src")

    already_has_file_handler = any(
        isinstance(h, logging.handlers.RotatingFileHandler)
        for h in src_logger.handlers
    )
    if already_has_file_handler:
        _LOGGER.debug(
            "RotatingFileHandler already present on 'src' logger. Skipping."
        )
        return

    detailed_fmt = logging.Formatter(
        fmt=(
            "%(asctime)s | %(levelname)-8s | %(name)-40s"
            " | %(filename)s:%(lineno)d | %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    error_fmt = logging.Formatter(
        fmt=(
            "%(asctime)s | %(levelname)-8s | %(name)s"
            " | %(filename)s:%(lineno)d | %(funcName)s() | %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log_path   = logs_dir / config.logging.log_filename
    error_path = logs_dir / config.logging.error_log_filename

    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_path,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
        delay=True,
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_fmt)

    error_handler = logging.handlers.RotatingFileHandler(
        filename=error_path,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
        delay=True,
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(error_fmt)

    src_logger.addHandler(file_handler)
    src_logger.addHandler(error_handler)

    _LOGGER.info(
        "File logging enabled: %s (DEBUG+), %s (ERROR+)",
        log_path,
        error_path,
    )


def print_startup_summary(
    config: Config,
    env: EnvironmentInfo,
    dir_report: ProjectStructureReport,
) -> None:
    """
    Print the ASCII startup banner to standard output.

    Uses only ASCII characters (no Unicode) for Windows compatibility.
    Prints directly to stdout so the banner is always visible regardless
    of the logging system's current handler configuration.

    Args:
        config:     Initialized Config object.
        env:        EnvironmentInfo from validate_environment().
        dir_report: ProjectStructureReport from DirectoryManager.verify_all().
    """
    project_name    = getattr(config.project, "name",    "river_morphology")
    project_version = getattr(config.project, "version", "unknown")

    sections: list[str] = [
        _separator("="),
        f" {project_name} - v{project_version}",
        _separator("="),
        " Environment Validation",
        _separator("-"),
        *_build_env_lines(env),
        "",
        " Configuration",
        _separator("-"),
        *_build_config_lines(config),
        "",
        " Project Directories",
        _separator("-"),
        *_build_dir_lines(dir_report),
        _separator("="),
    ]

    print("\n".join(sections), flush=True)


def bootstrap(
    config_path: str | Path,
    env_file: str | Path | None = None,
    strict_env: bool = False,
    print_summary: bool = True,
) -> Config:
    """
    Initialize the River Morphology Segmentation System.

    Performs all pre-flight checks and returns a validated Config object.
    This function does NOT create any directories and does NOT modify the
    filesystem. Call DirectoryManager(config).create_all() after bootstrap().

    Startup sequence:
        1. Load .env file (non-fatal if absent or python-dotenv not installed).
        2. Validate runtime environment (Python, PyTorch, CUDA, rasterio).
        3. Load and validate config.yaml through Config.__init__().
           Config.__init__ initializes the logging system as a side effect.
        4. Verify project directory structure (read-only).
        5. Print the ASCII startup banner (when print_summary=True).
        6. Return the Config object.

    Args:
        config_path:   Path to config/config.yaml.
        env_file:      Path to .env file containing GEE_PROJECT_ID etc.
                       None means auto-discover .env from the current
                       working directory upward.
        strict_env:    When True, raise EnvironmentValidationError if any
                       critical dependency is missing or incompatible.
                       CUDA absence is always a warning, never a hard error.
        print_summary: Print the ASCII startup banner to stdout. Set False
                       when embedding in Colab cells or during unit tests.

    Returns:
        Fully initialized and validated Config object.

    Raises:
        EnvironmentValidationError: strict_env=True and a critical check fails.
        FileNotFoundError:          config.yaml does not exist at config_path.
        ConfigurationError:         config.yaml is invalid or incomplete.
        RiverMorphologyError:       Any other project-level failure.
    """
    # Step 1: Load .env. Non-fatal regardless of outcome.
    _load_env_file(env_file)

    # Step 2: Validate runtime environment.
    env = validate_environment(strict=strict_env)

    for warning in env.warnings:
        _LOGGER.warning("Environment warning: %s", warning)
    for error in env.errors:
        _LOGGER.error("Environment error: %s", error)

    # Step 3: Load and validate configuration.
    # Config.__init__ initializes the Python logging system as a side effect.
    config = Config(config_path=config_path)

    # Step 4: Verify project directory structure (read-only).
    manager    = DirectoryManager(config)
    dir_report = manager.verify_all()

    # Step 5: Print startup banner.
    if print_summary:
        print_startup_summary(config, env, dir_report)

    _LOGGER.info(
        "Bootstrap complete. config=%s, env_errors=%d, missing_dirs=%d.",
        config.config_path,
        len(env.errors),
        len(dir_report.missing),
    )

    return config