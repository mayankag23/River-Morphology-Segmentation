"""
Runtime environment validation for the River Morphology Segmentation System.

Checks that all required dependencies are installed at compatible versions
and that the hardware configuration meets minimum requirements.

This module is intentionally standalone — it does not import from any other
project module so it can be run before Config is initialized.

Usage:

    from src.core.environment import validate_environment

    # Non-strict: returns info with warnings list, never raises
    env = validate_environment(strict=False)
    print(env.summary())

    # Strict: raises EnvironmentValidationError on any critical failure
    env = validate_environment(strict=True)

    # Run as a script to print a full environment report
    python -m src.core.environment

Attributes checked:
    Python   >= 3.11
    PyTorch  >= 2.0
    CUDA     (optional but strongly recommended for training)
    rasterio >= 1.3 (validates GDAL is accessible)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any

from src.core.exceptions import EnvironmentValidationError

__all__ = ["EnvironmentInfo", "validate_environment"]

# Minimum version requirements
_MIN_PYTHON: tuple[int, int] = (3, 11)
_MIN_TORCH:  tuple[int, int] = (2, 0)
_MIN_RASTERIO: tuple[int, int] = (1, 3)


# ==============================================================================
# EnvironmentInfo
# ==============================================================================

@dataclass
class EnvironmentInfo:
    """
    Structured report of the current runtime environment.

    Produced by validate_environment(). All fields are populated regardless
    of whether dependencies are installed — missing packages are represented
    as None with installed=False flags.

    Attributes:
        python_version:     Tuple of (major, minor, micro) version integers.
        python_ok:          True if Python >= 3.11.
        torch_installed:    True if PyTorch is importable.
        torch_version:      PyTorch version string, e.g. "2.2.2". None if not installed.
        torch_ok:           True if torch_installed and version >= 2.0.
        cuda_available:     True if torch.cuda.is_available() returns True.
        cuda_version:       CUDA toolkit version string, e.g. "12.1". None if unavailable.
        cuda_device_count:  Number of CUDA-capable devices. 0 if unavailable.
        cuda_device_name:   Name of the first CUDA device. None if unavailable.
        rasterio_installed: True if rasterio is importable.
        rasterio_version:   rasterio version string. None if not installed.
        rasterio_ok:        True if rasterio_installed and version >= 1.3.
        gdal_version:       GDAL version string from rasterio. None if unavailable.
        warnings:           List of non-fatal warnings about optional features.
        errors:             List of critical failures that will prevent pipeline execution.
    """

    python_version:     tuple[int, ...]
    python_ok:          bool

    torch_installed:    bool
    torch_version:      str | None
    torch_ok:           bool

    cuda_available:     bool
    cuda_version:       str | None
    cuda_device_count:  int
    cuda_device_name:   str | None

    rasterio_installed: bool
    rasterio_version:   str | None
    rasterio_ok:        bool
    gdal_version:       str | None

    warnings: list[str] = field(default_factory=list)
    errors:   list[str] = field(default_factory=list)

    @property
    def is_gpu_available(self) -> bool:
        """True if CUDA is available and at least one GPU device is present."""
        return self.cuda_available and self.cuda_device_count > 0

    @property
    def is_ready_for_training(self) -> bool:
        """
        True if all critical dependencies are satisfied for model training.
        GPU availability is checked as a warning, not a hard requirement
        (CPU training is supported but very slow).
        """
        return len(self.errors) == 0

    def summary(self) -> str:
        """
        Return a human-readable summary of the environment check results.

        Each line uses a [OK] / [ERROR] / [WARN] prefix:
            [OK] — check passed
            [ERROR] — critical failure
            [WARN] — warning (non-fatal)
        """
        lines: list[str] = ["=" * 60, "Environment Validation Report", "=" * 60]

        # Python
        py_str = ".".join(str(v) for v in self.python_version)
        status = "[OK]" if self.python_ok else "[ERROR]"
        req = f">= {_MIN_PYTHON[0]}.{_MIN_PYTHON[1]}"
        lines.append(f"{status} Python {py_str} (required {req})")

        # PyTorch
        if self.torch_installed:
            status = "[OK]" if self.torch_ok else "[ERROR]"
            req = f">= {_MIN_TORCH[0]}.{_MIN_TORCH[1]}"
            lines.append(f"{status} PyTorch {self.torch_version} (required {req})")
        else:
            lines.append("[ERROR] PyTorch — NOT INSTALLED")

        # CUDA
        if self.cuda_available:
            lines.append(
                f"[OK] CUDA {self.cuda_version} — "
                f"{self.cuda_device_count} device(s) — "
                f"{self.cuda_device_name}"
            )
        else:
            lines.append("[WARN] CUDA - not available (training will run on CPU; very slow)")

        # rasterio / GDAL
        if self.rasterio_installed:
            status = "[OK]" if self.rasterio_ok else "[ERROR]"
            req = f">= {_MIN_RASTERIO[0]}.{_MIN_RASTERIO[1]}"
            gdal_str = f" (GDAL {self.gdal_version})" if self.gdal_version else ""
            lines.append(
                f"{status} rasterio {self.rasterio_version}{gdal_str} (required {req})"
            )
        else:
            lines.append("[ERROR] rasterio — NOT INSTALLED")

        # Warnings
        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  [WARN] {w}")

        # Errors
        if self.errors:
            lines.append("")
            lines.append("Critical Errors:")
            for e in self.errors:
                lines.append(f"  [ERROR] {e}")
        else:
            lines.append("")
            lines.append("[OK] All critical checks passed.")

        lines.append("=" * 60)
        return "\n".join(lines)
    

# ==============================================================================
# Private check helpers
# ==============================================================================

def _check_python() -> dict[str, Any]:
    """Check the running Python version against the minimum requirement."""
    version = sys.version_info
    ok = (version.major, version.minor) >= _MIN_PYTHON
    return {
        "version": (version.major, version.minor, version.micro),
        "ok": ok,
    }


def _check_torch() -> dict[str, Any]:
    """
    Check PyTorch installation, version, and CUDA availability.

    Uses importlib to avoid a hard import dependency — this module must
    remain importable even when torch is not installed.
    """
    try:
        torch = import_module("torch")
    except ImportError:
        return {
            "installed": False,
            "version": None,
            "ok": False,
            "cuda_available": False,
            "cuda_version": None,
            "cuda_device_count": 0,
            "cuda_device_name": None,
        }

    raw_version: str = getattr(torch, "__version__", "0.0.0")

    # Strip CUDA suffix: "2.2.2+cu121" → "2.2.2"
    clean_version = raw_version.split("+")[0]
    try:
        parts = clean_version.split(".")
        major, minor = int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        major, minor = 0, 0

    version_ok = (major, minor) >= _MIN_TORCH

    cuda_available: bool = bool(torch.cuda.is_available())
    cuda_version: str | None = None
    cuda_device_count: int = 0
    cuda_device_name: str | None = None

    if cuda_available:
        cuda_version = getattr(torch.version, "cuda", None)
        try:
            cuda_device_count = int(torch.cuda.device_count())
            if cuda_device_count > 0:
                cuda_device_name = torch.cuda.get_device_name(0)
        except Exception:
            cuda_device_count = 0

    return {
        "installed": True,
        "version": raw_version,
        "ok": version_ok,
        "cuda_available": cuda_available,
        "cuda_version": cuda_version,
        "cuda_device_count": cuda_device_count,
        "cuda_device_name": cuda_device_name,
    }


def _check_rasterio() -> dict[str, Any]:
    """
    Check rasterio installation, version, and accessible GDAL version.

    rasterio bundles GDAL; GDAL availability is inferred from rasterio's
    __gdal_version__ attribute.
    """
    try:
        rasterio = import_module("rasterio")
    except ImportError:
        return {
            "installed": False,
            "version": None,
            "ok": False,
            "gdal_version": None,
        }

    raw_version: str = getattr(rasterio, "__version__", "0.0.0")

    try:
        parts = raw_version.split(".")
        major, minor = int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        major, minor = 0, 0

    version_ok = (major, minor) >= _MIN_RASTERIO
    gdal_version: str | None = getattr(rasterio, "__gdal_version__", None)

    return {
        "installed": True,
        "version": raw_version,
        "ok": version_ok,
        "gdal_version": gdal_version,
    }


# ==============================================================================
# Public API
# ==============================================================================

def validate_environment(strict: bool = False) -> EnvironmentInfo:
    """
    Validate the runtime environment for the River Morphology pipeline.

    Checks Python version, PyTorch, CUDA, and rasterio/GDAL. Results are
    collected into an EnvironmentInfo dataclass with a human-readable summary.

    Args:
        strict:
            If True, raises EnvironmentValidationError on any critical failure
            (Python too old, PyTorch missing or incompatible, rasterio missing).
            CUDA absence is always a warning, never a hard error.
            If False, returns the EnvironmentInfo with the errors list populated
            but does not raise.

    Returns:
        EnvironmentInfo with all check results populated.

    Raises:
        EnvironmentValidationError: Only when strict=True and a critical check
            fails. The exception message lists all failures, not just the first.

    Example:
        env = validate_environment(strict=False)
        print(env.summary())
        if not env.is_gpu_available:
            print("Warning: no GPU found. Training will be very slow.")
    """
    warnings: list[str] = []
    errors:   list[str] = []

    # Python
    py = _check_python()
    if not py["ok"]:
        errors.append(
            f"Python {py['version'][0]}.{py['version'][1]}.{py['version'][2]} "
            f"found but >= {_MIN_PYTHON[0]}.{_MIN_PYTHON[1]} is required. "
            f"Upgrade Python and recreate your virtual environment."
        )

    # PyTorch
    torch_info = _check_torch()
    if not torch_info["installed"]:
        errors.append(
            "PyTorch is not installed. "
            "Install it from https://pytorch.org/get-started/locally/ "
            "with CUDA support matching your driver."
        )
    elif not torch_info["ok"]:
        errors.append(
            f"PyTorch {torch_info['version']} found but "
            f">= {_MIN_TORCH[0]}.{_MIN_TORCH[1]} is required. "
            f"Run: pip install torch>=2.0"
        )

    # CUDA (warning only — CPU fallback is supported)
    if not torch_info.get("cuda_available", False):
        warnings.append(
            "CUDA is not available. Training will use CPU, which is 10-50x slower. Use Google Colab (T4 GPU) or a local CUDA-capable machine for training."
        )

    # rasterio / GDAL
    rasterio_info = _check_rasterio()
    if not rasterio_info["installed"]:
        errors.append(
            "rasterio is not installed (provides GDAL for GeoTIFF I/O). "
            "On Windows: conda install -c conda-forge rasterio=1.3.9. "
            "On Linux/macOS/Colab: pip install rasterio==1.3.9"
        )
    elif not rasterio_info["ok"]:
        errors.append(
            f"rasterio {rasterio_info['version']} found but "
            f">= {_MIN_RASTERIO[0]}.{_MIN_RASTERIO[1]} is required. "
            f"Run: pip install rasterio>=1.3"
        )

    info = EnvironmentInfo(
        python_version=py["version"],
        python_ok=py["ok"],
        torch_installed=torch_info["installed"],
        torch_version=torch_info.get("version"),
        torch_ok=torch_info.get("ok", False),
        cuda_available=torch_info.get("cuda_available", False),
        cuda_version=torch_info.get("cuda_version"),
        cuda_device_count=torch_info.get("cuda_device_count", 0),
        cuda_device_name=torch_info.get("cuda_device_name"),
        rasterio_installed=rasterio_info["installed"],
        rasterio_version=rasterio_info.get("version"),
        rasterio_ok=rasterio_info.get("ok", False),
        gdal_version=rasterio_info.get("gdal_version"),
        warnings=warnings,
        errors=errors,
    )

    if strict and errors:
        combined = "\n".join(f"  • {e}" for e in errors)
        raise EnvironmentValidationError(
            check="full_environment",
            details=f"The following critical checks failed:\n{combined}",
        )

    return info


# ==============================================================================
# Script entry point
# ==============================================================================

if __name__ == "__main__":
    env = validate_environment(strict=False)
    print(env.summary())
    sys.exit(0 if env.is_ready_for_training else 1)