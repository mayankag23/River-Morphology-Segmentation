"""
Google Earth Engine client package for the River Morphology Segmentation System.

ARCHITECTURE CONTRACT:
    All future modules must import from src.gee only.
    No other module may import earthengine-api (import ee) directly.
    EarthEngineClient is the sole GEE interface.

Public API:
    EarthEngineClient  - Central EE interface; use this in all future modules
    RetryConfig        - Exponential backoff configuration for transient errors
    HealthChecker      - Connectivity and permission verification
    HealthReport       - Structured health check result container
    HealthCheckItem    - Single check result (name, status, message)
    HealthStatus       - Status constants: OK, WARN, ERROR, SKIP
    detect_runtime     - Identify Colab vs local execution environment
    RuntimeEnvironment - Enum for runtime environment values

GEE-Specific Exception Hierarchy:
    RiverMorphologyError
    +-- GEECredentialError       (from src.core.exceptions)
    |   +-- GEEAuthenticationError
    +-- GEENotInstalledError
    +-- GEEInitializationError
    +-- GEEConnectionError
    +-- GEEAPIError
    |   +-- GEEQuotaError
    +-- GEEGeometryError

Exceptions are defined in this __init__.py BEFORE submodules are imported
so that auth.py, health.py, and client.py can import them from src.gee
without triggering a circular import.
"""

from src.core.exceptions import GEECredentialError, RiverMorphologyError

__all__ = [
    # Client
    "EarthEngineClient",
    "RetryConfig",
    # Health
    "HealthChecker",
    "HealthReport",
    "HealthCheckItem",
    "HealthStatus",
    # Auth utilities
    "AuthManager",
    "RuntimeEnvironment",
    "detect_runtime",
    # Exceptions
    "GEENotInstalledError",
    "GEEAuthenticationError",
    "GEEInitializationError",
    "GEEConnectionError",
    "GEEAPIError",
    "GEEQuotaError",
    "GEEGeometryError",
]


# ==============================================================================
# GEE-Specific Exception Hierarchy
# Defined here so submodules can import from src.gee without circular imports.
# ==============================================================================

class GEENotInstalledError(RiverMorphologyError):
    """
    Raised when the earthengine-api package is not installed.

    Install with: pip install earthengine-api==0.1.390
    """


class GEEAuthenticationError(GEECredentialError):
    """
    Raised when Google Earth Engine authentication fails.

    Causes: invalid credentials, expired token, no EE account access,
    service account key file missing or malformed.
    """


class GEEInitializationError(RiverMorphologyError):
    """
    Raised when ee.Initialize() fails after authentication succeeds.

    Causes: invalid project ID, EE API not enabled on the project,
    account lacks project access.
    """


class GEEConnectionError(RiverMorphologyError):
    """
    Raised when network connectivity to EE servers cannot be established.

    Causes: no internet access, firewall blocking, DNS failure.
    """


class GEEAPIError(RiverMorphologyError):
    """
    Raised when an EE API call fails for a non-transient reason.

    Args:
        operation: Name of the EE operation that failed.
        reason:    Human-readable explanation of the failure.
    """

    def __init__(self, operation: str, reason: str) -> None:
        self.operation = operation
        self.reason = reason
        super().__init__(
            f"EE API call failed [{operation}]: {reason}"
        )


class GEEQuotaError(GEEAPIError):
    """
    Raised when the EE quota or rate limit is exceeded.

    This is a transient error. The retry mechanism will back off and
    retry automatically. If it persists, check your project quota at
    https://console.cloud.google.com/iam-admin/quotas
    """


class GEEGeometryError(RiverMorphologyError):
    """
    Raised when an AOI bounding box cannot be converted to an EE geometry.

    Causes: null AOI coordinates in config, coordinates outside valid
    WGS84 ranges, or EE rejection of the geometry.
    """


# ==============================================================================
# Deferred submodule imports
# Submodules import GEE exceptions from src.gee (this file).
# By the time these imports execute, all exception classes above are
# already registered in this module's namespace, so the circular
# reference resolves correctly from the submodule side.
# ==============================================================================

from src.gee.auth import (  # noqa: E402
    AuthManager,
    RuntimeEnvironment,
    detect_runtime,
)
from src.gee.client import EarthEngineClient, RetryConfig  # noqa: E402
from src.gee.health import (  # noqa: E402
    HealthCheckItem,
    HealthChecker,
    HealthReport,
    HealthStatus,
)