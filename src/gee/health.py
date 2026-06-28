"""
Health check system for the Google Earth Engine client.

Provides:
    HealthStatus   - ASCII status constants: OK, WARN, ERROR, SKIP.
    HealthCheckItem - Single immutable check result (name, status, message).
    HealthReport    - Ordered collection of check results with computed properties.
    HealthChecker   - Runs all checks and returns a HealthReport.

Check sequence and dependencies:
    1. ee_installed   - Is earthengine-api importable? (no auth required)
    2. authentication - Is the EE client initialized?
    3. project_id     - Is GEE_PROJECT_ID set and non-empty?
    4. connectivity   - Can we reach earthengine.googleapis.com?
    5. api_access     - Can we execute a minimal EE computation? (requires 1+2+3+4)
    6. permissions    - Can we access the configured Landsat collection? (requires 5)

Items 5 and 6 return status=SKIP if any prerequisite check failed.
This prevents misleading ERROR results caused by upstream failures.

Type annotations for EarthEngineClient use TYPE_CHECKING to avoid a
circular import at runtime (health.py is imported by client.py).
"""

from __future__ import annotations

import logging
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.gee.client import EarthEngineClient

_LOGGER: logging.Logger = logging.getLogger(__name__)

# URL used for the connectivity ping check.
_EE_CONNECTIVITY_URL: str = "https://earthengine.googleapis.com"

# Timeout in seconds for the connectivity HTTP request.
_CONNECTIVITY_TIMEOUT_SECONDS: int = 8

# The lightest possible EE computation used to verify API access.
_API_TEST_NUMBER: int = 1

__all__ = [
    "HealthStatus",
    "HealthCheckItem",
    "HealthReport",
    "HealthChecker",
]


# ==============================================================================
# HealthStatus
# ==============================================================================

class HealthStatus(str, Enum):
    """
    ASCII status values for individual health check results.

    Values are plain ASCII strings so they can be used directly in
    console output, log messages, and JSON serialization.
    """

    OK    = "OK"
    WARN  = "WARN"
    ERROR = "ERROR"
    SKIP  = "SKIP"


# ==============================================================================
# HealthCheckItem
# ==============================================================================

@dataclass(frozen=True)
class HealthCheckItem:
    """
    Immutable result of a single health check.

    Attributes:
        name:    Short machine-readable identifier, e.g. "ee_installed".
        status:  One of HealthStatus values: "OK", "WARN", "ERROR", "SKIP".
        message: Human-readable result description.
        details: Optional extended context (error messages, suggestions).
    """

    name:    str
    status:  str
    message: str
    details: str = ""

    def is_ok(self) -> bool:
        """True if this check passed without issues."""
        return self.status == HealthStatus.OK

    def is_error(self) -> bool:
        """True if this check found a critical problem."""
        return self.status == HealthStatus.ERROR

    def is_skipped(self) -> bool:
        """True if this check was skipped due to a prerequisite failure."""
        return self.status == HealthStatus.SKIP


# ==============================================================================
# HealthReport
# ==============================================================================

@dataclass
class HealthReport:
    """
    Aggregate result of all health checks.

    Produced by HealthChecker.check_all(). All computed properties
    are derived from the items list.

    Attributes:
        items: Ordered list of HealthCheckItem results.
    """

    items: list[HealthCheckItem] = field(default_factory=list)

    @property
    def overall(self) -> str:
        """
        Overall health status derived from individual check results.

        Rules (applied in priority order):
            ERROR present -> "ERROR"
            WARN present (no ERROR) -> "WARN"
            All OK or SKIP -> "OK"
        """
        statuses = {item.status for item in self.items}
        if HealthStatus.ERROR in statuses:
            return HealthStatus.ERROR
        if HealthStatus.WARN in statuses:
            return HealthStatus.WARN
        return HealthStatus.OK

    @property
    def is_healthy(self) -> bool:
        """
        True if no check returned ERROR.

        WARN means degraded but operational (e.g., slow connection).
        SKIP means a check was not evaluated due to an upstream failure.
        Both are acceptable for declaring the system operational.
        """
        return not any(item.is_error() for item in self.items)

    @property
    def error_items(self) -> list[HealthCheckItem]:
        """All items with status ERROR."""
        return [item for item in self.items if item.is_error()]

    @property
    def warn_items(self) -> list[HealthCheckItem]:
        """All items with status WARN."""
        return [
            item for item in self.items
            if item.status == HealthStatus.WARN
        ]

    def summary_lines(self) -> list[str]:
        """
        Return one ASCII-formatted line per check result.

        Format (fixed-width tag for aligned output):
            [OK]    name: message
            [WARN]  name: message
            [ERROR] name: message  (details appended on next line if present)
            [SKIP]  name: message

        Returns:
            List of ASCII-only strings, one per check item.
        """
        lines: list[str] = []
        for item in self.items:
            tag = {
                HealthStatus.OK:    "[OK]   ",
                HealthStatus.WARN:  "[WARN] ",
                HealthStatus.ERROR: "[ERROR]",
                HealthStatus.SKIP:  "[SKIP] ",
            }.get(item.status, "[?????]")

            lines.append(f"  {tag}  {item.name}: {item.message}")
            if item.details and item.status == HealthStatus.ERROR:
                lines.append(f"           -> {item.details}")

        lines.append("")
        lines.append(f"  Overall: {self.overall}")
        return lines


# ==============================================================================
# HealthChecker
# ==============================================================================

class HealthChecker:
    """
    Runs all health checks against the EE client and returns a HealthReport.

    Checks are executed in dependency order. Each check is isolated in its
    own method and wrapped in a broad try/except so that a failure in one
    check never prevents subsequent checks from running.

    Args:
        client: The EarthEngineClient instance to check. Does not need to
                be initialized; auth-dependent checks return SKIP if not.
        config_collections: Optional list of GEE collection IDs to verify
                            read access for. Defaults to Landsat 8 C2 L2.
    """

    _DEFAULT_COLLECTIONS: tuple[str, ...] = (
        "LANDSAT/LC08/C02/T1_L2",
    )

    def __init__(
        self,
        client: EarthEngineClient,
        config_collections: list[str] | None = None,
    ) -> None:
        self._client = client
        self._collections: list[str] = (
            config_collections
            if config_collections is not None
            else list(self._DEFAULT_COLLECTIONS)
        )
        self._logger: logging.Logger = logging.getLogger(__name__)

    def check_all(self) -> HealthReport:
        """
        Execute all health checks and return a combined HealthReport.

        Checks run in dependency order. Auth-dependent checks are skipped
        (status=SKIP) if the authentication check failed.

        Returns:
            HealthReport with one HealthCheckItem per check performed.
        """
        self._logger.info("Running full EE health check.")
        items: list[HealthCheckItem] = []

        ee_item    = self._check_ee_installed()
        auth_item  = self._check_authentication()
        proj_item  = self._check_project_id()
        conn_item  = self._check_connectivity()

        items.extend([ee_item, auth_item, proj_item, conn_item])

        auth_ok = (
            ee_item.status == HealthStatus.OK
            and auth_item.status == HealthStatus.OK
            and proj_item.status == HealthStatus.OK
            and conn_item.status == HealthStatus.OK
        )

        if auth_ok:
            api_item  = self._check_api_access()
            perm_item = self._check_permissions()
        else:
            skip_reason = (
                "Skipped: prerequisite check(s) failed "
                "(ee_installed / authentication / project_id / connectivity)"
            )
            api_item = HealthCheckItem(
                name="api_access",
                status=HealthStatus.SKIP,
                message=skip_reason,
            )
            perm_item = HealthCheckItem(
                name="permissions",
                status=HealthStatus.SKIP,
                message=skip_reason,
            )

        items.extend([api_item, perm_item])

        report = HealthReport(items=items)
        self._logger.info(
            "Health check complete. overall=%s, errors=%d",
            report.overall,
            len(report.error_items),
        )
        return report

    # ------------------------------------------------------------------
    # Individual check methods
    # ------------------------------------------------------------------

    def _check_ee_installed(self) -> HealthCheckItem:
        """
        Verify that earthengine-api is importable.

        Returns OK if import succeeds, ERROR otherwise.
        Also checks that the package version attribute is present.
        """
        name = "ee_installed"
        try:
            import ee
            version = getattr(ee, "__version__", "unknown")
            return HealthCheckItem(
                name=name,
                status=HealthStatus.OK,
                message=f"earthengine-api {version} is installed.",
            )
        except ImportError as exc:
            return HealthCheckItem(
                name=name,
                status=HealthStatus.ERROR,
                message="earthengine-api is NOT installed.",
                details=(
                    f"ImportError: {exc}. "
                    "Install with: pip install earthengine-api==0.1.390"
                ),
            )
        except Exception as exc:
            return HealthCheckItem(
                name=name,
                status=HealthStatus.ERROR,
                message="Unexpected error checking earthengine-api.",
                details=str(exc),
            )

    def _check_authentication(self) -> HealthCheckItem:
        """
        Verify that the EE client has been initialized.

        Returns OK if client.is_initialized is True, ERROR otherwise.
        Does NOT attempt to initialize -- this is a status check only.
        """
        name = "authentication"
        try:
            if self._client.is_initialized:
                return HealthCheckItem(
                    name=name,
                    status=HealthStatus.OK,
                    message="EE client is authenticated and initialized.",
                )
            return HealthCheckItem(
                name=name,
                status=HealthStatus.ERROR,
                message="EE client is NOT initialized.",
                details=(
                    "Call EarthEngineClient.initialize() before running "
                    "any EE operations."
                ),
            )
        except Exception as exc:
            return HealthCheckItem(
                name=name,
                status=HealthStatus.ERROR,
                message="Unexpected error checking authentication state.",
                details=str(exc),
            )

    def _check_project_id(self) -> HealthCheckItem:
        """
        Verify that the GEE_PROJECT_ID environment variable is set.

        Reads the project ID via config.gee_project_id (which reads from
        the environment variable). Does not validate the ID against GEE
        servers -- that is covered by api_access.
        """
        name = "project_id"
        try:
            project_id = self._client.project_id
            if project_id:
                return HealthCheckItem(
                    name=name,
                    status=HealthStatus.OK,
                    message=f"GEE project ID is set: {project_id}",
                )
            return HealthCheckItem(
                name=name,
                status=HealthStatus.ERROR,
                message="GEE project ID is empty.",
                details=(
                    "Set GEE_PROJECT_ID in your .env file: "
                    "GEE_PROJECT_ID=your-project-id"
                ),
            )
        except Exception as exc:
            return HealthCheckItem(
                name=name,
                status=HealthStatus.ERROR,
                message="GEE_PROJECT_ID environment variable is not set.",
                details=str(exc),
            )

    def _check_connectivity(self) -> HealthCheckItem:
        """
        Verify network connectivity to earthengine.googleapis.com.

        Sends an HTTP request to the EE API root URL. Any response from
        the server (including 403 Forbidden) indicates connectivity.
        Only network-level failures (DNS, timeout, no route) return ERROR.
        """
        name = "connectivity"
        try:
            urllib.request.urlopen(
                _EE_CONNECTIVITY_URL,
                timeout=_CONNECTIVITY_TIMEOUT_SECONDS,
            )
            return HealthCheckItem(
                name=name,
                status=HealthStatus.OK,
                message=f"Reached {_EE_CONNECTIVITY_URL} successfully.",
            )
        except urllib.error.HTTPError as exc:
            # HTTP errors mean the server was reachable (network is up).
            return HealthCheckItem(
                name=name,
                status=HealthStatus.OK,
                message=(
                    f"Server reachable at {_EE_CONNECTIVITY_URL} "
                    f"(HTTP {exc.code})."
                ),
            )
        except urllib.error.URLError as exc:
            return HealthCheckItem(
                name=name,
                status=HealthStatus.ERROR,
                message=f"Cannot reach {_EE_CONNECTIVITY_URL}.",
                details=(
                    f"URLError: {exc.reason}. "
                    "Check your internet connection or firewall settings."
                ),
            )
        except TimeoutError:
            return HealthCheckItem(
                name=name,
                status=HealthStatus.WARN,
                message=f"Connection to {_EE_CONNECTIVITY_URL} timed out.",
                details=(
                    f"Timeout after {_CONNECTIVITY_TIMEOUT_SECONDS}s. "
                    "Server may be slow or connection degraded."
                ),
            )
        except Exception as exc:
            return HealthCheckItem(
                name=name,
                status=HealthStatus.ERROR,
                message="Unexpected connectivity check error.",
                details=str(exc),
            )

    def _check_api_access(self) -> HealthCheckItem:
        """
        Execute a minimal EE computation to verify API access.

        Calls ee.Number(1).getInfo() which is the lightest possible
        server-side computation. Success confirms:
            - Authentication token is valid
            - Project ID is accepted by EE
            - The EE computation API is responsive
        """
        name = "api_access"
        try:
            import ee
            result: Any = ee.Number(_API_TEST_NUMBER).getInfo()
            if result == _API_TEST_NUMBER:
                return HealthCheckItem(
                    name=name,
                    status=HealthStatus.OK,
                    message=(
                        f"EE API responded correctly: "
                        f"ee.Number({_API_TEST_NUMBER}).getInfo() = {result}"
                    ),
                )
            return HealthCheckItem(
                name=name,
                status=HealthStatus.WARN,
                message=(
                    f"EE API returned unexpected value: "
                    f"expected {_API_TEST_NUMBER}, got {result}"
                ),
            )
        except ImportError:
            return HealthCheckItem(
                name=name,
                status=HealthStatus.ERROR,
                message="earthengine-api import failed during API check.",
            )
        except Exception as exc:
            return HealthCheckItem(
                name=name,
                status=HealthStatus.ERROR,
                message="EE API computation failed.",
                details=str(exc),
            )

    def _check_permissions(self) -> HealthCheckItem:
        """
        Verify read access to the configured Landsat image collections.

        Calls .size().getInfo() on the first configured collection limited
        to a small region. This confirms asset-level read permissions
        without downloading any imagery.
        """
        name = "permissions"
        if not self._collections:
            return HealthCheckItem(
                name=name,
                status=HealthStatus.SKIP,
                message="No collections configured to check.",
            )

        collection_id = self._collections[0]
        try:
            import ee
            size: Any = (
                ee.ImageCollection(collection_id)
                .limit(1)
                .size()
                .getInfo()
            )
            return HealthCheckItem(
                name=name,
                status=HealthStatus.OK,
                message=(
                    f"Read access confirmed for '{collection_id}' "
                    f"(query returned {size} result(s))."
                ),
            )
        except ImportError:
            return HealthCheckItem(
                name=name,
                status=HealthStatus.ERROR,
                message="earthengine-api import failed during permissions check.",
            )
        except Exception as exc:
            err_msg = str(exc).lower()
            if "access" in err_msg or "permission" in err_msg or "denied" in err_msg:
                return HealthCheckItem(
                    name=name,
                    status=HealthStatus.ERROR,
                    message=(
                        f"Permission denied accessing '{collection_id}'."
                    ),
                    details=(
                        f"Error: {exc}. "
                        "Verify your EE account has access to Landsat public datasets."
                    ),
                )
            return HealthCheckItem(
                name=name,
                status=HealthStatus.ERROR,
                message=f"Failed to access '{collection_id}'.",
                details=str(exc),
            )