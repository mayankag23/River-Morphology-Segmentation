"""
Central Earth Engine client for the River Morphology Segmentation System.

EarthEngineClient is the ONLY class that any other module in this project
may use to interact with Google Earth Engine. No other module may import
or call the earthengine-api (ee) package directly.

Design principles:
    - Single initialization: call initialize() once, reuse the instance.
    - All EE exceptions are caught and re-raised as project GEE exceptions.
    - Transient failures are retried automatically with exponential backoff.
    - All operations are logged at appropriate levels.
    - The client is safe to construct without triggering authentication;
      initialize() must be called explicitly before any EE operation.

Typical usage in future modules:

    from src.gee import EarthEngineClient
    from src.core.config import Config

    config = Config("config/config.yaml")
    client = EarthEngineClient(config)
    client.initialize()

    geometry   = client.get_aoi_geometry()
    collection = client.get_image_collection("LANDSAT/LC08/C02/T1_L2")
    report     = client.health_check()
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

from src.core.config import Config
from src.core.exceptions import MissingFieldError
from src.gee.auth import AuthManager
from src.gee.health import HealthChecker

_LOGGER: logging.Logger = logging.getLogger(__name__)

T = TypeVar("T")

# Substrings in exception messages that indicate a TRANSIENT (retryable) error.
_TRANSIENT_PATTERNS: tuple[str, ...] = (
    "quota exceeded",
    "rate limit",
    "rate_limit",
    "service unavailable",
    "internal error",
    "internal_error",
    "backend error",
    "timed out",
    "timeout",
    "too many requests",
    "temporarily unavailable",
    "503",
    "500",
    "429",
    "connection reset",
    "connection aborted",
    "remote end closed",
    "broken pipe",
)

# Substrings that indicate a PERMANENT (non-retryable) error.
# Checked before transient patterns; permanent takes precedence.
_PERMANENT_PATTERNS: tuple[str, ...] = (
    "unauthorized",
    "unauthenticated",
    "forbidden",
    "access denied",
    "permission denied",
    "invalid credentials",
    "token expired",
    "invalid project",
    "project not found",
    "not found",
    "invalid argument",
    "invalid_argument",
    "malformed",
)

__all__ = [
    "RetryConfig",
    "EarthEngineClient",
]


# ==============================================================================
# RetryConfig
# ==============================================================================

@dataclass
class RetryConfig:
    """
    Configuration for the exponential backoff retry mechanism.

    Attributes:
        max_attempts:       Total number of attempts including the first.
                            Must be >= 1. A value of 1 disables retries.
        base_delay_seconds: Wait time (seconds) before the first retry.
        max_delay_seconds:  Upper bound on the computed delay. Prevents
                            unbounded wait times on many-retry scenarios.
        backoff_factor:     Multiplier applied to the delay after each
                            failed attempt. 2.0 = classic exponential backoff.
        jitter:             Add random noise (up to 50% of computed delay)
                            to prevent thundering herd when multiple clients
                            retry simultaneously.
    """

    max_attempts:       int   = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds:  float = 30.0
    backoff_factor:     float = 2.0
    jitter:             bool  = True


# ==============================================================================
# Private utilities
# ==============================================================================

def _is_transient_error(exc: Exception) -> bool:
    """
    Classify an exception as transient (retryable) or permanent (fatal).

    Strategy:
        1. Check message against PERMANENT_PATTERNS first (higher priority).
           If matched -> permanent, return False immediately.
        2. Check message against TRANSIENT_PATTERNS.
           If matched -> transient, return True.
        3. Default -> permanent (conservative: don't retry unknown errors).

    Args:
        exc: The exception to classify.

    Returns:
        True if the error is likely transient and retrying may succeed.
        False if the error is permanent and retrying will not help.
    """
    msg = str(exc).lower()

    if any(pattern in msg for pattern in _PERMANENT_PATTERNS):
        return False

    return any(pattern in msg for pattern in _TRANSIENT_PATTERNS)


def _compute_delay(
    attempt: int,
    config: RetryConfig,
) -> float:
    """
    Compute the wait duration before a retry attempt.

    Formula: min(base * factor^(attempt-1), max_delay)
    With optional jitter: delay *= uniform(0.5, 1.0)

    Args:
        attempt: The current attempt number (1-indexed). Pass the attempt
                 that just failed; delay is for before the NEXT attempt.
        config:  RetryConfig controlling the backoff parameters.

    Returns:
        Wait duration in seconds.
    """
    raw_delay = config.base_delay_seconds * (
        config.backoff_factor ** (attempt - 1)
    )
    delay = min(raw_delay, config.max_delay_seconds)

    if config.jitter:
        delay *= 0.5 + random.random() * 0.5

    return delay


# ==============================================================================
# EarthEngineClient
# ==============================================================================

class EarthEngineClient:
    """
    Central interface to Google Earth Engine for the River Morphology system.

    This class is the ONLY permitted way to interact with the EE API in
    this project. Future modules receive an EarthEngineClient instance
    and use its methods exclusively.

    Lifecycle:
        1. Construct: EarthEngineClient(config)           -- no auth, no network
        2. Initialize: client.initialize()                -- auth + ee.Initialize
        3. Use:        client.get_aoi_geometry() etc.     -- EE operations
        4. Health:     client.health_check()              -- diagnostic report

    Thread safety:
        Not thread-safe. Create one client per thread if using concurrency.

    Args:
        config:       Fully initialized Config object. GEE credentials are
                      read from environment variables via config.gee_project_id.
        retry_config: Optional RetryConfig. Uses safe defaults if not provided.

    Raises:
        Nothing at construction time. All failures are deferred to initialize().
    """

    def __init__(
        self,
        config: Config,
        retry_config: RetryConfig | None = None,
    ) -> None:
        self._config:       Config       = config
        self._retry_config: RetryConfig  = retry_config or RetryConfig()
        self._initialized:  bool         = False
        self._project_id:   str | None   = None
        self._auth_manager: AuthManager | None = None
        self._logger: logging.Logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Public: lifecycle
    # ------------------------------------------------------------------

    @property
    def is_initialized(self) -> bool:
        """True if initialize() completed without error."""
        return self._initialized

    @property
    def project_id(self) -> str | None:
        """
        The GEE Cloud project ID used during initialization.

        Returns None if initialize() has not yet been called.
        Returns the project ID string after successful initialization.
        """
        return self._project_id

    def initialize(self) -> None:
        """
        Authenticate with Google Earth Engine and call ee.Initialize().

        Reads credentials from environment variables via the Config object:
            GEE_PROJECT_ID          -- required; the Cloud project ID.
            GEE_SERVICE_ACCOUNT_KEY -- optional; path to service account JSON.

        Idempotent: calling initialize() on an already-initialized client
        logs a debug message and returns immediately without re-authenticating.

        Raises:
            GEECredentialError:    GEE_PROJECT_ID is not set or empty.
            GEENotInstalledError:  earthengine-api is not installed.
            GEEAuthenticationError: Authentication failed.
            GEEInitializationError: ee.Initialize() failed.
        """
        if self._initialized:
            self._logger.debug(
                "EarthEngineClient already initialized. Skipping."
            )
            return

        project_id  = self._config.gee_project_id
        service_key = self._config.gee_service_account_key

        self._logger.info(
            "Initializing EarthEngineClient. project=%s", project_id
        )

        auth = AuthManager(
            project_id=project_id,
            service_account_key=service_key,
        )
        auth.authenticate_and_initialize()

        self._auth_manager = auth
        self._project_id   = project_id
        self._initialized  = True

        self._logger.info(
            "EarthEngineClient initialized. project=%s", project_id
        )

    # ------------------------------------------------------------------
    # Public: geometry
    # ------------------------------------------------------------------

    def get_aoi_geometry(self) -> Any:
        """
        Convert the configured AOI bounding box to an ee.Geometry.Rectangle.

        Reads min_lon, min_lat, max_lon, max_lat from config.aoi.
        The rectangle is defined in WGS84 (EPSG:4326).

        Returns:
            ee.Geometry.Rectangle instance representing the AOI.

        Raises:
            MissingFieldError:   AOI coordinates are null in config.yaml.
            GEENotInstalledError: earthengine-api is not installed.
            GEEGeometryError:    EE rejected the geometry or coordinate
                                 conversion failed.
        """
        from src.gee import GEEGeometryError, GEENotInstalledError

        if not self._config.has_aoi:
            raise MissingFieldError(
                field="aoi.[min_lon, min_lat, max_lon, max_lat]",
                context=(
                    "Set all four AOI coordinates in config.yaml before "
                    "calling get_aoi_geometry(). "
                    "Example:\n"
                    "  aoi:\n"
                    "    min_lon: 87.0\n"
                    "    min_lat: 26.0\n"
                    "    max_lon: 87.5\n"
                    "    max_lat: 26.5"
                ),
            )

        aoi = self._config.aoi

        try:
            min_lon = float(aoi.min_lon)
            min_lat = float(aoi.min_lat)
            max_lon = float(aoi.max_lon)
            max_lat = float(aoi.max_lat)
        except (TypeError, ValueError) as exc:
            raise MissingFieldError(
                field="aoi coordinates",
                context=f"Could not convert AOI coordinates to float: {exc}",
            ) from exc

        try:
            import ee
        except ImportError as exc:
            raise GEENotInstalledError(
                "earthengine-api is not installed. "
                "Install with: pip install earthengine-api==0.1.390"
            ) from exc

        try:
            geometry = ee.Geometry.Rectangle(
                [min_lon, min_lat, max_lon, max_lat]
            )
        except Exception as exc:
            raise GEEGeometryError(
                f"Failed to create ee.Geometry.Rectangle from AOI "
                f"[{min_lon}, {min_lat}, {max_lon}, {max_lat}]: {exc}"
            ) from exc

        self._logger.debug(
            "AOI geometry created: lon=[%s, %s] lat=[%s, %s]",
            min_lon, max_lon, min_lat, max_lat,
        )
        return geometry

    # ------------------------------------------------------------------
    # Public: image collections
    # ------------------------------------------------------------------

    def get_image_collection(self, collection_id: str) -> Any:
        """
        Return an ee.ImageCollection for the given collection asset ID.

        This method performs NO filtering. Callers (preprocessing modules)
        apply date, cloud cover, and spatial filters on the returned object.

        Args:
            collection_id: GEE collection asset path, e.g.
                           "LANDSAT/LC08/C02/T1_L2".

        Returns:
            ee.ImageCollection instance (unfiltered).

        Raises:
            RuntimeError:        Client is not initialized.
            GEENotInstalledError: earthengine-api is not installed.
            GEEAPIError:         EE rejected the collection ID.
        """
        from src.gee import GEEAPIError, GEENotInstalledError

        if not self._initialized:
            raise RuntimeError(
                "EarthEngineClient is not initialized. "
                "Call initialize() before get_image_collection()."
            )

        if not collection_id or not collection_id.strip():
            raise GEEAPIError(
                operation="get_image_collection",
                reason="collection_id must be a non-empty string.",
            )

        try:
            import ee
        except ImportError as exc:
            raise GEENotInstalledError(
                "earthengine-api is not installed."
            ) from exc

        try:
            collection = ee.ImageCollection(collection_id.strip())
        except Exception as exc:
            raise GEEAPIError(
                operation="get_image_collection",
                reason=f"Failed to create ImageCollection for '{collection_id}': {exc}",
            ) from exc

        self._logger.debug(
            "ImageCollection created for: %s", collection_id
        )
        return collection

    # ------------------------------------------------------------------
    # Public: retry execution
    # ------------------------------------------------------------------

    def execute_with_retry(
        self,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """
        Execute a callable with exponential backoff retry on transient errors.

        Behavior:
            - Calls func(*args, **kwargs).
            - On success: returns the result immediately.
            - On transient error: waits and retries up to max_attempts.
            - On permanent error: wraps in GEEAPIError and re-raises immediately.
            - After max_attempts exhausted: raises GEEAPIError.

        This method is the recommended way for future modules to call any
        EE operation that may fail transiently.

        Args:
            func:    The callable to execute. Should be an EE API call or
                     a lambda/partial wrapping one.
            *args:   Positional arguments passed to func.
            **kwargs: Keyword arguments passed to func.

        Returns:
            Whatever func returns on success.

        Raises:
            GEEAPIError:   Permanent error (not retried) or max retries exceeded.
            GEEQuotaError: Quota exceeded after all retry attempts.
        """
        from src.gee import GEEAPIError, GEEQuotaError

        operation_name = getattr(func, "__name__", repr(func))
        last_exc: Exception | None = None

        for attempt in range(1, self._retry_config.max_attempts + 1):
            try:
                result: T = func(*args, **kwargs)
                if attempt > 1:
                    self._logger.info(
                        "EE operation succeeded on attempt %d/%d: %s",
                        attempt,
                        self._retry_config.max_attempts,
                        operation_name,
                    )
                return result

            except Exception as exc:
                last_exc = exc

                if not _is_transient_error(exc):
                    self._logger.error(
                        "Permanent EE error (no retry) in %s: %s",
                        operation_name, exc,
                    )
                    exc_msg = str(exc).lower()
                    if "quota" in exc_msg or "429" in exc_msg:
                        raise GEEQuotaError(
                            operation=operation_name,
                            reason=str(exc),
                        ) from exc
                    raise GEEAPIError(
                        operation=operation_name,
                        reason=str(exc),
                    ) from exc

                if attempt == self._retry_config.max_attempts:
                    break

                delay = _compute_delay(attempt, self._retry_config)
                self._logger.warning(
                    "Transient EE error on attempt %d/%d in %s. "
                    "Retrying in %.1fs. Error: %s",
                    attempt,
                    self._retry_config.max_attempts,
                    operation_name,
                    delay,
                    exc,
                )
                time.sleep(delay)

        exc_msg_lower = str(last_exc).lower() if last_exc else ""
        if "quota" in exc_msg_lower or "429" in exc_msg_lower:
            raise GEEQuotaError(
                operation=operation_name,
                reason=(
                    f"Quota exceeded; max retry attempts "
                    f"({self._retry_config.max_attempts}) exhausted. "
                    f"Last error: {last_exc}"
                ),
            ) from last_exc

        raise GEEAPIError(
            operation=operation_name,
            reason=(
                f"Max retry attempts ({self._retry_config.max_attempts}) "
                f"exhausted. Last error: {last_exc}"
            ),
        ) from last_exc

    # ------------------------------------------------------------------
    # Public: health check
    # ------------------------------------------------------------------
    
    def health_check(
        self,
        config_collections: list[str] | None = None,
    ) -> Any:
        """
        Run a full health check against the EE environment and return a report.

        Checks:
            1. ee_installed:   Is earthengine-api importable?
            2. authentication: Is this client initialized?
            3. project_id:     Is GEE_PROJECT_ID set?
            4. connectivity:   Can we reach earthengine.googleapis.com?
            5. api_access:     Does a minimal EE computation succeed? (if 1-4 OK)
            6. permissions:    Can we access the Landsat collection? (if 5 OK)

        Args:
            config_collections: Override the collections to check for
                                permissions. Defaults to Landsat 8 C2 L2.

        Returns:
            HealthReport with one HealthCheckItem per check performed.
        """
        # from src.gee.health import HealthChecker

        collections = config_collections
        if collections is None:
            try:
                collections = list(self._config.satellite.collections)
            except AttributeError:
                collections = None

        checker = HealthChecker(
            client=self,
            config_collections=collections,
        )
        return checker.check_all()

    # ------------------------------------------------------------------
    # Private utilities
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"EarthEngineClient("
            f"project={self._project_id!r}, "
            f"initialized={self._initialized})"
        )