"""
Authentication management for Google Earth Engine.

Provides:
    RuntimeEnvironment - Enum identifying the execution environment.
    detect_runtime()   - Detect whether code runs in Colab or locally.
    AuthManager        - Orchestrates ee.Authenticate() and ee.Initialize().

Authentication workflows:
    Service account (non-interactive):
        Used when GEE_SERVICE_ACCOUNT_KEY env var is set to a JSON key path.
        Suitable for CI/CD, server environments, and headless Colab.

    Interactive OAuth (browser-based):
        Used when no service account key is provided.
        Suitable for local development and standard Colab notebooks.

All earthengine-api (import ee) calls are deferred to method bodies so
that the module can be imported and tested without ee being installed.
All EE exceptions are wrapped in project-specific GEE exception types.
"""

from __future__ import annotations

import json
import logging
import os
from enum import Enum
from pathlib import Path
from typing import Any

_LOGGER: logging.Logger = logging.getLogger(__name__)

__all__ = [
    "RuntimeEnvironment",
    "detect_runtime",
    "AuthManager",
]


# ==============================================================================
# Runtime Detection
# ==============================================================================

class RuntimeEnvironment(Enum):
    """Identifies the Python execution environment."""

    COLAB = "colab"
    LOCAL = "local"


def detect_runtime() -> RuntimeEnvironment:
    """
    Detect whether code is executing in Google Colab or a local environment.

    Detection strategy (applied in priority order):
        1. Presence of COLAB_RELEASE_TAG environment variable (set by Colab).
        2. Successful import of google.colab (always available in Colab).
        3. Fall back to LOCAL.

    Returns:
        RuntimeEnvironment.COLAB or RuntimeEnvironment.LOCAL.
    """
    if os.environ.get("COLAB_RELEASE_TAG"):
        _LOGGER.debug(
            "Colab environment detected via COLAB_RELEASE_TAG env var."
        )
        return RuntimeEnvironment.COLAB

    try:
        import google.colab  # noqa: F401
        _LOGGER.debug(
            "Colab environment detected via google.colab import."
        )
        return RuntimeEnvironment.COLAB
    except ImportError:
        pass

    _LOGGER.debug("LOCAL environment detected.")
    return RuntimeEnvironment.LOCAL


# ==============================================================================
# AuthManager
# ==============================================================================

class AuthManager:
    """
    Manages Google Earth Engine authentication and initialization.

    Supports two mutually exclusive authentication workflows:
        1. Service account: reads a JSON key file, creates
           ee.ServiceAccountCredentials. Non-interactive, suitable for
           automated pipelines, CI/CD, and headless environments.
        2. Interactive OAuth: calls ee.Authenticate() which opens a browser
           window (local) or displays an authorization link (Colab).

    The workflow is selected automatically:
        - If service_account_key is not None -> service account workflow.
        - If service_account_key is None     -> interactive OAuth workflow.

    State:
        is_initialized tracks whether THIS instance has completed the full
        authentication + initialization sequence. The EE Python API also
        maintains its own global state, but AuthManager tracks separately
        to support clean re-initialization if needed.

    Args:
        project_id:          GEE Cloud project ID. Required. Never empty.
        service_account_key: Absolute path string to a service account JSON
                             key file. None triggers interactive OAuth.

    Raises:
        ValueError: project_id is empty or whitespace.
    """

    def __init__(
        self,
        project_id: str,
        service_account_key: str | None = None,
    ) -> None:
        if not project_id or not project_id.strip():
            raise ValueError(
                "project_id must be a non-empty string. "
                "Set the GEE_PROJECT_ID environment variable."
            )

        self._project_id:          str                = project_id.strip()
        self._service_account_key: str | None         = service_account_key
        self._initialized:         bool               = False
        self._runtime:             RuntimeEnvironment = detect_runtime()
        self._logger: logging.Logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def is_initialized(self) -> bool:
        """True if authenticate_and_initialize() completed successfully."""
        return self._initialized

    @property
    def runtime(self) -> RuntimeEnvironment:
        """The detected execution environment for this session."""
        return self._runtime

    @property
    def project_id(self) -> str:
        """The GEE Cloud project ID used for initialization."""
        return self._project_id

    def authenticate_and_initialize(self) -> None:
        """
        Execute the full GEE authentication and initialization sequence.

        Selects service account or interactive OAuth based on whether
        service_account_key was provided at construction time.

        On success, sets is_initialized = True.
        On failure, is_initialized remains False and a GEE exception is raised.

        Raises:
            GEENotInstalledError:    earthengine-api is not installed.
            GEEAuthenticationError:  The authentication step failed.
            GEEInitializationError:  ee.Initialize() raised an exception.
        """
        from src.gee import (
            GEEAuthenticationError,
            GEEInitializationError,
            GEENotInstalledError,
        )

        try:
            import ee
        except ImportError as exc:
            raise GEENotInstalledError(
                "earthengine-api is not installed.\n"
                "Install with: pip install earthengine-api==0.1.390\n"
                "On Colab: !pip install earthengine-api==0.1.390 -q"
            ) from exc

        self._logger.info(
            "Starting EE auth sequence. runtime=%s, project=%s, "
            "service_account=%s",
            self._runtime.value,
            self._project_id,
            "yes" if self._service_account_key else "no (interactive OAuth)",
        )

        if self._service_account_key:
            credentials = self._authenticate_service_account(ee)
        else:
            credentials = self._authenticate_interactive(ee)

        self._initialize(ee, credentials)
        self._initialized = True

        self._logger.info(
            "Earth Engine initialized. project=%s", self._project_id
        )

    # ------------------------------------------------------------------
    # Private authentication helpers
    # ------------------------------------------------------------------

    def _authenticate_service_account(self, ee: Any) -> Any:
        """
        Build EE credentials from a service account JSON key file.

        Reads the client_email from the key file so that
        ee.ServiceAccountCredentials can be constructed.

        Args:
            ee: The earthengine-api module object.

        Returns:
            An ee.ServiceAccountCredentials instance.

        Raises:
            GEEAuthenticationError: Key file not found, unreadable,
                                    invalid JSON, missing client_email,
                                    or credential creation failed.
        """
        from src.gee import GEEAuthenticationError

        key_path = Path(self._service_account_key)

        if not key_path.exists():
            raise GEEAuthenticationError(
                f"Service account key file not found: {key_path}\n"
                "Check the GEE_SERVICE_ACCOUNT_KEY environment variable "
                "in your .env file."
            )

        if not key_path.is_file():
            raise GEEAuthenticationError(
                f"Service account key path is not a file: {key_path}"
            )

        try:
            with open(key_path, "r", encoding="utf-8") as fh:
                key_data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise GEEAuthenticationError(
                f"Service account key file contains invalid JSON: {key_path}\n"
                f"Parse error: {exc}"
            ) from exc
        except OSError as exc:
            raise GEEAuthenticationError(
                f"Cannot read service account key file: {key_path}\n"
                f"OS error: {exc}"
            ) from exc

        email: str | None = key_data.get("client_email")
        if not email:
            raise GEEAuthenticationError(
                f"Service account key file is missing 'client_email': {key_path}\n"
                "Ensure this is a valid Google service account JSON key."
            )

        self._logger.debug(
            "Building ServiceAccountCredentials for: %s", email
        )

        try:
            credentials = ee.ServiceAccountCredentials(email, str(key_path))
        except Exception as exc:
            raise GEEAuthenticationError(
                f"Failed to create EE ServiceAccountCredentials: {exc}\n"
                f"Service account email: {email}\n"
                f"Key file: {key_path}"
            ) from exc

        self._logger.debug(
            "ServiceAccountCredentials created successfully."
        )
        return credentials

    def _authenticate_interactive(self, ee: Any) -> None:
        """
        Run interactive OAuth authentication via ee.Authenticate().

        Behavior by environment:
            Colab: displays an authorization link in the notebook output.
            Local: opens the default browser for Google account sign-in.

        Args:
            ee: The earthengine-api module object.

        Returns:
            None (interactive auth does not return a credentials object;
            the token is stored in the user's credential cache).

        Raises:
            GEEAuthenticationError: User denied access, network error,
                                    or ee.Authenticate() raised.
        """
        from src.gee import GEEAuthenticationError

        self._logger.debug(
            "Running interactive OAuth via ee.Authenticate(). "
            "runtime=%s",
            self._runtime.value,
        )

        try:
            ee.Authenticate(quiet=False)
        except Exception as exc:
            raise GEEAuthenticationError(
                f"Interactive EE authentication failed: {exc}\n"
                "Ensure you have a Google account registered for EE at "
                "https://earthengine.google.com and internet access."
            ) from exc

        self._logger.debug("Interactive OAuth completed.")
        return None

    def _initialize(self, ee: Any, credentials: Any) -> None:
        """
        Call ee.Initialize() with the project ID and optional credentials.

        Args:
            ee:          The earthengine-api module object.
            credentials: ee.ServiceAccountCredentials instance, or None for
                         interactive-auth sessions where token is cached.

        Raises:
            GEEInitializationError: ee.Initialize() raised an exception.
        """
        from src.gee import GEEInitializationError

        self._logger.debug(
            "Calling ee.Initialize(project=%s, credentials=%s)",
            self._project_id,
            "provided" if credentials is not None else "None (cached token)",
        )

        try:
            if credentials is not None:
                ee.Initialize(
                    credentials=credentials,
                    project=self._project_id,
                )
            else:
                ee.Initialize(project=self._project_id)
        except Exception as exc:
            raise GEEInitializationError(
                f"ee.Initialize() failed for project '{self._project_id}': {exc}\n"
                "Verify that:\n"
                "  1. GEE_PROJECT_ID is a valid Google Cloud project ID.\n"
                "  2. The project has the Earth Engine API enabled.\n"
                "     Enable at: https://console.cloud.google.com/apis/library\n"
                "  3. Your account has access to this project."
            ) from exc

        self._logger.debug("ee.Initialize() completed successfully.")