"""
Unit tests for src/gee/auth.py.

All earthengine-api (import ee) calls are intercepted by patching sys.modules.
No real GEE authentication is required to run these tests.

Run:
    pytest tests/test_auth.py -v
    pytest tests/test_auth.py -v --cov=src/gee/auth --cov-report=term-missing
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.gee import (
    GEEAuthenticationError,
    GEEInitializationError,
    GEENotInstalledError,
)
from src.gee.auth import AuthManager, RuntimeEnvironment, detect_runtime
import builtins

_ORIGINAL_IMPORT = builtins.__import__


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def mock_ee() -> MagicMock:
    """
    Return a MagicMock standing in for the earthengine-api (ee) module.

    Each test that needs to simulate EE behaviour patches sys.modules['ee']
    with this mock using unittest.mock.patch.dict.
    """
    ee = MagicMock()
    ee.__version__ = "0.1.390"
    ee.Authenticate.return_value = None
    ee.Initialize.return_value = None
    return ee


@pytest.fixture
def service_account_key_file(tmp_path: Path) -> Path:
    """
    Write a minimal valid service account JSON key file to tmp_path.

    Returns the absolute path to the file.
    """
    key_data = {
        "type":         "service_account",
        "client_email": "test@test-project.iam.gserviceaccount.com",
        "private_key":  "-----BEGIN RSA PRIVATE KEY-----\nFAKE\n-----END RSA PRIVATE KEY-----\n",
        "project_id":   "test-project",
    }
    key_file = tmp_path / "service_account.json"
    key_file.write_text(json.dumps(key_data), encoding="utf-8")
    return key_file


@pytest.fixture
def auth_manager_interactive() -> AuthManager:
    """Return an AuthManager configured for interactive OAuth (no service account)."""
    return AuthManager(project_id="test-project-123")


@pytest.fixture
def auth_manager_service_account(
    service_account_key_file: Path,
) -> AuthManager:
    """Return an AuthManager configured for service account authentication."""
    return AuthManager(
        project_id="test-project-123",
        service_account_key=str(service_account_key_file),
    )


# ==============================================================================
# detect_runtime tests
# ==============================================================================

class TestDetectRuntime:
    """Tests for the detect_runtime() function."""

    def test_returns_local_when_no_colab_signals(self) -> None:
        """When no Colab indicators are present, return LOCAL."""
        env_without_colab = {
            k: v for k, v in os.environ.items()
            if k != "COLAB_RELEASE_TAG"
        }
        # Remove google.colab from sys.modules if present.
        modules_without_colab = {
            k: v for k, v in sys.modules.items()
            if "google.colab" not in k
        }
        with (
            patch.dict(os.environ, env_without_colab, clear=True),
            patch.dict(sys.modules, modules_without_colab, clear=True),
        ):
            result = detect_runtime()
        assert result == RuntimeEnvironment.LOCAL

    def test_returns_colab_when_env_var_present(self) -> None:
        """COLAB_RELEASE_TAG env var indicates Colab environment."""
        with patch.dict(os.environ, {"COLAB_RELEASE_TAG": "v1.0"}):
            result = detect_runtime()
        assert result == RuntimeEnvironment.COLAB

    def test_returns_colab_when_google_colab_importable(self) -> None:
        """Importable google.colab module indicates Colab environment."""
        fake_colab = MagicMock()
        env_without_tag = {
            k: v for k, v in os.environ.items()
            if k != "COLAB_RELEASE_TAG"
        }
        with (
            patch.dict(os.environ, env_without_tag, clear=True),
            patch.dict(sys.modules, {"google.colab": fake_colab}),
        ):
            result = detect_runtime()
        assert result == RuntimeEnvironment.COLAB

    def test_env_var_takes_priority_over_import(self) -> None:
        """COLAB_RELEASE_TAG check runs before the import attempt."""
        with patch.dict(os.environ, {"COLAB_RELEASE_TAG": "v2.0"}):
            # Even without google.colab in sys.modules, returns COLAB.
            result = detect_runtime()
        assert result == RuntimeEnvironment.COLAB

    def test_runtime_enum_values_are_ascii(self) -> None:
        """RuntimeEnvironment values must be ASCII strings."""
        for member in RuntimeEnvironment:
            assert all(ord(c) < 128 for c in member.value)


# ==============================================================================
# AuthManager construction tests
# ==============================================================================

class TestAuthManagerConstruction:
    """Tests for AuthManager.__init__() validation."""

    def test_valid_construction_interactive(self) -> None:
        manager = AuthManager(project_id="my-project")
        assert manager.project_id == "my-project"
        assert manager.is_initialized is False

    def test_valid_construction_service_account(self, tmp_path: Path) -> None:
        key = str(tmp_path / "key.json")
        manager = AuthManager(project_id="my-project", service_account_key=key)
        assert manager.project_id == "my-project"

    def test_strips_whitespace_from_project_id(self) -> None:
        manager = AuthManager(project_id="  my-project  ")
        assert manager.project_id == "my-project"

    def test_empty_project_id_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            AuthManager(project_id="")

    def test_whitespace_only_project_id_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            AuthManager(project_id="   ")

    def test_runtime_property_returns_enum(self) -> None:
        manager = AuthManager(project_id="my-project")
        assert isinstance(manager.runtime, RuntimeEnvironment)

    def test_is_initialized_false_on_construction(self) -> None:
        manager = AuthManager(project_id="my-project")
        assert manager.is_initialized is False


# ==============================================================================
# AuthManager.authenticate_and_initialize — interactive OAuth tests
# ==============================================================================

class TestAuthManagerInteractive:
    """Tests for interactive OAuth authentication workflow."""

    def test_interactive_auth_success(
        self,
        auth_manager_interactive: AuthManager,
        mock_ee: MagicMock,
    ) -> None:
        """Successful interactive auth sets is_initialized to True."""
        with patch.dict(sys.modules, {"ee": mock_ee}):
            auth_manager_interactive.authenticate_and_initialize()
        assert auth_manager_interactive.is_initialized is True

    def test_interactive_auth_calls_authenticate(
        self,
        auth_manager_interactive: AuthManager,
        mock_ee: MagicMock,
    ) -> None:
        """ee.Authenticate() must be called for interactive workflow."""
        with patch.dict(sys.modules, {"ee": mock_ee}):
            auth_manager_interactive.authenticate_and_initialize()
        mock_ee.Authenticate.assert_called_once()

    def test_interactive_auth_calls_initialize_with_project(
        self,
        auth_manager_interactive: AuthManager,
        mock_ee: MagicMock,
    ) -> None:
        """ee.Initialize() must receive the project keyword argument."""
        with patch.dict(sys.modules, {"ee": mock_ee}):
            auth_manager_interactive.authenticate_and_initialize()
        call_kwargs = mock_ee.Initialize.call_args[1]
        assert call_kwargs.get("project") == "test-project-123"

    def test_interactive_auth_failure_raises_gee_authentication_error(
        self,
        auth_manager_interactive: AuthManager,
        mock_ee: MagicMock,
    ) -> None:
        """Failed ee.Authenticate() raises GEEAuthenticationError."""
        mock_ee.Authenticate.side_effect = Exception("Auth refused")
        with patch.dict(sys.modules, {"ee": mock_ee}):
            with pytest.raises(GEEAuthenticationError, match="Auth refused"):
                auth_manager_interactive.authenticate_and_initialize()
        assert auth_manager_interactive.is_initialized is False

    def test_initialize_failure_raises_gee_initialization_error(
        self,
        auth_manager_interactive: AuthManager,
        mock_ee: MagicMock,
    ) -> None:
        """Failed ee.Initialize() raises GEEInitializationError."""
        mock_ee.Initialize.side_effect = Exception("Invalid project")
        with patch.dict(sys.modules, {"ee": mock_ee}):
            with pytest.raises(GEEInitializationError, match="Invalid project"):
                auth_manager_interactive.authenticate_and_initialize()
        assert auth_manager_interactive.is_initialized is False
    
    # import builtins
    # _ORIGINAL_IMPORT = builtins.__import__
    def test_ee_not_installed_raises_gee_not_installed_error(
        self,
        auth_manager_interactive: AuthManager,
    ) -> None:
        """Missing earthengine-api raises GEENotInstalledError."""
        modules_without_ee = {
            k: v for k, v in sys.modules.items() if k != "ee"
        }
        with patch.dict(sys.modules, modules_without_ee, clear=True):
            with patch("builtins.__import__", side_effect=_block_ee_import):
                with pytest.raises(GEENotInstalledError):
                    auth_manager_interactive.authenticate_and_initialize()

    def test_initialize_error_message_includes_project_id(
        self,
        auth_manager_interactive: AuthManager,
        mock_ee: MagicMock,
    ) -> None:
        """GEEInitializationError message must include the project ID."""
        mock_ee.Initialize.side_effect = Exception("Bad project")
        with patch.dict(sys.modules, {"ee": mock_ee}):
            with pytest.raises(GEEInitializationError) as exc_info:
                auth_manager_interactive.authenticate_and_initialize()
        assert "test-project-123" in str(exc_info.value)


# ==============================================================================
# AuthManager.authenticate_and_initialize — service account tests
# ==============================================================================

class TestAuthManagerServiceAccount:
    """Tests for service account authentication workflow."""

    def test_service_account_success(
        self,
        auth_manager_service_account: AuthManager,
        mock_ee: MagicMock,
    ) -> None:
        """Successful service account auth sets is_initialized to True."""
        mock_ee.ServiceAccountCredentials.return_value = MagicMock()
        with patch.dict(sys.modules, {"ee": mock_ee}):
            auth_manager_service_account.authenticate_and_initialize()
        assert auth_manager_service_account.is_initialized is True

    def test_service_account_does_not_call_authenticate(
        self,
        auth_manager_service_account: AuthManager,
        mock_ee: MagicMock,
    ) -> None:
        """Service account workflow must NOT call ee.Authenticate()."""
        mock_ee.ServiceAccountCredentials.return_value = MagicMock()
        with patch.dict(sys.modules, {"ee": mock_ee}):
            auth_manager_service_account.authenticate_and_initialize()
        mock_ee.Authenticate.assert_not_called()

    def test_service_account_calls_initialize_with_credentials(
        self,
        auth_manager_service_account: AuthManager,
        mock_ee: MagicMock,
    ) -> None:
        """ee.Initialize() must receive the credentials keyword argument."""
        fake_creds = MagicMock()
        mock_ee.ServiceAccountCredentials.return_value = fake_creds
        with patch.dict(sys.modules, {"ee": mock_ee}):
            auth_manager_service_account.authenticate_and_initialize()
        call_kwargs = mock_ee.Initialize.call_args[1]
        assert call_kwargs.get("credentials") is fake_creds

    def test_key_file_not_found_raises_gee_authentication_error(
        self,
        tmp_path: Path,
        mock_ee: MagicMock,
    ) -> None:
        """Non-existent key file raises GEEAuthenticationError."""
        manager = AuthManager(
            project_id="test-project",
            service_account_key=str(tmp_path / "missing.json"),
        )
        with patch.dict(sys.modules, {"ee": mock_ee}):
            with pytest.raises(GEEAuthenticationError, match="not found"):
                manager.authenticate_and_initialize()

    def test_invalid_json_key_file_raises_gee_authentication_error(
        self,
        tmp_path: Path,
        mock_ee: MagicMock,
    ) -> None:
        """Malformed JSON key file raises GEEAuthenticationError."""
        bad_key = tmp_path / "bad.json"
        bad_key.write_text("not_valid_json{{{{", encoding="utf-8")
        manager = AuthManager(
            project_id="test-project",
            service_account_key=str(bad_key),
        )
        with patch.dict(sys.modules, {"ee": mock_ee}):
            with pytest.raises(GEEAuthenticationError, match="invalid JSON"):
                manager.authenticate_and_initialize()

    def test_key_file_missing_client_email_raises(
        self,
        tmp_path: Path,
        mock_ee: MagicMock,
    ) -> None:
        """Key file without client_email raises GEEAuthenticationError."""
        bad_key = tmp_path / "no_email.json"
        bad_key.write_text(
            json.dumps({"type": "service_account"}), encoding="utf-8"
        )
        manager = AuthManager(
            project_id="test-project",
            service_account_key=str(bad_key),
        )
        with patch.dict(sys.modules, {"ee": mock_ee}):
            with pytest.raises(GEEAuthenticationError, match="client_email"):
                manager.authenticate_and_initialize()

    def test_service_account_credential_creation_failure_raises(
        self,
        auth_manager_service_account: AuthManager,
        mock_ee: MagicMock,
    ) -> None:
        """ee.ServiceAccountCredentials() failure raises GEEAuthenticationError."""
        mock_ee.ServiceAccountCredentials.side_effect = Exception("Bad key format")
        with patch.dict(sys.modules, {"ee": mock_ee}):
            with pytest.raises(GEEAuthenticationError, match="Bad key format"):
                auth_manager_service_account.authenticate_and_initialize()


# ==============================================================================
# Private helper
# ==============================================================================

# def _block_ee_import(name: str, *args, **kwargs):
#     """Import hook that raises ImportError for the 'ee' package only."""
#     if name == "ee":
#         raise ImportError("Simulated: ee not installed")
#     return __builtins__.__import__(name, *args, **kwargs)
# import builtins
# _ORIGINAL_IMPORT = builtins.__import__

def _block_ee_import(name, *args, **kwargs):
    if name == "ee":
        raise ImportError("Simulated: ee not installed")
    return _ORIGINAL_IMPORT(name, *args, **kwargs)