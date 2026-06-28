"""
Unit tests for src/gee/client.py.

Tests cover:
    - RetryConfig defaults and custom values
    - _is_transient_error() classification logic
    - _compute_delay() backoff formula
    - EarthEngineClient lifecycle (init, is_initialized, project_id)
    - execute_with_retry() success, retry, max-attempts, permanent errors
    - get_aoi_geometry() with configured and null AOI
    - get_image_collection() success and error paths
    - health_check() delegation to HealthChecker

Run:
    pytest tests/test_client.py -v
    pytest tests/test_client.py -v --cov=src/gee/client --cov-report=term-missing
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest
import yaml

from src.core.config import Config
from src.core.exceptions import MissingFieldError
from src.gee import (
    GEEAPIError,
    GEEGeometryError,
    GEENotInstalledError,
    GEEQuotaError,
)
from src.gee.client import (
    EarthEngineClient,
    RetryConfig,
    _compute_delay,
    _is_transient_error,
)
from tests.conftest import make_valid_config, write_config


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def mock_ee() -> MagicMock:
    """Mock earthengine-api module."""
    ee = MagicMock()
    ee.__version__ = "0.1.390"
    ee.Authenticate.return_value = None
    ee.Initialize.return_value = None
    return ee


@pytest.fixture
def config_with_aoi(tmp_path: Path) -> Config:
    """Config with all AOI coordinates set."""
    data = make_valid_config()
    data["aoi"].update({
        "min_lon": 87.0, "min_lat": 26.0,
        "max_lon": 87.5, "max_lat": 26.5,
    })
    return Config(config_path=write_config(tmp_path, data))


@pytest.fixture
def config_no_aoi(tmp_path: Path) -> Config:
    """Config with null AOI coordinates."""
    return Config(config_path=write_config(tmp_path, make_valid_config()))


@pytest.fixture
def initialized_client(
    config_with_aoi: Config,
    mock_ee: MagicMock,
) -> EarthEngineClient:
    """Return an EarthEngineClient that has been initialized with a mocked EE."""
    client = EarthEngineClient(config_with_aoi)
    with (
        patch.dict(sys.modules, {"ee": mock_ee}),
        patch.dict(
            __import__("os").environ,
            {"GEE_PROJECT_ID": "test-project"},
        ),
    ):
        client.initialize()
    return client


# ==============================================================================
# RetryConfig tests
# ==============================================================================

class TestRetryConfig:
    """Tests for the RetryConfig dataclass."""

    def test_default_values(self) -> None:
        cfg = RetryConfig()
        assert cfg.max_attempts       == 3
        assert cfg.base_delay_seconds == 1.0
        assert cfg.max_delay_seconds  == 30.0
        assert cfg.backoff_factor     == 2.0
        assert cfg.jitter             is True

    def test_custom_values(self) -> None:
        cfg = RetryConfig(
            max_attempts=5,
            base_delay_seconds=0.5,
            max_delay_seconds=60.0,
            backoff_factor=3.0,
            jitter=False,
        )
        assert cfg.max_attempts       == 5
        assert cfg.base_delay_seconds == 0.5
        assert cfg.max_delay_seconds  == 60.0
        assert cfg.backoff_factor     == 3.0
        assert cfg.jitter             is False

    def test_max_attempts_one_means_no_retry(self) -> None:
        cfg = RetryConfig(max_attempts=1)
        assert cfg.max_attempts == 1


# ==============================================================================
# _is_transient_error tests
# ==============================================================================

class TestIsTransientError:
    """Tests for the _is_transient_error() classification function."""

    @pytest.mark.parametrize("message", [
        "Quota exceeded",
        "QUOTA EXCEEDED",
        "Rate limit reached",
        "Service unavailable",
        "Internal error occurred",
        "Request timed out",
        "Too many requests",
        "Backend error",
        "Connection reset by peer",
        "Error 429",
        "Error 503",
        "Error 500",
        "Temporarily unavailable",
    ])
    def test_transient_messages_return_true(self, message: str) -> None:
        exc = Exception(message)
        assert _is_transient_error(exc) is True, (
            f"Expected transient for: {message!r}"
        )

    @pytest.mark.parametrize("message", [
        "Unauthorized access",
        "Forbidden",
        "Permission denied",
        "Access denied",
        "Invalid credentials",
        "Token expired",
        "Invalid project",
        "Project not found",
        "Not found",
        "Invalid argument",
        "Malformed request",
        "Unauthenticated",
    ])
    def test_permanent_messages_return_false(self, message: str) -> None:
        exc = Exception(message)
        assert _is_transient_error(exc) is False, (
            f"Expected permanent for: {message!r}"
        )

    def test_unknown_error_returns_false(self) -> None:
        """Default classification for unknown errors is permanent (conservative)."""
        assert _is_transient_error(Exception("Something completely unknown")) is False

    def test_permanent_takes_priority_over_transient(self) -> None:
        """If message matches both patterns, permanent takes precedence."""
        exc = Exception("Unauthorized quota exceeded")
        assert _is_transient_error(exc) is False


# ==============================================================================
# _compute_delay tests
# ==============================================================================

class TestComputeDelay:
    """Tests for the _compute_delay() backoff formula."""

    def test_first_attempt_uses_base_delay_without_jitter(self) -> None:
        cfg = RetryConfig(
            base_delay_seconds=1.0,
            backoff_factor=2.0,
            max_delay_seconds=30.0,
            jitter=False,
        )
        delay = _compute_delay(attempt=1, config=cfg)
        assert delay == pytest.approx(1.0)

    def test_second_attempt_doubles_delay(self) -> None:
        cfg = RetryConfig(
            base_delay_seconds=1.0,
            backoff_factor=2.0,
            max_delay_seconds=30.0,
            jitter=False,
        )
        delay = _compute_delay(attempt=2, config=cfg)
        assert delay == pytest.approx(2.0)

    def test_delay_capped_at_max(self) -> None:
        cfg = RetryConfig(
            base_delay_seconds=1.0,
            backoff_factor=2.0,
            max_delay_seconds=5.0,
            jitter=False,
        )
        delay = _compute_delay(attempt=10, config=cfg)
        assert delay == pytest.approx(5.0)

    def test_jitter_reduces_delay(self) -> None:
        cfg = RetryConfig(
            base_delay_seconds=10.0,
            backoff_factor=1.0,
            max_delay_seconds=30.0,
            jitter=True,
        )
        with patch("src.gee.client.random.random", return_value=0.0):
            delay = _compute_delay(attempt=1, config=cfg)
        # With random=0.0: delay *= 0.5 + 0.0*0.5 = 0.5
        assert delay == pytest.approx(5.0)

    def test_jitter_does_not_exceed_full_delay(self) -> None:
        cfg = RetryConfig(
            base_delay_seconds=10.0,
            backoff_factor=1.0,
            max_delay_seconds=30.0,
            jitter=True,
        )
        with patch("src.gee.client.random.random", return_value=1.0):
            delay = _compute_delay(attempt=1, config=cfg)
        # With random=1.0: delay *= 0.5 + 1.0*0.5 = 1.0
        assert delay == pytest.approx(10.0)


# ==============================================================================
# EarthEngineClient lifecycle tests
# ==============================================================================

class TestEarthEngineClientLifecycle:
    """Tests for construction, initialization, and property access."""

    def test_construction_does_not_initialize(
        self, config_no_aoi: Config
    ) -> None:
        client = EarthEngineClient(config_no_aoi)
        assert client.is_initialized is False

    def test_construction_project_id_is_none(
        self, config_no_aoi: Config
    ) -> None:
        client = EarthEngineClient(config_no_aoi)
        assert client.project_id is None

    def test_initialize_sets_is_initialized(
        self,
        config_with_aoi: Config,
        mock_ee: MagicMock,
    ) -> None:
        client = EarthEngineClient(config_with_aoi)
        with (
            patch.dict(sys.modules, {"ee": mock_ee}),
            patch.dict(
                __import__("os").environ,
                {"GEE_PROJECT_ID": "my-project"},
            ),
        ):
            client.initialize()
        assert client.is_initialized is True

    def test_initialize_sets_project_id(
        self,
        config_with_aoi: Config,
        mock_ee: MagicMock,
    ) -> None:
        client = EarthEngineClient(config_with_aoi)
        with (
            patch.dict(sys.modules, {"ee": mock_ee}),
            patch.dict(
                __import__("os").environ,
                {"GEE_PROJECT_ID": "my-project-id"},
            ),
        ):
            client.initialize()
        assert client.project_id == "my-project-id"

    def test_initialize_is_idempotent(
        self,
        config_with_aoi: Config,
        mock_ee: MagicMock,
    ) -> None:
        """Calling initialize() twice must not re-authenticate."""
        client = EarthEngineClient(config_with_aoi)
        with (
            patch.dict(sys.modules, {"ee": mock_ee}),
            patch.dict(
                __import__("os").environ,
                {"GEE_PROJECT_ID": "my-project"},
            ),
        ):
            client.initialize()
            client.initialize()
        assert mock_ee.Initialize.call_count == 1

    def test_missing_project_id_raises(
        self, config_with_aoi: Config
    ) -> None:
        """Missing GEE_PROJECT_ID env var raises GEECredentialError."""
        from src.core.exceptions import GEECredentialError
        client = EarthEngineClient(config_with_aoi)
        env_without_gee = {
            k: v for k, v in __import__("os").environ.items()
            if k != "GEE_PROJECT_ID"
        }
        with patch.dict(__import__("os").environ, env_without_gee, clear=True):
            with pytest.raises(GEECredentialError):
                client.initialize()

    def test_repr_contains_initialized_state(
        self, config_no_aoi: Config
    ) -> None:
        client = EarthEngineClient(config_no_aoi)
        r = repr(client)
        assert "initialized=False" in r
        assert all(ord(c) < 128 for c in r), "repr must be ASCII-only"

    def test_custom_retry_config(self, config_no_aoi: Config) -> None:
        retry = RetryConfig(max_attempts=5, jitter=False)
        client = EarthEngineClient(config_no_aoi, retry_config=retry)
        assert client._retry_config.max_attempts == 5

    def test_default_retry_config_applied(self, config_no_aoi: Config) -> None:
        client = EarthEngineClient(config_no_aoi)
        assert client._retry_config.max_attempts == 3


# ==============================================================================
# EarthEngineClient.get_aoi_geometry tests
# ==============================================================================

class TestGetAoiGeometry:
    """Tests for EarthEngineClient.get_aoi_geometry()."""

    def test_returns_geometry_when_aoi_configured(
        self,
        config_with_aoi: Config,
        mock_ee: MagicMock,
    ) -> None:
        mock_geometry = MagicMock()
        mock_ee.Geometry.Rectangle.return_value = mock_geometry

        client = EarthEngineClient(config_with_aoi)
        with patch.dict(sys.modules, {"ee": mock_ee}):
            result = client.get_aoi_geometry()

        assert result is mock_geometry

    def test_calls_rectangle_with_correct_coords(
        self,
        config_with_aoi: Config,
        mock_ee: MagicMock,
    ) -> None:
        client = EarthEngineClient(config_with_aoi)
        with patch.dict(sys.modules, {"ee": mock_ee}):
            client.get_aoi_geometry()

        call_args = mock_ee.Geometry.Rectangle.call_args[0][0]
        assert call_args == [87.0, 26.0, 87.5, 26.5]

    def test_raises_missing_field_error_when_aoi_null(
        self,
        config_no_aoi: Config,
        mock_ee: MagicMock,
    ) -> None:
        client = EarthEngineClient(config_no_aoi)
        with patch.dict(sys.modules, {"ee": mock_ee}):
            with pytest.raises(MissingFieldError):
                client.get_aoi_geometry()

    def test_raises_gee_not_installed_when_ee_missing(
        self,
        config_with_aoi: Config,
    ) -> None:
        client = EarthEngineClient(config_with_aoi)
        modules_without_ee = {k: v for k, v in sys.modules.items() if k != "ee"}
        with patch.dict(sys.modules, modules_without_ee, clear=True):
            with patch("builtins.__import__", side_effect=_block_ee_import):
                with pytest.raises(GEENotInstalledError):
                    client.get_aoi_geometry()

    def test_geometry_creation_failure_raises_gee_geometry_error(
        self,
        config_with_aoi: Config,
        mock_ee: MagicMock,
    ) -> None:
        mock_ee.Geometry.Rectangle.side_effect = Exception("Invalid coords")
        client = EarthEngineClient(config_with_aoi)
        with patch.dict(sys.modules, {"ee": mock_ee}):
            with pytest.raises(GEEGeometryError, match="Invalid coords"):
                client.get_aoi_geometry()


# ==============================================================================
# EarthEngineClient.get_image_collection tests
# ==============================================================================

class TestGetImageCollection:
    """Tests for EarthEngineClient.get_image_collection()."""

    def test_returns_image_collection(
        self,
        initialized_client: EarthEngineClient,
        mock_ee: MagicMock,
    ) -> None:
        mock_collection = MagicMock()
        mock_ee.ImageCollection.return_value = mock_collection

        with patch.dict(sys.modules, {"ee": mock_ee}):
            result = initialized_client.get_image_collection(
                "LANDSAT/LC08/C02/T1_L2"
            )
        assert result is mock_collection

    def test_raises_runtime_error_when_not_initialized(
        self,
        config_no_aoi: Config,
        mock_ee: MagicMock,
    ) -> None:
        client = EarthEngineClient(config_no_aoi)
        with patch.dict(sys.modules, {"ee": mock_ee}):
            with pytest.raises(RuntimeError, match="not initialized"):
                client.get_image_collection("LANDSAT/LC08/C02/T1_L2")

    def test_empty_collection_id_raises_gee_api_error(
        self,
        initialized_client: EarthEngineClient,
        mock_ee: MagicMock,
    ) -> None:
        with patch.dict(sys.modules, {"ee": mock_ee}):
            with pytest.raises(GEEAPIError, match="non-empty"):
                initialized_client.get_image_collection("   ")

    def test_ee_rejection_raises_gee_api_error(
        self,
        initialized_client: EarthEngineClient,
        mock_ee: MagicMock,
    ) -> None:
        mock_ee.ImageCollection.side_effect = Exception("Asset not found")
        with patch.dict(sys.modules, {"ee": mock_ee}):
            with pytest.raises(GEEAPIError, match="Asset not found"):
                initialized_client.get_image_collection("INVALID/COLLECTION")


# ==============================================================================
# EarthEngineClient.execute_with_retry tests
# ==============================================================================

class TestExecuteWithRetry:
    """Tests for the exponential backoff retry mechanism."""

    def test_success_on_first_attempt(
        self,
        config_no_aoi: Config,
    ) -> None:
        client = EarthEngineClient(config_no_aoi)
        func   = MagicMock(return_value=42)

        with patch("src.gee.client.time.sleep"):
            result = client.execute_with_retry(func)

        assert result == 42
        assert func.call_count == 1

    def test_retries_on_transient_error(
        self,
        config_no_aoi: Config,
    ) -> None:
        """Transient error on first attempt, success on second."""
        client = EarthEngineClient(
            config_no_aoi,
            retry_config=RetryConfig(max_attempts=3, jitter=False),
        )
        func = MagicMock(
            side_effect=[Exception("quota exceeded"), "success"]
        )

        with patch("src.gee.client.time.sleep"):
            result = client.execute_with_retry(func)

        assert result == "success"
        assert func.call_count == 2

    def test_raises_gee_api_error_after_max_attempts(
        self,
        config_no_aoi: Config,
    ) -> None:
        """Max retries exceeded raises GEEAPIError."""
        client = EarthEngineClient(
            config_no_aoi,
            retry_config=RetryConfig(max_attempts=3, jitter=False),
        )
        func = MagicMock(side_effect=Exception("service unavailable"))

        with patch("src.gee.client.time.sleep"):
            with pytest.raises(GEEAPIError, match="Max retry attempts"):
                client.execute_with_retry(func)

        assert func.call_count == 3

    def test_permanent_error_raises_immediately_without_retry(
        self,
        config_no_aoi: Config,
    ) -> None:
        """Non-transient error must raise immediately on first failure."""
        client = EarthEngineClient(
            config_no_aoi,
            retry_config=RetryConfig(max_attempts=3, jitter=False),
        )
        func = MagicMock(side_effect=Exception("Unauthorized access"))

        with patch("src.gee.client.time.sleep") as mock_sleep:
            with pytest.raises(GEEAPIError, match="Unauthorized"):
                client.execute_with_retry(func)

        assert func.call_count == 1
        mock_sleep.assert_not_called()

    def test_quota_error_raises_gee_quota_error(
        self,
        config_no_aoi: Config,
    ) -> None:
        """Quota-related transient errors raise GEEQuotaError after max attempts."""
        client = EarthEngineClient(
            config_no_aoi,
            retry_config=RetryConfig(max_attempts=2, jitter=False),
        )
        func = MagicMock(side_effect=Exception("429 quota exceeded"))

        with patch("src.gee.client.time.sleep"):
            with pytest.raises(GEEQuotaError):
                client.execute_with_retry(func)

    def test_sleep_is_called_between_retries(
        self,
        config_no_aoi: Config,
    ) -> None:
        """time.sleep() must be called between retry attempts."""
        client = EarthEngineClient(
            config_no_aoi,
            retry_config=RetryConfig(
                max_attempts=3, base_delay_seconds=1.0, jitter=False
            ),
        )
        func = MagicMock(side_effect=Exception("service unavailable"))

        with patch("src.gee.client.time.sleep") as mock_sleep:
            with pytest.raises(GEEAPIError):
                client.execute_with_retry(func)

        assert mock_sleep.call_count == 2  # Called between attempts 1->2 and 2->3

    def test_no_retry_when_max_attempts_is_one(
        self,
        config_no_aoi: Config,
    ) -> None:
        """RetryConfig(max_attempts=1) means no retries, just one attempt."""
        client = EarthEngineClient(
            config_no_aoi,
            retry_config=RetryConfig(max_attempts=1, jitter=False),
        )
        func = MagicMock(side_effect=Exception("quota exceeded"))

        with patch("src.gee.client.time.sleep") as mock_sleep:
            with pytest.raises(GEEAPIError):
                client.execute_with_retry(func)

        assert func.call_count == 1
        mock_sleep.assert_not_called()

    def test_lambda_is_accepted_as_func(
        self,
        config_no_aoi: Config,
    ) -> None:
        """execute_with_retry must accept a lambda, not just named functions."""
        client = EarthEngineClient(config_no_aoi)
        result = client.execute_with_retry(lambda: "lambda_result")
        assert result == "lambda_result"


# ==============================================================================
# EarthEngineClient.health_check tests
# ==============================================================================

class TestHealthCheck:
    """Tests for health_check() delegation to HealthChecker."""

    def test_health_check_returns_health_report(
        self,
        config_no_aoi: Config,
    ) -> None:
        from src.gee.health import HealthReport

        client      = EarthEngineClient(config_no_aoi)
        mock_report = MagicMock(spec=HealthReport)
        mock_checker_instance = MagicMock()
        mock_checker_instance.check_all.return_value = mock_report

        with patch("src.gee.client.HealthChecker") as mock_checker_cls:
            mock_checker_cls.return_value = mock_checker_instance
            result = client.health_check()

        assert result is mock_report
        mock_checker_cls.assert_called_once()
        mock_checker_instance.check_all.assert_called_once()

    def test_health_check_passes_collections_from_config(
        self,
        tmp_path: Path,
    ) -> None:
        data = make_valid_config()
        data["aoi"].update({
            "min_lon": 87.0, "min_lat": 26.0,
            "max_lon": 87.5, "max_lat": 26.5,
        })
        config = Config(config_path=write_config(tmp_path, data))
        client = EarthEngineClient(config)

        mock_checker_instance = MagicMock()
        mock_checker_instance.check_all.return_value = MagicMock()

        with patch("src.gee.client.HealthChecker") as mock_checker_cls:
            mock_checker_cls.return_value = mock_checker_instance
            client.health_check()

        ctor_kwargs = mock_checker_cls.call_args[1]
        collections = ctor_kwargs.get("config_collections", [])
        assert "LANDSAT/LC08/C02/T1_L2" in collections


# ==============================================================================
# Private helper
# ==============================================================================
# import builtins

# _original_import = builtins.__import__
# def _block_ee_import(name: str, *args, **kwargs):
#     """Import side-effect that blocks only the 'ee' package."""
#     if name == "ee":
#         raise ImportError("Simulated: ee not installed")
#     import builtins
#     return builtins.__import__(name, *args, **kwargs)
import builtins

_original_import = builtins.__import__

def _block_ee_import(name, *args, **kwargs):
    if name == "ee":
        raise ImportError("Simulated: ee not installed")
    return _original_import(name, *args, **kwargs)