"""
Unit tests for src/gee/health.py.

Tests cover:
    - HealthStatus enum values and ASCII compliance
    - HealthCheckItem frozen dataclass
    - HealthReport computed properties and summary_lines()
    - HealthChecker individual check methods (mocked EE)
    - HealthChecker.check_all() sequencing and SKIP propagation

Run:
    pytest tests/test_health.py -v
    pytest tests/test_health.py -v --cov=src/gee/health --cov-report=term-missing
"""

from __future__ import annotations

import sys
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.gee.health import (
    HealthCheckItem,
    HealthChecker,
    HealthReport,
    HealthStatus,
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
    ee.Number.return_value.getInfo.return_value = 1
    ee.ImageCollection.return_value.limit.return_value.size.return_value.getInfo.return_value = 5
    return ee


@pytest.fixture
def uninitialized_client(tmp_path: Path) -> MagicMock:
    """Mock EarthEngineClient that is NOT initialized."""
    from src.gee.client import EarthEngineClient
    client = MagicMock(spec=EarthEngineClient)
    client.is_initialized = False
    client.project_id     = None
    return client


@pytest.fixture
def initialized_client_mock(tmp_path: Path) -> MagicMock:
    """Mock EarthEngineClient that IS initialized with a valid project ID."""
    from src.gee.client import EarthEngineClient
    client = MagicMock(spec=EarthEngineClient)
    client.is_initialized = True
    client.project_id     = "test-project-123"
    return client


# ==============================================================================
# HealthStatus tests
# ==============================================================================

class TestHealthStatus:
    """Tests for the HealthStatus string enum."""

    def test_all_values_are_ascii(self) -> None:
        for status in HealthStatus:
            assert all(ord(c) < 128 for c in status.value), (
                f"Non-ASCII character in HealthStatus.{status.name}: {status.value!r}"
            )

    def test_values_match_expected_strings(self) -> None:
        assert HealthStatus.OK    == "OK"
        assert HealthStatus.WARN  == "WARN"
        assert HealthStatus.ERROR == "ERROR"
        assert HealthStatus.SKIP  == "SKIP"

    def test_is_str_subclass(self) -> None:
        assert isinstance(HealthStatus.OK, str)

    def test_comparison_with_plain_string(self) -> None:
        assert HealthStatus.OK == "OK"
        assert HealthStatus.ERROR != "OK"


# ==============================================================================
# HealthCheckItem tests
# ==============================================================================

class TestHealthCheckItem:
    """Tests for the frozen HealthCheckItem dataclass."""

    def test_construction(self) -> None:
        item = HealthCheckItem(
            name="test_check",
            status=HealthStatus.OK,
            message="Everything is fine.",
        )
        assert item.name    == "test_check"
        assert item.status  == HealthStatus.OK
        assert item.message == "Everything is fine."
        assert item.details == ""

    def test_with_details(self) -> None:
        item = HealthCheckItem(
            name="test",
            status=HealthStatus.ERROR,
            message="Failed.",
            details="Connection refused on port 443.",
        )
        assert item.details == "Connection refused on port 443."

    def test_frozen_prevents_mutation(self) -> None:
        item = HealthCheckItem(
            name="test", status=HealthStatus.OK, message="ok"
        )
        with pytest.raises((AttributeError, TypeError)):
            item.status = HealthStatus.ERROR  # type: ignore[misc]

    def test_is_ok_true_for_ok_status(self) -> None:
        item = HealthCheckItem(name="x", status=HealthStatus.OK, message="ok")
        assert item.is_ok() is True

    def test_is_ok_false_for_error(self) -> None:
        item = HealthCheckItem(name="x", status=HealthStatus.ERROR, message="bad")
        assert item.is_ok() is False

    def test_is_error_true_for_error_status(self) -> None:
        item = HealthCheckItem(name="x", status=HealthStatus.ERROR, message="bad")
        assert item.is_error() is True

    def test_is_skipped_true_for_skip_status(self) -> None:
        item = HealthCheckItem(name="x", status=HealthStatus.SKIP, message="skip")
        assert item.is_skipped() is True


# ==============================================================================
# HealthReport tests
# ==============================================================================

class TestHealthReport:
    """Tests for HealthReport computed properties and summary output."""

    def _make_item(
        self,
        name: str = "check",
        status: str = HealthStatus.OK,
        message: str = "ok",
    ) -> HealthCheckItem:
        return HealthCheckItem(name=name, status=status, message=message)

    def test_all_ok_overall_is_ok(self) -> None:
        report = HealthReport(items=[
            self._make_item("a", HealthStatus.OK),
            self._make_item("b", HealthStatus.OK),
        ])
        assert report.overall == HealthStatus.OK

    def test_one_warn_overall_is_warn(self) -> None:
        report = HealthReport(items=[
            self._make_item("a", HealthStatus.OK),
            self._make_item("b", HealthStatus.WARN),
        ])
        assert report.overall == HealthStatus.WARN

    def test_one_error_overall_is_error_regardless_of_others(self) -> None:
        report = HealthReport(items=[
            self._make_item("a", HealthStatus.OK),
            self._make_item("b", HealthStatus.WARN),
            self._make_item("c", HealthStatus.ERROR),
        ])
        assert report.overall == HealthStatus.ERROR

    def test_error_takes_priority_over_warn(self) -> None:
        report = HealthReport(items=[
            self._make_item("a", HealthStatus.WARN),
            self._make_item("b", HealthStatus.ERROR),
        ])
        assert report.overall == HealthStatus.ERROR

    def test_skip_only_overall_is_ok(self) -> None:
        report = HealthReport(items=[
            self._make_item("a", HealthStatus.SKIP),
        ])
        assert report.overall == HealthStatus.OK

    def test_is_healthy_true_when_no_errors(self) -> None:
        report = HealthReport(items=[
            self._make_item("a", HealthStatus.OK),
            self._make_item("b", HealthStatus.WARN),
        ])
        assert report.is_healthy is True

    def test_is_healthy_false_when_any_error(self) -> None:
        report = HealthReport(items=[
            self._make_item("a", HealthStatus.OK),
            self._make_item("b", HealthStatus.ERROR),
        ])
        assert report.is_healthy is False

    def test_is_healthy_true_for_empty_items(self) -> None:
        report = HealthReport(items=[])
        assert report.is_healthy is True

    def test_error_items_property(self) -> None:
        ok_item  = self._make_item("a", HealthStatus.OK)
        err_item = self._make_item("b", HealthStatus.ERROR)
        report   = HealthReport(items=[ok_item, err_item])
        assert report.error_items == [err_item]

    def test_warn_items_property(self) -> None:
        ok_item   = self._make_item("a", HealthStatus.OK)
        warn_item = self._make_item("b", HealthStatus.WARN)
        report    = HealthReport(items=[ok_item, warn_item])
        assert report.warn_items == [warn_item]

    def test_summary_lines_returns_list_of_strings(self) -> None:
        report = HealthReport(items=[
            self._make_item("a", HealthStatus.OK),
        ])
        lines = report.summary_lines()
        assert isinstance(lines, list)
        assert all(isinstance(line, str) for line in lines)

    def test_summary_lines_are_ascii_only(self) -> None:
        report = HealthReport(items=[
            self._make_item("a", HealthStatus.OK),
            self._make_item("b", HealthStatus.ERROR, "bad"),
            self._make_item("c", HealthStatus.SKIP, "skipped"),
        ])
        for line in report.summary_lines():
            assert all(ord(c) < 128 for c in line), (
                f"Non-ASCII in summary line: {line!r}"
            )

    def test_summary_lines_contain_check_names(self) -> None:
        report = HealthReport(items=[
            self._make_item("my_special_check", HealthStatus.OK),
        ])
        lines = report.summary_lines()
        assert any("my_special_check" in line for line in lines)

    def test_summary_lines_contain_overall(self) -> None:
        report = HealthReport(items=[
            self._make_item("a", HealthStatus.OK),
        ])
        lines    = report.summary_lines()
        combined = " ".join(lines)
        assert "Overall" in combined

    def test_summary_lines_error_shows_details(self) -> None:
        report = HealthReport(items=[
            HealthCheckItem(
                name="bad_check",
                status=HealthStatus.ERROR,
                message="Something failed.",
                details="Root cause here.",
            )
        ])
        lines    = report.summary_lines()
        combined = " ".join(lines)
        assert "Root cause here." in combined

    def test_ok_item_does_not_show_details(self) -> None:
        report = HealthReport(items=[
            HealthCheckItem(
                name="good_check",
                status=HealthStatus.OK,
                message="All good.",
                details="Unnecessary detail.",
            )
        ])
        lines    = report.summary_lines()
        combined = " ".join(lines)
        assert "Unnecessary detail." not in combined


# ==============================================================================
# HealthChecker individual check tests
# ==============================================================================

class TestHealthCheckerCheckEeInstalled:
    """Tests for HealthChecker._check_ee_installed()."""

    def test_ok_when_ee_importable(
        self,
        initialized_client_mock: MagicMock,
        mock_ee: MagicMock,
    ) -> None:
        checker = HealthChecker(initialized_client_mock)
        with patch.dict(sys.modules, {"ee": mock_ee}):
            item = checker._check_ee_installed()
        assert item.status == HealthStatus.OK
        assert item.name   == "ee_installed"

    def test_error_when_ee_not_importable(
        self,
        initialized_client_mock: MagicMock,
    ) -> None:
        checker = HealthChecker(initialized_client_mock)
        modules_without_ee = {k: v for k, v in sys.modules.items() if k != "ee"}
        with patch.dict(sys.modules, modules_without_ee, clear=True):
            with patch("builtins.__import__", side_effect=_block_ee_import):
                item = checker._check_ee_installed()
        assert item.status == HealthStatus.ERROR

    def test_includes_version_in_message(
        self,
        initialized_client_mock: MagicMock,
        mock_ee: MagicMock,
    ) -> None:
        checker = HealthChecker(initialized_client_mock)
        with patch.dict(sys.modules, {"ee": mock_ee}):
            item = checker._check_ee_installed()
        assert "0.1.390" in item.message


class TestHealthCheckerCheckAuthentication:
    """Tests for HealthChecker._check_authentication()."""

    def test_ok_when_client_initialized(
        self, initialized_client_mock: MagicMock
    ) -> None:
        checker = HealthChecker(initialized_client_mock)
        item    = checker._check_authentication()
        assert item.status == HealthStatus.OK

    def test_error_when_client_not_initialized(
        self, uninitialized_client: MagicMock
    ) -> None:
        checker = HealthChecker(uninitialized_client)
        item    = checker._check_authentication()
        assert item.status == HealthStatus.ERROR


class TestHealthCheckerCheckProjectId:
    """Tests for HealthChecker._check_project_id()."""

    def test_ok_when_project_id_set(
        self, initialized_client_mock: MagicMock
    ) -> None:
        initialized_client_mock.project_id = "my-project"
        checker = HealthChecker(initialized_client_mock)
        item    = checker._check_project_id()
        assert item.status == HealthStatus.OK
        assert "my-project" in item.message

    def test_error_when_project_id_none(
        self, uninitialized_client: MagicMock
    ) -> None:
        uninitialized_client.project_id = None
        checker = HealthChecker(uninitialized_client)
        item    = checker._check_project_id()
        assert item.status == HealthStatus.ERROR

    def test_error_when_project_id_raises(
        self, initialized_client_mock: MagicMock
    ) -> None:
        type(initialized_client_mock).project_id = property(
            lambda self: (_ for _ in ()).throw(Exception("Env not set"))
        )
        checker = HealthChecker(initialized_client_mock)
        item    = checker._check_project_id()
        assert item.status == HealthStatus.ERROR


class TestHealthCheckerCheckConnectivity:
    """Tests for HealthChecker._check_connectivity()."""

    def test_ok_on_successful_http_response(
        self, initialized_client_mock: MagicMock
    ) -> None:
        checker = HealthChecker(initialized_client_mock)
        with patch("src.gee.health.urllib.request.urlopen"):
            item = checker._check_connectivity()
        assert item.status == HealthStatus.OK

    def test_ok_on_http_error_response(
        self, initialized_client_mock: MagicMock
    ) -> None:
        """HTTP 403 still means server is reachable."""
        checker = HealthChecker(initialized_client_mock)
        http_err = urllib.error.HTTPError(
            url="https://earthengine.googleapis.com",
            code=403, msg="Forbidden",
            hdrs=None, fp=None,
        )
        with patch(
            "src.gee.health.urllib.request.urlopen",
            side_effect=http_err,
        ):
            item = checker._check_connectivity()
        assert item.status == HealthStatus.OK
        assert "403" in item.message

    def test_error_on_url_error(
        self, initialized_client_mock: MagicMock
    ) -> None:
        checker  = HealthChecker(initialized_client_mock)
        url_err  = urllib.error.URLError(reason="Name or service not known")
        with patch(
            "src.gee.health.urllib.request.urlopen",
            side_effect=url_err,
        ):
            item = checker._check_connectivity()
        assert item.status == HealthStatus.ERROR

    def test_warn_on_timeout(
        self, initialized_client_mock: MagicMock
    ) -> None:
        checker = HealthChecker(initialized_client_mock)
        with patch(
            "src.gee.health.urllib.request.urlopen",
            side_effect=TimeoutError("timed out"),
        ):
            item = checker._check_connectivity()
        assert item.status == HealthStatus.WARN


class TestHealthCheckerCheckApiAccess:
    """Tests for HealthChecker._check_api_access()."""

    def test_ok_when_ee_responds_correctly(
        self,
        initialized_client_mock: MagicMock,
        mock_ee: MagicMock,
    ) -> None:
        mock_ee.Number.return_value.getInfo.return_value = 1
        checker = HealthChecker(initialized_client_mock)
        with patch.dict(sys.modules, {"ee": mock_ee}):
            item = checker._check_api_access()
        assert item.status == HealthStatus.OK

    def test_error_when_ee_call_raises(
        self,
        initialized_client_mock: MagicMock,
        mock_ee: MagicMock,
    ) -> None:
        mock_ee.Number.return_value.getInfo.side_effect = Exception("API down")
        checker = HealthChecker(initialized_client_mock)
        with patch.dict(sys.modules, {"ee": mock_ee}):
            item = checker._check_api_access()
        assert item.status == HealthStatus.ERROR

    def test_warn_when_unexpected_value_returned(
        self,
        initialized_client_mock: MagicMock,
        mock_ee: MagicMock,
    ) -> None:
        mock_ee.Number.return_value.getInfo.return_value = 999
        checker = HealthChecker(initialized_client_mock)
        with patch.dict(sys.modules, {"ee": mock_ee}):
            item = checker._check_api_access()
        assert item.status == HealthStatus.WARN


class TestHealthCheckerCheckPermissions:
    """Tests for HealthChecker._check_permissions()."""

    def test_ok_when_collection_accessible(
        self,
        initialized_client_mock: MagicMock,
        mock_ee: MagicMock,
    ) -> None:
        checker = HealthChecker(
            initialized_client_mock,
            config_collections=["LANDSAT/LC08/C02/T1_L2"],
        )
        with patch.dict(sys.modules, {"ee": mock_ee}):
            item = checker._check_permissions()
        assert item.status == HealthStatus.OK

    def test_skip_when_no_collections_configured(
        self, initialized_client_mock: MagicMock
    ) -> None:
        checker = HealthChecker(
            initialized_client_mock,
            config_collections=[],
        )
        item = checker._check_permissions()
        assert item.status == HealthStatus.SKIP

    def test_error_on_permission_denied(
        self,
        initialized_client_mock: MagicMock,
        mock_ee: MagicMock,
    ) -> None:
        mock_ee.ImageCollection.return_value.limit.return_value.size.return_value.getInfo.side_effect = (
            Exception("access denied to asset")
        )
        checker = HealthChecker(
            initialized_client_mock,
            config_collections=["LANDSAT/LC08/C02/T1_L2"],
        )
        with patch.dict(sys.modules, {"ee": mock_ee}):
            item = checker._check_permissions()
        assert item.status == HealthStatus.ERROR

    def test_error_on_general_collection_failure(
        self,
        initialized_client_mock: MagicMock,
        mock_ee: MagicMock,
    ) -> None:
        mock_ee.ImageCollection.side_effect = Exception("Collection error")
        checker = HealthChecker(
            initialized_client_mock,
            config_collections=["LANDSAT/LC08/C02/T1_L2"],
        )
        with patch.dict(sys.modules, {"ee": mock_ee}):
            item = checker._check_permissions()
        assert item.status == HealthStatus.ERROR


# ==============================================================================
# HealthChecker.check_all() sequencing tests
# ==============================================================================

class TestHealthCheckerCheckAll:
    """Tests for the full check_all() sequencing and SKIP propagation."""

    def test_returns_health_report(
        self,
        uninitialized_client: MagicMock,
        mock_ee: MagicMock,
    ) -> None:
        checker = HealthChecker(uninitialized_client)
        with patch.dict(sys.modules, {"ee": mock_ee}):
            with patch("src.gee.health.urllib.request.urlopen"):
                report = checker.check_all()
        assert isinstance(report, HealthReport)

    def test_report_has_six_items(
        self,
        initialized_client_mock: MagicMock,
        mock_ee: MagicMock,
    ) -> None:
        checker = HealthChecker(initialized_client_mock)
        with patch.dict(sys.modules, {"ee": mock_ee}):
            with patch("src.gee.health.urllib.request.urlopen"):
                report = checker.check_all()
        assert len(report.items) == 6

    def test_api_and_permissions_skipped_when_auth_fails(
        self,
        uninitialized_client: MagicMock,
        mock_ee: MagicMock,
    ) -> None:
        """auth check fails -> api_access and permissions must be SKIP."""
        checker = HealthChecker(uninitialized_client)
        with patch.dict(sys.modules, {"ee": mock_ee}):
            with patch("src.gee.health.urllib.request.urlopen"):
                report = checker.check_all()

        item_map = {item.name: item for item in report.items}
        assert item_map["authentication"].status  == HealthStatus.ERROR
        assert item_map["api_access"].status      == HealthStatus.SKIP
        assert item_map["permissions"].status     == HealthStatus.SKIP

    def test_all_checks_run_independently(
        self,
        initialized_client_mock: MagicMock,
        mock_ee: MagicMock,
    ) -> None:
        """Failure in connectivity must not prevent other checks from running."""
        checker = HealthChecker(initialized_client_mock)
        url_err = urllib.error.URLError(reason="No network")

        with patch.dict(sys.modules, {"ee": mock_ee}):
            with patch(
                "src.gee.health.urllib.request.urlopen",
                side_effect=url_err,
            ):
                report = checker.check_all()

        item_names = [item.name for item in report.items]
        expected   = [
            "ee_installed", "authentication", "project_id",
            "connectivity", "api_access", "permissions",
        ]
        assert item_names == expected

    def test_summary_lines_ascii_only(
        self,
        initialized_client_mock: MagicMock,
        mock_ee: MagicMock,
    ) -> None:
        checker = HealthChecker(initialized_client_mock)
        with patch.dict(sys.modules, {"ee": mock_ee}):
            with patch("src.gee.health.urllib.request.urlopen"):
                report = checker.check_all()

        for line in report.summary_lines():
            assert all(ord(c) < 128 for c in line), (
                f"Non-ASCII in line: {line!r}"
            )


# ==============================================================================
# Private helper
# ==============================================================================

def _block_ee_import(name: str, *args, **kwargs):
    """Import side-effect that raises ImportError only for 'ee'."""
    if name == "ee":
        raise ImportError("Simulated: ee not installed")
    import builtins
    return builtins.__import__(name, *args, **kwargs)