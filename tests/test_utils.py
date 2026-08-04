import datetime
import subprocess

import pytest
from unittest import mock

from openrecall import utils


# ---------------------------------------------------------------------------
# human_readable_time
# ---------------------------------------------------------------------------

def _fixed_now(now):
    class _FixedDateTime(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return now

    return _FixedDateTime


def test_human_readable_time_seconds_ago():
    now = datetime.datetime(2026, 1, 1, 12, 0, 30)
    timestamp = int(datetime.datetime(2026, 1, 1, 12, 0, 0).timestamp())
    with mock.patch("openrecall.utils.datetime.datetime", _fixed_now(now)):
        assert utils.human_readable_time(timestamp) == "30 seconds ago"


def test_human_readable_time_minutes_ago():
    now = datetime.datetime(2026, 1, 1, 12, 10, 0)
    timestamp = int(datetime.datetime(2026, 1, 1, 12, 0, 0).timestamp())
    with mock.patch("openrecall.utils.datetime.datetime", _fixed_now(now)):
        assert utils.human_readable_time(timestamp) == "10 minutes ago"


def test_human_readable_time_hours_ago():
    now = datetime.datetime(2026, 1, 1, 15, 0, 0)
    timestamp = int(datetime.datetime(2026, 1, 1, 12, 0, 0).timestamp())
    with mock.patch("openrecall.utils.datetime.datetime", _fixed_now(now)):
        assert utils.human_readable_time(timestamp) == "3 hours ago"


def test_human_readable_time_days_ago():
    now = datetime.datetime(2026, 1, 5, 12, 0, 0)
    timestamp = int(datetime.datetime(2026, 1, 1, 12, 0, 0).timestamp())
    with mock.patch("openrecall.utils.datetime.datetime", _fixed_now(now)):
        assert utils.human_readable_time(timestamp) == "4 days ago"


def test_human_readable_time_single_day_is_singular():
    now = datetime.datetime(2026, 1, 2, 12, 0, 0)
    timestamp = int(datetime.datetime(2026, 1, 1, 12, 0, 0).timestamp())
    with mock.patch("openrecall.utils.datetime.datetime", _fixed_now(now)):
        assert utils.human_readable_time(timestamp) == "1 day ago"


def test_human_readable_time_future_timestamp_is_not_hours_ago():
    # Clock skew or a DST shift can put a stored timestamp slightly ahead of
    # now; the timedelta then normalizes to days=-1 with a large .seconds.
    now = datetime.datetime(2026, 1, 1, 12, 0, 0)
    timestamp = int(datetime.datetime(2026, 1, 1, 12, 5, 0).timestamp())
    with mock.patch("openrecall.utils.datetime.datetime", _fixed_now(now)):
        assert utils.human_readable_time(timestamp) == "just now"


# ---------------------------------------------------------------------------
# timestamp_to_human_readable
# ---------------------------------------------------------------------------

def test_timestamp_to_human_readable_valid():
    timestamp = int(datetime.datetime(2026, 1, 1, 12, 30, 45).timestamp())
    assert utils.timestamp_to_human_readable(timestamp) == "2026-01-01 12:30:45"


def test_timestamp_to_human_readable_invalid_returns_empty_string():
    class _RaisingDateTime(datetime.datetime):
        @classmethod
        def fromtimestamp(cls, timestamp, tz=None):
            raise OverflowError

    with mock.patch("openrecall.utils.datetime.datetime", _RaisingDateTime):
        assert utils.timestamp_to_human_readable(0) == ""


# ---------------------------------------------------------------------------
# get_active_app_name / get_active_window_title platform dispatch
# ---------------------------------------------------------------------------

def test_get_active_app_name_windows_dispatch():
    with mock.patch("sys.platform", "win32"):
        with mock.patch(
            "openrecall.utils.get_active_app_name_windows", return_value="chrome.exe"
        ) as fn:
            assert utils.get_active_app_name() == "chrome.exe"
            fn.assert_called_once()


def test_get_active_app_name_darwin_dispatch():
    with mock.patch("sys.platform", "darwin"):
        with mock.patch(
            "openrecall.utils.get_active_app_name_osx", return_value="Safari"
        ) as fn:
            assert utils.get_active_app_name() == "Safari"
            fn.assert_called_once()


def test_get_active_app_name_linux_dispatch():
    with mock.patch("sys.platform", "linux"):
        with mock.patch(
            "openrecall.utils.get_active_app_name_linux", return_value="firefox"
        ) as fn:
            assert utils.get_active_app_name() == "firefox"
            fn.assert_called_once()


def test_get_active_app_name_unsupported_platform_raises():
    with mock.patch("sys.platform", "freebsd"):
        with pytest.raises(NotImplementedError):
            utils.get_active_app_name()


def test_get_active_window_title_windows_dispatch():
    with mock.patch("sys.platform", "win32"):
        with mock.patch(
            "openrecall.utils.get_active_window_title_windows",
            return_value="Untitled - Notepad",
        ) as fn:
            assert utils.get_active_window_title() == "Untitled - Notepad"
            fn.assert_called_once()


def test_get_active_window_title_darwin_dispatch():
    with mock.patch("sys.platform", "darwin"):
        with mock.patch(
            "openrecall.utils.get_active_window_title_osx", return_value="Finder"
        ) as fn:
            assert utils.get_active_window_title() == "Finder"
            fn.assert_called_once()


def test_get_active_window_title_linux_dispatch():
    with mock.patch("sys.platform", "linux"):
        with mock.patch(
            "openrecall.utils.get_active_window_title_linux", return_value="Terminal"
        ) as fn:
            assert utils.get_active_window_title() == "Terminal"
            fn.assert_called_once()


def test_get_active_window_title_unsupported_platform_raises():
    with mock.patch("sys.platform", "freebsd"):
        with pytest.raises(NotImplementedError):
            utils.get_active_window_title()


# ---------------------------------------------------------------------------
# is_user_active platform dispatch
# ---------------------------------------------------------------------------

def test_is_user_active_windows_dispatch():
    with mock.patch("sys.platform", "win32"):
        with mock.patch(
            "openrecall.utils.is_user_active_windows", return_value=True
        ) as fn:
            assert utils.is_user_active() is True
            fn.assert_called_once()


def test_is_user_active_darwin_dispatch():
    with mock.patch("sys.platform", "darwin"):
        with mock.patch(
            "openrecall.utils.is_user_active_osx", return_value=False
        ) as fn:
            assert utils.is_user_active() is False
            fn.assert_called_once()


def test_is_user_active_linux_dispatch():
    with mock.patch("sys.platform", "linux"):
        with mock.patch(
            "openrecall.utils.is_user_active_linux", return_value=True
        ) as fn:
            assert utils.is_user_active() is True
            fn.assert_called_once()


def test_is_user_active_unsupported_platform_raises():
    with mock.patch("sys.platform", "freebsd"):
        with pytest.raises(NotImplementedError):
            utils.is_user_active()


# ---------------------------------------------------------------------------
# is_user_active_osx: idle-time thresholding via ioreg output
# ---------------------------------------------------------------------------

def _ioreg_output(idle_ns):
    return f'"HIDIdleTime" = {idle_ns}\n'.encode()


def test_is_user_active_osx_active_when_idle_below_threshold():
    with mock.patch(
        "openrecall.utils.subprocess.check_output",
        return_value=_ioreg_output(1_000_000_000),  # 1 second
    ):
        assert utils.is_user_active_osx() is True


def test_is_user_active_osx_inactive_when_idle_above_threshold():
    with mock.patch(
        "openrecall.utils.subprocess.check_output",
        return_value=_ioreg_output(10_000_000_000),  # 10 seconds
    ):
        assert utils.is_user_active_osx() is False


def test_is_user_active_osx_assumes_active_on_timeout():
    with mock.patch(
        "openrecall.utils.subprocess.check_output",
        side_effect=subprocess.TimeoutExpired(cmd="ioreg", timeout=1),
    ):
        assert utils.is_user_active_osx() is True


def test_is_user_active_osx_assumes_active_on_missing_field():
    with mock.patch(
        "openrecall.utils.subprocess.check_output",
        return_value=b"some other output\n",
    ):
        assert utils.is_user_active_osx() is True


# ---------------------------------------------------------------------------
# get_active_app_name_osx / get_active_window_title_osx
# ---------------------------------------------------------------------------

def test_get_active_app_name_osx_returns_name():
    fake_workspace = mock.MagicMock()
    fake_workspace.sharedWorkspace.return_value.activeApplication.return_value = {
        "NSApplicationName": "Safari"
    }
    with mock.patch("openrecall.utils.NSWorkspace", fake_workspace):
        assert utils.get_active_app_name_osx() == "Safari"


def test_get_active_app_name_osx_unavailable_returns_empty_string():
    with mock.patch("openrecall.utils.NSWorkspace", None):
        assert utils.get_active_app_name_osx() == ""


def test_get_active_window_title_osx_unavailable_returns_empty_string():
    with mock.patch("openrecall.utils.CGWindowListCopyWindowInfo", None):
        assert utils.get_active_window_title_osx() == ""
