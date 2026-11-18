from unittest.mock import patch

import pytest

from transformer_diagnostics import proc_memory as pm


def sample(rss, hwm, ru=999999):
    return {
        name: dict(status="available", value_kib=value)
        for name, value in (("VmRSS", rss), ("VmHWM", hwm), ("ru_maxrss", ru))
    }


def test_parser_explicit_available_and_missing():
    assert pm.parse_status("VmRSS:\t123 kB\nVmHWM: 456 kB\n")["VmRSS"]["value_kib"] == 123
    assert pm.parse_status("")["VmRSS"]["status"] == "unavailable"


@pytest.mark.parametrize(
    "text", ["VmRSS: nope kB", "VmRSS: 2 MB", "VmRSS: -1 kB", "VmRSS: 1 kB\nVmRSS: 2 kB"]
)
def test_parser_rejects_ambiguous_or_bad_metric(text):
    assert pm.parse_status(text)["VmRSS"]["status"] == "unavailable"


def test_unavailable_platform_and_denied_reset():
    with patch.object(pm.sys, "platform", "win32"):
        assert all(v["status"] == "unavailable" for v in pm.observe().values())
        assert pm.reset_peak()["status"] == "unavailable"
    with (
        patch.object(pm.sys, "platform", "linux"),
        patch.object(pm.Path, "write_text", side_effect=PermissionError),
    ):
        assert pm.reset_peak() == dict(status="unavailable", reason="PermissionError")


def test_window_does_not_substitute_ru_maxrss_or_clamp_negative_deltas():
    start, end = sample(100, 100), sample(90, 99)
    assert pm.difference(end, start, "VmRSS")["value_kib"] == -10
    assert pm.window_status(dict(status="available"), start, end)["status"] == "observed"
    missing = dict(start, VmHWM=pm.unavailable("missing"))
    assert pm.window_status(dict(status="available"), missing, end)["status"] == "unavailable"
    assert pm.window_status(pm.unavailable("denied"), start, end)["status"] == "unavailable"
    assert (
        pm.window_status(dict(status="available"), sample(100, 99999), end)["status"]
        == "unavailable"
    )


@pytest.mark.parametrize("growth,status", [(4, "passed"), (1, "failed")])
def test_positive_control_coarse_status_with_mock_observations(growth, status):
    samples = [sample(100, 100), sample(100 + growth, 100 + growth), sample(100, 100 + growth)]
    with (
        patch.object(pm, "observe", side_effect=samples),
        patch.object(pm, "reset_peak", return_value=dict(status="available")),
    ):
        result = pm.positive_control(size_bytes=4096, minimum_growth_kib=3)
    assert result["status"] == status
    assert result["after_release"]["VmRSS"]["value_kib"] == 100
    assert result["while_live"]["ru_maxrss"]["value_kib"] == 999999


def test_positive_control_denial_is_unavailable_not_pass():
    samples = [sample(100, 100), sample(104, 104), sample(100, 104)]
    with (
        patch.object(pm, "observe", side_effect=samples),
        patch.object(pm, "reset_peak", return_value=pm.unavailable("denied")),
    ):
        assert pm.positive_control(4096, 3)["status"] == "unavailable"
