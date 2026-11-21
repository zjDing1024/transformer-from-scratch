"""Contract and saved-result replay for the cache comparison; no live timing."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from transformer_diagnostics.cache_tradeoff import (
    ARMS,
    audit,
    clockwise,
    distribution,
    expected_retention,
    measure_cell,
    timed_call,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "results/cache-tradeoff-v1"


def test_timed_call_holds_result_until_after_stop():
    trace = []

    class Thing:
        def __del__(self):
            trace.append("release")

    def clock():
        trace.append("clock")
        return len(trace) * 10

    def callable_():
        trace.append("call")
        return Thing()

    assert timed_call(callable_, clock) == 20
    assert trace == ["clock", "call", "clock", "release"]


def test_fake_measurement_balances_position_and_records_all_calls():
    clock_values = iter([0, 10, 20, 40, 50, 90, 100, 130])
    calls = []
    rows = []
    summary = measure_cell(
        {arm: lambda a=arm: calls.append(a) for arm in ARMS},
        cell_index=0,
        warmups=2,
        repeats=2,
        rows=rows,
        clock=lambda: next(clock_values),
    )
    assert calls == list(ARMS + ARMS[::-1] + ARMS + ARMS[::-1])
    assert rows == [
        dict(index=0, order=list(ARMS), duration_ns=dict(dynamic=10, preallocated=20)),
        dict(index=1, order=list(ARMS[::-1]), duration_ns=dict(preallocated=40, dynamic=30)),
    ]
    assert summary["paired_preallocated_over_dynamic"]["median"] == (2 + 4 / 3) / 2
    assert summary["paired_preallocated_minus_dynamic_ns_median"] == 10
    assert summary["preallocated_higher_pairs"] == 2
    assert summary["median_ratio_by_first_arm"] == dict(dynamic=2, preallocated=4 / 3)
    assert clockwise(0, 1) == list(ARMS[::-1])


def test_partial_pair_remains_available_after_failure():
    rows = []
    clock_values = iter([0, 10, 20])

    def fail():
        raise RuntimeError("fixture failure")

    with pytest.raises(RuntimeError, match="fixture failure"):
        measure_cell(
            dict(dynamic=lambda: None, preallocated=fail),
            cell_index=0,
            warmups=0,
            repeats=30,
            rows=rows,
            clock=lambda: next(clock_values),
        )
    assert rows == [dict(index=0, order=list(ARMS), duration_ns={"dynamic": 10})]


def test_nearest_rank_and_storage_contract():
    assert distribution(list(range(1, 31))) == dict(
        n=30, minimum=1, median=15.5, p90_nearest_rank=27, maximum=30
    )
    for prompt, count, active in ((16, 16, 63488), (32, 32, 129024)):
        dynamic = expected_retention(count, prompt, "dynamic")
        static = expected_retention(count, prompt, "preallocated")
        assert (
            dynamic["populated"]["active_bytes"]
            == static["populated"]["active_bytes"]
            == active
        )
        assert dynamic["reset"]["unique_backing_bytes"] == 0
        assert static["reset"]["unique_backing_bytes"] == 131072


def test_fresh_import_does_not_import_torch():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from transformer_diagnostics.cache_tradeoff import identities; "
            "assert identities(); "
            "assert 'torch' not in sys.modules",
        ],
        check=True,
        timeout=30,
        env=dict(os.environ, PYTHONPATH=str(ROOT / "src")),
    )


def test_affinity_rejects_already_imported_torch(monkeypatch):
    from transformer_diagnostics.cache_tradeoff import affinity

    monkeypatch.setitem(sys.modules, "torch", object())
    with pytest.raises(RuntimeError, match="Torch imported before"):
        affinity()


def test_original_raw_deviation_is_retained_and_ineligible():
    directory = ARTIFACTS / "deviations"
    receipt = json.loads((directory / "pre-affinity-import-receipt.json").read_text())
    raw = (directory / receipt["raw_file"]).read_bytes()
    assert receipt["status"] == "protocol_deviation" and not receipt["eligible_primary"]
    assert receipt["raw_execution_completed"]
    assert len(raw) == receipt["raw_bytes"]
    assert hashlib.sha256(raw).hexdigest() == receipt["raw_sha256"]


def read_report():
    return json.loads((ARTIFACTS / "report.json").read_text())


def test_published_report_replays_exact_audit():
    assert audit(read_report()) == json.loads((ARTIFACTS / "audit.json").read_text())


@pytest.mark.parametrize(
    "fault",
    ["fixture", "order", "clock", "summary", "capacity", "last_byte", "source", "partial"],
)
def test_audit_rejects_corrupted_evidence(fault):
    report = read_report()
    first = report["cells"][0]
    if fault == "fixture":
        report["fixture"]["tensor_sha256"] = "0" * 64
    elif fault == "order":
        first["samples"][0]["order"].reverse()
    elif fault == "clock":
        first["samples"][0]["duration_ns"]["dynamic"] = 0
    elif fault == "summary":
        first["summary"]["arm_ns"]["dynamic"]["median"] += 1
    elif fault == "capacity":
        for stage in ("before", "after"):
            report[stage][0]["retention"]["preallocated"]["reset"]["unique_backing_bytes"] = 0
    elif fault == "last_byte":
        for stage in ("before", "after"):
            report[stage][0]["last_generated_byte_forwarded"] = True
    elif fault == "source":
        report["source_sha256"].clear()
    else:
        first["samples"].pop()
    with pytest.raises((AssertionError, ValueError)):
        audit(report)
