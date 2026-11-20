"""Offline evidence auditing and actual portable CLI; no primary profiling rerun."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from transformer_diagnostics.preallocated import audit

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "results/preallocated-kv-v1/report.json"


def test_published_preallocated_report_recomputes_exact_audit():
    expected = json.loads((ROOT / "results/preallocated-kv-v1/audit.json").read_text())
    assert audit(json.loads(REPORT.read_text())) == expected


@pytest.mark.parametrize(
    "change",
    [
        "fixture",
        "source",
        "missing_source",
        "write_count",
        "write_length",
        "write_allocation",
        "token_cat",
        "capacity",
        "active",
        "pointer",
        "late_site",
        "first_batch",
        "reset",
        "poison",
        "calibration",
        "unscoped_cat",
    ],
)
def test_preallocated_audit_rejects_evidence_mutation(change):
    report = json.loads(REPORT.read_text())
    row = report["resource"]["cases"][0]
    trace = row["trace"]
    if change == "fixture":
        report["fixture"]["tensor_sha256"] = "0" * 64
    elif change == "source":
        report["source_sha256"]["transformer_core/preallocated.py"] = "0" * 64
    elif change == "missing_source":
        report["source_sha256"].pop("transformer_core/preallocated.py")
    elif change == "write_count":
        trace["write_events"].pop()
    elif change == "write_length":
        trace["write_events"][0]["scope"]["chunk"] += 1
    elif change == "write_allocation":
        trace["write_events"][0]["self_bytes"] = 4
    elif change == "token_cat":
        event = next(e for e in trace["selected_events"] if e["operator"] == "aten::cat")
        event["self_bytes"] += 16
    elif change == "capacity":
        row["storage"]["unique_backing_bytes"] += 4
    elif change == "active":
        row["storage"]["active_bytes"] += 4
    elif change == "pointer":
        row["backing_pointers_stable"] = False
    elif change == "late_site":
        report["failures"]["late_failures"][0]["late_sites"].pop()
    elif change == "first_batch":
        report["failures"]["late_failures"][2]["retry_storage"]["batch"] = 2
    elif change == "reset":
        report["failures"]["late_failures"][0]["reset_storage"]["active_bytes"] = 4
    elif change == "poison":
        report["failures"]["poisoned_tail_and_inactive_batch"]["valid_prefix_values_equal"] = (
            False
        )
    elif change == "calibration":
        report["resource"]["copy_control"]["write_events"][0]["self_bytes"] = 24
    else:
        trace["unscoped_operator_counts"]["aten::cat"] = 1
    with pytest.raises((AssertionError, ValueError)):
        audit(report)


def test_actual_preallocated_cli_without_profile_and_refuses_overwrite(tmp_path):
    path = tmp_path / "actual.json"
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    args = [
        sys.executable,
        "-m",
        "transformer_diagnostics.preallocated",
        "--check-only",
        "--output",
        str(path),
    ]
    run = subprocess.run(args, env=env, capture_output=True, text=True, timeout=90)
    assert run.returncode == 0, run.stderr
    report = json.loads(path.read_text())
    published = json.loads(REPORT.read_text())
    assert report["status"] == "passed" and report["measurements"] == "not_requested"
    assert "resource" not in report

    # Runtime checks the numeric tolerance. CPU kernels can produce slightly
    # different error maxima; identities, outputs and transaction facts stay exact.
    def without_error_maxima(value):
        if isinstance(value, dict):
            return {
                k: without_error_maxima(v)
                for k, v in value.items()
                if not k.endswith("max_absolute_error")
            }
        if isinstance(value, list):
            return [without_error_maxima(v) for v in value]
        return value

    for key in ("fixture", "parity", "failures", "source_sha256"):
        assert without_error_maxima(report[key]) == without_error_maxima(published[key])
    original = path.read_bytes()
    repeated = subprocess.run(args, env=env, capture_output=True, text=True, timeout=90)
    assert repeated.returncode != 0 and "output exists" in repeated.stderr
    assert path.read_bytes() == original


def test_cli_preserves_failure_receipt(tmp_path):
    wrong = json.loads(REPORT.read_text())
    wrong["status"] = "failed"
    incoming, outgoing = tmp_path / "bad.json", tmp_path / "failure.json"
    incoming.write_text(json.dumps(wrong))
    args = [
        sys.executable,
        "-m",
        "transformer_diagnostics.preallocated",
        "--audit",
        str(incoming),
        "--output",
        str(outgoing),
    ]
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    run = subprocess.run(args, env=env, capture_output=True, text=True, timeout=90)
    assert run.returncode == 2
    result = json.loads(outgoing.read_text())
    assert result["status"] == "failed" and result["error_type"] == "AssertionError"
