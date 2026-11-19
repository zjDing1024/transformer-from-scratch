"""Offline published observation checks; no primary fixture profiler rerun."""

import copy
import json
from pathlib import Path

import pytest

from transformer_diagnostics.allocation import audit

ROOT = Path(__file__).resolve().parents[1]


def report():
    return json.loads((ROOT / "results/operator-allocation-v1/report.json").read_text())


def test_published_report_recomputes_exact_audit():
    expected = json.loads((ROOT / "results/operator-allocation-v1/audit.json").read_text())
    assert audit(report()) == expected


@pytest.mark.parametrize(
    "change", ["fixture", "event", "source", "output", "phase", "inventory", "reset"]
)
def test_published_audit_rejects_changed_evidence(change):
    value = copy.deepcopy(report())
    if change == "fixture":
        value["fixture"]["tensor_sha256"] = "0" * 64
    elif change == "event":
        value["cases"][0]["trace"]["selected_events"][1]["self_bytes"] += 1
    elif change == "source":
        value["source_sha256"]["transformer_core/allocation_scopes.py"] = "0" * 64
    elif change == "output":
        value["cases"][0]["output_hex"] = ["00", "00"]
    elif change == "phase":
        value["cases"][0]["trace"]["ranges"]["append"] += 1
    elif change == "inventory":
        value["snapshot"]["inventory"]["payload_bytes"] += 1
    else:
        row = next(x for x in value["negative_controls"] if x["arm"] == "reset_only")
        row["trace"]["unscoped_operator_counts"]["aten::clone"] = 1
    with pytest.raises((AssertionError, ValueError)):
        audit(value)
