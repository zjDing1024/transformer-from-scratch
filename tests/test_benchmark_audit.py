"""Audit committed telemetry without running inference or a timing experiment."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "cpu_audit", ROOT / "examples/audit_cpu_benchmark.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def report():
    return json.loads((ROOT / "results/cpu-kv-v1/report.json").read_text())


def test_saved_report_summary_recomputes_exactly():
    expected = json.loads((ROOT / "results/cpu-kv-v1/summary.json").read_text())
    assert module.audit(report()) == expected


@pytest.mark.parametrize("change", ["time", "order", "output", "identity", "work"])
def test_audit_rejects_changed_records(change):
    value = copy.deepcopy(report())
    sample = value["cases"][0]["warm_request"]["samples"][0]
    if change == "time":
        sample["ns_per_call"]["cached"] = 1
    elif change == "order":
        sample["order"] = list(reversed(sample["order"]))
    elif change == "output":
        value["memory_workers"][0]["output_hex"] = ["00"]
    elif change == "identity":
        value["config"]["original_identity_matched"] = True
    else:
        value["cases"][0]["parity_before_and_after"]["analytic_work"]["forward_calls"] += 1
    with pytest.raises(AssertionError):
        module.audit(value)
