"""Scalar ancestry/byte-accounting contracts and tiny actual profiler controls."""

import json
from types import SimpleNamespace

import pytest

from transformer_core.allocation_scopes import _ENABLED, PREFIX, allocation_scopes, scope
from transformer_diagnostics.allocation import (
    calibration,
    check_trace,
    positive_bytes,
    summarize,
)


def event(name, parent=None, own=0, inclusive=0):
    return SimpleNamespace(
        name=name, cpu_parent=parent, self_cpu_memory_usage=own, cpu_memory_usage=inclusive
    )


def label(site, **metadata):
    return PREFIX + json.dumps(dict(site=site, **metadata))


def test_nested_self_allocations_are_not_double_counted():
    outer = event(label("snapshot_deepcopy"), inclusive=24)
    clone = event("aten::clone", outer, inclusive=24)
    empty = event("aten::empty_strided", clone, own=24, inclusive=24)
    copy = event("aten::copy_", clone)
    report = summarize([outer, clone, empty, copy])
    assert positive_bytes(report, "snapshot_deepcopy") == 24
    assert report["selected_events"][0]["inclusive_bytes_diagnostic_only"] == 24
    assert report["selected_events"][0]["self_bytes"] == 0


def test_nearest_site_owns_bytes_and_negative_scope_self_is_retained():
    outer = event(label("append"), own=-8, inclusive=32)
    inner = event(label("kv_cat", layer=0, kind="key", start=2, chunk=1), outer, inclusive=40)
    cat = event("aten::cat", inner, own=40, inclusive=40)
    report = summarize([outer, inner, cat])
    assert positive_bytes(report, "append") == 0
    assert positive_bytes(report, "kv_cat") == 40
    assert (
        next(p for p in report["producers"] if p["site"] == "append")["negative_self_bytes"]
        == -8
    )


def test_missing_ranges_is_unavailable_not_zero():
    report = summarize([event("aten::cat", own=40, inclusive=40)])
    assert report["attribution_status"] == "unavailable"
    assert report["unscoped_operator_counts"] == {"aten::cat": 1}
    with pytest.raises(ValueError, match="unavailable"):
        check_trace(report, batch=2, prompt=16, count=1, arm="cached")


def test_scopes_default_off_restore_and_nested_failure():
    assert _ENABLED.get() is False
    with scope("ignored", anything=object()):
        pass
    with allocation_scopes():
        assert _ENABLED.get() is True
        with pytest.raises(RuntimeError):
            with allocation_scopes():
                raise RuntimeError("restore")
        assert _ENABLED.get() is True
    assert _ENABLED.get() is False


@pytest.mark.parametrize("value", [[], {}, object(), 2.5, None])
def test_enabled_metadata_rejects_nonscalar_values(value):
    with allocation_scopes(), pytest.raises(ValueError, match="scalars"):
        scope("not_a_tensor_holder", payload=value)


def trace_for(prompt=16, count=16, arm="cached"):
    events = []
    if arm == "cached":
        events.append(event(label("prefill")))
        for start in range(prompt, prompt + count - 1):
            append = event(label("append"))
            events.append(append)
            for layer in range(2):
                for kind in ("key", "value"):
                    parent = event(
                        label("kv_cat", layer=layer, kind=kind, start=start, chunk=1), append
                    )
                    events += [parent, event("aten::cat", parent, own=512 * (start + 1))]
    for length in range(prompt + 1, prompt + count + 1):
        parent = event(label("token_cat", length=length, arm=arm))
        events += [parent, event("aten::cat", parent, own=16 * length)]
    return summarize(events)


@pytest.mark.parametrize(
    "prompt,count,expected_calls,expected_bytes", [(16, 16, 60, 737280), (32, 32, 124, 3047424)]
)
def test_independent_frozen_count_and_payload_oracle(
    prompt, count, expected_calls, expected_bytes
):
    result = check_trace(
        trace_for(prompt, count), batch=2, prompt=prompt, count=count, arm="cached"
    )
    assert result["kv_cat_events"] == expected_calls
    assert result["kv_cat_positive_self_bytes"] == expected_bytes


@pytest.mark.parametrize("arm,count", [("cached", 1), ("full_prefix", 1), ("full_prefix", 16)])
def test_negative_controls_have_no_kv_events(arm, count):
    result = check_trace(
        trace_for(count=count, arm=arm), batch=2, prompt=16, count=count, arm=arm
    )
    assert result["kv_cat_events"] == result["kv_cat_positive_self_bytes"] == 0


@pytest.mark.parametrize(
    "mutation", ["bytes", "count", "arm", "unscoped", "aggregate", "phase"]
)
def test_tampered_trace_is_rejected(mutation):
    report = trace_for()
    if mutation == "bytes":
        report["selected_events"][0]["self_bytes"] += 1
    elif mutation == "count":
        report["selected_events"].pop(0)
    elif mutation == "arm":
        report["selected_events"][-1]["scope"]["arm"] = "full_prefix"
    elif mutation == "unscoped":
        report["unscoped_operator_counts"]["aten::cat"] = 1
    elif mutation == "aggregate":
        report["producers"][0]["count"] += 1  # alter append below instead
    else:
        report["ranges"]["append"] += 1
    if mutation == "aggregate":
        next(p for p in report["producers"] if p["operator"] == "aten::cat")[
            "positive_self_bytes"
        ] += 1
    with pytest.raises(ValueError):
        check_trace(report, batch=2, prompt=16, count=16, arm="cached")


def test_actual_tiny_cpu_profiler_calibration():
    result = calibration()
    assert result["status"] == "passed"
    assert result["observed"] == dict(
        control_f32_cat=40, control_i64_cat=80, control_view=0, control_clone=24
    )
    assert result["disabled_scope_control"]["attribution_status"] == "unavailable"
