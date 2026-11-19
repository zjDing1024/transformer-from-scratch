"""Original CPU operator self-allocation attribution, not RSS/latency/traffic."""

import argparse
import hashlib
import json
import platform
from collections import Counter, defaultdict
from importlib.resources import files
from pathlib import Path

from transformer_core.allocation_scopes import PREFIX, allocation_scopes, scope

PROFILE_OPTIONS = dict(profile_memory=True, record_shapes=False, with_stack=False)


def annotate(event):
    """Nearest site ancestry; no stack/shape inference or fabricated attribution."""
    current = event
    while current is not None:
        if current.name.startswith(PREFIX):
            return json.loads(current.name[len(PREFIX) :])
        current = current.cpu_parent
    return None


def summarize(events):
    ranges = Counter()
    groups = defaultdict(lambda: dict(count=0, positive_self_bytes=0, negative_self_bytes=0))
    selected, unscoped = [], Counter()
    for event in events:
        is_range = event.name.startswith(PREFIX)
        metadata = annotate(event)
        if is_range:
            ranges[metadata["site"]] += 1
        if metadata is None:
            unscoped[event.name] += 1
            continue
        # Include a range's OWN self bytes, never its inclusive children again.
        name = "[scope self]" if is_range else event.name
        item = groups[(metadata["site"], name)]
        value = int(event.self_cpu_memory_usage)
        item["count"] += 1
        item["positive_self_bytes"] += max(0, value)
        item["negative_self_bytes"] += min(0, value)
        if event.name == "aten::cat" or (
            event.name == "aten::clone"
            and metadata["site"] in ("snapshot_deepcopy", "control_clone")
        ):
            selected.append(
                dict(
                    scope=metadata,
                    operator=event.name,
                    self_bytes=value,
                    inclusive_bytes_diagnostic_only=int(event.cpu_memory_usage),
                )
            )
    return dict(
        attribution_status="observed" if ranges else "unavailable",
        reason=None if ranges else "no instrumented site ranges observed",
        ranges=dict(sorted(ranges.items())),
        producers=[dict(site=s, operator=o, **v) for (s, o), v in sorted(groups.items())],
        selected_events=selected,
        unscoped_operator_counts=dict(sorted(unscoped.items())),
        accounting="sum positive self bytes once by nearest site; inclusive bytes not summed",
    )


def profile_call(function, *, enabled=True):
    from contextlib import nullcontext

    import torch

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU], **PROFILE_OPTIONS
    ) as profiler:
        with allocation_scopes() if enabled else nullcontext():
            result = function()
    # Keep result alive through stop; do not retain profile objects/tensor inputs.
    return result, summarize(profiler.events())


def site_events(trace, site, operator="aten::cat"):
    return [
        e
        for e in trace["selected_events"]
        if e["scope"]["site"] == site and e["operator"] == operator
    ]


def positive_bytes(trace, site):
    return sum(p["positive_self_bytes"] for p in trace["producers"] if p["site"] == site)


def check_trace(trace, *, batch, prompt, count, layers=2, width=64, arm):
    """Independent per-event oracle; do not use benchmark.analytic_work."""
    if trace["attribution_status"] != "observed":
        raise ValueError("site attribution unavailable")
    if arm not in ("cached", "full_prefix"):
        raise ValueError("unknown arm")
    kv, token = site_events(trace, "kv_cat"), site_events(trace, "token_cat")
    expected = []
    if arm == "cached":
        for start in range(prompt, prompt + count - 1):
            for layer in range(layers):
                for kind in ("key", "value"):
                    expected.append((layer, kind, start, 1, batch * width * (start + 1) * 4))
    observed = [
        (
            e["scope"]["layer"],
            e["scope"]["kind"],
            e["scope"]["start"],
            e["scope"]["chunk"],
            e["self_bytes"],
        )
        for e in kv
    ]
    if observed != expected:
        raise ValueError("K/V site event count/order/bytes mismatch")
    expected_tokens = [(n, batch * n * 8) for n in range(prompt + 1, prompt + count + 1)]
    if [(e["scope"]["length"], e["self_bytes"]) for e in token] != expected_tokens:
        raise ValueError("int64 token cat event count/order/bytes mismatch")
    if any(e["scope"]["arm"] != arm for e in token):
        raise ValueError("token cat attributed to wrong arm")
    all_cat = [e for e in trace["selected_events"] if e["operator"] == "aten::cat"]
    if len(all_cat) != len(kv) + len(token) or trace["unscoped_operator_counts"].get(
        "aten::cat", 0
    ):
        raise ValueError("unexpected or unattributed cat event")
    expected_ranges = {"token_cat": count}
    if arm == "cached":
        expected_ranges.update(prefill=1)
        if count > 1:
            expected_ranges.update(append=count - 1, kv_cat=2 * layers * (count - 1))
    if trace["ranges"] != expected_ranges:
        raise ValueError("request phase/site range counts mismatch")
    for site, events in (("kv_cat", kv), ("token_cat", token)):
        producers = [
            p for p in trace["producers"] if p["site"] == site and p["operator"] == "aten::cat"
        ]
        if sum(p["count"] for p in producers) != len(events) or sum(
            p["positive_self_bytes"] for p in producers
        ) != sum(max(0, e["self_bytes"]) for e in events):
            raise ValueError("selected events and producer aggregate disagree")
    return dict(
        status="passed",
        kv_cat_events=len(kv),
        kv_cat_positive_self_bytes=sum(e["self_bytes"] for e in kv),
        token_cat_events=len(token),
        token_cat_positive_self_bytes=sum(e["self_bytes"] for e in token),
        final_kv_payload_bytes=(
            2 * layers * batch * width * 4 * (prompt + count - 1) if arm == "cached" else 0
        ),
    )


def calibration():
    import torch

    x = torch.arange(6, dtype=torch.float32)
    y = torch.arange(4, dtype=torch.float32)
    a, b = x.long(), y.long()

    def operations():
        with scope("control_f32_cat"):
            one = torch.cat((x, y))
        with scope("control_i64_cat"):
            two = torch.cat((a, b))
        with scope("control_view"):
            three = x.view(2, 3)
        with scope("control_clone"):
            four = x.clone()
        return one, two, three, four

    _, trace = profile_call(operations)
    expected = dict(control_f32_cat=40, control_i64_cat=80, control_view=0, control_clone=24)
    actual = {s: positive_bytes(trace, s) for s in expected}
    expected_ops = {
        "control_f32_cat": "aten::cat",
        "control_i64_cat": "aten::cat",
        "control_view": "aten::view",
        "control_clone": "aten::clone",
    }
    ops_ok = all(
        sum(p["count"] for p in trace["producers"] if p["site"] == site and p["operator"] == op)
        == 1
        for site, op in expected_ops.items()
    )
    if actual != expected or trace["ranges"] != dict.fromkeys(expected, 1) or not ops_ok:
        return dict(
            status="unavailable",
            reason="operator byte calibration failed",
            expected=expected,
            observed=actual,
            trace=trace,
        )
    _, disabled = profile_call(lambda: torch.cat((x, y)), enabled=False)
    if (
        disabled["attribution_status"] != "unavailable"
        or disabled["unscoped_operator_counts"].get("aten::cat") != 1
    ):
        raise AssertionError("disabled-scope negative control failed")
    return dict(
        status="passed",
        expected=expected,
        observed=actual,
        trace=trace,
        disabled_scope_control=disabled,
    )


def snapshot_inventory(model, session):
    import torch

    source = dict(model.named_parameters())
    target = dict(session._model.named_parameters())
    if (
        source.keys() != target.keys()
        or list(model.buffers())
        or list(session._model.buffers())
    ):
        raise AssertionError("fixture parameter/buffer inventory changed")
    records = []
    for name, tensor in source.items():
        cloned = target[name]
        if tensor.data_ptr() == cloned.data_ptr() or not torch.equal(tensor, cloned):
            raise AssertionError("snapshot isolation/value check failed")
        records.append(dict(name=name, shape=list(tensor.shape), bytes=tensor.numel() * 4))
    return dict(
        tensors=records,
        parameter_tensors=len(records),
        payload_bytes=sum(r["bytes"] for r in records),
        buffers=0,
        distinct_storage_equal_values=True,
    )


def run(*, smoke=False):
    import torch

    from transformer_core.benchmark import cached_request, full_prefix_request
    from transformer_core.inference import CachedByteDecoder
    from transformer_core.portable_fixture import manifest, verified_fixture
    from transformer_core.portable_parity import verify_parity
    from transformer_core.stability import state_digest

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    controls = calibration()
    if controls["status"] != "passed":
        return dict(status="unavailable", calibration=controls)
    if smoke:
        return dict(status="passed", mode="actual-profiler-calibration", calibration=controls)
    config = manifest()
    model, artifact, identity = verified_fixture()
    del artifact
    parity = verify_parity(model, config)
    with allocation_scopes():
        enabled_parity = verify_parity(model, config)
    if parity != enabled_parity:
        raise AssertionError("opt-in scopes changed parity")
    session, cold = profile_call(lambda: CachedByteDecoder(model, max_context=64))
    inventory = snapshot_inventory(model, session)
    clones = site_events(cold, "snapshot_deepcopy", "aten::clone")
    if len(clones) != inventory["parameter_tensors"]:
        raise AssertionError("snapshot clone count mismatch")
    if positive_bytes(cold, "snapshot_deepcopy") != inventory["payload_bytes"]:
        raise AssertionError("snapshot allocation attribution does not match fixed inventory")
    snapshot_pointer_ids = [p.data_ptr() for p in session._model.parameters()]
    cases, negatives = [], []
    for case, oracle in zip(config["cases"], parity["cases"], strict=True):
        prompt = torch.tensor([list(bytes.fromhex(h)) for h in case["prompt_hex"]])
        p, g = prompt.shape[1], case["generated_per_row"]
        for count in (g, 1):
            for arm in ("cached", "full_prefix"):
                fn = (
                    (lambda: cached_request(session, prompt, count))
                    if arm == "cached"
                    else (lambda: full_prefix_request(model, prompt, count))
                )
                output, trace = profile_call(fn)
                expected = full_prefix_request(model, prompt, count)
                if not torch.equal(output, expected):
                    raise AssertionError("profiled generated IDs differ")
                output_hex = [bytes(row).hex() for row in output.tolist()]
                if count == g and output_hex != oracle["output_hex"]:
                    raise AssertionError("profiled output differs from parity oracle")
                check = check_trace(trace, batch=2, prompt=p, count=count, arm=arm)
                cache = session.cache_info() if arm == "cached" else None
                if cache and cache["kv_payload_bytes"] != check["final_kv_payload_bytes"]:
                    raise AssertionError("last-byte forwarding or cache payload mismatch")
                record = dict(
                    id=case["id"],
                    arm=arm,
                    generated_per_row=count,
                    prompt_length=p,
                    check=check,
                    cache=cache,
                    output_hex=output_hex,
                    trace=trace,
                )
                (cases if count == g else negatives).append(record)
        session.reset()
        _, prefill = profile_call(lambda: session.prefill(prompt))
        if site_events(prefill, "kv_cat") or site_events(prefill, "token_cat"):
            raise AssertionError("prefill-only concat negative control failed")

        def reset_only():
            with scope("reset_only"):
                session.reset()

        _, reset = profile_call(reset_only)
        if (
            any(x["operator"] in ("aten::clone", "aten::cat") for x in reset["producers"])
            or any(
                reset["unscoped_operator_counts"].get(op, 0)
                for op in ("aten::clone", "aten::cat")
            )
            or site_events(reset, "snapshot_deepcopy", "aten::clone")
            or site_events(reset, "kv_cat")
            or session.cache_info()["length"] != 0
            or snapshot_pointer_ids != [p.data_ptr() for p in session._model.parameters()]
        ):
            raise AssertionError("reset-only control failed")
        negatives.append(
            dict(
                id=case["id"],
                arm="prefill_only",
                trace=prefill,
                kv_cat_events=0,
                token_cat_events=0,
            )
        )
        negatives.append(
            dict(
                id=case["id"],
                arm="reset_only",
                trace=reset,
                cache=session.cache_info(),
                snapshot_pointer_ids_unchanged=True,
            )
        )
    if state_digest(model.state_dict()) != identity["tensor_sha256"]:
        raise AssertionError("source fixture mutated")
    if state_digest(session._model.state_dict()) != identity["tensor_sha256"]:
        raise AssertionError("snapshot fixture mutated")
    paths = [
        "transformer_core/" + x
        for x in (
            "allocation_scopes.py",
            "inference.py",
            "benchmark.py",
            "portable_fixture.py",
            "portable_parity.py",
        )
    ]
    paths += [
        "transformer_diagnostics/allocation.py",
        "transformer_diagnostics/data/portable-v1.json",
    ]
    hashes = {
        p: hashlib.sha256(
            files(p.split("/")[0]).joinpath("/".join(p.split("/")[1:])).read_bytes()
        ).hexdigest()
        for p in paths
    }
    return dict(
        schema_version=1,
        status="passed",
        fixture=identity,
        environment=dict(
            python=platform.python_version(),
            torch=torch.__version__,
            platform=platform.platform(),
            threads=1,
            interop_threads=1,
            device="cpu",
            dtype="float32",
        ),
        profiler_options=PROFILE_OPTIONS,
        source_sha256=hashes,
        parity=parity,
        enabled_parity_equal=True,
        calibration=controls,
        snapshot=dict(inventory=inventory, trace=cold),
        cases=cases,
        negative_controls=negatives,
        training_performed=False,
        timing_reported=False,
        rss_measured=False,
        scope="operator self-allocation attribution only; not savings, peak or traffic",
    )


def audit(report):
    """Offline record consistency, not rerunning inference or verifying human truth."""
    config = json.loads(
        files("transformer_diagnostics").joinpath("data/portable-v1.json").read_text()
    )
    assert report["status"] == "passed" and report["fixture"] == config["fixture"]
    assert report["profiler_options"] == PROFILE_OPTIONS
    assert report["enabled_parity_equal"] is True and report["parity"]["status"] == "passed"
    assert (
        not report["training_performed"]
        and not report["timing_reported"]
        and not report["rss_measured"]
    )
    controls = report["calibration"]
    assert controls["status"] == "passed"
    assert (
        controls["expected"]
        == controls["observed"]
        == dict(control_f32_cat=40, control_i64_cat=80, control_view=0, control_clone=24)
    )
    assert controls["disabled_scope_control"]["attribution_status"] == "unavailable"
    inventory, cold = report["snapshot"]["inventory"], report["snapshot"]["trace"]
    assert inventory["parameter_tensors"] == len(inventory["tensors"]) == 35
    assert inventory["payload_bytes"] == sum(t["bytes"] for t in inventory["tensors"]) == 399872
    assert inventory["buffers"] == 0 and inventory["distinct_storage_equal_values"]
    assert cold["ranges"] == dict(finite_validation=1, snapshot_deepcopy=1)
    assert len(site_events(cold, "snapshot_deepcopy", "aten::clone")) == 35
    assert positive_bytes(cold, "snapshot_deepcopy") == 399872
    assert len(report["cases"]) == 4 and len(report["negative_controls"]) == 8
    rows = []
    for collection in (report["cases"], report["negative_controls"]):
        for row in collection:
            arm = row["arm"]
            if arm in ("cached", "full_prefix"):
                result = check_trace(
                    row["trace"],
                    batch=2,
                    prompt=row["prompt_length"],
                    count=row["generated_per_row"],
                    arm=arm,
                )
                assert result == row["check"]
                if arm == "cached":
                    assert row["cache"]["kv_payload_bytes"] == result["final_kv_payload_bytes"]
                rows.append(
                    dict(id=row["id"], arm=arm, count=row["generated_per_row"], **result)
                )
            else:
                trace = row["trace"]
                assert trace["attribution_status"] == "observed"
                assert trace["ranges"] == {
                    "prefill" if arm == "prefill_only" else "reset_only": 1
                }
                assert not site_events(trace, "kv_cat") and not site_events(trace, "token_cat")
                assert not trace["unscoped_operator_counts"].get("aten::cat", 0)
                if arm == "reset_only":
                    assert row["cache"]["length"] == 0 and row["snapshot_pointer_ids_unchanged"]
                    assert not any(p["operator"] == "aten::clone" for p in trace["producers"])
                    assert not trace["unscoped_operator_counts"].get("aten::clone", 0)
    for case in config["cases"]:
        variants = [r for r in report["cases"] if r["id"] == case["id"]]
        assert [r["arm"] for r in variants] == ["cached", "full_prefix"]
        oracle = next(p for p in report["parity"]["cases"] if p["id"] == case["id"])
        assert all(r["output_hex"] == oracle["output_hex"] for r in variants)
    for path, digest in report["source_sha256"].items():
        package, relative = path.split("/", 1)
        assert (
            hashlib.sha256(files(package).joinpath(relative).read_bytes()).hexdigest() == digest
        )
    return dict(
        status="passed",
        mode="offline-record-audit",
        inference_performed=False,
        snapshot_positive_self_bytes=positive_bytes(cold, "snapshot_deepcopy"),
        validation_positive_self_bytes=positive_bytes(cold, "finite_validation"),
        cases=rows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--smoke", action="store_true", help="actual tiny allocation controls only"
    )
    group.add_argument("--audit", type=Path, help="offline audit of a full saved report")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; preserve previous observations")
    try:
        result = (
            audit(json.loads(args.audit.read_text())) if args.audit else run(smoke=args.smoke)
        )
    except (AssertionError, ValueError, RuntimeError) as exc:
        result = dict(
            status="failed",
            error_type=type(exc).__name__,
            error=str(exc),
            scope="failure receipt; no successful allocation claim",
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(dict(status=result["status"], output=str(args.output))))
    if result["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
