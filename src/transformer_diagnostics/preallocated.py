"""Reproducible correctness/failure/storage checks for opt-in preallocated K/V."""

import argparse
import hashlib
import json
import math
import platform
from importlib.resources import files
from pathlib import Path

import torch

from transformer_core.allocation_scopes import allocation_scopes, scope
from transformer_core.benchmark import cached_request
from transformer_core.inference import CachedByteDecoder
from transformer_core.portable_fixture import manifest, verified_fixture
from transformer_core.preallocated import PreallocatedByteDecoder
from transformer_core.stability import state_digest
from transformer_diagnostics.allocation import (
    PROFILE_OPTIONS,
    annotate,
    positive_bytes,
    summarize,
)


def pointers(session):
    return [t.untyped_storage().data_ptr() for kv in session._storage for t in kv]


def storage_observation(session):
    """Independent storage enumeration, not cache_info's accounting alone."""
    actual = {}
    for pair in session._storage:
        for tensor in pair:
            storage = tensor.untyped_storage()
            actual[storage.data_ptr()] = storage.nbytes()
    active = sum(t.numel() * t.element_size() for kv in session._cache for t in kv)
    info = session.cache_info()
    assert info["backing_storage_bytes"] == sum(actual.values())
    assert info["kv_payload_bytes"] == active
    return dict(
        active_bytes=active,
        unique_backing_bytes=sum(actual.values()),
        unique_backing_count=len(actual),
        per_storage_bytes=list(actual.values()),
        length=info["length"],
        batch=info["batch"],
    )


def capture(function):
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU], **PROFILE_OPTIONS
    ) as profiler:
        with allocation_scopes():
            result = function()
    events = profiler.events()
    record = summarize(events)
    record["write_events"] = [
        dict(scope=annotate(e), self_bytes=int(e.self_cpu_memory_usage))
        for e in events
        if e.name == "aten::copy_"
        and annotate(e) is not None
        and annotate(e)["site"] in ("static_kv_write", "control_copy")
    ]
    return result, record


def close(actual, expected):
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    return float((actual - expected).abs().max())


@torch.inference_mode()
def parity(model):
    data = torch.tensor([[(17 * i + 31 * r) % 256 for i in range(64)] for r in range(2)])
    teacher = []
    full = model(data)
    for chunks in ([64], [1] * 64, [5, 13, 1, 17, 28]):
        static = PreallocatedByteDecoder(model, max_batch_size=2)
        dynamic = CachedByteDecoder(model)
        start = 0
        differences = []
        identities = pointers(static)
        for size in chunks:
            tokens = data[:, start : start + size]
            one = (static.prefill if start == 0 else static.append)(tokens)
            two = (dynamic.prefill if start == 0 else dynamic.append)(tokens)
            differences += [
                close(one, two),
                close(one, full[:, start : start + size]),
                close(one, model(data[:, : start + size])[:, start:]),
            ]
            start += size
            assert pointers(static) == identities
        teacher.append(
            dict(
                chunks=chunks,
                max_absolute_error=max(differences),
                backing_pointers_stable=True,
                storage=storage_observation(static),
            )
        )
    cases = []
    for case in manifest()["cases"]:
        prompt = torch.tensor([list(bytes.fromhex(h)) for h in case["prompt_hex"]])
        static = PreallocatedByteDecoder(model, max_batch_size=2)
        dynamic = CachedByteDecoder(model)
        one, two = static.prefill(prompt), dynamic.prefill(prompt)
        sequence = prompt.clone()
        errors = []
        for i in range(case["generated_per_row"]):
            expected = model(sequence)[:, -1]
            errors += [close(one[:, -1], expected), close(one[:, -1], two[:, -1])]
            token = expected.argmax(-1, keepdim=True)
            assert torch.equal(token, one[:, -1].argmax(-1, keepdim=True))
            assert torch.equal(token, two[:, -1].argmax(-1, keepdim=True))
            sequence = torch.cat((sequence, token), dim=1)
            if i + 1 < case["generated_per_row"]:
                one, two = static.append(token), dynamic.append(token)
        cases.append(
            dict(
                id=case["id"],
                max_absolute_error=max(errors),
                exact_greedy_ids=True,
                output_hex=[bytes(r).hex() for r in sequence.tolist()],
                storage=storage_observation(static),
            )
        )
    return dict(status="passed", teacher=teacher, cases=cases, atol=1e-5, rtol=1e-5)


@torch.inference_mode()
def failure_checks(model):
    rows = []
    base = torch.tensor([[(17 * i + 31 * r) % 256 for i in range(16)] for r in range(2)])
    bad = torch.tensor([[200, 201, 202, 203], [210, 211, 212, 213]])
    for first in (False, True):
        for mode in ("raise", "nan"):
            session = PreallocatedByteDecoder(model, max_batch_size=2)
            if not first:
                session.prefill(base)
            before = session.cache_info()
            ids = pointers(session)
            prefix = [t.clone() for kv in session._cache for t in kv]
            backing = [t.clone() for kv in session._storage for t in kv]
            reached = []
            handles = [
                b.attention.v_proj.register_forward_hook(
                    lambda mod, args, out: reached.append("value")
                )
                for b in session._model.blocks
            ]

            def fault(mod, args, out):
                reached.append("output")
                if mode == "raise":
                    raise RuntimeError("injected late output failure")
                return torch.full_like(out, float("nan"))

            handles.append(session._model.output.register_forward_hook(fault))
            try:
                (session.prefill if first else session.append)(bad)
            except (ValueError, RuntimeError) as exc:
                error = str(exc)
            else:
                raise AssertionError("late fault was not observed")
            finally:
                for h in handles:
                    h.remove()
            assert reached == ["value", "value", "output"]
            assert before == session.cache_info() and ids == pointers(session)
            assert all(
                torch.equal(a, b)
                for a, b in zip(prefix, (t for kv in session._cache for t in kv))
            )
            tail_changed = any(
                not torch.equal(a, b)
                for a, b in zip(backing, (t for kv in session._storage for t in kv))
            )
            assert tail_changed  # physical rollback was NOT promised
            clean = CachedByteDecoder(model)
            if first:
                retry = torch.tensor([[41, 42]])
                a, b = session.prefill(retry), clean.prefill(retry)
                errors = [close(a, b), close(a, model(retry))]
                assert session.cache_info()["batch"] == 1
                current = retry
            else:
                clean.prefill(base)
                current = base
                errors = []
                for retry in (torch.tensor([[31], [41]]), torch.tensor([[32, 33], [42, 43]])):
                    current = torch.cat((current, retry), dim=1)
                    a, b = session.append(retry), clean.append(retry)
                    errors += [close(a, b), close(a, model(current)[:, -retry.shape[1] :])]
            committed = storage_observation(session)
            before_reset = [t.clone() for kv in session._storage for t in kv]
            session.reset()
            assert session.cache_info()["length"] == 0 and session.cache_info()["batch"] is None
            assert pointers(session) == ids
            assert all(
                torch.equal(a, b)
                for a, b in zip(before_reset, (t for kv in session._storage for t in kv))
            )
            reset_storage = storage_observation(session)
            fresh = torch.tensor([[71, 72, 73]])
            errors.append(close(session.prefill(fresh), model(fresh)))
            rows.append(
                dict(
                    first_prefill=first,
                    fault=mode,
                    error=error,
                    late_sites=reached,
                    committed_prefix_and_metadata_unchanged=True,
                    inactive_backing_changed=True,
                    retry_max_absolute_error=max(errors),
                    retry_storage=committed,
                    reset_storage=reset_storage,
                    reset_preserves_bytes_and_pointers=True,
                    new_batch_after_reset=1,
                )
            )
    sessions = [PreallocatedByteDecoder(model, max_batch_size=2) for _ in range(2)]
    prompt = base[:1]
    for session, poison in zip(sessions, (float("nan"), 12345.0)):
        session.prefill(prompt)
        for kv in session._storage:
            for tensor in kv:
                tensor[:1, :, 16:].fill_(poison)
                tensor[1:].fill_(poison)
    retry = torch.tensor([[91, 92]])
    one, two = [s.append(retry) for s in sessions]
    poison_error = max(
        close(one, two), close(one, model(torch.cat((prompt, retry), 1))[:, -2:])
    )
    assert all(
        torch.equal(a, b)
        for ka, kb in zip(sessions[0]._cache, sessions[1]._cache)
        for a, b in zip(ka, kb)
    )
    return dict(
        status="passed",
        late_failures=rows,
        poisoned_tail_and_inactive_batch=dict(
            status="passed",
            max_absolute_error=poison_error,
            active_length=18,
            batch=1,
            valid_prefix_values_equal=True,
        ),
    )


def check_write_trace(trace, prompt, count):
    assert trace["attribution_status"] == "observed"
    expected = []
    for start, chunk in [(0, prompt)] + [(n, 1) for n in range(prompt, prompt + count - 1)]:
        for layer in range(2):
            for kind in ("key", "value"):
                expected.append(
                    dict(
                        scope=dict(
                            site="static_kv_write",
                            layer=layer,
                            kind=kind,
                            start=start,
                            chunk=chunk,
                            batch=2,
                            logical_bytes=512 * chunk,
                        ),
                        self_bytes=0,
                    )
                )
    assert trace["write_events"] == expected
    producers = [
        p
        for p in trace["producers"]
        if p["site"] == "static_kv_write" and p["operator"] == "aten::copy_"
    ]
    assert sum(p["count"] for p in producers) == len(expected)
    assert sum(p["positive_self_bytes"] for p in producers) == 0
    assert trace["ranges"].get("static_kv_write") == 4 * count
    assert not trace["ranges"].get("kv_cat", 0)
    cats = [e for e in trace["selected_events"] if e["operator"] == "aten::cat"]
    assert len(cats) == count
    assert [(e["scope"]["site"], e["scope"]["length"], e["self_bytes"]) for e in cats] == [
        ("token_cat", n, 16 * n) for n in range(prompt + 1, prompt + count + 1)
    ]
    assert not trace["unscoped_operator_counts"].get("aten::cat", 0)
    return dict(
        write_events=len(expected),
        write_site_positive_self_bytes=positive_bytes(trace, "static_kv_write"),
        logical_write_payload_bytes=sum(e["scope"]["logical_bytes"] for e in expected),
        kv_cat_events=0,
        token_cat_events=len(cats),
        token_cat_positive_self_bytes=sum(e["self_bytes"] for e in cats),
    )


def run(*, check_only=False):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    model, artifact, identity = verified_fixture()
    del artifact
    correct = parity(model)
    faults = failure_checks(model)
    result = dict(
        schema_version=1,
        status="passed",
        fixture=identity,
        parity=correct,
        failures=faults,
        environment=dict(
            python=platform.python_version(),
            torch=torch.__version__,
            platform=platform.platform(),
            threads=1,
            interop_threads=1,
        ),
        measurements="not_requested" if check_only else "performed",
        training=False,
        latency=False,
        rss=False,
    )
    if not check_only:
        source = torch.arange(6, dtype=torch.float32)
        dest = torch.zeros_like(source)

        def copy_control():
            with scope("control_copy"):
                dest.copy_(source)
            return dest

        out, control = capture(copy_control)
        assert torch.equal(out, source) and len(control["write_events"]) == 1
        assert control["write_events"][0]["self_bytes"] == 0
        assert positive_bytes(control, "control_copy") == 0
        control["destination_payload_bytes"] = dest.numel() * dest.element_size()
        session, cold = capture(lambda: PreallocatedByteDecoder(model, max_batch_size=2))
        ids = pointers(session)
        reserve = storage_observation(session)
        assert reserve["unique_backing_bytes"] == 131072 and reserve["active_bytes"] == 0
        assert positive_bytes(cold, "static_reserve") == 131072
        rows = []
        for case, oracle in zip(manifest()["cases"], correct["cases"]):
            prompt = torch.tensor([list(bytes.fromhex(h)) for h in case["prompt_hex"]])
            count = case["generated_per_row"]
            output, trace = capture(lambda: cached_request(session, prompt, count))
            assert [bytes(r).hex() for r in output.tolist()] == oracle["output_hex"]
            assert pointers(session) == ids
            observed = storage_observation(session)
            assert observed == oracle["storage"]
            rows.append(
                dict(
                    id=case["id"],
                    prompt_length=prompt.shape[1],
                    count=count,
                    check=check_write_trace(trace, prompt.shape[1], count),
                    storage=observed,
                    backing_pointers_stable=True,
                    trace=trace,
                )
            )
        result["resource"] = dict(
            copy_control=control,
            constructor=cold,
            initial_storage=reserve,
            cases=rows,
            units="operator self allocation vs logical writes vs backing bytes; not peak/RSS",
        )
    if not check_only:
        assert state_digest(session._model.state_dict()) == identity["tensor_sha256"]
        result["resource"]["snapshot_tensor_identity_unchanged"] = True
    assert state_digest(model.state_dict()) == identity["tensor_sha256"]
    paths = [
        "transformer_core/preallocated.py",
        "transformer_core/inference.py",
        "transformer_core/benchmark.py",
        "transformer_core/portable_fixture.py",
        "transformer_diagnostics/preallocated.py",
        "transformer_diagnostics/allocation.py",
        "transformer_diagnostics/data/portable-v1.json",
    ]
    result["source_sha256"] = {
        p: hashlib.sha256(
            files(p.split("/")[0]).joinpath(p.split("/", 1)[1]).read_bytes()
        ).hexdigest()
        for p in paths
    }
    return result


def audit(report):
    assert report["status"] == "passed" and report["fixture"] == manifest()["fixture"]
    assert report["parity"]["status"] == report["failures"]["status"] == "passed"
    assert report["measurements"] == "performed"
    assert report["parity"]["atol"] == report["parity"]["rtol"] == 1e-5
    assert [r["chunks"] for r in report["parity"]["teacher"]] == [
        [64],
        [1] * 64,
        [5, 13, 1, 17, 28],
    ]
    assert [r["id"] for r in report["parity"]["cases"]] == [
        c["id"] for c in manifest()["cases"]
    ]
    for r in report["parity"]["teacher"] + report["parity"]["cases"]:
        assert math.isfinite(r["max_absolute_error"]) and r["max_absolute_error"] >= 0
    assert all(r["exact_greedy_ids"] for r in report["parity"]["cases"])
    assert len(report["failures"]["late_failures"]) == 4
    assert all(
        r["committed_prefix_and_metadata_unchanged"]
        and r["inactive_backing_changed"]
        and r["reset_preserves_bytes_and_pointers"]
        for r in report["failures"]["late_failures"]
    )
    assert not report["training"] and not report["latency"] and not report["rss"]
    assert [(r["first_prefill"], r["fault"]) for r in report["failures"]["late_failures"]] == [
        (False, "raise"),
        (False, "nan"),
        (True, "raise"),
        (True, "nan"),
    ]
    for r in report["failures"]["late_failures"]:
        assert r["late_sites"] == ["value", "value", "output"]
        assert r["reset_storage"]["active_bytes"] == r["reset_storage"]["length"] == 0
        assert r["reset_storage"]["batch"] is None
        assert r["reset_storage"]["unique_backing_bytes"] == 131072
        assert r["retry_storage"]["batch"] == (1 if r["first_prefill"] else 2)
        assert r["retry_storage"]["length"] == (2 if r["first_prefill"] else 19)
    assert report["failures"]["poisoned_tail_and_inactive_batch"]["status"] == "passed"
    assert report["failures"]["poisoned_tail_and_inactive_batch"]["valid_prefix_values_equal"]
    resource = report["resource"]
    assert resource["snapshot_tensor_identity_unchanged"]
    control = resource["copy_control"]
    assert control["destination_payload_bytes"] == 24
    assert len(control["write_events"]) == 1 and control["write_events"][0]["self_bytes"] == 0
    assert positive_bytes(control, "control_copy") == 0
    assert resource["initial_storage"]["unique_backing_bytes"] == 131072
    assert positive_bytes(resource["constructor"], "static_reserve") == 131072
    assert len(resource["cases"]) == 2
    summary = []
    assert [r["id"] for r in resource["cases"]] == [c["id"] for c in manifest()["cases"]]
    assert [(r["prompt_length"], r["count"]) for r in resource["cases"]] == [(16, 16), (32, 32)]
    for row, oracle in zip(resource["cases"], report["parity"]["cases"]):
        assert row["storage"] == oracle["storage"]
        check = check_write_trace(row["trace"], row["prompt_length"], row["count"])
        assert check == row["check"]
        assert row["storage"]["active_bytes"] == 2048 * (
            row["prompt_length"] + row["count"] - 1
        )
        assert (
            row["storage"]["unique_backing_bytes"] == 131072 and row["backing_pointers_stable"]
        )
        summary.append(dict(id=row["id"], **check, **row["storage"]))
    assert set(report["source_sha256"]) == {
        "transformer_core/preallocated.py",
        "transformer_core/inference.py",
        "transformer_core/benchmark.py",
        "transformer_core/portable_fixture.py",
        "transformer_diagnostics/preallocated.py",
        "transformer_diagnostics/allocation.py",
        "transformer_diagnostics/data/portable-v1.json",
    }
    for p, h in report["source_sha256"].items():
        assert (
            hashlib.sha256(
                files(p.split("/")[0]).joinpath(p.split("/", 1)[1]).read_bytes()
            ).hexdigest()
            == h
        )
    return dict(status="passed", mode="offline-consistency-audit", cases=summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--check-only",
        action="store_true",
        help="actual fixture parity/fault checks, no profiler",
    )
    group.add_argument(
        "--audit", type=Path, help="offline consistency audit of saved full report"
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; retain prior evidence")
    try:
        result = (
            audit(json.loads(args.audit.read_text()))
            if args.audit
            else run(check_only=args.check_only)
        )
    except (AssertionError, ValueError, RuntimeError) as exc:
        result = dict(status="failed", error_type=type(exc).__name__, error=str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(dict(status=result["status"], output=str(args.output))))
    if result["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
