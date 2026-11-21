"""Frozen, unprofiled CPU comparison of the two existing K/V session policies."""

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
from importlib.resources import files
from importlib.util import find_spec
from pathlib import Path

ARMS = ("dynamic", "preallocated")
DATA_PACKAGE = "transformer_diagnostics"
SOURCES = (
    "transformer_core/allocation_scopes.py",
    "transformer_core/attention.py",
    "transformer_core/benchmark.py",
    "transformer_core/inference.py",
    "transformer_core/language_model.py",
    "transformer_core/layers.py",
    "transformer_core/portable_fixture.py",
    "transformer_core/preallocated.py",
    "transformer_core/stability.py",
    "transformer_diagnostics/cache_tradeoff.py",
    "transformer_diagnostics/data/cache-tradeoff-v1.json",
    "transformer_diagnostics/data/portable-v1.json",
)
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def config():
    return json.loads(files(DATA_PACKAGE).joinpath("data/cache-tradeoff-v1.json").read_bytes())


def identities():
    result = {}
    for p in SOURCES:
        package, rel = p.split("/", 1)
        # Resource lookup by module can execute transformer_core.__init__,
        # importing Torch before affinity. These are top-level package specs.
        spec = find_spec(package)
        if spec is None or spec.origin is None:
            raise RuntimeError(f"source package not found: {package}")
        result[p] = hashlib.sha256((Path(spec.origin).parent / rel).read_bytes()).hexdigest()
    return result


def clockwise(index, cell_index):
    return list(ARMS if (index + cell_index) % 2 == 0 else ARMS[::-1])


def timed_call(function, clock=time.perf_counter_ns):
    start = clock()
    result = function()
    duration = clock() - start
    del result  # lifetime includes the stop clock, not teardown
    if type(duration) is not int or duration <= 0:
        raise ValueError("elapsed time must be positive integer ns")
    return duration


def distribution(values):
    if not values or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("positive finite values required")
    ordered = sorted(values)
    return {
        "n": len(values),
        "minimum": ordered[0],
        "median": statistics.median(values),
        "p90_nearest_rank": ordered[math.ceil(0.9 * len(values)) - 1],
        "maximum": ordered[-1],
    }


def summarize(pairs):
    ratios = [p["duration_ns"]["preallocated"] / p["duration_ns"]["dynamic"] for p in pairs]
    differences = [
        p["duration_ns"]["preallocated"] - p["duration_ns"]["dynamic"] for p in pairs
    ]
    return dict(
        arm_ns={arm: distribution([p["duration_ns"][arm] for p in pairs]) for arm in ARMS},
        paired_preallocated_over_dynamic=distribution(ratios),
        paired_preallocated_minus_dynamic_ns_median=statistics.median(differences),
        preallocated_lower_pairs=sum(v < 0 for v in differences),
        preallocated_higher_pairs=sum(v > 0 for v in differences),
        ties=sum(v == 0 for v in differences),
        median_ratio_by_first_arm={
            arm: statistics.median([r for p, r in zip(pairs, ratios) if p["order"][0] == arm])
            for arm in ARMS
        },
    )


def measure_cell(functions, *, cell_index, warmups, repeats, rows, clock=time.perf_counter_ns):
    for index in range(warmups):
        for arm in clockwise(index, cell_index):
            functions[arm]()
    for index in range(repeats):
        order = clockwise(index, cell_index)
        row = dict(index=index, order=order, duration_ns={})
        rows.append(row)
        for arm in order:
            row["duration_ns"][arm] = timed_call(functions[arm], clock)
    return summarize(rows)


def affinity():
    if "torch" in sys.modules:
        raise RuntimeError("Torch imported before primary CPU affinity")
    before = sorted(os.sched_getaffinity(0))
    if not before:
        raise RuntimeError("no eligible CPU")
    os.sched_setaffinity(0, {before[0]})
    after = sorted(os.sched_getaffinity(0))
    if after != [before[0]]:
        raise RuntimeError("primary CPU affinity did not apply")
    return dict(initial_eligible=before, pinned=after, torch_imported_before_pin=False)


def native_thread_affinity(pinned):
    masks = []
    for entry in Path("/proc/self/task").iterdir():
        try:
            masks.append(sorted(os.sched_getaffinity(int(entry.name))))
        except ProcessLookupError:
            continue  # thread exited after task listing
    if not masks or any(mask != pinned for mask in masks):
        raise RuntimeError("native-thread affinity differs from the frozen CPU pin")
    return dict(observed_thread_count=len(masks), all_eligible_masks_equal_pin=True)


def cpu_model():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except (OSError, UnicodeError):
        pass
    return platform.processor() or "unreported"


def canonical_hex(sequence):
    return [bytes(row).hex() for row in sequence.tolist()]


def observe(session):
    active = tuple(t for pair in session._cache for t in pair)
    physical = (
        tuple(t for pair in session._storage for t in pair)
        if hasattr(session, "_storage")
        else active
    )
    backing = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes() for t in physical}
    payload = sum(t.numel() * t.element_size() for t in active)
    info = session.cache_info()
    if payload != info["kv_payload_bytes"]:
        raise AssertionError("active payload accounting mismatch")
    if not {t.untyped_storage().data_ptr() for t in active}.issubset(backing):
        raise AssertionError("active views are outside session backing")
    if "backing_storage_bytes" in info and info["backing_storage_bytes"] != sum(
        backing.values()
    ):
        raise AssertionError("preallocated backing accounting mismatch")
    return dict(
        batch=info["batch"],
        length=info["length"],
        active_bytes=payload,
        unique_backing_bytes=sum(backing.values()),
        unique_storage_sizes=sorted(backing.values()),
    ), tuple(backing)


def expected_retention(count, prompt_length, arm):
    length = prompt_length + count - 1
    active = 2048 * length
    reserve = [32768] * 4 if arm == "preallocated" else []
    used = reserve or [active // 4] * 4

    def state(length, batch, active, sizes):
        return dict(
            length=length,
            batch=batch,
            active_bytes=active,
            unique_backing_bytes=sum(sizes),
            unique_storage_sizes=sizes,
        )

    return dict(
        constructed=state(0, None, 0, reserve),
        populated=state(length, 2, active, used),
        reset=state(0, None, 0, reserve),
        backing_stable=arm == "preallocated",
    )


def runtime(report, check_only):
    import torch

    from transformer_core.allocation_scopes import _ENABLED
    from transformer_core.benchmark import cached_request, full_prefix_request
    from transformer_core.inference import CachedByteDecoder
    from transformer_core.portable_fixture import manifest, verified_fixture
    from transformer_core.preallocated import PreallocatedByteDecoder
    from transformer_core.stability import state_digest

    if _ENABLED.get():
        raise RuntimeError("allocation scopes must be disabled")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    settings = report["protocol"]
    clk = time.get_clock_info("perf_counter")
    report["environment"].update(
        cpu_model=cpu_model(),
        python=platform.python_version(),
        torch=torch.__version__,
        torch_build_commit=torch.version.git_version,
        platform=platform.platform(),
        intraop_threads=torch.get_num_threads(),
        interop_threads=torch.get_num_interop_threads(),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        python_cyclic_gc_enabled=gc.isenabled(),
        profiler=False,
        allocation_scopes=False,
        thread_environment={name: os.environ.get(name) for name in THREAD_ENV},
        clock=dict(
            implementation=clk.implementation,
            resolution_seconds=clk.resolution,
            monotonic=clk.monotonic,
            adjustable=clk.adjustable,
        ),
    )
    model, artifact, identity = verified_fixture()
    if identity["name"] != settings["fixture"]:
        raise AssertionError("fixture identity changed")
    del artifact
    report["fixture"] = identity
    ctx = settings["max_context"]
    factories = dict(
        dynamic=lambda: CachedByteDecoder(model, max_context=ctx),
        preallocated=lambda: PreallocatedByteDecoder(
            model, max_batch_size=settings["max_batch_size"], max_context=ctx
        ),
    )
    cases = {
        c["id"]: (
            torch.tensor([list(bytes.fromhex(p)) for p in c["prompt_hex"]]),
            c["generated_per_row"],
        )
        for c in manifest()["cases"]
    }
    atol, rtol = manifest()["parity_atol"], manifest()["parity_rtol"]

    def assert_snapshot(session):
        if state_digest(session._model.state_dict()) != identity["tensor_sha256"]:
            raise AssertionError("snapshot identity changed")

    @torch.inference_mode()
    def parity_and_retention():
        rows = []
        for name, (prompt, count) in cases.items():
            sessions = {a: f() for a, f in factories.items()}
            logits = {a: s.prefill(prompt) for a, s in sessions.items()}
            sequence = prompt.clone()
            errors = dict(dynamic_vs_full=0.0, preallocated_vs_full=0.0, arm_vs_arm=0.0)
            for i in range(count):
                full = model(sequence)[:, -1]
                dynamic, static = (logits[a][:, -1] for a in ARMS)
                token = full.argmax(dim=-1, keepdim=True)
                for key, left, right in (
                    ("dynamic_vs_full", dynamic, full),
                    ("preallocated_vs_full", static, full),
                    ("arm_vs_arm", static, dynamic),
                ):
                    torch.testing.assert_close(left, right, atol=atol, rtol=rtol)
                    errors[key] = max(errors[key], (left - right).abs().max().item())
                if not all(
                    torch.equal(t.argmax(-1, keepdim=True), token) for t in (dynamic, static)
                ):
                    raise AssertionError("greedy step IDs differ")
                sequence = torch.cat((sequence, token), dim=1)
                if i + 1 < count:
                    logits = {a: s.append(token) for a, s in sessions.items()}
            if not torch.equal(full_prefix_request(model, prompt, count, ctx), sequence):
                raise AssertionError("full request changed output")
            retention = {}
            for arm, factory in factories.items():
                assert_snapshot(sessions[arm])
                s = factory()
                initial, identities_0 = observe(s)
                output = cached_request(s, prompt, count, ctx)
                populated, identities_1 = observe(s)
                s.reset()
                reset, identities_2 = observe(s)
                row = dict(
                    constructed=initial,
                    populated=populated,
                    reset=reset,
                    backing_stable=identities_0 == identities_1 == identities_2,
                )
                if row != expected_retention(count, prompt.shape[1], arm):
                    raise AssertionError("cache bookkeeping differs from expected reserve")
                if not torch.equal(output, sequence):
                    raise AssertionError("cached request changed output")
                assert_snapshot(s)
                retention[arm] = row
            if state_digest(model.state_dict()) != identity["tensor_sha256"]:
                raise AssertionError("source model identity changed")
            rows.append(
                dict(
                    case=name,
                    batch=prompt.shape[0],
                    prompt_length=prompt.shape[1],
                    generated=count,
                    prefill_calls=1,
                    append_calls=count - 1,
                    last_generated_byte_forwarded=False,
                    output_hex=canonical_hex(sequence),
                    exact_ids=True,
                    max_absolute_logit_error=errors,
                    atol=atol,
                    rtol=rtol,
                    retention=retention,
                    snapshot_identity_unchanged=True,
                )
            )
        return rows

    def new_request(factory, prompt, count):
        s = factory()
        return s, cached_request(s, prompt, count, ctx)

    if not check_only:
        pin = report["environment"]["affinity"]
        pin["before_parity"] = native_thread_affinity(pin["pinned"])
    report["before"] = parity_and_retention()
    if check_only:
        return
    empty = [timed_call(lambda: None) for _ in range(settings["empty_boundary_pairs"])]
    report["empty_boundary"] = dict(samples_ns=empty, summary_ns=distribution(empty))
    report["cells"] = []
    for index, cell in enumerate(settings["cells"]):
        phase, case = cell["phase"], cell["case"]
        row = dict(cell, samples=[])
        report["cells"].append(row)
        if phase == "constructor_only":
            functions = factories
        elif phase == "new_session_request":
            prompt, count = cases[case]
            functions = {
                arm: lambda f=f: new_request(f, prompt, count) for arm, f in factories.items()
            }
        elif phase == "reused_session_request":
            prompt, count = cases[case]
            active_sessions = {arm: f() for arm, f in factories.items()}
            functions = {
                arm: lambda s=s: cached_request(s, prompt, count, ctx)
                for arm, s in active_sessions.items()
            }
        else:
            raise ValueError("unknown frozen phase")
        row["summary"] = measure_cell(
            functions,
            cell_index=index,
            warmups=settings["warmup_pairs_per_cell"],
            repeats=settings["recorded_pairs_per_cell"],
            rows=row["samples"],
        )
        del functions
        if phase == "reused_session_request":
            for s in active_sessions.values():
                assert_snapshot(s)
            del s, active_sessions
    pin["after_timing"] = native_thread_affinity(pin["pinned"])
    report["after"] = parity_and_retention()
    if report["before"] != report["after"]:
        raise AssertionError("same-process pre/post parity or retention drift")


def positive_times(values, count):
    if len(values) != count or any(type(t) is not int or t <= 0 for t in values):
        raise ValueError("incomplete or invalid raw durations")


def audit(report):
    """Standard-library arithmetic evidence audit; no time acquisition or Torch import."""
    cfg = config()
    assert report["status"] == "passed" and report["mode"] == "paired_timing"
    assert report["protocol"] == cfg and report["source_sha256"] == identities()
    locked = json.loads(files(DATA_PACKAGE).joinpath("data/portable-v1.json").read_bytes())
    assert report["fixture"] == locked["fixture"]
    env = report["environment"]
    assert env["intraop_threads"] == env["interop_threads"] == 1
    assert (
        env["deterministic_algorithms"] and not env["profiler"] and not env["allocation_scopes"]
    )
    pin = env["affinity"]
    assert pin["pinned"] == [min(pin["initial_eligible"])]
    assert pin["torch_imported_before_pin"] is False
    for stage in ("before_parity", "after_timing"):
        assert pin[stage]["observed_thread_count"] >= 1
        assert pin[stage]["all_eligible_masks_equal_pin"] is True
    assert env["thread_environment"] == dict.fromkeys(THREAD_ENV, "1")
    assert report["before"] == report["after"]
    cases = report["before"]
    assert len(cases) == len(locked["cases"])
    for row, source in zip(cases, locked["cases"]):
        prompt = [bytes.fromhex(h) for h in source["prompt_hex"]]
        length, count = len(prompt[0]), source["generated_per_row"]
        assert (row["case"], row["batch"], row["prompt_length"], row["generated"]) == (
            source["id"],
            2,
            length,
            count,
        )
        assert row["prefill_calls"] == 1 and row["append_calls"] == count - 1
        assert not row["last_generated_byte_forwarded"]
        assert row["exact_ids"] and row["snapshot_identity_unchanged"]
        assert row["atol"] == locked["parity_atol"] and row["rtol"] == locked["parity_rtol"]
        assert set(row["max_absolute_logit_error"]) == {
            "dynamic_vs_full",
            "preallocated_vs_full",
            "arm_vs_arm",
        }
        assert all(
            v >= 0 and math.isfinite(v) for v in row["max_absolute_logit_error"].values()
        )
        assert len(row["output_hex"]) == 2
        for output, p in zip(row["output_hex"], prompt):
            seq = bytes.fromhex(output)
            assert seq.startswith(p) and len(seq) == length + count
        assert row["retention"] == {arm: expected_retention(count, length, arm) for arm in ARMS}
    empty = report["empty_boundary"]
    positive_times(empty["samples_ns"], cfg["empty_boundary_pairs"])
    assert empty["summary_ns"] == distribution(empty["samples_ns"])
    cells = report["cells"]
    assert len(cells) == len(cfg["cells"])
    summary = []
    for i, (cell, locked_cell) in enumerate(zip(cells, cfg["cells"])):
        assert {k: cell[k] for k in locked_cell} == locked_cell
        rows = cell["samples"]
        assert len(rows) == cfg["recorded_pairs_per_cell"]
        for j, row in enumerate(rows):
            assert row["index"] == j and row["order"] == clockwise(j, i)
            assert list(row["duration_ns"]) == row["order"]
            positive_times(list(row["duration_ns"].values()), len(ARMS))
        rebuilt = summarize(rows)
        assert cell["summary"] == rebuilt
        summary.append(dict(locked_cell, summary=rebuilt))
    return dict(status="passed", mode="offline_audit", cells=summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    flags = parser.add_mutually_exclusive_group()
    flags.add_argument("--check-only", action="store_true")
    flags.add_argument("--audit", type=Path)
    args = parser.parse_args()
    if not __debug__:
        parser.error("diagnostic assertions must be enabled (no Python -O)")
    if args.output.exists():
        parser.error("output exists; retain prior evidence")
    report = dict(status="failed")
    try:
        if args.audit:
            report = audit(json.loads(args.audit.read_text()))
        else:
            report.update(
                schema_version=1,
                mode="check_only" if args.check_only else "paired_timing",
                protocol=config(),
                source_sha256=identities(),
                environment={},
            )
            if not args.check_only:
                for name in THREAD_ENV:
                    os.environ[name] = "1"
                report["environment"]["affinity"] = affinity()
            runtime(report, args.check_only)
            report["status"] = "passed"
    except (AssertionError, KeyError, OSError, RuntimeError, ValueError) as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(dict(status=report["status"], output=str(args.output))))
    if report["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
