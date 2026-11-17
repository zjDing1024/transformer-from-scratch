"""Frozen CPU timings and separate fresh-process RSS; no quality experiment."""

import argparse
import gc
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from transformer_core.benchmark import (
    analytic_work,
    cached_prefill,
    cached_request,
    distribution,
    full_prefill,
    full_prefix_request,
    paired_measurement,
    validate_request,
)
from transformer_core.inference import CachedByteDecoder
from transformer_core.inference_cli import load_checkpoint
from transformer_core.stability import file_digest, state_digest, write_json

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "experiments/cpu-kv-v1.json"


def configure(config):
    if torch.__version__ != config["torch_version"]:
        raise ValueError("benchmark requires the pinned CPU torch version")
    if any(
        os.environ.get(name) != "1"
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    ):
        raise ValueError(
            "set OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1 before start"
        )
    torch.set_num_threads(config["intraop_threads"])
    torch.set_num_interop_threads(config["interop_threads"])
    torch.use_deterministic_algorithms(True)


def prompt_tensor(case):
    rows = [list(bytes.fromhex(value)) for value in case["prompt_hex"]]
    if len({len(row) for row in rows}) != 1:
        raise ValueError("all prompt rows must have equal byte length")
    return torch.tensor(rows, dtype=torch.long)


@torch.inference_mode()
def parity(model, case, config):
    """Every used last-position logit plus exact emitted IDs, outside timing."""
    prompt = prompt_tensor(case)
    count, budget = case["generated_per_row"], config["max_context"]
    validate_request(prompt, count, budget)
    session = CachedByteDecoder(model, max_context=budget)
    current = prompt.clone()
    cached = session.prefill(current)
    differences = []
    for index in range(count):
        full = model(current)[:, -1]
        last = cached[:, -1]
        torch.testing.assert_close(
            last, full, atol=config["parity_atol"], rtol=config["parity_rtol"]
        )
        if not torch.equal(last.argmax(-1), full.argmax(-1)):
            raise AssertionError("greedy mismatch; do not benchmark differing trajectories")
        differences.append(float((last - full).abs().max()))
        token = full.argmax(-1, keepdim=True)
        current = torch.cat((current, token), dim=1)
        if index + 1 < count:
            cached = session.append(token)
    expected = full_prefix_request(model, prompt, count, budget)
    actual = cached_request(session, prompt, count, budget)
    if not torch.equal(current, expected) or not torch.equal(expected, actual):
        raise AssertionError("timed helpers differ from audited trajectory")
    work = analytic_work(
        prompt.shape[0], prompt.shape[1], count, model.config["width"], model.config["layers"]
    )
    info = session.cache_info()
    if (
        info["length"] != work["final_cached_positions"]
        or info["kv_payload_bytes"] != work["final_kv_payload_bytes"]
    ):
        raise AssertionError("cache length/payload mismatch")
    return dict(
        last_logits_max_absolute_error=max(differences),
        per_step_max_abs=differences,
        exact_output_ids=True,
        output_hex=[bytes(row).hex() for row in current.tolist()],
        final_cache=info,
        analytic_work=work,
    )


def memory_worker(checkpoint, case, arm, config):
    """Fresh process only. No hashing, parity pass or prior warmup here."""
    if sys.platform != "linux":
        raise ValueError("ru_maxrss units/protocol implemented for Linux only")
    model = load_checkpoint(checkpoint)
    after_load = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    session = (
        CachedByteDecoder(model, max_context=config["max_context"]) if arm == "cached" else None
    )
    prompt = prompt_tensor(case)
    after_setup = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if arm == "cached":
        result = cached_request(
            session, prompt, case["generated_per_row"], config["max_context"]
        )
    else:
        result = full_prefix_request(
            model, prompt, case["generated_per_row"], config["max_context"]
        )
    after_request = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Output encoding and any report allocations happen AFTER all memory readings.
    return dict(
        arm=arm,
        case_id=case["id"],
        peak_rss_kib=dict(
            after_model_load=after_load,
            after_session_setup=after_setup,
            after_request=after_request,
        ),
        output_hex=[bytes(row).hex() for row in result.tolist()],
        kv_payload_bytes=0 if session is None else session.cache_info()["kv_payload_bytes"],
        source_model_retained=True,
        snapshot_owned=arm == "cached",
    )


def measure_constructor(model, config):
    for _ in range(config["constructor_warmups"]):
        session = CachedByteDecoder(model, max_context=config["max_context"])
        del session
    values = []
    for _ in range(config["constructor_repetitions"]):
        start = time.perf_counter_ns()
        session = CachedByteDecoder(model, max_context=config["max_context"])
        elapsed = time.perf_counter_ns() - start
        values.append(elapsed)
        del session
    return dict(
        samples_ns=values,
        summary_ns=distribution(values),
        scope="new-session validation/snapshot construction; NOT cold process or request",
    )


def environment():
    cpu = (
        next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            "unknown",
        )
        if sys.platform == "linux"
        else platform.processor()
    )
    clock = time.get_clock_info("perf_counter")
    return dict(
        python=platform.python_version(),
        torch=torch.__version__,
        platform=platform.platform(),
        cpu=cpu,
        cpu_count=os.cpu_count(),
        affinity=sorted(os.sched_getaffinity(0)),
        intraop_threads=torch.get_num_threads(),
        interop_threads=torch.get_num_interop_threads(),
        thread_environment={
            key: os.environ[key]
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        gc_enabled=gc.isenabled(),
        clock=dict(
            implementation=clock.implementation,
            resolution_seconds=clock.resolution,
            monotonic=clock.monotonic,
        ),
        shared_host=True,
        exclusive_hardware=False,
        cpu_frequency_controlled=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--memory-worker", action="store_true")
    parser.add_argument("--case-id")
    parser.add_argument("--arm", choices=("cached", "full_prefix"))
    args = parser.parse_args()
    config = json.loads(CONFIG_PATH.read_text())
    configure(config)
    if args.memory_worker:
        case = next(c for c in config["cases"] if c["id"] == args.case_id)
        if args.arm is None:
            raise ValueError("memory worker requires --arm")
        print(json.dumps(memory_worker(args.checkpoint, case, args.arm, config)))
        return
    if args.output_dir is None or args.output_dir.exists():
        raise ValueError("provide a NEW output directory; preserve all earlier attempts")
    if file_digest(args.checkpoint) != config["checkpoint_sha256"]:
        raise ValueError("exact checkpoint artifact mismatch")
    model = load_checkpoint(args.checkpoint)
    if state_digest(model.state_dict()) != config["tensor_sha256"]:
        raise ValueError("exact checkpoint tensors mismatch")
    args.output_dir.mkdir(parents=True)
    sources = [
        "src/transformer_core/benchmark.py",
        "examples/benchmark_inference.py",
        "docs/cpu-kv-protocol.md",
        "experiments/cpu-kv-v1.json",
        "src/transformer_core/inference.py",
        "src/transformer_core/language_model.py",
        "src/transformer_core/inference_cli.py",
        "src/transformer_core/attention.py",
    ]
    report = dict(
        schema_version=1,
        config=config,
        environment=environment(),
        source_sha256={p: file_digest(ROOT / p) for p in sources},
        checkpoint_verified=True,
        checkpoint_verification_scope="frozen amended candidate, NOT original identity",
        started_utc=datetime.now(timezone.utc).isoformat(),
        status="started; parity and timing incomplete",
    )
    write_json(args.output_dir / "started.json", report)
    before = {c["id"]: parity(model, c, config) for c in config["cases"]}
    report["new_session_constructor"] = measure_constructor(model, config)
    rows = []
    for case in config["cases"]:
        prompt = prompt_tensor(case)
        session = CachedByteDecoder(model, max_context=config["max_context"])
        options = dict(
            warmups=config["warmup_pairs"],
            repeats=config["paired_repetitions"],
            inner=config["requests_per_timed_sample"],
        )
        prefills = paired_measurement(
            lambda: full_prefill(model, prompt),
            lambda: cached_prefill(session, prompt),
            **options,
        )
        requests = paired_measurement(
            lambda: full_prefix_request(
                model, prompt, case["generated_per_row"], config["max_context"]
            ),
            lambda: cached_request(
                session, prompt, case["generated_per_row"], config["max_context"]
            ),
            **options,
        )
        after = parity(model, case, config)
        if before[case["id"]] != after:
            raise AssertionError("parity evidence changed after timing")
        rows.append(
            dict(
                case_id=case["id"],
                parity_before_and_after=after,
                warm_prefill=prefills,
                warm_request=requests,
            )
        )
        write_json(args.output_dir / "timing-progress.json", rows)
    report["cases"] = rows
    report["timings_finished_utc"] = datetime.now(timezone.utc).isoformat()
    memories = []
    for case in config["cases"]:
        for repeat in range(config["memory_workers_per_case_arm"]):
            order = ["full_prefix", "cached"] if repeat % 2 == 0 else ["cached", "full_prefix"]
            for arm in order:
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--checkpoint",
                    str(args.checkpoint.resolve()),
                    "--memory-worker",
                    "--case-id",
                    case["id"],
                    "--arm",
                    arm,
                ]
                child = subprocess.run(
                    command, capture_output=True, text=True, check=True, env=os.environ.copy()
                )
                row = json.loads(child.stdout)
                if row["output_hex"] != before[case["id"]]["output_hex"]:
                    raise AssertionError("memory worker has different output")
                row["repeat"] = repeat
                row["order"] = order
                memories.append(row)
                write_json(args.output_dir / "memory-progress.json", memories)
    report["memory_workers"] = memories
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    report["status"] = "complete; descriptive shared-host timings only"
    write_json(args.output_dir / "report.json", report)
    print(
        json.dumps(
            dict(
                report_sha256=hashlib.sha256(
                    (args.output_dir / "report.json").read_bytes()
                ).hexdigest(),
                status=report["status"],
            )
        )
    )


if __name__ == "__main__":
    main()
