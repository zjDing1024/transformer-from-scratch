"""Installed lightweight launcher for portable CPU parity and scoped Linux memory."""

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
from importlib.resources import files
from pathlib import Path

from .proc_memory import difference, observe, positive_control, reset_peak, window_status


def configuration():
    return json.loads(
        files("transformer_diagnostics").joinpath("data/portable-v1.json").read_text()
    )


def cpu_description():
    value = platform.processor()
    if value:
        return value
    if sys.platform == "linux":
        try:
            return next(
                (
                    line.split(":", 1)[1].strip()
                    for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")
                ),
                "unreported",
            )
        except (OSError, UnicodeError):
            pass
    return "unreported"


def worker(mode, case_id=None, arm=None):
    # Keep Torch entirely out of the launcher process and import it only here.
    import torch

    from transformer_core.benchmark import cached_request, full_prefix_request
    from transformer_core.inference import CachedByteDecoder
    from transformer_core.portable_fixture import verified_fixture
    from transformer_core.portable_parity import verify_parity

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    config = configuration()
    model, artifact, identity = verified_fixture()
    del artifact
    gc.collect()
    env = dict(
        python=platform.python_version(),
        torch=torch.__version__,
        platform=platform.platform(),
        cpu=cpu_description(),
        threads=torch.get_num_threads(),
        interop_threads=torch.get_num_interop_threads(),
        device="cpu",
        dtype="float32",
    )
    if mode == "parity":
        return dict(fixture=identity, environment=env, parity=verify_parity(model, config))
    case = next(c for c in config["cases"] if c["id"] == case_id)
    if arm not in ("cached", "full_prefix"):
        raise ValueError("unknown memory arm")
    after_model = observe()
    session = (
        CachedByteDecoder(model, max_context=config["max_context"]) if arm == "cached" else None
    )
    prompt = torch.tensor([list(bytes.fromhex(h)) for h in case["prompt_hex"]])
    after_setup = observe()
    reset = reset_peak()
    baseline = observe()
    if session is None:
        output = full_prefix_request(
            model, prompt, case["generated_per_row"], config["max_context"]
        )
    else:
        output = cached_request(
            session, prompt, case["generated_per_row"], config["max_context"]
        )
    after_request = observe()
    observation = window_status(
        reset, baseline, after_request, config["post_reset_alignment_kib"]
    )
    # Control is a DIFFERENT window, after recording the request, never a request peak.
    control = positive_control(
        config["positive_control_bytes"], config["minimum_positive_growth_kib"]
    )
    if control["status"] != "passed":
        observation = dict(
            status="unavailable", reason="coarse resident positive control did not pass"
        )
    return dict(
        fixture=identity,
        environment=env,
        case_id=case_id,
        arm=arm,
        request_window_status=observation,
        after_model=after_model,
        after_setup=after_setup,
        request_reset=reset,
        request_baseline=baseline,
        after_request=after_request,
        raw_rss_change_kib=difference(after_request, baseline, "VmRSS"),
        raw_hwm_minus_baseline_rss_kib=peak_above_rss(after_request, baseline),
        request_phase_peak_kib=(
            after_request["VmHWM"]
            if observation["status"] == "observed"
            else dict(status="unavailable", reason="request window/control not verified")
        ),
        raw_delta_scope="snapshot differences only; NOT allocations or verified phase metrics",
        positive_control=control,
        source_model_retained=True,
        snapshot_owned=session is not None,
        kv_payload_bytes=0 if session is None else session.cache_info()["kv_payload_bytes"],
        output_hex=[bytes(row).hex() for row in output.tolist()],
    )


def peak_above_rss(after, baseline):
    if after["VmHWM"]["status"] != "available" or baseline["VmRSS"]["status"] != "available":
        return dict(status="unavailable", reason="VmHWM and baseline VmRSS required")
    return dict(
        status="available",
        value_kib=after["VmHWM"]["value_kib"] - baseline["VmRSS"]["value_kib"],
    )


def invoke(mode, case_id=None, arm=None):
    args = [sys.executable, "-m", "transformer_diagnostics.cli", "--worker", mode]
    if case_id is not None:
        args += ["--case-id", case_id, "--arm", arm]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    completed = subprocess.run(args, env=env, capture_output=True, text=True, timeout=120)
    if completed.returncode:
        raise RuntimeError(f"{mode} worker failed: {completed.stderr[-1500:]}")
    return json.loads(completed.stdout)


def run(include_memory=True):
    config = configuration()
    if "torch" in sys.modules:
        raise RuntimeError("launcher must not import Torch; use a fresh process")
    parity = invoke("parity")
    cases = {case["id"]: case for case in parity["parity"]["cases"]}
    memories = []
    if include_memory:
        for index, case in enumerate(config["cases"]):
            order = ("full_prefix", "cached") if index % 2 == 0 else ("cached", "full_prefix")
            for arm in order:
                result = invoke("memory", case["id"], arm)
                if (
                    result["fixture"] != config["fixture"]
                    or result["output_hex"] != cases[case["id"]]["output_hex"]
                ):
                    raise AssertionError(
                        "memory worker fixture/output differs from parity worker"
                    )
                expected_payload = (
                    cases[case["id"]]["analytic_work"]["final_kv_payload_bytes"]
                    if arm == "cached"
                    else 0
                )
                if result["kv_payload_bytes"] != expected_payload:
                    raise AssertionError("unexpected resident cache payload")
                memories.append(result)
    source_paths = [
        "transformer_core/portable_fixture.py",
        "transformer_core/portable_parity.py",
        "transformer_core/inference.py",
        "transformer_core/benchmark.py",
        "transformer_diagnostics/cli.py",
        "transformer_diagnostics/proc_memory.py",
        "transformer_diagnostics/data/portable-v1.json",
    ]
    package_root = Path(__file__).resolve().parent.parent
    return dict(
        schema_version=1,
        config=config,
        lightweight_launcher_no_torch=True,
        source_sha256={
            p: hashlib.sha256((package_root / p).read_bytes()).hexdigest() for p in source_paths
        },
        fixture=parity["fixture"],
        environment=parity["environment"],
        parity=parity["parity"],
        memory_status="not_requested"
        if not include_memory
        else (
            "observed"
            if all(m["request_window_status"]["status"] == "observed" for m in memories)
            else "partly_or_fully_unavailable"
        ),
        memory_workers=memories,
        timing_performed=False,
        training_performed=False,
        scope="finite parity/process metrics only; no savings, speed or quality claim",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--parity-only", action="store_true")
    parser.add_argument("--worker", choices=("parity", "memory"), help=argparse.SUPPRESS)
    parser.add_argument("--case-id", help=argparse.SUPPRESS)
    parser.add_argument("--arm", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        raise ValueError("output exists; preserve earlier observations")
    result = (
        worker(args.worker, args.case_id, args.arm)
        if args.worker
        else run(not args.parity_only)
    )
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as target:
            target.write(text)
        print(
            json.dumps(
                dict(
                    output=str(args.output),
                    parity=result["parity"]["status"],
                    memory=result["memory_status"],
                )
            )
        )


if __name__ == "__main__":
    main()
