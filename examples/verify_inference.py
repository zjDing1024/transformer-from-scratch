"""Frozen real-weight equivalence checks, not text quality or performance scoring."""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import torch

from transformer_core.inference import CachedByteDecoder, greedy_cached
from transformer_core.inference_cli import load_checkpoint
from transformer_core.stability import configure_cpu, file_digest, state_digest, write_json

ROOT = Path(__file__).resolve().parents[1]
ATOL = RTOL = 1e-5


def compare(actual, expected):
    torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)
    return float((actual - expected).abs().max())


@torch.inference_mode()
def verify(model):
    tokens = torch.tensor([[(17 * i + 31 * row) % 256 for i in range(64)] for row in range(2)])
    reference = model(tokens)
    schedules = [[64], [1] * 64, [5, 13, 1, 17, 28], [16, 48]]
    probes = []
    for chunks in schedules:
        session = CachedByteDecoder(model, max_context=64)
        seen, rows = 0, []
        for size in chunks:
            result = (session.prefill if seen == 0 else session.append)(
                tokens[:, seen : seen + size]
            )
            prefix = model(tokens[:, : seen + size])[:, seen:]
            rows.append(
                dict(
                    offset=seen,
                    size=size,
                    versus_fresh_prefix_max_abs=compare(result, prefix),
                    versus_full64_slice_max_abs=compare(
                        result, reference[:, seen : seen + size]
                    ),
                )
            )
            seen += size
        info = session.cache_info()
        if info["kv_payload_bytes"] != 131072 or info["length"] != 64:
            raise AssertionError("unexpected cache payload/length")
        expected_shape = [2, 4, 64, 16]
        if info["layers"] != [dict(key=expected_shape, value=expected_shape)] * 2:
            raise AssertionError("unexpected cache shape")
        try:
            session.append(tokens[:, :1])
        except ValueError:
            pass
        else:
            raise AssertionError("context overflow was not rejected")
        if session.cache_info() != info:
            raise AssertionError("rejected overflow advanced the cache")
        probes.append(
            dict(chunks=chunks, comparisons=rows, final_cache=info, overflow_rejected=True)
        )
    prompts = [b"The application ", b"HTTP request:   "]
    prompt = torch.tensor([list(p) for p in prompts])
    sequence = prompt.clone()
    session = CachedByteDecoder(model, max_context=64)
    result = session.prefill(prompt)
    greedy_rows = []
    for step in range(16):
        full = model(sequence)[:, -1]
        cached = result[:, -1]
        delta = compare(cached, full)
        cached_token = cached.argmax(-1, keepdim=True)
        full_token = full.argmax(-1, keepdim=True)
        if not torch.equal(cached_token, full_token):
            raise AssertionError("greedy token mismatch")
        greedy_rows.append(
            dict(
                step=step, max_abs_logit_difference=delta, token_ids=cached_token[:, 0].tolist()
            )
        )
        sequence = torch.cat((sequence, full_token), dim=1)
        if step + 1 < 16:
            result = session.append(cached_token)
    if not torch.equal(greedy_cached(model, prompt, count=16), sequence):
        raise AssertionError("public greedy helper differs from audited loop")
    return dict(
        tolerances=dict(atol=ATOL, rtol=RTOL, bitwise_logits_required=False),
        teacher_forced_input_hex=[bytes(row.tolist()).hex() for row in tokens],
        teacher_forced=probes,
        greedy=dict(
            prompt_hex=[p.hex() for p in prompts],
            generated_per_row=16,
            all_token_ids_equal=True,
            steps=greedy_rows,
            full_output_hex=[bytes(row.tolist()).hex() for row in sequence],
        ),
        maximum_absolute_logit_error=max(
            [
                max(r["versus_fresh_prefix_max_abs"], r["versus_full64_slice_max_abs"])
                for p in probes
                for r in p["comparisons"]
            ]
            + [r["max_abs_logit_difference"] for r in greedy_rows]
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output already exists; preserve previous evidence")
    configure_cpu()
    selection_path = ROOT / "results/position-v1/selection.json"
    selection_hash = "4a12794b6df715539657cbb951a7a52c4423f7102a578eaf7ca52c7739da4b9b"
    if file_digest(selection_path) != selection_hash:
        raise ValueError("original selection lock changed")
    selection = json.loads(selection_path.read_text())
    row = next(
        r for r in selection["runs"] if r["seed"] == 20261114 and r["arm"] == "sinusoidal"
    )
    artifact_hash = file_digest(args.checkpoint)
    if artifact_hash != row["checkpoint_sha256"]:
        raise ValueError("checkpoint artifact mismatch")
    model = load_checkpoint(args.checkpoint)
    tensor_hash = state_digest(model.state_dict())
    if tensor_hash != row["selected_state_sha256"]:
        raise ValueError("checkpoint tensor mismatch")
    if model.config != dict(selection["config"]["model"], use_positions=True):
        raise ValueError("model configuration mismatch")
    report = dict(
        schema_version=1,
        purpose="bounded cached-vs-full-prefix correctness, no quality benchmark",
        source_sha256={
            name: file_digest(ROOT / name)
            for name in (
                "src/transformer_core/inference.py",
                "src/transformer_core/inference_cli.py",
                "src/transformer_core/language_model.py",
                "examples/verify_inference.py",
                "docs/inference-protocol.md",
            )
        },
        checkpoint=dict(
            seed=20261114,
            arm="sinusoidal",
            selected_step=row["selected_step"],
            original_selection_sha256=selection_hash,
            artifact_sha256=artifact_hash,
            original_tensor_sha256=tensor_hash,
            artifact_matches_original=True,
            tensors_match_original=True,
        ),
        environment=dict(
            python=platform.python_version(),
            torch=torch.__version__,
            device="cpu",
            threads=1,
            deterministic_algorithms=True,
        ),
        evidence=verify(model),
        performance_claim="none; no timing benchmark performed",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(
        json.dumps(
            dict(
                report_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
                max_error=report["evidence"]["maximum_absolute_logit_error"],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
