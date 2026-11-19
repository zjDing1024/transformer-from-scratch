"""Offline standard-library audit of frozen CPU benchmark records; no inference."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def stats(values):
    assert values and all(math.isfinite(v) and v > 0 for v in values)
    ordered = sorted(values)
    return dict(
        n=len(values),
        median=statistics.median(values),
        p90_nearest_rank=ordered[math.ceil(len(values) * 0.9) - 1],
        minimum=min(values),
        maximum=max(values),
    )


def audit(report):
    config = report["config"]
    assert report["status"] == "complete; descriptive shared-host timings only"
    assert config["original_identity_matched"] is False
    assert report["checkpoint_verified"] is True
    assert config["checkpoint_sha256"] != config["original_requested_checkpoint_sha256"]
    assert config["tensor_sha256"] != config["original_requested_tensor_sha256"]
    assert config == json.loads((ROOT / "experiments/cpu-kv-v1.json").read_text())
    for name, sha in report["source_sha256"].items():
        # Nov20 opt-in ranges change these current sources. Audit the exact
        # archived versions actually used in this historical run, not new code.
        archived = {
            "src/transformer_core/inference.py": "inference.py.txt",
            "src/transformer_core/benchmark.py": "benchmark.py.txt",
        }
        source = (
            ROOT / "experiments/cpu-kv-v1-source" / archived[name]
            if name in archived
            else ROOT / name
        )
        assert hashlib.sha256(source.read_bytes()).hexdigest() == sha, name
    constructor = report["new_session_constructor"]
    assert len(constructor["samples_ns"]) == config["constructor_repetitions"]
    assert constructor["summary_ns"] == stats(constructor["samples_ns"])
    case_ids = [c["id"] for c in config["cases"]]
    assert [r["case_id"] for r in report["cases"]] == case_ids
    assert (
        len(report["memory_workers"])
        == len(case_ids) * 2 * config["memory_workers_per_case_arm"]
    )
    summary = dict(
        schema_version=1,
        inference_performed=False,
        constructor_median_ms=constructor["summary_ns"]["median"] / 1e6,
        constructor_p90_ms=constructor["summary_ns"]["p90_nearest_rank"] / 1e6,
        original_checkpoint_identity_matched=False,
        cases=[],
    )
    for case, row in zip(config["cases"], report["cases"]):
        batch, length, count = (
            len(case["prompt_hex"]),
            len(bytes.fromhex(case["prompt_hex"][0])),
            case["generated_per_row"],
        )
        assert length + count <= config["max_context"]
        parity = row["parity_before_and_after"]
        assert parity["exact_output_ids"] is True
        assert len(parity["per_step_max_abs"]) == count
        assert parity["last_logits_max_absolute_error"] == max(parity["per_step_max_abs"])
        assert len(parity["output_hex"]) == batch
        assert all(len(bytes.fromhex(h)) == length + count for h in parity["output_hex"])
        assert all(h.startswith(p) for h, p in zip(parity["output_hex"], case["prompt_hex"]))
        lengths = range(length, length + count)
        expected_work = dict(
            forward_calls=count,
            cached_prefill_calls=1,
            cached_append_calls=count - 1,
            full_prefix_positions_per_row=sum(lengths),
            cached_positions_per_row=length + count - 1,
            full_dense_qk_entries_per_row_head_layer=sum(v * v for v in lengths),
            cached_dense_qk_entries_per_row_head_layer=length * length
            + sum(range(length + 1, length + count)),
            final_cached_positions=length + count - 1,
            final_kv_payload_bytes=1024 * batch * (length + count - 1),
            concatenation_output_payload_bytes=1024
            * batch
            * sum(range(length + 1, length + count)),
            old_prefix_payload_copied_bytes=1024
            * batch
            * sum(range(length, length + count - 1)),
        )
        assert parity["analytic_work"] == expected_work
        assert (
            parity["final_cache"]["kv_payload_bytes"] == expected_work["final_kv_payload_bytes"]
        )
        item = dict(
            case_id=case["id"],
            max_logit_error=parity["last_logits_max_absolute_error"],
            exact_generated_ids=batch * count,
            kv_payload_bytes=expected_work["final_kv_payload_bytes"],
        )
        for phase in ("warm_prefill", "warm_request"):
            values = row[phase]
            assert len(values["samples"]) == config["paired_repetitions"]
            for index, sample in enumerate(values["samples"]):
                assert sample["index"] == index
                assert sample["order"] == (
                    ["full_prefix", "cached"] if index % 2 == 0 else ["cached", "full_prefix"]
                )
                assert set(sample["ns_per_call"]) == {"full_prefix", "cached"}
            for arm in ("full_prefix", "cached"):
                assert values["summary_ns"][arm] == stats(
                    [s["ns_per_call"][arm] for s in values["samples"]]
                )
            ratios = [
                s["ns_per_call"]["cached"] / s["ns_per_call"]["full_prefix"]
                for s in values["samples"]
            ]
            assert values["paired_cached_over_full_ratio"] == stats(ratios)
            item[phase] = {
                arm: {
                    key: value / 1e6
                    for key, value in values["summary_ns"][arm].items()
                    if key != "n"
                }
                for arm in ("full_prefix", "cached")
            }
            item[phase]["paired_cached_over_full_median"] = statistics.median(ratios)
            item[phase]["units"] = "milliseconds per call, statistics of five-call means"
        memory_rows = [m for m in report["memory_workers"] if m["case_id"] == case["id"]]
        assert {(m["repeat"], m["arm"]) for m in memory_rows} == {
            (i, a)
            for i in range(config["memory_workers_per_case_arm"])
            for a in ("full_prefix", "cached")
        }
        for mem in memory_rows:
            assert mem["output_hex"] == parity["output_hex"]
            assert mem["source_model_retained"] is True
            assert mem["snapshot_owned"] == (mem["arm"] == "cached")
            assert mem["kv_payload_bytes"] == (
                expected_work["final_kv_payload_bytes"] if mem["arm"] == "cached" else 0
            )
            rss = mem["peak_rss_kib"]
            assert (
                0
                < rss["after_model_load"]
                <= rss["after_session_setup"]
                <= rss["after_request"]
            )
        item["process_peak_rss_kib_after_request"] = {
            arm: [m["peak_rss_kib"]["after_request"] for m in memory_rows if m["arm"] == arm]
            for arm in ("full_prefix", "cached")
        }
        summary["cases"].append(item)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "results/cpu-kv-v1/report.json")
    args = parser.parse_args()
    print(json.dumps(audit(json.loads(args.report.read_text())), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
