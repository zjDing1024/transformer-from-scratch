"""Contract and fake-clock tests, never latency thresholds or real benchmark runs."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from transformer_core.benchmark import (
    analytic_work,
    cached_prefill,
    cached_request,
    distribution,
    full_prefill,
    full_prefix_request,
    measure_calls,
    paired_measurement,
    validate_request,
)
from transformer_core.inference import CachedByteDecoder
from transformer_core.language_model import ByteLanguageModel

ROOT = Path(__file__).resolve().parents[1]


def model():
    torch.manual_seed(118)
    return ByteLanguageModel(width=12, heads=3, hidden=17, layers=2).double().eval()


@pytest.mark.parametrize("batch,prompt_length,count", [(1, 3, 1), (2, 5, 4), (2, 32, 32)])
def test_equal_requests_no_final_forward_and_projection_work(batch, prompt_length, count):
    torch.set_num_threads(1)
    net = model()
    prompt = torch.arange(batch * prompt_length).reshape(batch, prompt_length) % 256
    session = CachedByteDecoder(net)
    seen_full, seen_cached = [], []
    a = net.blocks[0].attention.q_proj.register_forward_hook(
        lambda mod, args, out: seen_full.append(args[0].shape[1])
    )
    b = session._model.blocks[0].attention.q_proj.register_forward_hook(
        lambda mod, args, out: seen_cached.append(args[0].shape[1])
    )
    expected = full_prefix_request(net, prompt, count)
    result = cached_request(session, prompt, count)
    assert torch.equal(expected, result)
    assert seen_full == list(range(prompt_length, prompt_length + count))
    assert seen_cached == [prompt_length] + [1] * (count - 1)
    work = analytic_work(batch, prompt_length, count, 12, 2, 8)
    assert sum(seen_full) == work["full_prefix_positions_per_row"]
    assert sum(seen_cached) == work["cached_positions_per_row"]
    assert session.cache_info()["length"] == prompt_length + count - 1
    assert session.cache_info()["kv_payload_bytes"] == work["final_kv_payload_bytes"]
    # Every timed helper invocation must replay from reset, not keep growing.
    again = cached_request(session, prompt, count)
    assert torch.equal(result, again)
    assert session.cache_info()["length"] == prompt_length + count - 1
    a.remove()
    b.remove()


def test_prefill_resets_and_matches_reference():
    net = model()
    session = CachedByteDecoder(net)
    prompt = torch.tensor([[1, 2, 3], [3, 2, 1]])
    for _ in range(3):
        torch.testing.assert_close(
            cached_prefill(session, prompt), full_prefill(net, prompt), atol=1e-12, rtol=1e-12
        )
        assert session.cache_info()["length"] == 3


def test_preflight_work_counts():
    work = analytic_work(2, 16, 16, 64, 2)
    assert work == dict(
        forward_calls=16,
        cached_prefill_calls=1,
        cached_append_calls=15,
        full_prefix_positions_per_row=376,
        cached_positions_per_row=31,
        full_dense_qk_entries_per_row_head_layer=9176,
        cached_dense_qk_entries_per_row_head_layer=616,
        final_cached_positions=31,
        final_kv_payload_bytes=62 * 1024,
        concatenation_output_payload_bytes=720 * 1024,
        old_prefix_payload_copied_bytes=690 * 1024,
    )


def test_fake_clock_records_order_means_and_all_samples():
    calls = []
    times = iter(range(0, 10000, 100))
    result = paired_measurement(
        lambda: calls.append("F"),
        lambda: calls.append("C"),
        warmups=2,
        repeats=4,
        inner=5,
        clock=lambda: next(times),
    )
    assert calls[:4] == ["F", "C", "C", "F"]
    assert calls[4:] == (["F"] * 5 + ["C"] * 5 + ["C"] * 5 + ["F"] * 5) * 2
    for i, row in enumerate(result["samples"]):
        assert row["order"] == (
            ["full_prefix", "cached"] if i % 2 == 0 else ["cached", "full_prefix"]
        )
        assert row["ns_per_call"] == dict(full_prefix=20, cached=20)
    assert result["summary_ns"]["full_prefix"]["n"] == 4
    assert result["paired_cached_over_full_ratio"]["median"] == 1


def test_summary_nearest_rank_and_no_outlier_removal():
    row = distribution([1, 2, 3, 100])
    assert row == dict(n=4, median=2.5, p90_nearest_rank=100, minimum=1, maximum=100)


@pytest.mark.parametrize("values", [[], [0], [float("inf")], [-1], [float("nan")]])
def test_bad_distribution(values):
    with pytest.raises(ValueError):
        distribution(values)


@pytest.mark.parametrize(
    "prompt,count,budget",
    [
        (torch.tensor([[1.0]]), 1, 64),
        (torch.tensor([[256]]), 1, 64),
        (torch.empty(1, 0, dtype=torch.long), 1, 64),
        (torch.tensor([[1]]), 0, 64),
        (torch.tensor([[1]]), True, 64),
        (torch.tensor([[1]]), 64, 64),
        (torch.tensor([[1]]), 1, True),
        (None, 1, 64),
    ],
)
def test_bad_request(prompt, count, budget):
    with pytest.raises(ValueError):
        validate_request(prompt, count, budget)


def test_bad_measurement_configuration():
    with pytest.raises(ValueError):
        measure_calls(lambda: None, 0)
    with pytest.raises(ValueError):
        paired_measurement(lambda: None, lambda: None, warmups=0, repeats=2, inner=1)
    with pytest.raises(ValueError):
        analytic_work(2, 16, 0, 64, 2)


def test_fixed_case_geometry_and_runner_parity_without_timing():
    spec = importlib.util.spec_from_file_location(
        "bench_runner", ROOT / "examples/benchmark_inference.py"
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    config = json.loads((ROOT / "experiments/cpu-kv-v1.json").read_text())
    assert [
        (runner.prompt_tensor(c).shape, c["generated_per_row"]) for c in config["cases"]
    ] == [(torch.Size([1, 16]), 16), (torch.Size([2, 16]), 16), (torch.Size([2, 32]), 32)]
    net = model()
    tiny = dict(id="synthetic", prompt_hex=["010203", "030201"], generated_per_row=3)
    proof = runner.parity(net.float(), tiny, config)
    assert proof["exact_output_ids"]
    assert proof["final_cache"]["length"] == 5
