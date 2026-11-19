"""Original same-semantics CPU benchmark helpers; no speed threshold tests."""

import math
import statistics
import time

import torch


def validate_request(prompt, count, max_context):
    if (
        not isinstance(prompt, torch.Tensor)
        or prompt.dtype != torch.long
        or prompt.device.type != "cpu"
        or prompt.ndim != 2
        or 0 in prompt.shape
        or bool(torch.any((prompt < 0) | (prompt > 255)))
    ):
        raise ValueError("prompt must be nonempty CPU [B,P] int64 byte IDs")
    if type(count) is not int or count < 1:
        raise ValueError("benchmark count must be a positive integer")
    if type(max_context) is not int or max_context < prompt.shape[1] + count:
        raise ValueError("prompt plus output must fit the total context budget")


@torch.inference_mode()
def full_prefix_request(model, prompt, count, max_context=64):
    """Exactly count full-prefix forwards; never forward the last emitted byte."""
    validate_request(prompt, count, max_context)
    sequence = prompt.clone()
    with torch.autocast(device_type="cpu", enabled=False):
        for _ in range(count):
            token = model(sequence)[:, -1].argmax(-1, keepdim=True)
            sequence = torch.cat((sequence, token), dim=1)
    return sequence


@torch.inference_mode()
def cached_request(session, prompt, count, max_context=64):
    """Warm snapshot, but reset sequence state on EVERY request, including warmup."""
    validate_request(prompt, count, max_context)
    session.reset()
    sequence = prompt.clone()
    logits = session.prefill(sequence)
    for index in range(count):
        token = logits[:, -1].argmax(-1, keepdim=True)
        sequence = torch.cat((sequence, token), dim=1)
        if index + 1 < count:
            logits = session.append(token)
    return sequence


@torch.inference_mode()
def full_prefill(model, prompt):
    with torch.autocast(device_type="cpu", enabled=False):
        return model(prompt)


@torch.inference_mode()
def cached_prefill(session, prompt):
    session.reset()
    return session.prefill(prompt)


def analytic_work(batch, prompt_length, count, width, layers, element_size=4):
    """Logical path/tensor counts, NOT latency, FLOPs, RSS or measured traffic."""
    if any(
        type(x) is not int or x < 1
        for x in (batch, prompt_length, count, width, layers, element_size)
    ):
        raise ValueError("work-count dimensions must be positive integers")
    lengths = list(range(prompt_length, prompt_length + count))
    append_lengths = lengths[1:]
    bytes_per_position = 2 * layers * batch * width * element_size
    return dict(
        forward_calls=count,
        cached_prefill_calls=1,
        cached_append_calls=count - 1,
        full_prefix_positions_per_row=sum(lengths),
        cached_positions_per_row=lengths[-1],
        full_dense_qk_entries_per_row_head_layer=sum(x * x for x in lengths),
        cached_dense_qk_entries_per_row_head_layer=prompt_length**2 + sum(append_lengths),
        final_cached_positions=lengths[-1],
        final_kv_payload_bytes=bytes_per_position * lengths[-1],
        concatenation_output_payload_bytes=bytes_per_position * sum(append_lengths),
        old_prefix_payload_copied_bytes=bytes_per_position * sum(x - 1 for x in append_lengths),
    )


def distribution(values):
    """Descriptive nearest-rank p90; no confidence interval or independence claim."""
    if not values or any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError("positive finite measurements required")
    ordered = sorted(values)
    return dict(
        n=len(values),
        median=statistics.median(values),
        p90_nearest_rank=ordered[math.ceil(0.9 * len(values)) - 1],
        minimum=ordered[0],
        maximum=ordered[-1],
    )


def measure_calls(function, inner, clock=time.perf_counter_ns):
    """Return batch mean ns/call. No correctness comparison or JSON in the interval."""
    if type(inner) is not int or inner < 1:
        raise ValueError("inner must be positive")
    start = clock()
    for _ in range(inner):
        result = function()
    elapsed = clock() - start
    # Keep final result alive through stop. Other result destruction occurs on
    # reassignment equally in both paths; session reset is part of cached timing.
    del result
    return elapsed / inner


def paired_measurement(full, cached, *, warmups, repeats, inner, clock=time.perf_counter_ns):
    """Interleave fixed trajectories, never time a growing uncontrolled cache."""
    if any(type(x) is not int or x < 1 for x in (warmups, repeats, inner)):
        raise ValueError("warmups/repeats/inner must be positive")
    functions = {"full_prefix": full, "cached": cached}
    for index in range(warmups):
        order = ["full_prefix", "cached"] if index % 2 == 0 else ["cached", "full_prefix"]
        for arm in order:
            functions[arm]()
    records = []
    for index in range(repeats):
        order = ["full_prefix", "cached"] if index % 2 == 0 else ["cached", "full_prefix"]
        timings = {arm: measure_calls(functions[arm], inner, clock) for arm in order}
        records.append(dict(index=index, order=order, ns_per_call=timings))
    summaries = {
        arm: distribution([r["ns_per_call"][arm] for r in records]) for arm in functions
    }
    ratios = [r["ns_per_call"]["cached"] / r["ns_per_call"]["full_prefix"] for r in records]
    return dict(
        samples=records,
        summary_ns=summaries,
        paired_cached_over_full_ratio=distribution(ratios),
        measurement="each sample is a batch mean, not an individual-request percentile",
    )
