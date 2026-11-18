"""Full-prefix reference checks on independently regenerable, untrained weights."""

import torch

from .benchmark import analytic_work, cached_request, full_prefix_request
from .inference import CachedByteDecoder


@torch.inference_mode()
def verify_parity(model, config):
    def error(actual, expected):
        torch.testing.assert_close(
            actual, expected, atol=config["parity_atol"], rtol=config["parity_rtol"]
        )
        return float((actual - expected).abs().max())

    data = torch.tensor([[(17 * i + 31 * row) % 256 for i in range(64)] for row in range(2)])
    reference = model(data)
    teacher = []
    for chunks in ([64], [1] * 64, [5, 13, 1, 17, 28]):
        session = CachedByteDecoder(model, max_context=config["max_context"])
        offset, differences = 0, []
        for size in chunks:
            actual = (session.prefill if offset == 0 else session.append)(
                data[:, offset : offset + size]
            )
            differences.append(error(actual, reference[:, offset : offset + size]))
            error(actual, model(data[:, : offset + size])[:, offset:])
            offset += size
        teacher.append(dict(chunks=chunks, max_absolute_error=max(differences)))
    cases = []
    for case in config["cases"]:
        prompt = torch.tensor([list(bytes.fromhex(h)) for h in case["prompt_hex"]])
        count = case["generated_per_row"]
        session = CachedByteDecoder(model, max_context=config["max_context"])
        sequence = prompt.clone()
        cached = session.prefill(sequence)
        differences = []
        for index in range(count):
            full = model(sequence)[:, -1]
            differences.append(error(cached[:, -1], full))
            token = full.argmax(-1, keepdim=True)
            if not torch.equal(token, cached[:, -1].argmax(-1, keepdim=True)):
                raise AssertionError("greedy mismatch; stop rather than changing fixture")
            sequence = torch.cat((sequence, token), dim=1)
            if index + 1 < count:
                cached = session.append(token)
        for actual in (
            full_prefix_request(model, prompt, count),
            cached_request(session, prompt, count),
        ):
            if not torch.equal(actual, sequence):
                raise AssertionError("request helper differs from reference")
        work = analytic_work(
            prompt.shape[0],
            prompt.shape[1],
            count,
            model.config["width"],
            model.config["layers"],
        )
        if session.cache_info()["length"] != work["final_cached_positions"]:
            raise AssertionError("extra/missing forward in cache")
        cases.append(
            dict(
                id=case["id"],
                max_absolute_error=max(differences),
                per_step_max_absolute_error=differences,
                exact_greedy_ids=True,
                output_hex=[bytes(row).hex() for row in sequence.tolist()],
                analytic_work=work,
                cache=session.cache_info(),
            )
        )
    return dict(
        status="passed",
        teacher_forced=teacher,
        cases=cases,
        atol=config["parity_atol"],
        rtol=config["parity_rtol"],
        bitwise_logits_required=False,
        claim="finite CPU correctness fixtures, not training quality or arbitrary-prompt proof",
    )
