"""Independent synthetic review of source alignment and format-control scoring."""

import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from transformer_core import format_control as control


def config(width=8):
    return {
        "context_bytes": width,
        "changed_cap_per_document": 1000,
        "identical_cap_per_document": 1000,
    }


def test_projection_literal_unicode_entity_and_dropped_span_alignment():
    raw = (
        "<h1>Heading</h1>\n<p>café 中文 &#65; &amp; literal "
        "<em>Word</em> <code>NeverCode</code><script>NeverScript</script>"
        "<style>NeverStyle</style><tcl>NeverTcl</tcl>After [macro] ^caret.</p>"
        "<pre>NeverPre</pre><p>Last paragraph</p>"
    ).encode()
    projected, origins, flags = control.ProseProjection(raw).render()
    assert b"Never" not in projected
    assert b"[macro] ^caret" in projected
    assert "café 中文 A & literal".encode() in projected
    assert len(projected) == len(origins) == len(flags)
    assert origins == sorted(origins)
    for j, flag in enumerate(flags):
        if flag:
            assert raw[origins[j]] == projected[j]
            assert chr(projected[j]).isascii() and chr(projected[j]).isalpha()
    entity_a = projected.index(b" A &") + 1
    assert not flags[entity_a]
    assert not any(flags[: len(b"Heading")])
    word = raw.index(b"Word")
    assert [
        origins[j] for j, flag in enumerate(flags) if flag and word <= origins[j] < word + 4
    ] == list(range(word, word + 4))


def test_pair_histories_exclude_target_future_and_have_exact_budget():
    prefix = b"<p>abcdefghijklmnopqrstuvwxyz <b>BoldWords</b> OrdinaryLettersAgain "
    one = prefix + b"AlphaOmega</p>"
    two = prefix + b"ZZZZZZZZZZ</p><script>ignored</script>"
    left = control.document_pairs(one, {"id": "synthetic"}, config())
    right = control.document_pairs(two, {"id": "synthetic"}, config())
    projected, origins, _ = control.ProseProjection(one).render()
    assert left["pairs"]["changed"] and left["pairs"]["identical"]
    indexed = {p["raw_offset"]: p for group in right["pairs"].values() for p in group}
    for kind, pairs in left["pairs"].items():
        a, b, y = control.histories(one, projected, pairs, 8)
        assert a.shape == b.shape == (len(pairs), 8)
        for i, p in enumerate(pairs):
            assert a[i].tolist() == list(one[p["raw_offset"] - 8 : p["raw_offset"]])
            assert b[i].tolist() == list(
                projected[p["projected_offset"] - 8 : p["projected_offset"]]
            )
            assert (
                max(origins[p["projected_offset"] - 8 : p["projected_offset"]])
                < p["raw_offset"]
            )
            assert y[i].item() == one[p["raw_offset"]] == projected[p["projected_offset"]]
            assert torch.equal(a[i], b[i]) == (kind == "identical")
            if p["raw_offset"] < len(prefix):
                assert p == indexed[p["raw_offset"]]


def test_stratified_sampling_indexes_and_early_exclusion():
    raw = b"<p>abcdefghijklmnopqrstuvwxyz <i>SeveralBoldLetters</i> MoreOrdinaryLetters</p>"
    all_pairs = control.document_pairs(raw, {"id": "s"}, config())
    capped_config = dict(config(), changed_cap_per_document=3, identical_cap_per_document=2)
    selected = control.document_pairs(raw, {"id": "s"}, capped_config)
    for kind, cap in [("changed", 3), ("identical", 2)]:
        total = all_pairs["pairs"][kind]
        indexes = [i * (len(total) - 1) // (cap - 1) for i in range(cap)]
        assert selected["pairs"][kind] == [total[i] for i in indexes]
        assert selected["counts"][kind + "_eligible"] == len(total)
    assert all_pairs["counts"]["early_projected"] > 0
    assert (
        all_pairs["counts"]["early_either"] + sum(len(v) for v in all_pairs["pairs"].values())
        == all_pairs["counts"]["literal_paragraph_ascii_targets"]
    )


class LastByteModel(nn.Module):
    def forward(self, inputs):
        logits = torch.zeros(*inputs.shape, 256, dtype=torch.float64)
        logits.scatter_(-1, ((inputs + 1) % 256).unsqueeze(-1), 2.0)
        return logits


def test_only_last_prediction_is_scored_and_batching_does_not_change_scores():
    x = torch.tensor([[1, 2, 3], [10, 20, 30], [4, 5, 6]])
    y = torch.tensor([4, 9, 7])
    model = LastByteModel()
    expected = F.cross_entropy(model(x)[:, -1], y, reduction="none").tolist()
    assert control.last_target_losses(model, x, y, 1) == expected
    assert control.last_target_losses(model, x, y, 2) == expected
    assert control.last_target_losses(model, x, y, 100) == expected
    m = control.condition_metrics(expected, expected)
    assert m["max_abs_pair_difference"] == m["projected_minus_raw_nll"] == 0
    assert m["raw"]["count"] == len(y)
    assert m["raw"]["nll"] == pytest.approx(math.fsum(expected) / len(y))


def test_history_tampering_and_early_window_are_rejected():
    raw = b"<p>abcdefghijklmnopqrstuvwxyz</p>"
    projected, _, _ = control.ProseProjection(raw).render()
    pairs = control.document_pairs(raw, {"id": "s"}, config())["pairs"]["identical"]
    bad = dict(pairs[0], raw_history_sha256="0" * 64)
    with pytest.raises(ValueError, match="digest"):
        control.histories(raw, projected, [bad], 8)
    bad = dict(pairs[0], projected_offset=2)
    with pytest.raises(ValueError, match="length"):
        control.histories(raw, projected, [bad], 8)


def test_nonfinite_final_logits_fail_closed():
    class Invalid(nn.Module):
        def forward(self, x):
            return torch.full((*x.shape, 256), float("nan"))

    with pytest.raises(ValueError, match="nonfinite"):
        control.last_target_losses(
            Invalid(), torch.ones(2, 8, dtype=torch.long), torch.ones(2, dtype=torch.long), 2
        )


def test_aggregate_uses_seed_paired_effects_not_unpaired_score_levels():
    runs = []
    for seed, arm, left, right in [
        (2, "no_position", [4, 6], [5, 7]),
        (1, "sinusoidal", [1, 3], [2, 4]),
        (2, "sinusoidal", [5, 7], [4, 6]),
        (1, "no_position", [2, 4], [4, 6]),
    ]:
        runs.append(
            {
                "seed": seed,
                "arm": arm,
                "pooled": {
                    "changed": {"model": control.condition_metrics(left, right)},
                    "identical": {"model": control.condition_metrics(left, left)},
                },
            }
        )
    result = control.aggregate_format(runs, [1, 2])
    assert [
        p["no_position_minus_sinusoidal_projection_effect"]
        for p in result["changed"]["paired_seeds"]
    ] == [1, 2]
    assert result["changed"]["paired_effect_difference"]["mean"] == 1.5
    assert result["changed"]["paired_effect_difference"]["sample_sd"] == pytest.approx(
        1 / math.sqrt(2)
    )
    assert result["identical"]["paired_effect_difference"]["mean"] == 0
    with pytest.raises(ValueError, match="roster"):
        control.aggregate_format(runs[:-1] + [runs[0]], [1, 2])
