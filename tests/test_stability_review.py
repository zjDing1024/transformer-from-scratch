"""Independent synthetic-only review of paired selection and locked scoring."""

import copy
import math

import pytest
import torch

from transformer_core import stability as study


@pytest.fixture
def tiny_config():
    return {
        "seeds": [31, 32],
        "arms": {"sinusoidal": True, "no_position": False},
        "model": {"width": 4, "heads": 1, "hidden": 7, "layers": 1},
        "steps": 2,
        "validate_every": 1,
        "batch": 2,
        "context": 3,
        "learning_rate": 0.001,
        "weight_decay": 0.01,
        "clip_norm": 1.0,
        "bigram_alpha": 0.1,
    }


@pytest.fixture
def synthetic_documents():
    return [torch.tensor([3, 1, 4, 1, 5, 9, 2, 6], dtype=torch.long)]


def metric(nll, tokens):
    return {
        "nll": nll,
        "nll_sum": nll * tokens,
        "tokens": tokens,
        "perplexity": math.exp(nll),
        "bits_per_byte": nll / math.log(2),
    }


def test_pooling_uses_targets_not_document_means():
    docs = [
        {"metrics": {"model": metric(1, 2), "bigram": metric(2, 2)}},
        {"metrics": {"model": metric(3, 6), "bigram": metric(4, 6)}},
    ]
    result = study.pool_documents(docs)
    assert result["model"]["nll"] == pytest.approx(2.5)
    assert result["bigram"]["nll"] == pytest.approx(3.5)
    assert result["model"]["tokens"] == 8
    assert result["model"]["perplexity"] == pytest.approx(math.exp(2.5))


def test_aggregate_preserves_pairing_and_arithmetic_perplexity_mean():
    runs = [
        {"seed": seed, "arm": arm, "pooled": {"model": metric(nll, 7)}}
        for seed, arm, nll in [
            (32, "no_position", 2.5),
            (31, "sinusoidal", 1),
            (31, "no_position", 2),
            (32, "sinusoidal", 3),
        ]
    ]
    result = study.aggregate_runs(runs, [31, 32])
    assert result["paired_nll_summary"]["mean"] == pytest.approx(0.25)
    assert result["paired_nll_summary"]["sample_sd"] == pytest.approx(1.5 / math.sqrt(2))
    assert result["arms"]["sinusoidal"]["perplexity"]["mean"] == pytest.approx(
        (math.exp(1) + math.exp(3)) / 2
    )
    with pytest.raises(ValueError):
        study.aggregate_runs(runs[:-1] + [runs[0]], [31, 32])


def test_equal_validation_scores_keep_earliest_checkpoint(
    tmp_path, monkeypatch, tiny_config, synthetic_documents
):
    monkeypatch.setattr(study, "evaluate", lambda *args: {"model": metric(5, 7)})
    row = study.train_candidate(
        synthetic_documents, synthetic_documents, tiny_config, 31, "sinusoidal", tmp_path
    )
    assert row["selected_step"] == 1
    assert [point["step"] for point in row["curve"]] == [1, 2]
    saved = torch.load(tmp_path / row["checkpoint"], weights_only=True)
    assert saved["selected_step"] == 1
    assert study.state_digest(saved["model"]) == row["selected_state_sha256"]


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf")])
def test_nonfinite_validation_aborts_before_checkpoint(
    tmp_path, monkeypatch, tiny_config, synthetic_documents, nonfinite
):
    monkeypatch.setattr(study, "evaluate", lambda *args: {"model": metric(nonfinite, 7)})
    with pytest.raises((RuntimeError, ValueError), match="finite"):
        study.train_candidate(
            synthetic_documents, synthetic_documents, tiny_config, 31, "sinusoidal", tmp_path
        )
    assert not list(tmp_path.glob("*.pt"))


def test_nonfinite_validation_after_good_checkpoint_still_aborts(
    tmp_path, monkeypatch, tiny_config, synthetic_documents
):
    scores = iter([metric(5, 7), metric(float("nan"), 7)])
    monkeypatch.setattr(study, "evaluate", lambda *args: {"model": next(scores)})
    with pytest.raises((RuntimeError, ValueError), match="finite"):
        study.train_candidate(
            synthetic_documents, synthetic_documents, tiny_config, 31, "sinusoidal", tmp_path
        )
    assert not list(tmp_path.glob("*.pt"))


def test_complete_synthetic_study_and_preflight_all_candidates(
    tmp_path, monkeypatch, tiny_config, synthetic_documents
):
    output = tmp_path / "study"
    provenance = {"synthetic_fixture": "review-v1"}
    digest = study.train_study(
        synthetic_documents, synthetic_documents, tiny_config, output, provenance
    )
    import json

    selected = json.loads((output / "selection.json").read_text())
    for seed in tiny_config["seeds"]:
        pair = [row for row in selected["runs"] if row["seed"] == seed]
        for field in ("initial_state_sha256", "sample_schedule_sha256", "parameter_count"):
            assert pair[0][field] == pair[1][field]
    heldout = [torch.tensor([11, 12, 13, 14]), torch.tensor([15, 16, 17, 18, 19, 20])]
    original_evaluate = study.evaluate
    calls = []

    def record(*args):
        calls.append(True)
        return original_evaluate(*args)

    monkeypatch.setattr(study, "evaluate", record)
    last_path = output / selected["runs"][-1]["checkpoint"]
    original_bytes = last_path.read_bytes()
    last_path.write_bytes(original_bytes + b"tampered")
    with pytest.raises(ValueError, match="digest"):
        study.score_study(
            synthetic_documents, heldout, ["a", "b"], tiny_config, output, provenance, digest
        )
    assert calls == [], "No candidate may be scored before the final candidate passes"
    last_path.write_bytes(original_bytes)
    changed_config = copy.deepcopy(tiny_config)
    changed_config["context"] += 1
    with pytest.raises(ValueError, match="provenance"):
        study.score_study(
            synthetic_documents, heldout, ["a", "b"], changed_config, output, provenance, digest
        )
    assert calls == []
    result = study.score_study(
        synthetic_documents, heldout, ["a", "b"], tiny_config, output, provenance, digest
    )
    assert len(calls) == 8
    assert len(result["runs"]) == 4
    for row in result["runs"]:
        assert row["pooled"]["model"]["tokens"] == 8
        assert row["pooled"]["bigram"]["tokens"] == 8
        total = sum(d["metrics"]["model"]["nll_sum"] for d in row["documents"])
        assert row["pooled"]["model"]["nll"] == pytest.approx(total / 8)
    assert len({row["pooled"]["bigram"]["nll"] for row in result["runs"]}) == 1
    with pytest.raises(ValueError, match="already exists"):
        study.score_study(
            synthetic_documents, heldout, ["a", "b"], tiny_config, output, provenance, digest
        )
