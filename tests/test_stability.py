import copy
import json
import math

import pytest
import torch

from transformer_core.language_model import ByteLanguageModel
from transformer_core.stability import (
    aggregate_runs,
    file_digest,
    pool_documents,
    score_study,
    state_digest,
    train_study,
    validate_config,
)


@pytest.fixture
def mini_study(tmp_path):
    config = dict(
        seeds=[11, 12],
        arms=dict(sinusoidal=True, no_position=False),
        model=dict(width=4, heads=1, hidden=8, layers=1),
        steps=2,
        validate_every=1,
        context=3,
        batch=2,
        learning_rate=0.003,
        weight_decay=0.01,
        clip_norm=1.0,
        bigram_alpha=0.1,
    )
    train = [torch.tensor(list(b"abcabcabc"))]
    validation = [torch.tensor(list(b"defghi"))]
    heldout = [torch.tensor(list(b"xyzxyz")), torch.tensor(list(b"xyz"))]
    provenance = dict(fixture="synthetic only")
    out = tmp_path / "study"
    digest = train_study(train, validation, config, out, provenance)
    return config, train, heldout, provenance, out, digest


@pytest.mark.parametrize("use_positions", [True, False])
def test_position_arm_remains_causal(use_positions):
    torch.manual_seed(55)
    model = ByteLanguageModel(
        width=8, heads=2, hidden=16, layers=1, use_positions=use_positions
    )
    x = torch.tensor([[1, 2, 3, 4]])
    altered = x.clone()
    altered[:, 2:] = 7
    torch.testing.assert_close(model(x)[:, :2], model(altered)[:, :2], atol=0, rtol=0)


def test_position_ablation_has_identical_parameters_and_different_forward():
    torch.manual_seed(8)
    a = ByteLanguageModel(width=8, heads=2, hidden=16, layers=1)
    torch.manual_seed(8)
    b = ByteLanguageModel(width=8, heads=2, hidden=16, layers=1, use_positions=False)
    assert state_digest(a.state_dict()) == state_digest(b.state_dict())
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())
    x = torch.tensor([[1, 2, 3]])
    assert not torch.equal(a(x), b(x))
    with pytest.raises(ValueError, match="bool"):
        ByteLanguageModel(use_positions="False")


def test_complete_study_controls_and_score(mini_study):
    config, train, heldout, provenance, out, digest = mini_study
    selected = json.loads((out / "selection.json").read_text())
    assert len(selected["runs"]) == 4
    assert not (out / "report.json").exists()
    for seed in config["seeds"]:
        a, b = [r for r in selected["runs"] if r["seed"] == seed]
        assert a["initial_state_sha256"] == b["initial_state_sha256"]
        assert a["sample_schedule_sha256"] == b["sample_schedule_sha256"]
        for row in (a, b):
            best = min(row["curve"], key=lambda p: p["validation"]["nll"])
            assert row["selected_step"] == best["step"]
    report = score_study(train, heldout, ["one", "two"], config, out, provenance, digest)
    assert len(report["runs"]) == 4
    assert report["runs"][0]["pooled"]["model"]["tokens"] == 7
    assert report["runs"][0]["pooled"]["bigram"]["tokens"] == 7
    assert report["aggregate"]["paired_nll_summary"]["n"] == 2
    with pytest.raises(ValueError, match="already exists"):
        score_study(train, heldout, ["one", "two"], config, out, provenance, digest)
    with pytest.raises(ValueError, match="empty"):
        train_study(train, heldout, config, out, provenance)


def test_tampered_final_checkpoint_rejected_before_any_scoring(mini_study, monkeypatch):
    config, train, heldout, provenance, out, digest = mini_study
    from transformer_core import stability

    def forbidden(*args, **kwargs):
        pytest.fail("held-out evaluation happened before all checkpoints were checked")

    monkeypatch.setattr(stability, "evaluate", forbidden)
    last = list(config["arms"])[-1]
    (out / f"{config['seeds'][-1]}-{last}.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checkpoint digest"):
        score_study(train, heldout, ["one", "two"], config, out, provenance, digest)


def test_selection_and_source_integrity(mini_study):
    config, train, heldout, provenance, out, digest = mini_study
    with pytest.raises(ValueError, match="selection digest"):
        score_study(train, heldout, ["one", "two"], config, out, provenance, "0" * 64)
    with pytest.raises(ValueError, match="provenance changed"):
        score_study(train, heldout, ["one", "two"], config, out, {}, digest)
    lock = out / "selection.json"
    selection = json.loads(lock.read_text())
    selection["runs"][-1] = selection["runs"][0]
    lock.write_text(json.dumps(selection))
    with pytest.raises(ValueError, match="roster"):
        score_study(train, heldout, ["one", "two"], config, out, provenance, file_digest(lock))


def test_weighted_pooling_and_paired_statistics():
    docs = [
        dict(metrics={k: dict(tokens=n, nll_sum=total) for k in ("model", "bigram")})
        for n, total in [(2, 2.0), (8, 24.0)]
    ]
    pooled = pool_documents(docs)
    assert pooled["model"]["nll"] == pytest.approx(2.6)
    assert pooled["model"]["perplexity"] == pytest.approx(math.exp(2.6))
    runs = []
    for seed, a, b in [(1, 1.0, 2.0), (2, 2.0, 4.0)]:
        for arm, nll in [("sinusoidal", a), ("no_position", b)]:
            runs.append(
                dict(
                    seed=seed,
                    arm=arm,
                    pooled=dict(
                        model=dict(
                            nll=nll, perplexity=math.exp(nll), bits_per_byte=nll / math.log(2)
                        )
                    ),
                )
            )
    stats = aggregate_runs(runs, [1, 2])
    assert stats["paired_nll_summary"]["mean"] == pytest.approx(1.5)
    assert stats["paired_nll_summary"]["sample_sd"] == pytest.approx(math.sqrt(0.5))
    assert stats["arms"]["sinusoidal"]["perplexity"]["mean"] == pytest.approx(
        (math.e + math.exp(2)) / 2
    )
    with pytest.raises(ValueError, match="seed-arm"):
        aggregate_runs(runs[:-1], [1, 2])


def test_config_rejects_missing_final_validation(mini_study):
    config = copy.deepcopy(mini_study[0])
    config["steps"], config["validate_every"] = 3, 2
    with pytest.raises(ValueError, match="final step"):
        validate_config(config)
