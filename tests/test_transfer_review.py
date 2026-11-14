"""Independent synthetic-only transfer identity and target-coverage checks."""

import hashlib
import json

import pytest
import torch

from transformer_core import transfer
from transformer_core.language_model import ByteLanguageModel
from transformer_core.stability import file_digest, state_digest, write_json


@pytest.fixture
def synthetic_locked_models(tmp_path):
    config = {
        "seeds": [41, 42, 43],
        "arms": {"sinusoidal": True, "no_position": False},
        "model": {"width": 4, "heads": 1, "hidden": 7, "layers": 1},
        "context": 3,
        "bigram_alpha": 0.1,
    }
    rows = []
    for seed in config["seeds"]:
        for arm, positions in config["arms"].items():
            torch.manual_seed(seed)
            model = ByteLanguageModel(**config["model"], use_positions=positions)
            path = tmp_path / f"{seed}-{arm}.pt"
            torch.save(
                {"config": model.config, "selected_step": 1, "model": model.state_dict()}, path
            )
            rows.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "selected_step": 1,
                    "checkpoint": path.name,
                    "checkpoint_sha256": file_digest(path),
                    "selected_state_sha256": state_digest(model.state_dict()),
                }
            )
    selection = {"config": config, "runs": rows, "provenance": {"sources": {}}}
    path = tmp_path / "synthetic-selection.json"
    write_json(path, selection)
    return path, file_digest(path), selection


def invoke_score(tmp_path, fixture):
    path, digest, _ = fixture
    train = [torch.tensor([1, 2, 1, 2, 3], dtype=torch.long)]
    external = [torch.tensor([91, 92, 93]), torch.tensor([94, 95, 96, 97, 98])]
    return transfer.score_transfer(
        path,
        digest,
        tmp_path,
        tmp_path,
        train,
        external,
        ["one", "two"],
        tmp_path / "synthetic-report.json",
        {"synthetic": True},
    )


@pytest.mark.parametrize("damage", ["tensor", "dtype", "step", "model_config"])
def test_final_checkpoint_damage_prevents_all_external_forwards(
    tmp_path, synthetic_locked_models, monkeypatch, damage
):
    _, _, selection = synthetic_locked_models
    path = tmp_path / selection["runs"][-1]["checkpoint"]
    saved = torch.load(path, weights_only=True)
    key = next(iter(saved["model"]))
    if damage == "tensor":
        saved["model"][key].view(-1)[0] += 1
    elif damage == "dtype":
        saved["model"][key] = saved["model"][key].double()
    elif damage == "step":
        saved["selected_step"] += 1
    else:
        saved["config"]["use_positions"] = True
    torch.save(saved, path)
    calls = []
    monkeypatch.setattr(transfer, "evaluate", lambda *args: calls.append(args))
    with pytest.raises(ValueError):
        invoke_score(tmp_path, synthetic_locked_models)
    assert not calls
    assert not (tmp_path / "synthetic-report.json").exists()


def test_reserialized_artifact_is_separately_reported_not_rejected(
    tmp_path, synthetic_locked_models
):
    selection_path, digest, selection = synthetic_locked_models
    path = tmp_path / selection["runs"][0]["checkpoint"]
    saved = torch.load(path, weights_only=True)
    torch.save(saved, path, _use_new_zipfile_serialization=False)
    assert file_digest(path) != selection["runs"][0]["checkpoint_sha256"]
    _, models, audit = transfer.checked_models(selection_path, digest, tmp_path, tmp_path)
    assert len(models) == 6
    assert audit[0]["tensor_identical"] is True
    assert audit[0]["artifact_identical"] is False
    assert audit[0]["observed_tensor_sha256"] == audit[0]["original_tensor_sha256"]


def test_all_models_and_bigram_receive_same_targets_and_train_only_baseline(
    tmp_path, synthetic_locked_models, monkeypatch
):
    original_fit = transfer.fit_bigram
    original_evaluate = transfer.evaluate
    fitted = []
    evaluations = []

    def fit(documents, alpha):
        fitted.append([x.tolist() for x in documents])
        assert alpha == 0.1
        return original_fit(documents, alpha)

    def evaluate(model, documents, context, baseline):
        evaluations.append(([x.tolist() for x in documents], context, id(baseline)))
        return original_evaluate(model, documents, context, baseline)

    monkeypatch.setattr(transfer, "fit_bigram", fit)
    monkeypatch.setattr(transfer, "evaluate", evaluate)
    report = invoke_score(tmp_path, synthetic_locked_models)
    assert fitted == [[[1, 2, 1, 2, 3]]]
    assert len(evaluations) == 12
    assert len({x[2] for x in evaluations}) == 1
    assert all(x[1] == 3 for x in evaluations)
    assert report["evaluation"]["total_targets"] == 6
    for run in report["runs"]:
        for name in ("model", "bigram"):
            assert [d["metrics"][name]["tokens"] for d in run["documents"]] == [2, 4]
            assert run["pooled"][name]["tokens"] == 6
    assert len({r["pooled"]["bigram"]["nll"] for r in report["runs"]}) == 1
    before = (tmp_path / "synthetic-report.json").read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        invoke_score(tmp_path, synthetic_locked_models)
    assert (tmp_path / "synthetic-report.json").read_bytes() == before


def test_original_loader_does_not_open_validation_or_test(tmp_path):
    raw = b"synthetic training bytes"
    (tmp_path / "train.txt").write_bytes(raw)
    manifest = {
        "documents": [
            {
                "split": "train",
                "upstream_path": "train.txt",
                "size_bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            },
            {"split": "validation", "upstream_path": "does-not-exist-validation"},
            {"split": "test", "upstream_path": "does-not-exist-test"},
        ]
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    docs = transfer.load_original_train(path, file_digest(path), tmp_path)
    assert len(docs) == 1 and docs[0].tolist() == list(raw)
