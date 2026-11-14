"""Offline tests: external scoring is separated from checkpoint selection."""

import copy
import hashlib
import io
import json

import pytest
import torch

from transformer_core import transfer
from transformer_core.language_model import ByteLanguageModel
from transformer_core.stability import file_digest, state_digest, write_json


@pytest.fixture
def locked(tmp_path):
    config = dict(
        seeds=[31, 32],
        arms=dict(sinusoidal=True, no_position=False),
        model=dict(width=4, heads=1, hidden=8, layers=1),
        context=3,
        bigram_alpha=0.1,
    )
    rows = []
    for seed in config["seeds"]:
        for arm, position in config["arms"].items():
            torch.manual_seed(seed)
            model = ByteLanguageModel(**config["model"], use_positions=position)
            name = f"{seed}-{arm}.pt"
            torch.save(
                dict(model=model.state_dict(), config=model.config, selected_step=2),
                tmp_path / name,
            )
            rows.append(
                dict(
                    seed=seed,
                    arm=arm,
                    selected_step=2,
                    checkpoint=name,
                    checkpoint_sha256=file_digest(tmp_path / name),
                    selected_state_sha256=state_digest(model.state_dict()),
                )
            )
    lock = tmp_path / "selection.json"
    write_json(lock, dict(config=config, runs=rows, provenance=dict(sources={})))
    return lock, file_digest(lock), tmp_path


def docs():
    return [torch.tensor(list(b"a<a>\n<tcl>never execute</tcl>")), torch.tensor(list(b"bcd"))]


def score(locked, output="external.json"):
    lock, sha, root = locked
    return transfer.score_transfer(
        lock,
        sha,
        root,
        root,
        [torch.tensor(list(b"abcabcabc"))],
        docs(),
        ["a", "b"],
        root / output,
        dict(fixture="synthetic-only"),
    )


def test_same_bytes_scoring_and_lock_preserved(locked):
    lock, sha, root = locked
    result = score(locked)
    assert all(
        r["artifact_identical"] and r["tensor_identical"] for r in result["checkpoint_identity"]
    )
    assert file_digest(lock) == sha
    targets = sum(len(d) - 1 for d in docs())
    for row in result["runs"]:
        assert row["pooled"]["model"]["tokens"] == targets
        assert row["pooled"]["bigram"]["tokens"] == targets
    assert len({r["pooled"]["bigram"]["nll"] for r in result["runs"]}) == 1
    with pytest.raises(ValueError, match="already exists"):
        score(locked)


def test_reencoded_identical_tensors_report_different_artifact(locked):
    lock, _, root = locked
    row = json.loads(lock.read_text())["runs"][0]
    path = root / row["checkpoint"]
    checkpoint = torch.load(path, weights_only=True)
    torch.save(checkpoint, path, _use_new_zipfile_serialization=False)
    result = score(locked)
    assert result["checkpoint_identity"][0]["artifact_identical"] is False
    assert result["checkpoint_identity"][0]["tensor_identical"] is True


def test_last_model_tensor_mismatch_stops_all_external_forwards(locked, monkeypatch):
    lock, _, root = locked
    row = json.loads(lock.read_text())["runs"][-1]
    path = root / row["checkpoint"]
    checkpoint = torch.load(path, weights_only=True)
    checkpoint["model"]["output.bias"][0] += 1
    torch.save(checkpoint, path)

    def forbidden(*args, **kwargs):
        pytest.fail("an external forward happened before every identity check passed")

    monkeypatch.setattr(transfer, "evaluate", forbidden)
    with pytest.raises(ValueError, match="tensor identity"):
        score(locked)
    assert not (root / "external.json").exists()


def test_original_lock_and_sources_are_checked(locked):
    lock, sha, root = locked
    with pytest.raises(ValueError, match="selection digest"):
        transfer.checked_models(lock, "0" * 64, root, root)
    item = json.loads(lock.read_text())
    (root / "source.txt").write_text("original")
    item["provenance"]["sources"]["source.txt"] = file_digest(root / "source.txt")
    write_json(lock, item)
    (root / "source.txt").write_text("changed")
    with pytest.raises(ValueError, match="source changed"):
        transfer.checked_models(lock, file_digest(lock), root, root)
    assert file_digest(lock) != sha


def test_train_only_loader_never_opens_validation_or_external(tmp_path):
    (tmp_path / "train").write_bytes(b"abcabc")
    items = [
        dict(
            upstream_path="train",
            split="train",
            size_bytes=6,
            sha256=hashlib.sha256(b"abcabc").hexdigest(),
        ),
        dict(upstream_path="not-present", split="test"),
    ]
    manifest = tmp_path / "manifest.json"
    write_json(manifest, dict(documents=items))
    train = transfer.load_original_train(manifest, file_digest(manifest), tmp_path)
    assert train[0].tolist() == list(b"abcabc")
    with pytest.raises(ValueError, match="manifest digest"):
        transfer.load_original_train(manifest, "0" * 64, tmp_path)


def test_raw_external_bytes_and_hash_verification(tmp_path):
    raw = b"<title>raw</title>\n<tcl>not executed</tcl>\r\n"
    (tmp_path / "page.in").write_bytes(raw)
    item = dict(
        id="one",
        upstream_path="page.in",
        size_bytes=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
        artifact_algorithm="sha3_256",
        artifact_hash=hashlib.sha3_256(raw).hexdigest(),
    )
    manifest = tmp_path / "manifest.json"
    write_json(manifest, dict(documents=[item]))
    ids, result = transfer.load_external(manifest, file_digest(manifest), tmp_path)
    assert ids == ["one"] and bytes(result[0].tolist()) == raw
    (tmp_path / "page.in").write_bytes(raw.replace(b"raw", b"bad"))
    with pytest.raises(ValueError, match="SHA256"):
        transfer.load_external(manifest, file_digest(manifest), tmp_path)
    with pytest.raises(ValueError, match="unsafe"):
        transfer.relative_path("../escape")


def test_downloader_checks_fossil_membership_and_keeps_inert_bytes(tmp_path, monkeypatch):
    raw = b"<tcl>do not execute</tcl>\n"
    artifact = hashlib.sha3_256(raw).hexdigest()
    commit = f"F pages/doc.in {artifact}\nF pages/copyright.in {artifact}\n".encode()
    checkin = hashlib.sha3_256(commit).hexdigest()
    item = dict(
        id="one",
        upstream_path="pages/doc.in",
        size_bytes=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
        artifact_algorithm="sha3_256",
        artifact_hash=artifact,
        source_url=transfer.SQLITE_RAW + artifact,
    )
    copyright_item = dict(item, upstream_path="pages/copyright.in")
    manifest_data = dict(
        upstream_commit=checkin,
        upstream_manifest_sha256=hashlib.sha256(commit).hexdigest(),
        documents=[item],
        copyright_document=copyright_item,
    )
    manifest = tmp_path / "manifest.json"
    write_json(manifest, manifest_data)
    seen = []

    def fetch(url, timeout):
        seen.append(url)
        return io.BytesIO(
            {transfer.SQLITE_RAW + checkin: commit, transfer.SQLITE_RAW + artifact: raw}[url]
        )

    monkeypatch.setattr(transfer, "urlopen", fetch)
    transfer.download_corpus(manifest, tmp_path / "corpus")
    assert (tmp_path / "corpus/pages/doc.in").read_bytes() == raw
    assert (tmp_path / "corpus/pages/copyright.in").read_bytes() == raw
    assert len(seen) == 3
    corrupted = copy.deepcopy(manifest_data)
    corrupted["documents"][0]["artifact_hash"] = "0" * 64
    write_json(manifest, corrupted)
    with pytest.raises(ValueError, match="not in pinned"):
        transfer.download_corpus(manifest, tmp_path / "another")
    assert not (tmp_path / "another").exists()


def test_bigram_digest_is_sensitive_to_full_float64_values():
    a = torch.zeros((256, 256), dtype=torch.float64)
    b = a.clone()
    b[0, 0] = 1e-100
    assert transfer.bigram_digest(a) != transfer.bigram_digest(b)
