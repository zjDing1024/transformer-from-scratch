"""Read-only transfer scoring of existing, identity-checked float32 checkpoints."""

import hashlib
import importlib
import json
import platform
import struct
from pathlib import Path, PurePosixPath
from urllib.request import urlopen

import torch

from .language_model import ByteLanguageModel, evaluate, fit_bigram
from .stability import (
    aggregate_runs,
    configure_cpu,
    file_digest,
    pool_documents,
    state_digest,
    write_json,
)

SQLITE_RAW = "https://sqlite.org/docsrc/raw/"


def relative_path(value):
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or not path.parts:
        raise ValueError("unsafe manifest path")
    return Path(*path.parts)


def verified_bytes(root, item):
    data = (Path(root) / relative_path(item["upstream_path"])).read_bytes()
    if len(data) != item["size_bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
        raise ValueError("document size/SHA256 mismatch")
    if "artifact_hash" in item:
        algorithm = item["artifact_algorithm"]
        if algorithm not in ("sha1", "sha3_256"):
            raise ValueError("unsupported Fossil artifact algorithm")
        if hashlib.new(algorithm, data).hexdigest() != item["artifact_hash"]:
            raise ValueError("Fossil document artifact mismatch")
    return data


def load_external(manifest_path, expected_sha, data_dir):
    if file_digest(manifest_path) != expected_sha:
        raise ValueError("external manifest digest mismatch")
    manifest = json.loads(Path(manifest_path).read_text())
    ids = [d["id"] for d in manifest["documents"]]
    paths = [d["upstream_path"] for d in manifest["documents"]]
    if not ids or len(set(ids)) != len(ids) or len(set(paths)) != len(paths):
        raise ValueError("external documents must be nonempty and distinct")
    raw = [verified_bytes(data_dir, d) for d in manifest["documents"]]
    if any(len(data) < 2 for data in raw):
        raise ValueError("each external document needs a target")
    return ids, [torch.tensor(list(data), dtype=torch.long) for data in raw]


def load_original_train(manifest_path, expected_sha, data_dir):
    """Intentionally never read original validation/test bodies for transfer scoring."""
    if file_digest(manifest_path) != expected_sha:
        raise ValueError("original train manifest digest mismatch")
    manifest = json.loads(Path(manifest_path).read_text())
    items = [d for d in manifest["documents"] if d["split"] == "train"]
    if not items:
        raise ValueError("no original training documents")
    return [torch.tensor(list(verified_bytes(data_dir, d)), dtype=torch.long) for d in items]


def verify_original_sources(root, selection):
    for path, digest in selection["provenance"]["sources"].items():
        if file_digest(Path(root) / relative_path(path)) != digest:
            raise ValueError("original model/training source changed")
        if path.startswith("src/transformer_core/") and path.endswith(".py"):
            stem = Path(path).stem
            name = "transformer_core" + ("" if stem == "__init__" else "." + stem)
            if file_digest(importlib.import_module(name).__file__) != digest:
                raise ValueError("imported model code differs from frozen source")


def checked_models(selection_path, selection_sha, checkpoint_dir, source_root):
    if file_digest(selection_path) != selection_sha:
        raise ValueError("original selection digest mismatch")
    selection = json.loads(Path(selection_path).read_text())
    verify_original_sources(source_root, selection)
    config = selection["config"]
    expected = {(s, a) for s in config["seeds"] for a in config["arms"]}
    keys = [(r["seed"], r["arm"]) for r in selection["runs"]]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("incomplete or duplicate original checkpoint roster")
    models, audit = [], []
    for row in selection["runs"]:
        name = f"{row['seed']}-{row['arm']}.pt"
        if row["checkpoint"] != name:
            raise ValueError("checkpoint filename mismatch")
        path = Path(checkpoint_dir) / name
        digest = file_digest(path)
        saved = torch.load(path, weights_only=True, map_location="cpu")
        expected_config = dict(config["model"], use_positions=config["arms"][row["arm"]])
        if saved["config"] != expected_config or saved["selected_step"] != row["selected_step"]:
            raise ValueError("checkpoint model/step metadata mismatch")
        if any(t.dtype != torch.float32 for t in saved["model"].values()):
            raise ValueError("legacy identity scheme requires original float32 tensors")
        tensor_digest = state_digest(saved["model"])
        if tensor_digest != row["selected_state_sha256"]:
            raise ValueError("original checkpoint tensor identity mismatch; abort transfer")
        model = ByteLanguageModel(**expected_config)
        model.load_state_dict(saved["model"], strict=True)
        models.append(model)
        audit.append(
            dict(
                seed=row["seed"],
                arm=row["arm"],
                selected_step=row["selected_step"],
                original_artifact_sha256=row["checkpoint_sha256"],
                observed_artifact_sha256=digest,
                artifact_identical=digest == row["checkpoint_sha256"],
                original_tensor_sha256=row["selected_state_sha256"],
                observed_tensor_sha256=tensor_digest,
                tensor_identical=True,
            )
        )
    return selection, models, audit


def bigram_digest(log_probabilities):
    values = log_probabilities.flatten().tolist()
    return hashlib.sha256(struct.pack(f"<{len(values)}d", *values)).hexdigest()


def score_transfer(
    selection_path,
    selection_sha,
    checkpoint_dir,
    source_root,
    train,
    external,
    ids,
    output_path,
    provenance,
):
    output_path = Path(output_path)
    if output_path.exists():
        raise ValueError("external report already exists; refusing overwrite")
    if not external or len(external) != len(ids) or len(set(ids)) != len(ids):
        raise ValueError("external IDs/documents must be aligned and distinct")
    configure_cpu()
    selection, models, audit = checked_models(
        selection_path, selection_sha, checkpoint_dir, source_root
    )
    config = selection["config"]
    baseline = fit_bigram(train, alpha=config["bigram_alpha"])
    runs = []
    # No external forward occurs until ALL six identities and metadata pass.
    for row, model in zip(selection["runs"], models):
        documents = [
            dict(id=name, metrics=evaluate(model, [doc], config["context"], baseline))
            for name, doc in zip(ids, external)
        ]
        runs.append(
            dict(
                seed=row["seed"],
                arm=row["arm"],
                selected_step=row["selected_step"],
                documents=documents,
                pooled=pool_documents(documents),
            )
        )
    if file_digest(selection_path) != selection_sha:
        raise ValueError("original selection changed during external scoring")
    result = dict(
        schema_version=1,
        original_selection_sha256=selection_sha,
        checkpoint_identity=audit,
        provenance=provenance,
        evaluation=dict(
            context=config["context"],
            total_targets=sum(len(d) - 1 for d in external),
            baseline="original Flask train-only add-0.1 byte bigram",
            bigram_log_probability_float64_sha256=bigram_digest(baseline),
        ),
        environment=dict(
            python=platform.python_version(), torch=torch.__version__, device="cpu", threads=1
        ),
        runs=runs,
        aggregate=aggregate_runs(runs, config["seeds"]),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, result)
    return result


def download_corpus(manifest_path, output_dir):
    """Download only pinned canonical artifacts; documentation is never executed."""
    manifest = json.loads(Path(manifest_path).read_text())
    checkin = manifest["upstream_commit"]
    with urlopen(SQLITE_RAW + checkin, timeout=30) as response:
        raw_manifest = response.read()
    if (
        hashlib.sha3_256(raw_manifest).hexdigest() != checkin
        or hashlib.sha256(raw_manifest).hexdigest() != manifest["upstream_manifest_sha256"]
    ):
        raise ValueError("canonical Fossil manifest digest mismatch")
    files = {
        line.split()[1]: line.split()[2]
        for line in raw_manifest.decode().splitlines()
        if line.startswith("F ")
    }
    for item in [*manifest["documents"], manifest["copyright_document"]]:
        if files.get(item["upstream_path"]) != item["artifact_hash"]:
            raise ValueError("document artifact not in pinned Fossil check-in")
        if item["source_url"] != SQLITE_RAW + item["artifact_hash"]:
            raise ValueError("noncanonical document URL")
        target = Path(output_dir) / relative_path(item["upstream_path"])
        with urlopen(item["source_url"], timeout=30) as response:
            data = response.read()
        algorithm = item["artifact_algorithm"]
        if algorithm not in ("sha1", "sha3_256"):
            raise ValueError("unsupported Fossil artifact algorithm")
        if (
            len(data) != item["size_bytes"]
            or hashlib.sha256(data).hexdigest() != item["sha256"]
            or hashlib.new(algorithm, data).hexdigest() != item["artifact_hash"]
        ):
            raise ValueError("downloaded document digest mismatch")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
