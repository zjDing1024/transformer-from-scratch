"""Score the locked original models on a separately frozen, external corpus."""

import argparse
import json
from pathlib import Path

import torch

from transformer_core import transfer as transfer_module
from transformer_core.stability import file_digest
from transformer_core.transfer import load_external, load_original_train, score_transfer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--train-data-dir", type=Path, required=True)
    parser.add_argument("--external-data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-origin", choices=["retained", "rebuilt"], required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if file_digest(transfer_module.__file__) != file_digest(
        root / "src/transformer_core/transfer.py"
    ):
        raise ValueError("imported transfer scorer differs from recorded source")
    config_path = root / "experiments/sqlite-transfer-v1.json"
    config = json.loads(config_path.read_text())
    if torch.__version__ != config["torch_version"]:
        parser.error(f"frozen transfer requires torch {config['torch_version']}")
    selection_path = root / config["selection"]
    if file_digest(selection_path) != config["selection_sha256"]:
        raise ValueError("original selection lock changed")
    selection = json.loads(selection_path.read_text())
    if (
        selection["config"]["seeds"] != config["seeds"]
        or list(selection["config"]["arms"]) != config["arms"]
        or selection["config"]["context"] != config["context"]
        or selection["config"]["bigram_alpha"] != config["bigram_alpha"]
    ):
        raise ValueError("transfer recipe disagrees with original frozen model protocol")
    manifest = root / config["external_manifest"]
    train_manifest = root / config["training_manifest"]
    train = load_original_train(
        train_manifest, selection["provenance"]["manifest_sha256"], args.train_data_dir
    )
    ids, docs = load_external(
        manifest, config["external_manifest_sha256"], args.external_data_dir
    )
    original_hashes = {
        d["sha256"]
        for d in json.loads(train_manifest.read_text())["documents"]
        if d["split"] == "train"
    }
    external_hashes = {d["sha256"] for d in json.loads(manifest.read_text())["documents"]}
    if original_hashes & external_hashes:
        raise ValueError("external document duplicates an original whole training file")
    sources = [
        root / "src/transformer_core/transfer.py",
        Path(__file__).resolve(),
        root / "examples/download_transfer.py",
    ]
    provenance = dict(
        config_sha256=file_digest(config_path),
        protocol_sha256=file_digest(root / config["protocol"]),
        external_manifest_sha256=file_digest(manifest),
        train_manifest_sha256=file_digest(train_manifest),
        sources={str(p.relative_to(root)): file_digest(p) for p in sources},
        checkpoint_origin=args.checkpoint_origin,
        checkpoint_origin_note="Identity checks do not prove uninterrupted artifact retention.",
        format_shift=config["format_shift"],
        new_training_trials=0,
    )
    result = score_transfer(
        selection_path,
        config["selection_sha256"],
        args.checkpoint_dir,
        root,
        train,
        docs,
        ids,
        args.output,
        provenance,
    )
    print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    main()
