"""Prepare paired source metadata first; then separately score frozen checkpoints."""

import argparse
import json
from pathlib import Path

import torch

from transformer_core import format_control, transfer
from transformer_core.stability import file_digest, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["prepare", "score"])
    parser.add_argument("--external-data-dir", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--pairs-sha256")
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--checkpoint-origin", choices=["retained", "rebuilt"])
    parser.add_argument("--train-data-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    recipe = root / "experiments/sqlite-format-v1.json"
    config = json.loads(recipe.read_text())
    for module in (format_control, transfer):
        expected = root / "src/transformer_core" / Path(module.__file__).name
        if file_digest(module.__file__) != file_digest(expected):
            raise ValueError("imported control/checker differs from recorded source")
    if torch.__version__ != config["torch_version"]:
        parser.error(f"requires {config['torch_version']}")
    sources = [
        root / "src/transformer_core/format_control.py",
        Path(__file__).resolve(),
        root / "src/transformer_core/transfer.py",
    ]
    provenance = dict(
        config_sha256=file_digest(recipe),
        protocol_sha256=file_digest(root / config["protocol"]),
        sources={str(p.relative_to(root)): file_digest(p) for p in sources},
        original_selection_sha256=config["original_selection_sha256"],
        torch=torch.__version__,
        device="cpu",
        threads=1,
    )
    manifest = root / config["corpus_manifest"]
    if args.phase == "prepare":
        if args.pairs.exists():
            raise ValueError("pair manifest already exists")
        result = format_control.prepare_pairs(
            manifest,
            config["corpus_manifest_sha256"],
            args.external_data_dir,
            config,
            provenance,
        )
        args.pairs.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.pairs, result)
        print("pairs_sha256=" + file_digest(args.pairs))
        print(json.dumps({d["id"]: d["counts"] for d in result["documents"]}, indent=2))
        return
    if not all(
        (
            args.pairs_sha256,
            args.checkpoint_dir,
            args.checkpoint_origin,
            args.train_data_dir,
            args.output,
        )
    ):
        parser.error(
            "score requires pairs-sha256, checkpoint-dir, checkpoint-origin, "
            "train-data-dir and output"
        )
    if file_digest(args.pairs) != args.pairs_sha256:
        raise ValueError("pair manifest digest changed")
    selection_path = root / config["original_selection"]
    if file_digest(selection_path) != config["original_selection_sha256"]:
        raise ValueError("original selection changed")
    selection = json.loads(selection_path.read_text())
    train = transfer.load_original_train(
        root / config["train_manifest"],
        selection["provenance"]["manifest_sha256"],
        args.train_data_dir,
    )
    result = format_control.score_format(
        config,
        json.loads(args.pairs.read_text()),
        selection_path,
        args.checkpoint_dir,
        root,
        train,
        manifest,
        args.external_data_dir,
        args.output,
        provenance,
        checkpoint_origin=args.checkpoint_origin,
    )
    print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    main()
