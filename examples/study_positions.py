"""Train all candidates, then score hash-locked selections in a separate process."""

import argparse
import json
from pathlib import Path

import torch

from transformer_core.language_model import load_documents
from transformer_core.stability import file_digest, score_study, train_study


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["train", "score"])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("experiments/position-stability-v1.json")
    )
    parser.add_argument(
        "--selection-sha256", help="digest printed by train; required for score"
    )
    args = parser.parse_args()
    if args.phase == "score" and not args.selection_sha256:
        parser.error("score requires --selection-sha256 from the completed train phase")
    root = Path(__file__).resolve().parents[1]
    config = json.loads(args.config.read_text())
    if torch.__version__ != config["torch_version"]:
        parser.error(f"frozen study requires torch {config['torch_version']}")
    manifest = root / config["manifest"]
    manifest_data = json.loads(manifest.read_text())
    test_paths = [
        d["upstream_path"] for d in manifest_data["documents"] if d["split"] == "test"
    ]
    if test_paths != config["holdout_paths"]:
        parser.error("manifest held-out paths disagree with the frozen protocol")
    provenance = dict(
        config_sha256=file_digest(args.config),
        manifest_sha256=file_digest(manifest),
        protocol_sha256=file_digest(root / "docs/stability-protocol.md"),
        sources={
            str(p.relative_to(root)): file_digest(p)
            for p in sorted(
                [*(root / "src/transformer_core").glob("*.py"), Path(__file__).resolve()]
            )
        },
    )
    docs = load_documents(manifest, args.data_dir)
    if args.phase == "train":
        digest = train_study(
            docs["train"], docs["validation"], config, args.output_dir, provenance
        )
        print(f"selection_sha256={digest}")
    else:
        result = score_study(
            docs["train"],
            docs["test"],
            test_paths,
            config,
            args.output_dir,
            provenance,
            args.selection_sha256,
        )
        print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    main()
