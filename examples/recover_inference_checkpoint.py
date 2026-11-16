"""Restore ONE fixed original checkpoint. This is not a new training experiment."""

import argparse
import json
from pathlib import Path
from urllib.request import urlopen

import torch

from transformer_core.stability import configure_cpu, file_digest, train_candidate, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("output directory must be empty; preserve earlier recovery attempts")
    config = json.loads((ROOT / "experiments/position-stability-v1.json").read_text())
    if torch.__version__ != config["torch_version"]:
        raise ValueError(f"exact recovery requires torch {config['torch_version']}")
    selection_path = ROOT / "results/position-v1/selection.json"
    if file_digest(selection_path) != (
        "4a12794b6df715539657cbb951a7a52c4423f7102a578eaf7ca52c7739da4b9b"
    ):
        raise ValueError("original selection lock mismatch")
    selection = json.loads(selection_path.read_text())
    expected = next(
        r for r in selection["runs"] if r["seed"] == 20261114 and r["arm"] == "sinusoidal"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / config["manifest"]
    manifest = json.loads(manifest_path.read_text())
    base = f"https://raw.githubusercontent.com/pallets/flask/{manifest['upstream_commit']}/"
    splits = {"train": [], "validation": []}
    # No test/SQLite documents are fetched, evaluated or used for restoration.
    items = [d for d in manifest["documents"] if d["split"] in splits]
    items.append(dict(upstream_path="LICENSE.txt", sha256=manifest["license_sha256"]))
    for item in items:
        path = args.output_dir / "corpus" / item["upstream_path"]
        with urlopen(base + item["upstream_path"], timeout=30) as response:
            data = response.read()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        if file_digest(path) != item["sha256"]:
            raise ValueError("frozen train/validation/license digest mismatch")
        if item.get("split") in splits:
            splits[item["split"]].append(torch.tensor(list(data), dtype=torch.long))
    configure_cpu()
    row = train_candidate(
        splits["train"], splits["validation"], config, 20261114, "sinusoidal", args.output_dir
    )
    checked = [key for key in expected if key != "training_and_validation_seconds"]
    mismatches = [key for key in checked if row[key] != expected[key]]
    report = dict(
        schema_version=1,
        origin="rebuilt from original train/validation recipe, not retained original file",
        selection_sha256=file_digest(selection_path),
        manifest_sha256=file_digest(manifest_path),
        checked_original_fields=checked,
        mismatches=mismatches,
        artifact_sha256_match=row["checkpoint_sha256"] == expected["checkpoint_sha256"],
        tensor_sha256_match=row["selected_state_sha256"] == expected["selected_state_sha256"],
        run=row,
    )
    write_json(args.output_dir / "recovery.json", report)
    if mismatches:
        raise ValueError(f"recovery differs from original: {mismatches}")
    print(json.dumps({k: v for k, v in report.items() if k != "run"}, indent=2))


if __name__ == "__main__":
    main()
