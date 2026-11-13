"""Explicit network preparation; frozen public docs only, never executes content."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/text-manifest.json",
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    base = f"https://raw.githubusercontent.com/pallets/flask/{manifest['upstream_commit']}/"
    items = [(d["upstream_path"], d["sha256"]) for d in manifest["documents"]]
    items.append(("LICENSE.txt", manifest["license_sha256"]))
    for path, digest in items:
        with urlopen(base + path, timeout=30) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"hash mismatch: {path}")
        target = args.output_dir / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        print(path, len(data), digest)


if __name__ == "__main__":
    main()
