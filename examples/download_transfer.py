"""Explicit network preparation for the pinned public-domain SQLite docs."""

import argparse
from pathlib import Path

from transformer_core.transfer import download_corpus

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    download_corpus(root / "data/sqlite-transfer-manifest.json", args.output_dir)
    print("Pinned SQLite documents and copyright evidence verified; no code executed.")
