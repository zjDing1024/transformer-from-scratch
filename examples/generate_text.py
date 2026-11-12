"""Generate from a locally produced weights-only checkpoint in a fresh process."""

import argparse
from pathlib import Path

import torch

from transformer_core.language_model import ByteLanguageModel, generate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", default="The application ")
    parser.add_argument("--bytes", type=int, default=160)
    args = parser.parse_args()
    torch.set_num_threads(1)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = ByteLanguageModel(**saved["config"])
    model.load_state_dict(saved["model"])
    print(
        generate(model, args.prompt.encode(), count=args.bytes).decode(
            "utf-8", errors="backslashreplace"
        )
    )


if __name__ == "__main__":
    main()
