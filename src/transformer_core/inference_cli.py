"""Installed, CPU-only greedy generation from a trusted local byte-LM checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .inference import greedy_cached
from .language_model import ByteLanguageModel


def load_checkpoint(path):
    """Use PyTorch's restricted tensor loader; still only open trusted local files."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint["model"]
    dtypes = {tensor.dtype for tensor in state.values()}
    if dtypes not in ({torch.float32}, {torch.float64}):
        raise ValueError("checkpoint weights must have one float32/float64 dtype")
    model = ByteLanguageModel(**checkpoint["config"]).to(dtype=next(iter(dtypes)))
    model.load_state_dict(state, strict=True)
    return model.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--prompt", default="The application ")
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument(
        "--max-context",
        type=int,
        default=64,
        help="total byte budget, not sliding; original training context was 64",
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    prompt = torch.tensor([list(args.prompt.encode("utf-8"))], dtype=torch.long)
    model = load_checkpoint(args.checkpoint)
    result = greedy_cached(model, prompt, count=args.count, max_context=args.max_context)
    data = bytes(result[0].tolist())
    print(
        json.dumps(
            dict(
                schema_version=1,
                checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                mode="cached greedy, untruncated absolute prefix positions",
                prompt_bytes=prompt.shape[1],
                generated_bytes=args.count,
                max_context=args.max_context,
                output_hex=data.hex(),
                display_utf8_replacement=data.decode("utf-8", errors="replace"),
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
