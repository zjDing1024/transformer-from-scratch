"""Run the frozen single-seed experiment; downloads are a separate explicit step."""

import argparse
import copy
import hashlib
import json
import platform
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from transformer_core.language_model import (
    ByteLanguageModel,
    evaluate,
    fit_bigram,
    generate,
    load_documents,
    sample_batch,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = root / "data/text-manifest.json"
    docs = load_documents(manifest, args.data_dir)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20261113)
    generator = torch.Generator().manual_seed(20261113)
    model = ByteLanguageModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    baseline = fit_bigram(docs["train"])
    best, selected, state, curve = float("inf"), None, None, []
    start = time.perf_counter()
    for step in range(1, 601):
        model.train()
        x, y = sample_batch(docs["train"], 64, 16, generator)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x).reshape(-1, 256), y.flatten())
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite loss; no result published")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step % 100 == 0:
            validation = evaluate(model, docs["validation"], bigram=baseline)
            row = dict(
                step=step,
                last_batch_train_nll=loss.item(),
                gradient_norm=norm.item(),
                validation=validation,
            )
            curve.append(row)
            print(json.dumps(row), flush=True)
            if validation["model"]["nll"] < best:
                best, selected = validation["model"]["nll"], step
                state = copy.deepcopy(model.state_dict())
    model.load_state_dict(state)
    test = evaluate(model, docs["test"], bigram=baseline)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "checkpoint.pt"
    torch.save({"model": state, "config": model.config, "selected_step": selected}, checkpoint)
    restored = torch.load(checkpoint, weights_only=True, map_location="cpu")
    reloaded = ByteLanguageModel(**restored["config"])
    reloaded.load_state_dict(restored["model"])
    model.eval()
    reloaded.eval()
    probe = torch.tensor([list(b"The application ")])
    with torch.no_grad():
        reload_error = (model(probe) - reloaded(probe)).abs().max().item()
    result = dict(
        protocol="docs/text-protocol.md",
        seed=20261113,
        steps=600,
        byte_target_exposures=600 * 16 * 64,
        selected_step=selected,
        parameter_count=sum(p.numel() for p in model.parameters()),
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        split_bytes={k: [len(d) for d in v] for k, v in docs.items()},
        curve=curve,
        test=test,
        checkpoint_reload_max_abs=reload_error,
        sample=generate(reloaded).decode("utf-8", errors="backslashreplace"),
        elapsed_seconds=time.perf_counter() - start,
        environment={
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
            "threads": 1,
        },
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    )
    (args.output_dir / "text-training.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["test"], indent=2))


if __name__ == "__main__":
    main()
