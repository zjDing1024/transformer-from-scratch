"""Offline new-process installed-module surface tests."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from transformer_core.inference import greedy_cached
from transformer_core.inference_cli import load_checkpoint
from transformer_core.language_model import ByteLanguageModel


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cli_checkpoint_generation_and_overflow(tmp_path, dtype):
    torch.manual_seed(17)
    model = ByteLanguageModel(width=8, heads=2, hidden=13, layers=1).to(dtype).eval()
    path = tmp_path / "local.pt"
    torch.save(dict(model=model.state_dict(), config=model.config), path)
    assert next(load_checkpoint(path).parameters()).dtype == dtype
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root / "src"))
    command = [
        sys.executable,
        "-m",
        "transformer_core.inference_cli",
        "--checkpoint",
        str(path),
        "--prompt",
        "ab",
        "--count",
        "3",
        "--max-context",
        "5",
    ]
    run = subprocess.run(command, capture_output=True, text=True, env=env, cwd=tmp_path)
    assert run.returncode == 0, run.stderr
    result = json.loads(run.stdout)
    expected = greedy_cached(model, torch.tensor([[97, 98]]), count=3, max_context=5)
    assert result["output_hex"] == bytes(expected[0].tolist()).hex()
    assert result["generated_bytes"] == 3
    bad = subprocess.run(
        command[:-1] + ["4"], capture_output=True, text=True, env=env, cwd=tmp_path
    )
    assert bad.returncode != 0
    assert "max_context" in bad.stderr


def test_loader_rejects_mixed_precision_weights(tmp_path):
    model = ByteLanguageModel(width=8, heads=2, hidden=13, layers=1).eval()
    state = model.state_dict()
    state["embedding.weight"] = state["embedding.weight"].double()
    path = tmp_path / "mixed.pt"
    torch.save(dict(model=state, config=model.config), path)
    with pytest.raises(ValueError, match="one float32/float64 dtype"):
        load_checkpoint(path)
