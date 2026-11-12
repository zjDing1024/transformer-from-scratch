"""CPU-only numerical checks and a tiny synthetic optimizer smoke test.

Run after installation: python examples/verify_mechanisms.py --output results/day1.json
No dataset downloads, network calls, credentials, or accelerator are needed.
"""

import argparse
import json
import platform
from pathlib import Path

import torch
from torch.nn import functional as F

from transformer_core import (
    EncoderBlock,
    LayerNorm,
    SinusoidalPositionalEncoding,
    causal_mask,
    scaled_dot_product_attention,
)


def max_error(left, right):
    return float((left - right).abs().max().detach())


def verify():
    torch.set_num_threads(1)
    torch.manual_seed(20261112)
    inputs = [
        torch.randn(2, 2, 5, 4, dtype=torch.float64, requires_grad=True) for _ in range(3)
    ]
    mask = causal_mask(5)
    actual, weights = scaled_dot_product_attention(*inputs, allowed_mask=mask)
    expected = F.scaled_dot_product_attention(*inputs, attn_mask=mask)
    probe = torch.randn_like(actual)
    grads = torch.autograd.grad((actual * probe).sum(), inputs, retain_graph=True)
    expected_grads = torch.autograd.grad((expected * probe).sum(), inputs)
    attention_error = max_error(actual, expected)
    gradient_error = max(max_error(a, b) for a, b in zip(grads, expected_grads))
    assert attention_error < 1e-10 and gradient_error < 1e-10
    assert torch.count_nonzero(weights.triu(1)) == 0

    norm = LayerNorm(8).double()
    x = torch.randn(2, 5, 8, dtype=torch.float64)
    norm_error = max_error(norm(x), F.layer_norm(x, (8,), norm.weight, norm.bias, norm.eps))
    assert norm_error < 1e-10

    # Fixed random embeddings and labels test optimizer wiring only. This is
    # not a language-model benchmark, corpus evaluation or held-out accuracy.
    model = torch.nn.Sequential(
        SinusoidalPositionalEncoding(16),
        EncoderBlock(16, 4, 32),
        torch.nn.Linear(16, 7),
    )
    features = torch.randn(4, 6, 16)
    labels = torch.randint(0, 7, (4, 6))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    losses = []
    for _ in range(10):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(features).flatten(0, 1), labels.flatten())
        loss.backward()
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
        )
        losses.append(float(loss.detach()))
        optimizer.step()
    with torch.no_grad():
        final_loss = float(F.cross_entropy(model(features).flatten(0, 1), labels.flatten()))
    assert final_loss < losses[0]
    return {
        "scope": (
            "Day 1 mechanism checks and synthetic optimizer smoke test, not paper reproduction"
        ),
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "device": "cpu",
            "threads": torch.get_num_threads(),
        },
        "seed": 20261112,
        "parity_dtype": "float64",
        "attention_max_abs_error": attention_error,
        "attention_gradient_max_abs_error": gradient_error,
        "layernorm_max_abs_error": norm_error,
        "causal_future_weight_count": int(torch.count_nonzero(weights.triu(1))),
        "smoke": {
            "task": "fixed synthetic token labels; no held-out set",
            "parameters": sum(p.numel() for p in model.parameters()),
            "steps": 10,
            "batch": 4,
            "length": 6,
            "d_model": 16,
            "heads": 4,
            "d_ff": 32,
            "dropout": 0.0,
            "optimizer": "SGD(lr=0.1)",
            "initial_loss": losses[0],
            "final_loss": final_loss,
            "step_losses": losses,
            "all_parameter_gradients_finite": True,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    result = json.dumps(verify(), indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result, encoding="utf-8")
    print(result, end="")


if __name__ == "__main__":
    main()
