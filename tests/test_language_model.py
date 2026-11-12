import hashlib
import json
import math

import pytest
import torch

from transformer_core.language_model import (
    ByteLanguageModel,
    evaluate,
    evaluation_blocks,
    fit_bigram,
    generate,
    load_documents,
    sample_batch,
)


def test_causal_and_gradients():
    torch.manual_seed(4)
    model = ByteLanguageModel(width=8, heads=2, hidden=16, layers=2)
    x = torch.randint(256, (2, 7))
    altered = x.clone()
    altered[:, 4:] = (altered[:, 4:] + 1) % 256
    torch.testing.assert_close(model(x)[:, :4], model(altered)[:, :4], atol=0, rtol=0)
    model(x).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_sampling_shift_and_document_isolation():
    docs = [torch.arange(10), torch.arange(100, 115)]
    x, y = sample_batch(docs, 4, 100, torch.Generator().manual_seed(4))
    assert torch.equal(x[:, 1:], y[:, :-1])
    assert torch.equal(x + 1, y)
    assert all((row < 10).all() or (row >= 100).all() for row in x)
    with pytest.raises(ValueError):
        sample_batch(docs, 100, 2, torch.Generator())


def test_eval_exact_target_coverage_and_tail():
    docs = [torch.arange(10), torch.arange(100, 105)]
    blocks = list(evaluation_blocks(docs, 4))
    assert [y.numel() for _, y in blocks] == [4, 4, 1, 4]
    assert torch.equal(
        torch.cat([y.flatten() for _, y in blocks]), torch.cat([d[1:] for d in docs])
    )
    for x, y in blocks:
        assert torch.equal(x + 1, y)


def test_bigram_counts_and_shared_scoring():
    docs = [torch.tensor([0, 1, 0]), torch.tensor([1, 1])]
    table = fit_bigram(docs)
    assert table.exp()[0, 1].item() == pytest.approx(1.1 / 26.6)
    # No artificial final0 -> initial1 across document boundary.
    assert table.exp()[1, 0].item() == pytest.approx(1.1 / 27.6)
    model = ByteLanguageModel(width=4, heads=1, hidden=8, layers=1)
    for p in model.parameters():
        p.data.zero_()
    result = evaluate(model, docs, context=2, bigram=table)
    assert result["model"]["tokens"] == result["bigram"]["tokens"] == 3
    assert result["model"]["perplexity"] == pytest.approx(256, rel=1e-6)
    expected = -sum(table[d[:-1], d[1:]].sum().item() for d in docs)
    assert result["bigram"]["nll_sum"] == pytest.approx(expected)
    assert result["model"]["bits_per_byte"] == pytest.approx(8, rel=1e-6)


def test_manifest_rejects_corruption(tmp_path):
    items = []
    for name in ["train", "validation", "test"]:
        data = b"abc"
        (tmp_path / name).write_bytes(data)
        items.append(
            dict(
                upstream_path=name,
                split=name,
                size_bytes=3,
                sha256=hashlib.sha256(data).hexdigest(),
            )
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dict(documents=items)))
    assert len(load_documents(manifest, tmp_path)["train"][0]) == 3
    (tmp_path / "test").write_bytes(b"abd")
    with pytest.raises(ValueError, match="hash"):
        load_documents(manifest, tmp_path)


def test_checkpoint_and_generation(tmp_path):
    model = ByteLanguageModel(width=4, heads=1, hidden=8, layers=1).eval()
    path = tmp_path / "checkpoint.pt"
    torch.save(dict(model=model.state_dict(), config=model.config), path)
    saved = torch.load(path, weights_only=True)
    restored = ByteLanguageModel(**saved["config"]).eval()
    restored.load_state_dict(saved["model"])
    x = torch.tensor([[0, 1, 2]])
    torch.testing.assert_close(model(x), restored(x), atol=0, rtol=0)
    sample = generate(restored, b"a", count=4, seed=3)
    assert sample == generate(model, b"a", count=4, seed=3)
    assert len(sample) == 5
    with pytest.raises(ValueError):
        generate(model, b"")


def test_optimizer_reduces_fixed_batch_loss():
    torch.manual_seed(8)
    model = ByteLanguageModel(width=8, heads=2, hidden=16, layers=1)
    x = torch.tensor([[1, 2, 1, 2]])
    y = torch.tensor([[2, 1, 2, 1]])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    losses = []
    for _ in range(10):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x).reshape(-1, 256), y.flatten())
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    assert math.isfinite(losses[-1]) and losses[-1] < losses[0]
