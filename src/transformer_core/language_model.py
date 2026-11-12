"""Original byte LM and document-local training/evaluation utilities."""

import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .attention import causal_mask
from .layers import EncoderBlock, SinusoidalPositionalEncoding


class ByteLanguageModel(nn.Module):
    """Decoder-only LM: causal self-attention, no encoder or cross-attention."""

    def __init__(self, width=64, heads=4, hidden=128, layers=2):
        super().__init__()
        self.config = dict(width=width, heads=heads, hidden=hidden, layers=layers)
        self.embedding = nn.Embedding(256, width)
        self.position = SinusoidalPositionalEncoding(width)
        # An encoder block with an always-on causal mask is a self-attention-only
        # decoder block. It is not a translation decoder with cross-attention.
        self.blocks = nn.ModuleList([EncoderBlock(width, heads, hidden) for _ in range(layers)])
        self.output = nn.Linear(width, 256)

    def forward(self, tokens):
        if tokens.ndim != 2 or tokens.shape[1] == 0 or tokens.dtype != torch.long:
            raise ValueError("tokens must be nonempty [B,L] int64 byte IDs")
        x = self.position(self.embedding(tokens) * math.sqrt(self.config["width"]))
        mask = causal_mask(tokens.shape[1], device=tokens.device)
        for block in self.blocks:
            x = block(x, allowed_mask=mask)
        return self.output(x)


def load_documents(manifest_path, root):
    """Read only files whose immutable manifest hashes match, before training."""
    manifest = json.loads(Path(manifest_path).read_text())
    splits = {name: [] for name in ("train", "validation", "test")}
    for item in manifest["documents"]:
        data = (Path(root) / item["upstream_path"]).read_bytes()
        if (
            hashlib.sha256(data).hexdigest() != item["sha256"]
            or len(data) != item["size_bytes"]
        ):
            raise ValueError(f"corpus hash/size mismatch: {item['upstream_path']}")
        splits[item["split"]].append(torch.tensor(list(data), dtype=torch.long))
    if any(not docs for docs in splits.values()):
        raise ValueError("every split needs a document")
    return splits


def sample_batch(documents, context, batch, generator):
    """Uniform over valid local offsets, never over concatenated documents."""
    counts = torch.tensor([max(0, len(d) - context) for d in documents])
    total = int(counts.sum())
    if total == 0:
        raise ValueError("no document has context+1 bytes")
    offsets = torch.randint(total, (batch,), generator=generator)
    ends = counts.cumsum(0)
    xs, ys = [], []
    for offset in offsets.tolist():
        index = int(torch.searchsorted(ends, torch.tensor(offset), right=True))
        start = offset - (int(ends[index - 1]) if index else 0)
        chunk = documents[index][start : start + context + 1]
        xs.append(chunk[:-1])
        ys.append(chunk[1:])
    return torch.stack(xs), torch.stack(ys)


def evaluation_blocks(documents, context):
    if context < 1:
        raise ValueError("context must be positive")
    for doc in documents:
        for start in range(0, len(doc) - 1, context):
            end = min(start + context, len(doc) - 1)
            yield doc[start:end].unsqueeze(0), doc[start + 1 : end + 1].unsqueeze(0)


def fit_bigram(documents, alpha=0.1):
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    counts = torch.full((256, 256), alpha, dtype=torch.float64)
    for doc in documents:
        counts += torch.bincount(doc[:-1] * 256 + doc[1:], minlength=65536).reshape(256, 256)
    return (counts / counts.sum(1, keepdim=True)).log()


@torch.no_grad()
def evaluate(model, documents, context=64, bigram=None):
    model.eval()
    total, count, baseline = 0.0, 0, 0.0
    for x, y in evaluation_blocks(documents, context):
        total += F.cross_entropy(model(x).reshape(-1, 256), y.flatten(), reduction="sum").item()
        count += y.numel()
        if bigram is not None:
            baseline -= bigram[x, y].sum().item()
    if not count:
        raise ValueError("no evaluation targets")

    def metric(value):
        return dict(
            nll_sum=value,
            tokens=count,
            nll=value / count,
            perplexity=math.exp(value / count),
            bits_per_byte=value / count / math.log(2),
        )

    result = {"model": metric(total)}
    if bigram is not None:
        result["bigram"] = metric(baseline)
    return result


@torch.no_grad()
def generate(model, prompt=b"The application ", count=160, context=64, seed=42):
    if not prompt:
        raise ValueError("prompt must not be empty")
    model.eval()
    sequence = list(prompt)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(count):
        x = torch.tensor([sequence[-context:]], dtype=torch.long)
        probability = model(x)[0, -1].softmax(-1)
        sequence.append(int(torch.multinomial(probability, 1, generator=generator)))
    return bytes(sequence)
