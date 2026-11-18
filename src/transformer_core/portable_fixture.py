"""Versioned procedural float32 weights, NOT trained or historical checkpoint weights."""

import hashlib
import json
import struct
from importlib.resources import files

import torch

from .language_model import ByteLanguageModel
from .stability import state_digest

NAME = "byte-lm-portable-f32-v1"
MODEL = dict(width=64, heads=4, hidden=128, layers=2, use_positions=True)
MAGIC = b"PORTABLE-BYTE-LM-F32-V1\n"


def build_fixture():
    """Exact binary fractions derived from SHAKE256, independent of torch RNG draws.

    Every parameter is overwritten. RNG state is restored after model construction.
    Tensor n's signed16 codes are SHAKE256(UTF8(NAME + NUL + n)), little endian.
    Embeddings use code/2**15, matrices code/2**18, biases code/2**21;
    norm weights use 1 + code/2**21. All are exactly representable in float32.
    """
    with torch.random.fork_rng(devices=[]):
        model = ByteLanguageModel(**MODEL).float().eval()
    records, payload = [], bytearray()
    with torch.no_grad():
        for name, tensor in sorted(model.state_dict().items()):
            raw = hashlib.shake_256((NAME + "\0" + name).encode()).digest(tensor.numel() * 2)
            codes = [
                int.from_bytes(raw[i : i + 2], "little") - 32768 for i in range(0, len(raw), 2)
            ]
            if name == "embedding.weight":
                values = [code / 2**15 for code in codes]
            elif name.endswith("norm.weight"):
                values = [1 + code / 2**21 for code in codes]
            elif name.endswith("bias"):
                values = [code / 2**21 for code in codes]
            else:
                values = [code / 2**18 for code in codes]
            tensor.copy_(torch.tensor(values, dtype=torch.float32).reshape(tensor.shape))
            records.append(
                dict(name=name, shape=list(tensor.shape), dtype="float32", numel=tensor.numel())
            )
            payload.extend(struct.pack(f"<{len(values)}f", *values))
    header = json.dumps(
        dict(name=NAME, model=MODEL, tensors=records),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    artifact = MAGIC + struct.pack("<I", len(header)) + header + payload
    identity = dict(
        name=NAME,
        model=MODEL,
        artifact_size_bytes=len(artifact),
        artifact_sha256=hashlib.sha256(artifact).hexdigest(),
        tensor_sha256=state_digest(model.state_dict()),
        parameters=sum(p.numel() for p in model.parameters()),
        trained=False,
        historical_checkpoint=False,
    )
    return model, artifact, identity


def manifest():
    return json.loads(
        files("transformer_diagnostics").joinpath("data/portable-v1.json").read_text()
    )


def verified_fixture():
    model, artifact, identity = build_fixture()
    expected = manifest()["fixture"]
    if identity != expected:
        raise ValueError(
            "portable fixture identity/configuration mismatch; do not replace the lock"
        )
    return model, artifact, identity
