import hashlib
import json
import struct

import torch

from transformer_core.portable_fixture import MAGIC, build_fixture, manifest, verified_fixture
from transformer_core.portable_parity import verify_parity


def test_portable_identity_canonical_layout_and_rng_isolation():
    torch.set_num_threads(1)
    torch.manual_seed(12345)
    before = torch.get_rng_state().clone()
    model, artifact, identity = verified_fixture()
    assert torch.equal(before, torch.get_rng_state())
    assert identity == manifest()["fixture"]
    assert hashlib.sha256(artifact).hexdigest() == identity["artifact_sha256"]
    assert artifact.startswith(MAGIC)
    length = struct.unpack("<I", artifact[len(MAGIC) : len(MAGIC) + 4])[0]
    start = len(MAGIC) + 4
    header = json.loads(artifact[start : start + length])
    assert header["model"] == model.config
    offset = start + length
    for row in header["tensors"]:
        count = row["numel"]
        values = struct.unpack(f"<{count}f", artifact[offset : offset + count * 4])
        assert list(values) == model.state_dict()[row["name"]].flatten().tolist()
        offset += count * 4
    assert offset == len(artifact) == 403088
    torch.manual_seed(56789)
    _, again, other = build_fixture()
    assert again == artifact and other == identity


def test_fixed_fixture_parity_and_no_last_byte_forward():
    model, _, _ = verified_fixture()
    evidence = verify_parity(model, manifest())
    assert evidence["status"] == "passed"
    assert [c["cache"]["length"] for c in evidence["cases"]] == [31, 63]
    assert [c["cache"]["kv_payload_bytes"] for c in evidence["cases"]] == [63488, 129024]
