"""Original synthetic tests for fixed storage and transactional logical validity."""

import copy

import pytest
import torch

from transformer_core.inference import CachedByteDecoder
from transformer_core.language_model import ByteLanguageModel
from transformer_core.preallocated import PreallocatedByteDecoder


def model(dtype=torch.float64, positions=True):
    return (
        ByteLanguageModel(width=12, heads=3, hidden=19, layers=2, use_positions=positions)
        .to(dtype)
        .eval()
    )


def pointers(session):
    return [t.untyped_storage().data_ptr() for pair in session._storage for t in pair]


def storage_copy(session):
    return [t.clone() for pair in session._storage for t in pair]


def assert_storage(session, saved):
    for actual, expected in zip((t for pair in session._storage for t in pair), saved):
        assert torch.equal(actual, expected)


def close(a, b):
    tol = 1e-5 if a.dtype == torch.float32 else 1e-12
    torch.testing.assert_close(a, b, atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("positions", [False, True])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("chunks", [[16], [1] * 16, [3, 5, 1, 7]])
def test_full_dynamic_chunk_parity_and_fixed_reserve(dtype, positions, batch, chunks):
    net = model(dtype, positions)
    static = PreallocatedByteDecoder(net, max_batch_size=3, max_context=16)
    dynamic = CachedByteDecoder(net, max_context=16)
    ptrs = pointers(static)
    tokens = torch.arange(batch * 16).reshape(batch, 16)
    with torch.no_grad():
        full = net(tokens)
        start = 0
        for size in chunks:
            new = tokens[:, start : start + size]
            a = (static.prefill if start == 0 else static.append)(new)
            b = (dynamic.prefill if start == 0 else dynamic.append)(new)
            close(a, b)
            close(a, full[:, start : start + size])
            close(a, net(tokens[:, : start + size])[:, start:])
            start += size
            assert pointers(static) == ptrs
            assert not a.requires_grad
    info = static.cache_info()
    element_bytes = torch.empty((), dtype=dtype).element_size()
    assert info["backing_storage_bytes"] == 2 * 2 * 3 * 16 * 12 * element_bytes
    assert info["kv_payload_bytes"] == 2 * 2 * batch * 16 * 12 * element_bytes
    assert info["backing_storage_count"] == 4


@pytest.mark.parametrize("failure", ["raise", "nan"])
@pytest.mark.parametrize("prefilled", [False, True])
def test_late_fault_shorter_different_retry_without_reset(failure, prefilled):
    net = model()
    session = PreallocatedByteDecoder(net, max_batch_size=2, max_context=24)
    oracle = CachedByteDecoder(net, max_context=24)
    prefix = torch.arange(32).reshape(2, 16)
    if prefilled:
        session.prefill(prefix)
        oracle.prefill(prefix)
    info, ptrs = session.cache_info(), pointers(session)
    old = [(k.clone(), v.clone()) for k, v in session._cache]
    values_seen = []
    handles = [
        block.attention.v_proj.register_forward_hook(
            lambda module, args, out: values_seen.append(True)
        )
        for block in session._model.blocks
    ]

    def fail(module, args, out):
        assert len(values_seen) == 2, "Fault must be after every layer value projection"
        if failure == "raise":
            raise RuntimeError("injected late fault")
        return torch.full_like(out, float("nan"))

    handles.append(session._model.output.register_forward_hook(fail))
    with pytest.raises((RuntimeError, ValueError)):
        (session.append if prefilled else session.prefill)(torch.full((2, 4), 250))
    for handle in handles:
        handle.remove()
    assert session.cache_info() == info and pointers(session) == ptrs
    for (k, v), (ok, ov) in zip(session._cache, old):
        assert torch.equal(k, ok) and torch.equal(v, ov)
    if prefilled:
        for suffix in [torch.tensor([[101], [102]]), torch.tensor([[103, 104], [105, 106]])]:
            prefix = torch.cat((prefix, suffix), dim=1)
            actual = session.append(suffix)
            close(actual, oracle.append(suffix))
            close(actual, net(prefix)[:, -suffix.shape[1] :])
    else:
        assert session.cache_info()["batch"] is None
        retry = torch.tensor([[101, 102]])
        close(session.prefill(retry), oracle.prefill(retry))
        close(
            session.append(torch.tensor([[103]])), net(torch.tensor([[101, 102, 103]]))[:, -1:]
        )
    assert pointers(session) == ptrs


def test_inactive_tail_and_unused_rows_are_not_read():
    net = model()
    sessions = [
        PreallocatedByteDecoder(net, max_batch_size=3, max_context=24) for _ in range(2)
    ]
    prefix = torch.arange(16).reshape(1, 16)
    for session, poison in zip(sessions, [float("nan"), 123456.0]):
        session.prefill(prefix)
        with torch.inference_mode():
            for pair in session._storage:
                for backing in pair:
                    backing[:1, :, 16:].fill_(poison)
                    backing[1:].fill_(poison)
    for suffix in [torch.tensor([[101]]), torch.tensor([[102, 103]])]:
        prefix = torch.cat((prefix, suffix), dim=1)
        left, right = [s.append(suffix) for s in sessions]
        close(left, right)
        close(left, net(prefix)[:, -suffix.shape[1] :])
        for pair_a, pair_b in zip(sessions[0]._cache, sessions[1]._cache):
            for a, b in zip(pair_a, pair_b):
                assert torch.equal(a, b)


@pytest.mark.parametrize(
    "bad",
    [
        torch.tensor([[256]]),
        torch.tensor([[-1]]),
        torch.tensor([[1.0]]),
        torch.tensor([1]),
        torch.empty(1, 0, dtype=torch.long),
        [[1]],
        torch.ones(1, 1, dtype=torch.long, device="meta"),
        torch.ones(3, 1, dtype=torch.long),
        torch.ones(2, 4, dtype=torch.long),
    ],
)
def test_rejected_append_preserves_every_backing_byte(bad):
    session = PreallocatedByteDecoder(model(), max_batch_size=2, max_context=4)
    session.prefill(torch.tensor([[1], [2]]))
    before, info, ptrs = storage_copy(session), session.cache_info(), pointers(session)
    with pytest.raises(ValueError):
        session.append(bad)
    assert_storage(session, before)
    assert session.cache_info() == info and pointers(session) == ptrs


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_invalid_batch_budget(budget):
    with pytest.raises(ValueError):
        PreallocatedByteDecoder(model(), max_batch_size=budget)


def test_initial_batch_cap_context_boundary_and_lifecycle():
    session = PreallocatedByteDecoder(model(), max_batch_size=2, max_context=2)
    before = storage_copy(session)
    with pytest.raises(ValueError, match="max_batch_size"):
        session.prefill(torch.ones(3, 1, dtype=torch.long))
    with pytest.raises(ValueError, match="prefill"):
        session.append(torch.ones(1, 1, dtype=torch.long))
    assert_storage(session, before)
    session.prefill(torch.ones(2, 2, dtype=torch.long))
    before = storage_copy(session)
    with pytest.raises(ValueError, match="max_context"):
        session.append(torch.ones(2, 1, dtype=torch.long))
    with pytest.raises(ValueError, match="empty"):
        session.prefill(torch.ones(2, 1, dtype=torch.long))
    assert_storage(session, before)


def test_reset_retains_bytes_and_snapshot_and_allows_new_batch():
    net = model()
    frozen = copy.deepcopy(net)
    session = PreallocatedByteDecoder(net, max_batch_size=2, max_context=8)
    session.prefill(torch.tensor([[1, 2, 3], [4, 5, 6]]))
    before, ptrs, reserved = (
        storage_copy(session),
        pointers(session),
        session.cache_info()["backing_storage_bytes"],
    )
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.add_(2)
    net.config["use_positions"] = False
    net.train()
    session.reset()
    assert_storage(session, before)
    assert pointers(session) == ptrs
    info = session.cache_info()
    assert info["length"] == 0 and info["batch"] is None and info["kv_payload_bytes"] == 0
    assert info["backing_storage_bytes"] == reserved
    tokens = torch.tensor([[9, 10, 11]])
    close(session.prefill(tokens), frozen(tokens))
    assert all(p.requires_grad and p.grad is None for p in net.parameters())


def test_metadata_is_detached_and_zero_kv_is_valid():
    net = model()
    with torch.no_grad():
        for block in net.blocks:
            for projection in [block.attention.k_proj, block.attention.v_proj]:
                projection.weight.zero_()
                projection.bias.zero_()
    session = PreallocatedByteDecoder(net, max_batch_size=2, max_context=8)
    tokens = torch.tensor([[1, 2, 3, 4]])
    close(session.prefill(tokens[:, :2]), net(tokens[:, :2]))
    info = session.cache_info()
    info["backing_shapes"][0]["key"][2] = 999
    info["layers"][0]["value"][2] = 999
    info["batch"] = 999
    close(session.append(tokens[:, 2:]), net(tokens)[:, 2:])
    assert session.cache_info()["length"] == 4
    assert session.cache_info()["backing_shapes"][0]["key"][2] == 8
    assert all(torch.count_nonzero(t) == 0 for pair in session._cache for t in pair)


def test_model_mode_dtype_and_nonfinite_restrictions():
    with pytest.raises(ValueError, match="eval"):
        PreallocatedByteDecoder(model().train(), max_batch_size=1)
    with pytest.raises(ValueError, match="float32/float64"):
        PreallocatedByteDecoder(model().half(), max_batch_size=1)
    net = model()
    with torch.no_grad():
        next(net.parameters()).fill_(float("nan"))
    with pytest.raises(ValueError, match="finite"):
        PreallocatedByteDecoder(net, max_batch_size=1)
