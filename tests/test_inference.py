"""Independent full-prefix oracles for bounded cached inference."""

import copy

import pytest
import torch

from transformer_core.inference import CachedByteDecoder, greedy_cached
from transformer_core.language_model import ByteLanguageModel


def model(use_positions=True, dtype=torch.float64):
    torch.manual_seed(17)
    return (
        ByteLanguageModel(width=12, heads=3, hidden=19, layers=2, use_positions=use_positions)
        .to(dtype)
        .eval()
    )


@pytest.mark.parametrize("positions", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("chunks", [[16], [1] * 16, [3, 5, 1, 7]])
def test_all_logits_match_full_prefix(positions, dtype, batch, chunks):
    torch.set_num_threads(1)
    net = model(positions, dtype)
    tokens = torch.arange(batch * 16).reshape(batch, 16) % 256
    session = CachedByteDecoder(net, max_context=16)
    seen = 0
    tolerance = 1e-5 if dtype == torch.float32 else 1e-12
    with torch.no_grad():
        expected_all = net(tokens)
        for size in chunks:
            fn = session.prefill if seen == 0 else session.append
            actual = fn(tokens[:, seen : seen + size])
            expected_prefix = net(tokens[:, : seen + size])[:, seen:]
            torch.testing.assert_close(actual, expected_prefix, atol=tolerance, rtol=tolerance)
            torch.testing.assert_close(
                actual, expected_all[:, seen : seen + size], atol=tolerance, rtol=tolerance
            )
            assert not actual.requires_grad
            seen += size
    info = session.cache_info()
    assert info["length"] == 16
    assert info["layers"] == [dict(key=[batch, 3, 16, 4], value=[batch, 3, 16, 4])] * 2
    assert (
        info["kv_payload_bytes"]
        == 2 * 2 * batch * 16 * 12 * torch.empty((), dtype=dtype).element_size()
    )


@pytest.mark.parametrize("positions", [True, False])
def test_greedy_tokens_equal_full_history(positions):
    net = model(positions)
    prompt = torch.tensor([[0, 2, 255], [3, 2, 1]])
    expected = prompt.clone()
    with torch.no_grad():
        for _ in range(13):
            expected = torch.cat((expected, net(expected)[:, -1].argmax(-1, keepdim=True)), 1)
    assert torch.equal(greedy_cached(net, prompt, count=13, max_context=16), expected)
    assert torch.equal(greedy_cached(net, prompt, count=0), prompt)


def test_snapshot_isolation_reset_and_no_source_mutation():
    net = model()
    frozen = copy.deepcopy(net)
    source_grad_flags = [p.requires_grad for p in net.parameters()]
    session = CachedByteDecoder(net, max_context=8)
    tokens = torch.tensor([[1, 3, 2, 4]])
    session.prefill(tokens[:, :2])
    with torch.no_grad():
        for p in net.parameters():
            p.add_(1)
        net.config["use_positions"] = False
        net.train()
        result = session.append(tokens[:, 2:])
        torch.testing.assert_close(result, frozen(tokens)[:, 2:], atol=1e-12, rtol=1e-12)
    session.reset()
    assert session.cache_info()["length"] == 0
    assert session.cache_info()["kv_payload_bytes"] == 0
    torch.testing.assert_close(session.prefill(tokens), frozen(tokens), atol=1e-12, rtol=1e-12)
    assert [p.requires_grad for p in net.parameters()] == source_grad_flags
    assert all(p.grad is None for p in net.parameters())


@pytest.mark.parametrize(
    "bad",
    [
        torch.tensor([[256]]),
        torch.tensor([[-1]]),
        torch.tensor([[1.0]]),
        torch.tensor([1]),
        torch.empty(1, 0, dtype=torch.long),
        torch.empty(0, 1, dtype=torch.long),
        [[1]],
        torch.ones(1, 1, dtype=torch.long, device="meta"),
    ],
)
def test_invalid_input_does_not_advance_cache(bad):
    session = CachedByteDecoder(model())
    session.prefill(torch.tensor([[1]]))
    before = session.cache_info()
    with pytest.raises(ValueError):
        session.append(bad)
    assert session.cache_info() == before
    assert session.append(torch.tensor([[2]])).shape == (1, 1, 256)


def test_lifecycle_batch_and_context_boundaries():
    session = CachedByteDecoder(model(), max_context=3)
    with pytest.raises(ValueError, match="prefill first"):
        session.append(torch.tensor([[1]]))
    session.prefill(torch.tensor([[1, 2]]))
    before = session.cache_info()
    with pytest.raises(ValueError, match="empty session"):
        session.prefill(torch.tensor([[1]]))
    with pytest.raises(ValueError, match="batch size"):
        session.append(torch.tensor([[1], [2]]))
    with pytest.raises(ValueError, match="max_context"):
        session.append(torch.tensor([[1, 2]]))
    assert before == session.cache_info()
    session.append(torch.tensor([[3]]))
    with pytest.raises(ValueError, match="max_context"):
        session.append(torch.tensor([[4]]))
    assert session.cache_info()["length"] == 3
    session.reset()
    session.prefill(torch.tensor([[1], [2]]))
    assert session.cache_info()["batch"] == 2


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_invalid_budget(budget):
    with pytest.raises(ValueError):
        CachedByteDecoder(model(), max_context=budget)


def test_unsupported_models_modes_dtypes_and_nonfinite_weights():
    with pytest.raises(TypeError):
        CachedByteDecoder(torch.nn.Linear(2, 2))
    net = model().train()
    with pytest.raises(ValueError, match="eval"):
        CachedByteDecoder(net)
    net.eval().blocks[0].train()
    with pytest.raises(ValueError, match="eval"):
        CachedByteDecoder(net)
    with pytest.raises(ValueError, match="float32/float64"):
        CachedByteDecoder(model().half())
    net = model()
    with torch.no_grad():
        next(net.parameters())[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        CachedByteDecoder(net)


def test_cache_metadata_is_not_mutable_state():
    session = CachedByteDecoder(model())
    session.prefill(torch.tensor([[1]]))
    info = session.cache_info()
    info["length"] = 200
    info["layers"][0]["key"][2] = 200
    assert session.cache_info()["length"] == 1
    assert session.cache_info()["layers"][0]["key"][2] == 1


def test_new_token_projection_only_and_prefix_keys_preserved():
    session = CachedByteDecoder(model(), max_context=8)
    observed = []
    handles = [
        p.register_forward_hook(lambda module, args, out: observed.append(args[0].shape[1]))
        for block in session._model.blocks
        for p in (block.attention.q_proj, block.attention.k_proj, block.attention.v_proj)
    ]
    session.prefill(torch.tensor([[1, 2, 3]]))
    old = [(k.clone(), v.clone()) for k, v in session._cache]
    session.append(torch.tensor([[4]]))
    assert observed == [3] * 6 + [1] * 6
    for (k, v), (old_k, old_v) in zip(session._cache, old):
        assert torch.equal(k[:, :, :3], old_k)
        assert torch.equal(v[:, :, :3], old_v)
    for handle in handles:
        handle.remove()


def test_autocast_is_disabled_and_argmax_ties_choose_zero():
    net = model(dtype=torch.float32)
    tokens = torch.tensor([[1, 2]])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        cached = CachedByteDecoder(net).prefill(tokens)
    torch.testing.assert_close(cached, net(tokens), atol=1e-5, rtol=1e-5)
    assert cached.dtype == torch.float32
    with torch.no_grad():
        net.output.weight.zero_()
        net.output.bias.zero_()
    assert torch.equal(greedy_cached(net, tokens, count=2), torch.tensor([[1, 2, 0, 0]]))


@pytest.mark.parametrize("count", [-1, True, 1.5, 64])
def test_invalid_generation_budget(count):
    with pytest.raises(ValueError):
        greedy_cached(model(), torch.tensor([[1, 2]]), count=count, max_context=64)
