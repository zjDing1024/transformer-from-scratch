"""Independent cache failure recovery, alias isolation, and causal chunk checks."""

import pytest
import torch

from transformer_core.inference import CachedByteDecoder
from transformer_core.language_model import ByteLanguageModel


def original():
    torch.manual_seed(77)
    return ByteLanguageModel(width=9, heads=3, hidden=13, layers=2).double().eval()


@pytest.mark.parametrize("failure", ["exception", "nonfinite"])
def test_failed_append_preserves_actual_cache_and_resumes(failure):
    model = original()
    session = CachedByteDecoder(model, max_context=8)
    tokens = torch.tensor([[5, 6, 7, 8, 9]])
    session.prefill(tokens[:, :2])
    old = [(k.clone(), v.clone()) for k, v in session._cache]
    info = session.cache_info()

    def fail(module, args, output):
        if failure == "exception":
            raise RuntimeError("synthetic failure after cache work")
        return torch.full_like(output, float("nan"))

    handle = session._model.output.register_forward_hook(fail)
    with pytest.raises((RuntimeError, ValueError)):
        session.append(tokens[:, 2:])
    assert session.cache_info() == info
    for (k, v), (old_k, old_v) in zip(session._cache, old):
        assert torch.equal(k, old_k) and torch.equal(v, old_v)
    handle.remove()
    actual = session.append(tokens[:, 2:])
    with torch.no_grad():
        expected = model(tokens)[:, 2:]
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


def test_failed_prefill_leaves_empty_reusable_session():
    model = original()
    session = CachedByteDecoder(model, max_context=2)
    with pytest.raises(ValueError, match="max_context"):
        session.prefill(torch.tensor([[1, 2, 3]]))
    assert session.cache_info()["length"] == 0
    assert session.cache_info()["batch"] is None
    assert session.cache_info()["kv_payload_bytes"] == 0
    actual = session.prefill(torch.tensor([[1, 2], [3, 4]]))
    torch.testing.assert_close(actual, model(torch.tensor([[1, 2], [3, 4]])))


def test_input_and_returned_logits_do_not_alias_cached_state():
    model = original()
    session = CachedByteDecoder(model, max_context=8)
    tokens = torch.tensor([[11, 12, 13, 14]])
    prefix = tokens[:, :2].clone()
    first_logits = session.prefill(prefix)
    prefix.fill_(255)
    with torch.inference_mode():
        first_logits.fill_(float("nan"))
    next_logits = session.append(tokens[:, 2:])
    torch.testing.assert_close(next_logits, model(tokens)[:, 2:], atol=1e-12, rtol=1e-12)


def test_append_chunk_cannot_see_future_even_with_odd_width_positions():
    model = original()
    a = CachedByteDecoder(model, max_context=9)
    b = CachedByteDecoder(model, max_context=9)
    prefix = torch.tensor([[1, 2, 3], [10, 20, 30]])
    a.prefill(prefix)
    b.prefill(prefix)
    one = torch.tensor([[4, 5, 6, 7], [40, 50, 60, 70]])
    two = one.clone()
    two[:, 2:] = 255
    left = a.append(one)
    right = b.append(two)
    torch.testing.assert_close(left[:, :2], right[:, :2], rtol=0, atol=0)
    expected = model(torch.cat((prefix, one), dim=1))[:, 3:]
    torch.testing.assert_close(left, expected, atol=1e-12, rtol=1e-12)
    a.reset()
    replacement = torch.tensor([[101, 102]])
    torch.testing.assert_close(
        a.prefill(replacement), model(replacement), atol=1e-12, rtol=1e-12
    )
