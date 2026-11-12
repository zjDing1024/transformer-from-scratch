import math

import pytest
import torch
from torch.nn import functional as F

from transformer_core import MultiHeadAttention, causal_mask, scaled_dot_product_attention


def tensors(dtype=torch.float64):
    return [torch.randn(2, 2, n, 4, dtype=dtype) for n in (3, 5, 5)]


def test_hand_computed_two_keys():
    q = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float64)
    k = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=torch.float64)
    v = torch.tensor([[[[2.0], [8.0]]]], dtype=torch.float64)
    output, weights = scaled_dot_product_attention(q, k, v)
    first = math.exp(1 / math.sqrt(2)) / (math.exp(1 / math.sqrt(2)) + 1)
    torch.testing.assert_close(
        weights, torch.tensor([[[[first, 1 - first]]]]).double(), rtol=1e-7, atol=1e-7
    )
    assert output.item() == pytest.approx(2 * first + 8 * (1 - first), abs=1e-12)


@pytest.mark.parametrize("mask_kind", ["none", "key", "mixed", "all"])
def test_sdpa_forward_and_input_gradients_match_reference(mask_kind):
    q, k, v = [t.requires_grad_() for t in tensors()]
    if mask_kind == "none":
        mask = None
    elif mask_kind == "key":
        mask = torch.tensor([[True, True, False, True, False]]).expand(3, -1)
    elif mask_kind == "mixed":
        mask = torch.rand(2, 2, 3, 5) > 0.3
        mask[0, 0, 0] = False
    else:
        mask = torch.zeros(3, 5, dtype=torch.bool)
    actual, weights = scaled_dot_product_attention(q, k, v, allowed_mask=mask)
    reference = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    torch.testing.assert_close(actual, reference, atol=1e-12, rtol=1e-10)
    probe = torch.randn_like(actual)
    grads = torch.autograd.grad((actual * probe).sum(), (q, k, v), retain_graph=True)
    expected = torch.autograd.grad((reference * probe).sum(), (q, k, v))
    for gradient, target in zip(grads, expected):
        assert torch.isfinite(gradient).all()
        torch.testing.assert_close(gradient, target, atol=1e-12, rtol=1e-10)
    assert torch.isfinite(weights).all()
    if mask_kind == "all":
        assert torch.count_nonzero(weights) == 0
        assert torch.count_nonzero(actual) == 0
        assert all(torch.count_nonzero(g) == 0 for g in grads)


def test_softmax_rows_and_blocked_probabilities():
    q, k, v = tensors()
    mask = torch.ones(3, 5, dtype=torch.bool)
    mask[:, -2:] = False
    _, weights = scaled_dot_product_attention(q, k, v, allowed_mask=mask)
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, 2, 3, dtype=q.dtype))
    assert torch.count_nonzero(weights[..., -2:]) == 0


def test_causal_attention_cannot_see_future():
    x = torch.randn(2, 3, 5, 4, dtype=torch.float64)
    mask = causal_mask(5)
    original, weights = scaled_dot_product_attention(x, x, x, allowed_mask=mask)
    changed = x.clone()
    changed[..., 3:, :] += 1000
    result, _ = scaled_dot_product_attention(changed, changed, changed, allowed_mask=mask)
    torch.testing.assert_close(original[..., :3, :], result[..., :3, :], rtol=0, atol=0)
    assert torch.count_nonzero(weights.triu(1)) == 0


def test_attention_gradcheck():
    inputs = tuple(
        torch.randn(1, 1, 2, 2, dtype=torch.float64, requires_grad=True) for _ in range(3)
    )
    mask = torch.tensor([[False, False], [True, False]])
    assert torch.autograd.gradcheck(
        lambda q, k, v: scaled_dot_product_attention(q, k, v, allowed_mask=mask)[0], inputs
    )


def test_dropout_is_training_only_and_returned_weights_are_pre_dropout():
    q, k, v = tensors()
    evaluation, p_eval = scaled_dot_product_attention(q, k, v, dropout_p=0.5)
    baseline, p_base = scaled_dot_product_attention(q, k, v)
    sampled, p_train = scaled_dot_product_attention(q, k, v, dropout_p=0.5, training=True)
    torch.testing.assert_close(evaluation, baseline, rtol=0, atol=0)
    torch.testing.assert_close(p_train, p_eval, rtol=0, atol=0)
    torch.testing.assert_close(p_base, p_eval, rtol=0, atol=0)
    assert not torch.allclose(sampled, evaluation)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_low_precision_dtype_and_finite_large_logits(dtype):
    q, k, v = tensors(dtype)
    actual, weights = scaled_dot_product_attention(q * 100, k * 100, v)
    expected, _ = scaled_dot_product_attention((q * 100).float(), (k * 100).float(), v.float())
    assert actual.dtype == weights.dtype == dtype
    assert torch.isfinite(actual).all() and torch.isfinite(weights).all()
    torch.testing.assert_close(actual, expected.to(dtype), rtol=0, atol=0)


@pytest.mark.parametrize(
    "mask",
    [
        torch.ones(3, 5),
        torch.ones(2, 3, 5, dtype=torch.bool),
        torch.ones(7, 5, dtype=torch.bool),
    ],
)
def test_invalid_masks_rejected(mask):
    with pytest.raises((TypeError, ValueError)):
        scaled_dot_product_attention(*tensors(), allowed_mask=mask)


@pytest.mark.parametrize("rate", [-0.1, 1.0, float("nan")])
def test_invalid_dropout_rejected(rate):
    with pytest.raises(ValueError):
        scaled_dot_product_attention(*tensors(), dropout_p=rate)


@pytest.mark.parametrize("length", [0, -1, 2.5, True])
def test_invalid_causal_lengths(length):
    with pytest.raises(ValueError):
        causal_mask(length)


def test_different_value_dimension_and_noncontiguous_inputs():
    q, k, _ = tensors()
    v = torch.randn(2, 2, 7, 5, dtype=torch.float64).transpose(-1, -2)
    actual, _ = scaled_dot_product_attention(q, k, v)
    assert actual.shape == (2, 2, 3, 7)
    torch.testing.assert_close(actual, F.scaled_dot_product_attention(q, k, v))


def test_qkv_validation():
    q, k, v = tensors()
    for bad in [
        (q[0], k, v),
        (q, k[:1], v),
        (q, k, v[..., :2, :]),
        (q, k.float(), v),
        (q.int(), k.int(), v.int()),
        (q[..., :0, :], k, v),
    ]:
        with pytest.raises((TypeError, ValueError)):
            scaled_dot_product_attention(*bad)


def load_attention_reference(custom, reference):
    with torch.no_grad():
        reference.in_proj_weight.copy_(
            torch.cat([p.weight for p in (custom.q_proj, custom.k_proj, custom.v_proj)])
        )
        reference.in_proj_bias.copy_(
            torch.cat([p.bias for p in (custom.q_proj, custom.k_proj, custom.v_proj)])
        )
        reference.out_proj.load_state_dict(custom.out_proj.state_dict())


@pytest.mark.parametrize("cross", [False, True])
def test_multihead_forward_and_parameter_gradient_parity(cross):
    layer = MultiHeadAttention(8, 2).double()
    ref = torch.nn.MultiheadAttention(8, 2, batch_first=True, dropout=0).double()
    load_attention_reference(layer, ref)
    q = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True)
    k = torch.randn(2, 5, 8, dtype=torch.float64, requires_grad=True) if cross else q
    v = torch.randn_like(k, requires_grad=True) if cross else q
    keep = torch.ones(2, k.shape[1], dtype=torch.bool)
    keep[0, -1] = False
    allowed = torch.ones(q.shape[1], k.shape[1], dtype=torch.bool)
    allowed[0, 1] = False
    actual, weights = layer(q, k, v, allowed_mask=allowed, key_keep_mask=keep)
    expected, expected_weights = ref(
        q, k, v, attn_mask=~allowed, key_padding_mask=~keep, average_attn_weights=False
    )
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(weights, expected_weights, rtol=1e-10, atol=1e-12)
    probe = torch.randn_like(actual)
    (actual * probe).sum().backward(retain_graph=True)
    q_grad = q.grad.clone()
    q.grad = None
    (expected * probe).sum().backward()
    torch.testing.assert_close(q_grad, q.grad, rtol=1e-10, atol=1e-12)
    joined = torch.cat([p.weight.grad for p in (layer.q_proj, layer.k_proj, layer.v_proj)])
    torch.testing.assert_close(joined, ref.in_proj_weight.grad, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(layer.out_proj.weight.grad, ref.out_proj.weight.grad)


def test_multihead_padded_keys_do_not_leak():
    layer = MultiHeadAttention(8, 2).eval()
    x = torch.randn(2, 4, 8)
    keep = torch.tensor([[True, True, False, False], [True, True, True, False]])
    changed = x.clone()
    changed[~keep] = 1000
    actual, _ = layer(x, key_keep_mask=keep)
    perturbed, _ = layer(changed, key_keep_mask=keep)
    torch.testing.assert_close(actual[keep], perturbed[keep], rtol=0, atol=0)


def test_multihead_empty_rows_are_bias_only():
    layer = MultiHeadAttention(8, 2)
    x = torch.randn(1, 3, 8)
    y, p = layer(x, key_keep_mask=torch.zeros(1, 3, dtype=torch.bool))
    torch.testing.assert_close(y, layer.out_proj.bias.expand_as(y))
    assert torch.count_nonzero(p) == 0


def test_multihead_argument_validation():
    with pytest.raises(ValueError):
        MultiHeadAttention(7, 2)
    layer = MultiHeadAttention(8, 2)
    x = torch.randn(1, 3, 8)
    with pytest.raises(ValueError):
        layer(x, x)
    with pytest.raises(ValueError):
        layer(x[..., :4])
    with pytest.raises(TypeError):
        layer(x, key_keep_mask=torch.ones(1, 3))
    with pytest.raises(ValueError):
        layer(x, key_keep_mask=torch.ones(2, 3, dtype=torch.bool))
