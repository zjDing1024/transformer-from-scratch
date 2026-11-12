"""Independent checks of masking, precision, and whole-block invariants."""

import pytest
import torch
from torch.nn import functional as F

from transformer_core import (
    EncoderBlock,
    LayerNorm,
    MultiHeadAttention,
    causal_mask,
    scaled_dot_product_attention,
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_layernorm_low_precision_affine_and_backward(dtype):
    torch.manual_seed(123)
    layer = LayerNorm(8).to(dtype)
    with torch.no_grad():
        layer.weight.copy_(torch.randn(8))
        layer.bias.copy_(torch.randn(8))
    x = (torch.randn(2, 4, 8) * 100).to(dtype).requires_grad_()
    y = layer(x)
    expected = F.layer_norm(x, (8,), layer.weight, layer.bias, layer.eps)
    torch.testing.assert_close(y, expected, rtol=0.02, atol=0.02)
    y.square().sum().backward()
    assert torch.isfinite(x.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in layer.parameters())


def test_head_specific_cross_attention_matches_separate_heads():
    torch.manual_seed(124)
    layer = MultiHeadAttention(12, 3, bias=False).double()
    q = torch.randn(2, 3, 12, dtype=torch.float64)
    k = torch.randn(2, 5, 12, dtype=torch.float64)
    v = torch.randn_like(k)
    allowed = torch.rand(2, 3, 3, 5) > 0.4
    allowed[0, 1, 1] = False
    keep = torch.tensor([[True, False, True, True, False], [True] * 5])
    actual, weights = layer(q, k, v, allowed_mask=allowed, key_keep_mask=keep)
    projections = [
        p(x).reshape(2, x.shape[1], 3, 4).transpose(1, 2)
        for p, x in [(layer.q_proj, q), (layer.k_proj, k), (layer.v_proj, v)]
    ]
    expected_heads = F.scaled_dot_product_attention(
        *projections, attn_mask=allowed & keep[:, None, None, :]
    )
    expected = layer.out_proj(expected_heads.transpose(1, 2).reshape(2, 3, 12))
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)
    assert torch.count_nonzero(weights[0, 1, 1]) == 0
    assert torch.count_nonzero(weights.masked_select(~keep[:, None, None, :])) == 0


def test_empty_attention_rows_have_finite_backward_through_returned_weights():
    torch.manual_seed(125)
    q, k, v = [
        torch.randn(2, 3, 4, 5, dtype=torch.float64, requires_grad=True) for _ in range(3)
    ]
    mask = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
    y, weights = scaled_dot_product_attention(q, k, v, allowed_mask=mask)
    (y.square().sum() + weights.sum()).backward()
    for x in [q, k, v]:
        assert torch.isfinite(x.grad).all()
        assert torch.count_nonzero(x.grad) == 0


def test_encoder_causal_prefix_and_padding_invariance():
    torch.manual_seed(126)
    model = EncoderBlock(8, 2, 19, dropout=0.5).double().eval()
    x = torch.randn(2, 6, 8, dtype=torch.float64)
    mask = causal_mask(6)
    original = model(x, allowed_mask=mask)
    changed = x.clone()
    changed[:, 3:] += 100
    perturbed = model(changed, allowed_mask=mask)
    torch.testing.assert_close(original[:, :3], perturbed[:, :3], rtol=0, atol=0)
    keep = torch.tensor([[True, True, True, False, False, False]]).expand(2, -1)
    original = model(x, key_keep_mask=keep)
    perturbed = model(changed, key_keep_mask=keep)
    torch.testing.assert_close(original[keep], perturbed[keep], rtol=0, atol=0)


def test_encoder_all_padding_training_backward_is_finite():
    torch.manual_seed(127)
    model = EncoderBlock(8, 2, 19, dropout=0.3).double().train()
    x = torch.randn(2, 4, 8, dtype=torch.float64, requires_grad=True)
    keep = torch.zeros(2, 4, dtype=torch.bool)
    y = model(x, key_keep_mask=keep)
    (y * torch.randn_like(y)).sum().backward()
    assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_attention_float32_accumulation_survives_outer_autocast():
    q = torch.full((1, 1, 2, 4), 300.0, dtype=torch.float16)
    v = torch.ones_like(q)
    with torch.autocast("cpu", dtype=torch.float16):
        y, weights = scaled_dot_product_attention(q, q, v)
    torch.testing.assert_close(y, v, rtol=0, atol=0)
    torch.testing.assert_close(weights, torch.full_like(weights, 0.5), rtol=0, atol=0)
