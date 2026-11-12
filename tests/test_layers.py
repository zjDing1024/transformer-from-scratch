import math

import pytest
import torch
from test_attention import load_attention_reference

from transformer_core import (
    EncoderBlock,
    FeedForward,
    LayerNorm,
    ResidualAddNorm,
    SinusoidalPositionalEncoding,
    causal_mask,
)


@pytest.mark.parametrize("d_model", [1, 5, 8])
def test_positions_match_scalar_formula(d_model):
    layer = SinusoidalPositionalEncoding(d_model)
    x = torch.zeros(2, 4, d_model, dtype=torch.float64)
    actual = layer(x, offset=2)
    expected = torch.tensor(
        [
            [
                (math.sin if j % 2 == 0 else math.cos)(p / 10000 ** (2 * (j // 2) / d_model))
                for j in range(d_model)
            ]
            for p in range(2, 6)
        ],
        dtype=x.dtype,
    )
    torch.testing.assert_close(actual, expected.expand_as(actual), atol=1e-14, rtol=1e-14)
    assert list(layer.parameters()) == []


def test_position_chunk_offset_matches_full_and_gradient_is_identity():
    layer = SinusoidalPositionalEncoding(6)
    x = torch.randn(2, 7, 6, dtype=torch.float64, requires_grad=True)
    full = layer(x)
    chunked = torch.cat([layer(x[:, :3]), layer(x[:, 3:], offset=3)], dim=1)
    torch.testing.assert_close(full, chunked, rtol=0, atol=0)
    full.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_position_dtype_preserved(dtype):
    assert SinusoidalPositionalEncoding(5)(torch.zeros(1, 3, 5, dtype=dtype)).dtype == dtype


def test_position_validation():
    layer = SinusoidalPositionalEncoding(4)
    for x, offset in [
        (torch.zeros(1, 3, 2), 0),
        (torch.zeros(1, 3, 4), -1),
        (torch.zeros(1, 3, 4), 0.5),
    ]:
        with pytest.raises(ValueError):
            layer(x, offset=offset)


def test_layernorm_forward_input_and_affine_gradient_parity():
    own = LayerNorm(5, eps=1e-5).double()
    ref = torch.nn.LayerNorm(5, eps=1e-5).double()
    with torch.no_grad():
        own.weight.copy_(torch.randn(5))
        own.bias.copy_(torch.randn(5))
        ref.load_state_dict(own.state_dict())
    x = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
    other = x.detach().clone().requires_grad_()
    y, expected = own(x), ref(other)
    torch.testing.assert_close(y, expected, atol=1e-12, rtol=1e-10)
    probe = torch.randn_like(y)
    (y * probe).sum().backward()
    (expected * probe).sum().backward()
    for a, b in [
        (x.grad, other.grad),
        (own.weight.grad, ref.weight.grad),
        (own.bias.grad, ref.bias.grad),
    ]:
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-10)


def test_layernorm_constant_and_single_feature_are_finite():
    for width in [1, 4]:
        x = torch.full((2, 3, width), 100.0, requires_grad=True)
        actual = LayerNorm(width)(x)
        assert torch.count_nonzero(actual) == 0
        actual.sum().backward()
        assert torch.isfinite(x.grad).all()


def test_layernorm_gradcheck():
    layer = LayerNorm(3).double()
    x = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(layer, (x,))


def test_residual_post_norm_and_exact_shape_contract():
    block = ResidualAddNorm(4, dropout=0.7).eval()
    x, update = torch.randn(2, 3, 4), torch.randn(2, 3, 4)
    torch.testing.assert_close(block(x, update), block.norm(x + update))
    torch.testing.assert_close(block(x, torch.zeros_like(x)), block.norm(x))
    with pytest.raises(ValueError):
        block(x, update[:1])


def test_feed_forward_is_position_wise():
    layer = FeedForward(4, 9).eval()
    x = torch.randn(2, 5, 4)
    order = torch.tensor([4, 2, 0, 1, 3])
    torch.testing.assert_close(layer(x[:, order]), layer(x)[:, order])


def test_encoder_forward_and_all_gradients_match_torch():
    own = EncoderBlock(8, 2, 17).double()
    ref = torch.nn.TransformerEncoderLayer(
        8, 2, dim_feedforward=17, dropout=0, batch_first=True, norm_first=False
    ).double()
    load_attention_reference(own.attention, ref.self_attn)
    pairs = [
        (own.feed_forward.input, ref.linear1),
        (own.feed_forward.output, ref.linear2),
        (own.attention_residual.norm, ref.norm1),
        (own.ffn_residual.norm, ref.norm2),
    ]
    for source, destination in pairs:
        destination.load_state_dict(source.state_dict())
    x = torch.randn(2, 4, 8, dtype=torch.float64, requires_grad=True)
    other = x.detach().clone().requires_grad_()
    mask = causal_mask(4)
    keep = torch.tensor([[True, True, True, False], [True, True, True, True]])
    y = own(x, allowed_mask=mask, key_keep_mask=keep)
    expected = ref(other, src_mask=~mask, src_key_padding_mask=~keep)
    torch.testing.assert_close(y, expected, rtol=1e-10, atol=1e-12)
    probe = torch.randn_like(y)
    (y * probe).sum().backward()
    (expected * probe).sum().backward()
    torch.testing.assert_close(x.grad, other.grad, rtol=1e-9, atol=1e-11)
    for source, destination in pairs + [(own.attention.out_proj, ref.self_attn.out_proj)]:
        for a, b in zip(source.parameters(), destination.parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=1e-9, atol=1e-11)
    for name in ("weight", "bias"):
        combined = torch.cat(
            [
                getattr(p, name).grad
                for p in (own.attention.q_proj, own.attention.k_proj, own.attention.v_proj)
            ]
        )
        torch.testing.assert_close(
            combined, getattr(ref.self_attn, "in_proj_" + name).grad, rtol=1e-9, atol=1e-11
        )


def test_encoder_eval_determinism_and_checkpoint_roundtrip():
    model = EncoderBlock(8, 2, 16, dropout=0.3).eval()
    clone = EncoderBlock(8, 2, 16, dropout=0.3).eval()
    clone.load_state_dict(model.state_dict())
    x = torch.randn(2, 4, 8)
    torch.testing.assert_close(model(x), model(x), rtol=0, atol=0)
    torch.testing.assert_close(model(x), clone(x), rtol=0, atol=0)
