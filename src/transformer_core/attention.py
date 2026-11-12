"""Explicit attention. Boolean masks use True=allowed everywhere in this package."""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def causal_mask(length: int, *, device: torch.device | str | None = None) -> Tensor:
    """Return [query, key] permissions including each token's own position."""
    if not isinstance(length, int) or isinstance(length, bool) or length < 1:
        raise ValueError("length must be a positive integer")
    positions = torch.arange(length, device=device)
    return positions[:, None] >= positions[None, :]


def _validate_qkv(query: Tensor, key: Tensor, value: Tensor) -> None:
    if any(t.ndim != 4 for t in (query, key, value)):
        raise ValueError("Q, K and V must be [batch, heads, sequence, features]")
    if any(not t.is_floating_point() for t in (query, key, value)):
        raise TypeError("Q, K and V must be floating point")
    if len({t.dtype for t in (query, key, value)}) != 1:
        raise ValueError("Q, K and V must have the same dtype")
    if len({t.device for t in (query, key, value)}) != 1:
        raise ValueError("Q, K and V must have the same device")
    if any(0 in t.shape for t in (query, key, value)):
        raise ValueError("empty Q, K and V dimensions are unsupported")
    if query.shape[:2] != key.shape[:2] or key.shape[:2] != value.shape[:2]:
        raise ValueError("Q, K and V batch and head counts must match")
    if query.shape[-1] != key.shape[-1] or key.shape[-2] != value.shape[-2]:
        raise ValueError("Q/K features and K/V sequence lengths must match")


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    allowed_mask: Tensor | None = None,
    dropout_p: float = 0.0,
    training: bool = False,
) -> tuple[Tensor, Tensor]:
    """Compute softmax(QK^T/sqrt(d_k))V using ordinary tensor operations.

    Q/K/V are [B,H,L,Dk], [B,H,S,Dk], [B,H,S,Dv]. Mask is bool,
    [L,S] or a 4D tensor broadcastable to [B,H,L,S]. A fully masked
    query has zero probabilities, zero output and finite, zero gradients.
    Returned weights are PRE-dropout probabilities. Dropout affects only
    the value aggregation, and only when training=True. Inputs must be finite.
    Half/bfloat16 products and softmax are accumulated in float32.
    """
    _validate_qkv(query, key, value)
    if not math.isfinite(dropout_p) or not 0 <= dropout_p < 1:
        raise ValueError("dropout_p must be finite and in [0, 1)")
    work_dtype = (
        torch.float32 if query.dtype in (torch.float16, torch.bfloat16) else query.dtype
    )
    # An outer AMP context would otherwise downcast matmul despite .to(float32).
    with torch.autocast(device_type=query.device.type, enabled=False):
        logits = (query.to(work_dtype) @ key.to(work_dtype).transpose(-2, -1)) / math.sqrt(
            query.shape[-1]
        )
        if allowed_mask is not None:
            if allowed_mask.dtype != torch.bool:
                raise TypeError("allowed_mask must be bool (True=allowed)")
            if allowed_mask.device != query.device:
                raise ValueError("allowed_mask must be on the Q/K/V device")
            if allowed_mask.ndim not in (2, 4):
                raise ValueError("allowed_mask must be 2D [L,S] or 4D [B,H,L,S]")
            try:
                allowed_mask = torch.broadcast_to(allowed_mask, logits.shape)
            except RuntimeError as exc:
                raise ValueError("allowed_mask cannot broadcast to [B,H,L,S]") from exc
            has_key = allowed_mask.any(dim=-1, keepdim=True)
            logits = logits.masked_fill(~allowed_mask, -torch.inf)
            # Never evaluate softmax on [-inf, ... , -inf], even in a dead branch:
            # masking NaNs after softmax would still poison its backward pass.
            logits = torch.where(has_key, logits, torch.zeros_like(logits))
            probabilities = torch.softmax(logits, dim=-1).masked_fill(~allowed_mask, 0.0)
        else:
            probabilities = torch.softmax(logits, dim=-1)
        sampled = F.dropout(probabilities, p=dropout_p, training=training)
        output = sampled @ value.to(work_dtype)
        return output.to(query.dtype), probabilities.to(query.dtype)


class MultiHeadAttention(nn.Module):
    """Batch-first Q/K/V projections, head splitting and output projection.

    This is not a wrapper around torch.nn.MultiheadAttention or fused SDPA.
    key_keep_mask [B,S] marks real keys True. It does not zero padded query
    outputs. If a query has no valid keys, attention is zero BEFORE out_proj;
    with bias=True the projected output can therefore equal out_proj.bias.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        if d_model < 1 or num_heads < 1 or d_model % num_heads:
            raise ValueError("d_model must be positive and divisible by positive num_heads")
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be finite and in [0, 1)")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def _split(self, tensor: Tensor) -> Tensor:
        batch, length, _ = tensor.shape
        return tensor.reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        query: Tensor,
        key: Tensor | None = None,
        value: Tensor | None = None,
        *,
        allowed_mask: Tensor | None = None,
        key_keep_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if (key is None) != (value is None):
            raise ValueError("provide both key and value, or neither for self-attention")
        key, value = (query, query) if key is None else (key, value)
        for name, tensor in (("query", query), ("key", key), ("value", value)):
            if tensor.ndim != 3 or tensor.shape[-1] != self.d_model:
                raise ValueError(f"{name} must be [B,L,{self.d_model}]")
        if allowed_mask is not None:
            if allowed_mask.dtype != torch.bool:
                raise TypeError("allowed_mask must be bool (True=allowed)")
            if allowed_mask.ndim not in (2, 4):
                raise ValueError("allowed_mask must be 2D or 4D")
        if key_keep_mask is not None:
            if key_keep_mask.dtype != torch.bool:
                raise TypeError("key_keep_mask must be bool (True=real key)")
            if key_keep_mask.shape != key.shape[:2] or key_keep_mask.device != key.device:
                raise ValueError("key_keep_mask must be [B,S] on the key device")
            padding = key_keep_mask[:, None, None, :]
            allowed_mask = padding if allowed_mask is None else allowed_mask & padding
        heads, weights = scaled_dot_product_attention(
            self._split(self.q_proj(query)),
            self._split(self.k_proj(key)),
            self._split(self.v_proj(value)),
            allowed_mask=allowed_mask,
            dropout_p=self.dropout,
            training=self.training,
        )
        merged = heads.transpose(1, 2).reshape(query.shape[0], query.shape[1], self.d_model)
        return self.out_proj(merged), weights
