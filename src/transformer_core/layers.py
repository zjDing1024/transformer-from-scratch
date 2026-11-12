"""Sinusoidal positions, population-variance normalization and residual blocks."""

import math

import torch
from torch import Tensor, nn

from .attention import MultiHeadAttention


class SinusoidalPositionalEncoding(nn.Module):
    """Add deterministic sin/cos positions to [B,L,D], including odd D.

    The input is already an embedding; sqrt(d_model) scaling, if required,
    belongs to the caller. Positions are computed on demand, not learned.
    """

    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        if d_model < 1:
            raise ValueError("d_model must be positive")
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, *, offset: int = 0) -> Tensor:
        if x.ndim != 3 or x.shape[-1] != self.d_model or not x.is_floating_point():
            raise ValueError(f"x must be floating point [B,L,{self.d_model}]")
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise ValueError("offset must be a nonnegative integer")
        work_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        positions = torch.arange(offset, offset + x.shape[1], device=x.device, dtype=work_dtype)
        dimensions = torch.arange(0, self.d_model, 2, device=x.device, dtype=work_dtype)
        angles = positions[:, None] * torch.exp(-math.log(10000.0) * dimensions / self.d_model)
        encoding = torch.empty((x.shape[1], self.d_model), device=x.device, dtype=work_dtype)
        encoding[:, 0::2] = angles.sin()
        encoding[:, 1::2] = angles[:, : self.d_model // 2].cos()
        return self.dropout(x + encoding.to(x.dtype))


class LayerNorm(nn.Module):
    """Normalize the last dimension with biased (population) variance."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        if d_model < 1 or not math.isfinite(eps) or eps <= 0:
            raise ValueError("d_model and finite eps must be positive")
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim < 1 or x.shape[-1] != self.d_model or not x.is_floating_point():
            raise ValueError(
                f"last dimension must be {self.d_model}, with floating point input"
            )
        work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        centered = work - work.mean(dim=-1, keepdim=True)
        variance = centered.square().mean(dim=-1, keepdim=True)
        normalized = centered * torch.rsqrt(variance + self.eps)
        return (normalized * self.weight + self.bias).to(x.dtype)


class ResidualAddNorm(nn.Module):
    """Paper-style post-norm: LayerNorm(x + Dropout(sublayer_output))."""

    def __init__(self, d_model: int, dropout: float = 0.0, eps: float = 1e-5):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(d_model, eps=eps)

    def forward(self, x: Tensor, update: Tensor) -> Tensor:
        if x.shape != update.shape:
            raise ValueError("residual and update shapes must match exactly")
        return self.norm(x + self.dropout(update))


class FeedForward(nn.Module):
    """Position-wise D -> d_ff -> D with ReLU and hidden dropout."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        if d_model < 1 or d_ff < 1:
            raise ValueError("d_model and d_ff must be positive")
        self.input = nn.Linear(d_model, d_ff)
        self.output = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.output(self.dropout(torch.relu(self.input(x))))


class EncoderBlock(nn.Module):
    """One post-norm encoder layer. This is not a complete translation model."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.attention_residual = ResidualAddNorm(d_model, dropout=dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout=dropout)
        self.ffn_residual = ResidualAddNorm(d_model, dropout=dropout)

    def forward(
        self,
        x: Tensor,
        *,
        allowed_mask: Tensor | None = None,
        key_keep_mask: Tensor | None = None,
    ) -> Tensor:
        update, _ = self.attention(x, allowed_mask=allowed_mask, key_keep_mask=key_keep_mask)
        x = self.attention_residual(x, update)
        return self.ffn_residual(x, self.feed_forward(x))
