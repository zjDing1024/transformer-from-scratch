"""Bounded, snapshot-owned CPU KV caching for the original byte language model.

No training path is changed. Cache tensors are private; inspect their shapes and
payload size through cache_info(), not by modifying keys or values in place.
"""

import copy
import math

import torch

from .attention import scaled_dot_product_attention
from .language_model import ByteLanguageModel


class CachedByteDecoder:
    """Inference-only, same-length unpadded batches with absolute prefix positions.

    The source must be in eval mode. This session owns a deep weight snapshot so
    later source training/load_state_dict cannot silently invalidate cached K/V.
    One session is one sequence batch, not thread-safe and not a serving server.
    CPU float32/float64 only. No sliding, eviction, padding, EOS or beam reordering.
    max_context is an explicit caller budget, not a learned architecture limit.
    """

    def __init__(self, model: ByteLanguageModel, *, max_context: int = 64):
        if type(model) is not ByteLanguageModel:
            raise TypeError("model must be an unmodified ByteLanguageModel")
        if type(max_context) is not int or max_context < 1:
            raise ValueError("max_context must be a positive integer")
        if any(module.training for module in model.modules()):
            raise ValueError("call model.eval() before creating an inference session")
        parameters = list(model.parameters())
        dtypes = {p.dtype for p in parameters}
        if any(p.device.type != "cpu" for p in parameters) or dtypes not in (
            {torch.float32},
            {torch.float64},
        ):
            raise ValueError("only uniform CPU float32/float64 model weights are supported")
        if any(not torch.isfinite(p).all() for p in parameters):
            raise ValueError("model weights must be finite")
        self._model = copy.deepcopy(model).requires_grad_(False)
        self._max_context = max_context
        self.reset()

    def reset(self):
        """Discard the sequence, retaining the same isolated weight snapshot."""
        self._cache = ()
        self._length = 0
        self._batch = None

    def cache_info(self):
        """JSON-compatible metadata only; no shared writable tensor aliases."""
        return dict(
            length=self._length,
            batch=self._batch,
            max_context=self._max_context,
            layers=[dict(key=list(k.shape), value=list(v.shape)) for k, v in self._cache],
            kv_payload_bytes=sum(
                t.numel() * t.element_size() for kv in self._cache for t in kv
            ),
        )

    def _validate_tokens(self, tokens):
        if (
            not isinstance(tokens, torch.Tensor)
            or tokens.ndim != 2
            or tokens.dtype != torch.long
            or tokens.device.type != "cpu"
            or 0 in tokens.shape
        ):
            raise ValueError("tokens must be nonempty CPU [B,L] int64 byte IDs")
        if torch.any((tokens < 0) | (tokens > 255)):
            raise ValueError("byte IDs must be in [0,255]")
        if self._batch is not None and tokens.shape[0] != self._batch:
            raise ValueError("batch size cannot change within a cached sequence")
        if self._length + tokens.shape[1] > self._max_context:
            raise ValueError("max_context exceeded; no implicit sliding or position reset")

    @torch.inference_mode()
    def prefill(self, tokens):
        """Start an empty sequence and return logits for every supplied token."""
        if self._length:
            raise ValueError("prefill requires an empty session; use append() or reset()")
        return self._append(tokens)

    @torch.inference_mode()
    def append(self, tokens):
        """Append one token or a causal chunk; return logits for NEW positions only."""
        if not self._length:
            raise ValueError("append requires prefill first")
        return self._append(tokens)

    def _append(self, tokens):
        # Validate and compute transactionally: a failure leaves the old state intact.
        self._validate_tokens(tokens)
        start, chunk = self._length, tokens.shape[1]
        model = self._model
        with torch.autocast(device_type="cpu", enabled=False):
            x = model.embedding(tokens) * math.sqrt(model.config["width"])
            if model.config["use_positions"]:
                x = model.position(x, offset=start)
            query_positions = torch.arange(start, start + chunk)
            key_positions = torch.arange(start + chunk)
            mask = query_positions[:, None] >= key_positions[None, :]
            pending = []
            for index, block in enumerate(model.blocks):
                attention = block.attention
                query = attention._split(attention.q_proj(x))
                key = attention._split(attention.k_proj(x))
                value = attention._split(attention.v_proj(x))
                if start:
                    old_key, old_value = self._cache[index]
                    key = torch.cat((old_key, key), dim=2)
                    value = torch.cat((old_value, value), dim=2)
                heads, _ = scaled_dot_product_attention(query, key, value, allowed_mask=mask)
                merged = heads.transpose(1, 2).reshape(
                    tokens.shape[0], chunk, attention.d_model
                )
                x = block.attention_residual(x, attention.out_proj(merged))
                x = block.ffn_residual(x, block.feed_forward(x))
                pending.append((key, value))
            logits = model.output(x)
        if not torch.isfinite(logits).all():
            raise ValueError("nonfinite inference output; cache not advanced")
        self._cache = tuple(pending)
        self._length += chunk
        self._batch = tokens.shape[0]
        return logits


@torch.inference_mode()
def greedy_cached(model, prompt, *, count: int, max_context: int = 64):
    """Return [B,prompt_length+count] byte IDs, with lowest-ID argmax tie breaking.

    A conservative total-output-length budget is enforced up front, even though
    the final sampled token is not forwarded. This is not the legacy sliding,
    multinomial generate() function and does not silently replace its semantics.
    """
    if type(count) is not int or count < 0:
        raise ValueError("count must be a nonnegative integer")
    session = CachedByteDecoder(model, max_context=max_context)
    session._validate_tokens(prompt)
    if prompt.shape[1] + count > max_context:
        raise ValueError("prompt plus requested output exceeds max_context")
    sequence = prompt.clone()
    if count:
        logits = session.prefill(sequence)
        for index in range(count):
            token = logits[:, -1].argmax(-1, keepdim=True)
            sequence = torch.cat((sequence, token), dim=1)
            if index + 1 < count:
                logits = session.append(token)
    return sequence
