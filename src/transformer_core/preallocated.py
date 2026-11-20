"""Opt-in bounded K/V storage with logical commit, not byte-for-byte rollback."""

import math

import torch

from .allocation_scopes import scope
from .attention import scaled_dot_product_attention
from .inference import CachedByteDecoder


class PreallocatedByteDecoder(CachedByteDecoder):
    """Snapshot-owned CPU inference, with explicit batch/context storage bounds.

    A failed call preserves committed prefix values, length and batch. It may
    modify INACTIVE backing bytes. Neither failed writes nor reset erase storage.
    Attention sees only [:batch, :, :prospective_length], never the unused tail.
    Reset retains storage and permits a new batch up to max_batch_size. This is
    not concurrent/thread-safe, a GPU/compiled path, or a secure erasure facility.
    The dynamic CachedByteDecoder remains unchanged and is still the default.
    """

    def __init__(self, model, *, max_batch_size: int, max_context: int = 64):
        if type(max_batch_size) is not int or max_batch_size < 1:
            raise ValueError("max_batch_size must be a positive integer")
        self._max_batch_size = max_batch_size
        super().__init__(model, max_context=max_context)
        parameter = next(self._model.parameters())
        storage = []
        with scope("static_reserve"):
            for block in self._model.blocks:
                attention = block.attention
                shape = (max_batch_size, attention.num_heads, max_context, attention.head_dim)
                storage.append(
                    tuple(torch.zeros(shape, dtype=parameter.dtype) for _ in range(2))
                )
        self._storage = tuple(storage)

    def reset(self):
        """Logical reset only: keep snapshot, backing tensors and their private bytes."""
        super().reset()

    def _validate_tokens(self, tokens):
        super()._validate_tokens(tokens)
        if tokens.shape[0] > self._max_batch_size:
            raise ValueError("max_batch_size exceeded; reserve cannot grow")

    def cache_info(self):
        info = super().cache_info()
        storages = {
            t.untyped_storage().data_ptr(): t.untyped_storage().nbytes()
            for pair in self._storage
            for t in pair
        }
        info.update(
            storage_policy="bounded_preallocated_logical_commit",
            max_batch_size=self._max_batch_size,
            backing_storage_bytes=sum(storages.values()),
            backing_storage_count=len(storages),
            backing_shapes=[
                dict(key=list(k.shape), value=list(v.shape)) for k, v in self._storage
            ],
            reset_erases_bytes=False,
        )
        return info

    def _append(self, tokens):
        # Input/budget validation must happen before any suffix write.
        self._validate_tokens(tokens)
        start, chunk, batch = self._length, tokens.shape[1], tokens.shape[0]
        end = start + chunk
        model = self._model
        with torch.autocast(device_type="cpu", enabled=False):
            x = model.embedding(tokens) * math.sqrt(model.config["width"])
            if model.config["use_positions"]:
                x = model.position(x, offset=start)
            query_positions = torch.arange(start, end)
            key_positions = torch.arange(end)
            mask = query_positions[:, None] >= key_positions[None, :]
            pending = []
            for index, block in enumerate(model.blocks):
                attention = block.attention
                query = attention._split(attention.q_proj(x))
                additions = (
                    attention._split(attention.k_proj(x)),
                    attention._split(attention.v_proj(x)),
                )
                for kind, backing, added in zip(
                    ("key", "value"), self._storage[index], additions
                ):
                    with scope(
                        "static_kv_write",
                        layer=index,
                        kind=kind,
                        start=start,
                        chunk=chunk,
                        batch=batch,
                        logical_bytes=added.numel() * added.element_size(),
                    ):
                        backing[:batch, :, start:end].copy_(added)
                # Slicing, not masking a full-capacity value tensor: 0 * NaN would
                # otherwise let a poisoned inactive value contaminate attention.
                key, value = (t[:batch, :, :end] for t in self._storage[index])
                heads, _ = scaled_dot_product_attention(query, key, value, allowed_mask=mask)
                merged = heads.transpose(1, 2).reshape(batch, chunk, attention.d_model)
                x = block.attention_residual(x, attention.out_proj(merged))
                x = block.ffn_residual(x, block.feed_forward(x))
                pending.append((key, value))
            logits = model.output(x)
        if not torch.isfinite(logits).all():
            raise ValueError("nonfinite inference output; logical cache not advanced")
        # Only Python state is published now. On failure, dirty suffix remains
        # private and a shorter retry must neither read it nor count it as valid.
        self._cache = tuple(pending)
        self._length = end
        self._batch = batch
        return logits
