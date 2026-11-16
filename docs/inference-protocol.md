# Bounded KV-cache correctness protocol

2026-11-17 MYT. Base public commit: `1f88bdb9eb83f3939611f0b37af8ab2dfa6192f2`.
This is an engineering validation, not a new training or held-out research study.
The original models, training/selection code, protocols and prior results stay unchanged.

## Scope fixed before checkpoint equivalence run

- Original `ByteLanguageModel` architecture; independently calculated full-prefix
  forward is the oracle. New code stores each layer's projected K/V, appends only
  new positions and uses absolute offset positions plus a rectangular causal mask.
- Public API: snapshot-owned `CachedByteDecoder.prefill/append/reset/cache_info`
  and `greedy_cached`. Fixed CPU float32/64, equal-length nonempty unpadded batches.
  No truncation, sliding, eviction, EOS, sampling, beam search, GPU or service API.
- A source model must already be in eval mode. Deep-copy its weights/config once
  per session; later source training/weight loading must not change that session.
  Cache state is private, and failed append operations must not advance it.
- Explicit `max_context` rejects overflow. Original weights trained with context64;
  validation stays within64. Allowing a caller's larger budget is not evidence of
  larger-context model quality or tested equivalence beyond these cases.
- Numerical acceptance: float32 `atol=rtol=1e-5`, float64 `1e-12`.
  Report measured maximum absolute errors; logits need NOT be bitwise identical.
  Greedy byte IDs must be exactly equal in tested cases. No universal argmax
  guarantee for unseen near-ties follows from floating-point closeness.

## Inputs and checkpoint identity

Synthetic tests cover position encoding on/off, float32/64, batches1/3, whole,
individual-token and irregular chunks. Additional tests cover dtype/input/lifecycle
errors, mutation isolation, cache payload accounting and new-position-only QKV work.

For real trained-weight evidence, restore ONLY original seed20261114/sinusoidal
using unchanged600-step train/validation recipe and PyTorch2.14.1+cpu, one CPU
thread, lower process priority. The prior local file is absent. Verify serialized
file SHA256 AND original tensor digest against immutable Nov14 selection. All
original training fields except wall time must also match. This is reconstruction,
not a new independent training run or better model. Download only original Flask
train/validation files and BSD-3-Clause license, never test or SQLite documents.

Real-weight equivalence probes, not accuracy evaluations:

1. Two64-token synthetic byte-ID rows: `(17*i + 31*row) % 256`.
   Chunk schedules `[64]`,64 one-token chunks,`[5,13,1,17,28]`,`[16,48]`.
   Compare EVERY new-position logit both to a fresh full-prefix forward and the
   matching slice of one full64-position forward. Record each chunk's maximum.
2. Greedy generation of16 bytes from each original16-byte prompt
   `The application ` and `HTTP request:   `, compared to full-history greedy.
   No downloaded prose is needed. Exact byte sequences and hash are reported.
3. At64 positions each K/V tensor must be `[2,4,64,16]`; tensor payload only
   is131072 bytes for2 layers/float32. This excludes weight snapshot, allocations,
   attention matrices and Python overhead, so it is NOT a process-memory benchmark.
4. New-process replay must reproduce the report byte-for-byte in this environment.

No latency/throughput benchmark or speedup claim. Concatenating cache tensors,
copying weights and explicit attention matrices are deliberately simple, not a
production memory-management strategy. No new hyperparameter/model selection,
SQLite losses or test-driven training occurs.

## Design references (not copied code)

- [Hugging Face Transformers v4.57.1 cache explanation](https://huggingface.co/docs/transformers/v4.57.1/cache_explanation),
  [Apache-2.0 repository license](https://github.com/huggingface/transformers/blob/v4.57.1/LICENSE):
  conceptual K/V reuse, growing attention mask and cache positions. No HF code,
  model weights, dependency or class implementation is copied/imported.
- [PyTorch inference mode](https://docs.pytorch.org/docs/2.14/generated/torch.autograd.grad_mode.inference_mode.html):
  autograd-free execution is distinct from setting eval mode. Existing PyTorch
  dependency remains under its own BSD-style license; new code is original MIT.
