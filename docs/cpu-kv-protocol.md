# CPU KV-cache timing and memory protocol v1

Frozen 2026-11-18 MYT before any benchmark timing. Base public main:
`1d7a4d4f2e085c1f0abc447e3f063590f6c23c71`. Configuration is
[`cpu-kv-v1.json`](../experiments/cpu-kv-v1.json). Original inference, weights,
training and scientific reports are unchanged. New benchmark code is original MIT.

## Question, workload and acceptance

For this tiny CPU model and specified finite requests, what does the existing
snapshot-owned cache path cost compared with full-prefix recomputation? This is
not an attention-only benchmark or evidence of general speedup/production capacity.
The original-identity preflight FAILED before any benchmark timing. One attempt to
rebuild seed20261114/sinusoidal with the unchanged600-step recipe had the same
initial-state and sampling-schedule digests and selected step600, but different
final weights/artifact and training trace. No retained original file was available.
The cause is not established; do not attribute it to hardware or promise cross-host
bitwise training reproducibility. Full expected/actual hashes and differences are
preserved in the preflight/recovery-mismatch artifacts.

AMENDMENT FROZEN BEFORE TIMING: use exactly that ONE already-built candidate for
BOTH arms, with its own file/tensor hashes fixed in JSON. No second training run,
ranking, quality testing, hyperparameter adjustment or outcome-based selection.
This is same-weight engineering timing of a known DIFFERENT checkpoint, not a
successful original-artifact reconstruction or a reproduction of yesterday's exact
trained-model benchmark. Verify the amended candidate's two digests before timing.
The provisional original-identity protocol and failed recovery are retained in
local commit history; only this explicit amended protocol may proceed.

Cases are fixed: batch1 and batch2 with16-byte prompts and16 emitted bytes, plus
batch2 with32-byte prompts and32 emitted bytes. Prompt bytes are fixed in JSON;
the longer case repeats each16-byte phrase twice. Both arms use float32, greedy
last-position argmax, untruncated absolute prefix positions, total output <=64,
and DO NOT forward the final emitted byte. Hence16/32 forwards total, respectively,
not17/33. No EOS or variable-length masks. Full-prefix baseline is not the legacy
sliding/multinomial generator. Both request functions clone the prompt, validate
shared request constraints and concatenate output bytes.

Before and after timing, compare full/cached output IDs exactly and every used
last-position logit under `atol=rtol=1e-5`; abort on failure. Check final cache length
and payload and count projection work with tests, outside timing. Preserve all
attempts if an implementation error requires correction; never rerun for a better
number. No correctness assertions, integrity hashes or JSON reporting inside timers.
Existing model/session input and finite-output checks remain part of their paths.

## Timed boundaries and order

1. **New-session snapshot construction:** an already-loaded eval source model,
   including constructor validation, deep copy, freezing parameters and empty
   reset. Excludes loading, imports, model hashing and destruction after timer stop.
   Five untimed constructions, then30 individual construction samples. This is a
   cold SESSION, not a fresh process, cold hardware cache or first-ever kernel call.
2. **Warm reset + prefill vs full prefill:** reuse one weight snapshot; cached
   function resets sequence state on EVERY call then processes the same prompt.
   No greedy argmax/decode. Full arm forwards the identical prompt. Reset/freeing
   existing cached tensors is included in the cached path.
3. **Warm complete request:** reuse snapshot, but reset state every call. Includes
   prompt clone, prefill, all argmax/byte concatenations, G-1 cache appends or full
   prefix recomputations. Includes cache validation/concatenation overhead, excludes
   snapshot construction. Never repeatedly append to one uncontrolled growing cache.

Per case and paired boundary: five untimed alternating warmup pairs, then30 paired
replicates. Order alternates full/cache then cache/full. Each measured arm sample
is the elapsed wall time for five complete identical calls divided by five. Use
`perf_counter_ns`; CPU execution is synchronous, no CUDA synchronization exists.
Record every raw paired sample/order plus median, nearest-rank p90, min and max.
These are distributions of five-call means, NOT individual-request tail latency.
Report paired cache/full ratios descriptively, not confidence intervals, a universal
speedup or independent experimental replications. No outlier removal, adaptive
repeat counts or post-result budget changes. Do not infer cold-request time by
adding independent medians. Preserve all measured scenarios, including slowdowns.

Fix PyTorch2.14.1+cpu, intraop/inter-op1, deterministic algorithms, external
OMP/MKL/OpenBLAS thread limits1. Report Python/runtime/platform/CPU/affinity and
clock resolution. No affinity/security/system setting changes. Ask other project
workers to avoid CPU work in the measurement window; this is still a shared cloud
host, not exclusive hardware. No claim of complete isolation or fixed CPU frequency.
GC remains in its normal enabled mode and is reported, not tuned after results.

## Independent memory workers

After timings, for each case launch three fresh serial Linux processes per arm,
with alternating arm order. Same module imports and checkpoint loading, same source
model retained until measurement ends. Cached arm additionally owns its required
snapshot; this real API cost is not hidden by deleting the source model. Baseline
has no artificial extra weight copy. No warmup or repeated inference in a memory
worker: one request only. Parent verifies checkpoint integrity; child does not
read_bytes/hash it. Each worker records RUSAGE_SELF ru_maxrss (Linux KiB) after
load, after session setup and after request, BEFORE output encoding/reporting.
Never use CLI process measurement, which adds checkpoint hashing/report overhead.

ru_maxrss is cumulative PROCESS peak RSS, including interpreter, imports, load,
allocator history, source and cache model weights and scratch tensors. A difference
between phase high-water marks is only an increase over the earlier peak, not
allocated bytes. No increase is inconclusive. Per-worker peaks can vary and do not
prove cache memory savings. Keep per-worker values and ranges; do not subtract
arm medians as if they were exact inference allocations. Linux-only memory path.

Separately report exact final K/V tensor payload and analytical logical path counts:
positions/projected rows, dense pre-mask QK entries per row/head/layer, concatenated
K/V output payload and old-prefix payload copied. These are not measured memory
traffic, bandwidth, FLOPs, RSS or latency. For batch2/P16/G16:16 calls, cache final31
positions/62KiB, full376 vs cached31 positions per row, dense QK9176 vs616,
concatenation output720KiB including690KiB old prefixes. Output byte32 is not cached.

## Sources, version separation and licensing

[PyTorch v2.9.1 Timer](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/utils/benchmark/utils/timer.py)
informed warmup/thread control/repeated-measurement discipline; this is a reference
version, NOT the runtime version. Its
[BSD-style license](https://github.com/pytorch/pytorch/blob/v2.9.1/LICENSE) remains
upstream. No Timer implementation is copied or imported; the harness is original.
[Linux getrusage](https://man7.org/linux/man-pages/man2/getrusage.2.html) defines
ru_maxrss process high-water semantics and KiB units. No documentation text copied.
