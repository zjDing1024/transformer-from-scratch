# CPU KV cache: same-semantics timing, with explicit limits

2026-11-18 MYT. This benchmark measures the existing tiny-model implementation on
one shared CPU host, not model quality, attention alone or production serving.
All original scientific reports, training/inference code and previous identities
remain unchanged. **This run did NOT recover the original checkpoint identity.**

## Provenance failure and pre-timing amendment

A single unchanged600-step recipe reconstruction was attempted. Initial parameters,
sampling schedule and selected step600 matched the old record, but final tensor
values, serialized file and parts of the training trace did not. The local previous
checkpoint was unavailable. The cause is unestablished; a different host is not
proof of a specific hardware/kernel cause. No further training, ranking, test/SQLite
scoring or hyperparameter change followed this mismatch.

| Identity | Original requested | Actual fixed candidate |
|---|---|---|
|File SHA256|`adab1bc0d8fdf760ab2bf567d6d3ff6dccafaf8f4c0c7dff4aab866133a0a00b`|`baec64d4ee594bc7062d039b3e3e75f58f6dda98cc00c569407e642706ff6601`|
|Tensor digest|`28b7836441b6f150dc677b221a66d43959c7767c9e3ece4d46095efc0a3f6ff3`|`a66094c4eaa1d7cbd7cc7f96b1d854cc9488c16f210e761bded1018b6a31a5b1`|

Before ANY benchmark timing, the protocol was explicitly amended to compare both
paths using that same already-created candidate. Local freeze chronology:
`d0fbee5` records the original-identity plan and failure; `d83491b` freezes the
fallback. These local commits document execution history, not separate published
benchmarks. [Full preflight](../results/cpu-kv-v1/preflight.json) and the unmodified
[recovery failure record](../results/cpu-kv-v1/recovery-mismatch.json) preserve both
the mismatch and all original training fields. This is not a successful restoration
of yesterday's artifact, and not a model-quality improvement or fresh experiment.

## What was actually measured

[Protocol](cpu-kv-protocol.md), [configuration](../experiments/cpu-kv-v1.json),
[raw paired samples, identities, source hashes and workers](../results/cpu-kv-v1/report.json),
[offline recomputed summary](../results/cpu-kv-v1/summary.json).

Python3.12.14, PyTorch2.14.1+cpu, float32, Intel Xeon Platinum8370C at reported2.80GHz.
One intraop and one interop thread; OMP/MKL/OpenBLAS limits1, deterministic algorithms.
GC remained enabled. Nine CPUs were visible, with no affinity or frequency changes.
Other portfolio workers paused CPU work during measurement; this did not reserve
exclusive hardware or eliminate host noise.

The three fixed requests use B=batch rows, P=prompt bytes, G=emitted bytes per row:
B1/P16/G16, B2/P16/G16, B2/P32/G32. Prompts are original short phrases, not benchmark
corpus excerpts. Both paths use the same greedy argmax and absolute full history,
never the old sliding/multinomial generator. They forward G times, with no forward
of the final emitted byte. Complete cached requests reset before every call.

Numerical/greedy parity was checked before AND after timing, not inside the timer.
Every used last-position logit met the preset combined `atol=rtol=1e-5` rule, with
maximum absolute difference3.337860107421875e-6 in each case. Generated IDs matched
exactly for each fixed trajectory. This is not a universal near-tie guarantee.

### Warm complete request results

Each arm had five alternating warmup pairs, then30 alternating-order paired
samples. Each sample is the mean of five full requests. Times include prompt clone,
prefill, greedy argmax, output concatenation, cache reset/validation/concatenation
or full recomputation as appropriate. They exclude imports, checkpoint loading,
integrity hashing, parity checks and cache snapshot construction.

|B / P / G|Full median ms|Full p90 ms|Cache median ms|Cache p90 ms|Median paired cache/full ratio|
|---|---:|---:|---:|---:|---:|
|1 /16 /16|27.103|31.330|19.842|27.406|0.759|
|2 /16 /16|27.997|38.084|19.771|23.536|0.671|
|2 /32 /32|80.628|124.365|40.961|57.318|0.512|

Ratios are medians of within-pair ratios, **not ratios of the two arm medians**.
P90 is nearest-rank p90 of five-request means, not single-request tail latency.
All raw points, including maxima, are retained. For example, the longest full-path
sample reached199.250ms/request mean and cache reached88.219ms/request mean.
These broad ranges reinforce the shared-host limitation. No outlier deletion,
adaptive repeat counts, retry-to-improve or statistical independence claim.

The cache path had lower warm complete-request medians in these three cases.
This does not establish universal speedup, attention-only acceleration, capacity,
GPU performance or production latency. It measures the whole implementation,
including projection/FFN/vocabulary work and Python validation/copying overhead.

### Snapshot construction and prefill are separate costs

New-session constructor median: **3.445ms**, p90 **4.349ms**,30 individual samples
after five untimed constructions. It includes validation and deep-copying the
already-loaded eval model. Destruction is outside this timer. It is not a cold
process/first kernel/model-loading measurement. Do not add independent medians and
call the sum measured cold-request latency. Warm reuse amortizes this ownership
choice; a one-shot caller does not get it for free.

|B / P|Full prefill median ms|Cache reset+prefill median ms|Median paired cache/full ratio|
|---|---:|---:|---:|
|1 /16|1.539|1.507|0.976|
|2 /16|1.563|1.546|1.048|
|2 /32|2.272|2.513|1.255|

Prefill is NOT uniformly better with caching. In B2/P16, the paired ratio and
ratio-of-medians have different directions; report both rather than substituting
one statistic. Snapshot reuse does not remove cached-path validation/reset costs.

## Memory: exact payload, but inconclusive RSS

Eighteen serial child processes were launched: three per arm/case. Each loads the
same checkpoint, keeps the source model alive, and runs one request without warmup.
The cached arm additionally owns its real weight snapshot. Integrity hashing was
kept in the parent; memory readings precede output encoding. Children recorded
`ru_maxrss` after load, setup and request.

**All54 phase readings were identical:294,548KiB.** The measurement has no
request/arm resolution here. Separate process launches do not demonstrate fresh
high-water accounting: inherited/pre-existing peaks across exec or the runtime's
accounting are possible explanations, not established causes. Linux preserves
resource usage statistics across exec; these readings are cumulative process
high-water marks, not instantaneous RSS or allocated bytes. They cannot establish
that cached/full inference use equal memory, that cache costs nothing, or that it
saves memory. No memory comparison conclusion and no replacement run were made.

The deterministic K/V tensor payload, separately verified, is informative:

|B / P / G|Final cached positions|Final K/V payload|
|---|---:|---:|
|1 /16 /16|31|31KiB|
|2 /16 /16|31|62KiB|
|2 /32 /32|63|126KiB|

It excludes weights/snapshot, output tensors, attention matrices, allocator and
Python overhead. In B2/P16/G16, full prefix processes376 positions/row versus31 new
cached positions, and9,176 versus616 dense pre-mask QK entries per row/head/layer.
Append concatenations logically produce720KiB of K/V payload including690KiB of
copied old prefixes. These are analytical path/tensor counts, NOT measured traffic,
bandwidth, FLOPs, peak process memory or speedup. The last output byte is not cached.

## Reproduction and audit

Final verification: **191 CPU tests passed** (163 previous + 28 new benchmark/audit
checks), Ruff/format/compileall passed. Independent review checked candidate
file/tensor hashes, all180 paired records and all18 workers; it also flagged the
inconclusive RSS measurement. Optional NumPy is absent and unused.

Offline evidence audit, with standard Python only and no inference/timing:

```bash
python examples/audit_cpu_benchmark.py
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```

Fresh timing requires the EXACT candidate hashes in the amended manifest and the
pinned runtime. The candidate's weights are not published. You can attempt the
existing original recipe, preserving its exit status and identity comparison:

```bash
# Explicit network/recovery operation; original train/validation and license only.
python examples/recover_inference_checkpoint.py --output-dir results/inference-local
```

On this run it exited nonzero for the original-identity mismatch, while preserving
the candidate and recovery record. Never suppress that failure or assume an
arbitrary rebuild is identical. If your candidate does not match the amended
manifest either, the benchmark intentionally refuses it. Establish a NEW explicitly
labeled protocol/identity for a different candidate rather than changing this frozen
result. Latencies themselves are not expected to reproduce byte-for-byte across
runs or hosts. With this run's matching candidate:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
python examples/benchmark_inference.py \
  --checkpoint results/inference-local/20261114-sinusoidal.pt \
  --output-dir results/cpu-kv-local
```

Output directory must not already exist. All attempted timing samples are retained;
no command silently overwrites previous evidence. The original generator and installed
CLI are unchanged and are not being stopwatch-benchmarked here.

## Reference boundary

The official [PyTorch v2.9.1 Timer source](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/utils/benchmark/utils/timer.py)
informed warmup/thread-control/repetition discipline, under its own
[BSD-style license](https://github.com/pytorch/pytorch/blob/v2.9.1/LICENSE). Version2.9.1
is a design reference, not this run's2.14.1 runtime. No code was copied or imported.
[Linux getrusage documentation](https://man7.org/linux/man-pages/man2/getrusage.2.html)
provides the process high-water/exec/unit semantics. New source/tests are original MIT.
