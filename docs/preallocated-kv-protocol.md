# Bounded preallocated K/V protocol v1

2026-11-21 MYT. Base `70c942b57a155529bc31931cc8cd776848d122d6`.
Original opt-in class; dynamic inference/benchmark/default CLI unchanged. No
training, historical checkpoint reconstruction, latency, RSS or GPU experiment.
Use the same locked portable fixture, CPU float32, deterministic, one thread.

## Contract fixed before the primary run

Explicit max_batch_size and max_context; allocate fixed zero-initialized K/V
reserve at construction. No growth, sliding, eviction or padding. Suffix-only
writes; attention slices to the prospective committed length and current batch.
Publish visible prefix/length/batch only after all layers, output and finite check.
Late failure may leave INACTIVE suffix bytes changed. This is logical rollback,
not whole-buffer byte rollback, and never uses nonzero slots to infer validity.
Reset clears logical state only, preserving backing bytes/pointers/model snapshot;
it permits a new batch within capacity. Not erasure, thread safety or service readiness.

## Frozen checks

- Complete two-row, 64-token teacher-forced logits, schedules [64], 64 singles and
  [5,13,1,17,28], compared with unchanged dynamic/full-prefix references.
- Original B2/P16/G16 and B2/P32/G32 greedy trajectories, no last-byte forward,
  full per-step logits, combined atol=rtol=1e-5 and exact selected IDs.
- Late fault modes: raise and NaN at the final output hook, proven reached after
  both layer value projections. Starting B2/P16, fail four different tokens;
  metadata/committed K/V exact unchanged, then retry one different token and two
  more, without reset. Compare each result to a clean dynamic/full-prefix oracle.
- Failed first B2/P4 prefill followed by B1/P2 without reset; no premature batch
  binding. Reset a populated session then B1/P3 and compare fresh reference.
- Two equal-prefix sessions with NaN versus finite poison in inactive positions
  and inactive batch rows; identical retry outputs and valid-prefix values.
- Hard token/batch caps, early invalid input, dtype/mode restrictions, source
  snapshot isolation, zero-valued K/V and metadata non-aliasing in tests.

## Resource check, not a latency/footprint result

Deduplicate actual untyped backing storages, measure nbytes, and verify fixed
pointers across valid writes, failures and reset. Record active-view numel bytes
separately. B2/C64 float32 reserve predicts 131,072 B; active after the two greedy
cases predicts 63,488 / 129,024 B. No sum of aliased views as backing capacity.

Use one scoped constructor profile and one warm request profile per frozen case
with CPU profile_memory, no shape/stack tracing. Scalar-only write site labels;
independently count actual aten::copy_ events and producer self bytes. For each
case expect four K/V writes per forward: 64 / 128 including prefill, logical write
payload 63,488 / 129,024 B. Zero K/V concat. Output-token cat remains 16 / 32
(6,272 / 24,832 B). Assert direct copy_ calibration (24 B destination, zero new
self allocation) before interpreting counts. No new dynamic allocation experiment;
old measurements remain historical. Zero allocation does not mean zero writes.

Default source/test paths stay intact; publish separate compact report/CLI/audit,
source hashes and wheel checks. Freeze this protocol and source in a LOCAL commit
before primary measurement; local freeze is not public preregistration. Preserve
failures, do not select reruns. Pinned HF StaticLayer is design reference only,
Apache-2.0; no imported/copied source, third-party weights or paid compute.
