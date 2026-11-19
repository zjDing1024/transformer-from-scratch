# CPU allocation attribution: which operation owns the bytes?

2026-11-20 MYT. A runnable, opt-in diagnostic on the existing **untrained portable
fixture**, not a new training result, latency benchmark or memory-saving claim.
The new evidence is observed operator/site attribution. Logical concatenation
formulas already existed; this report does not count those formulas as new work.

## Run and reproduce

```bash
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install .
# Full fixed-fixture profiling, parity and controls, no download or old checkpoint:
transformer-allocation-check --output allocation-local.json
# Small real CPU-profiler calibration only:
transformer-allocation-check --smoke --output allocation-smoke.json
# Recompute assertions from the published record, no inference:
transformer-allocation-check --audit results/operator-allocation-v1/report.json \
  --output allocation-audit.json
```

`python -m transformer_diagnostics.allocation` is equivalent. Existing output paths
are refused. A calibration failure returns explicit unavailable status; detected
identity, parity or attribution contradictions fail with a saved failure receipt,
not a successful zero. Smoke mode does **not** run the model. The offline audit
checks record consistency and source hashes, not a new physical measurement.

[Protocol](operator-allocation-protocol.md) was committed locally as
`a4cee8ae96164526fe3a7ac86ff8b70ad8c405b9` before the one primary run. This is a local
pre-run freeze, not prior public registration. [Raw observations](../results/operator-allocation-v1/report.json),
[offline audit](../results/operator-allocation-v1/audit.json) and
[verification](../results/operator-allocation-v1/verification.json) retain the evidence.

## Scope and accounting

Optional ContextVar ranges surround the **existing** finite-weight check, snapshot
deepcopy, prefill/append, each K/V concat and output-token concat. They do not replace
the inference algorithm or monkeypatch Torch. The default leaves tracing off; its
no-op Python contexts are not claimed to have zero runtime overhead. Labels retain
only scalar site/layer/start/chunk metadata, never tensors or tensor values.

The CPU profiler uses `profile_memory=True`, `record_shapes=False`, `with_stack=False`.
Shape/stack recording was deliberately disabled: profiler shape capture can retain
references and alter copy behavior. Each event's signed **self** memory is assigned
to its nearest annotated range exactly once, including any range's own self bytes.
Positive and negative self bytes are retained separately. Clone-inclusive values
are diagnostic fields only and are **not added** to their child allocations.

A positive-self sum is the sum of positive event-level memory accounting, not a
complete allocator transaction log, maximum live storage, physical memory traffic,
RSS, resident pages or process footprint. Some operators allocate and free within
one event. Unscoped operator counts are retained but not turned into a total model
memory estimate. In particular, this does not compare total full-prefix versus
cached allocation or diagnose the earlier process-level RSS plateau.

## Actual run

Python 3.12.14, PyTorch 2.14.1+cpu, Linux 6.18.57 x86-64/glibc 2.41, CPU float32,
one intraop/inter-op thread, deterministic algorithms. There was one primary run,
with no outcome-driven rerun and no historical training, timing or RSS experiment.

The same `byte-lm-portable-f32-v1` identity was verified before inference:

- Canonical artifact, 403,088 bytes:
  `72b3fcbbd1233973e8dadd98c7ab8f3d0d33c8b4b4f272d0e55c24cfb381e597`.
- Tensor SHA256:
  `9ab16b65155a55a2860c6f851f0d69a6ad072885b1342ac0fa4c754d9096813e`.

Complete parity records with scopes disabled/enabled matched outside profiling.
The existing full-sequence, chunked and per-step reference assertions passed;
maximum recorded greedy logit difference remained 1.0728836059570312e-6 within
combined atol=rtol=1e-5. Profiled request output IDs matched their unprofiled oracle
exactly; the final emitted byte was not forwarded. Source and snapshot tensor
digests were unchanged after the requests. These are finite fixture checks, not
language quality or universal output-equivalence guarantees.

### Cold snapshot decomposition

| Site | Observed operator evidence | Sum of positive self bytes |
| --- | --- | ---: |
| Finite-weight validation | 35 `aten::isfinite` calls and their actual producer decomposition | 699,951 |
| Snapshot deepcopy | 35 `aten::clone`, with allocation attributed to 35 child `aten::empty_strided` events | 399,872 |

All 35 clone events had zero self bytes and positive inclusive bytes. Their child
allocations summed to the exact 99,968-element float32 parameter payload. Adding
clone-inclusive bytes again would double-count the storage. Inventory verified
identical values and distinct source/snapshot parameter pointers, with no buffers.
Validation allocation is reported separately, not mislabeled as snapshot copying.
The observed validation number is exploratory/runtime-specific, not a portable
threshold. These quantities do not explain the previous mixed setup-RSS interval.

### Warm-snapshot request attribution

One observed request per arm/case; no request-latency comparison:

| B / P / G | Arm | K/V cat events | K/V cat positive self bytes | Token cat events | Token cat positive self bytes | Final K/V payload |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2 / 16 / 16 | cached | 60 | 737,280 | 16 | 6,272 | 63,488 |
| 2 / 16 / 16 | full prefix | 0 | 0 | 16 | 6,272 | 0 |
| 2 / 32 / 32 | cached | 124 | 3,047,424 | 32 | 24,832 | 129,024 |
| 2 / 32 / 32 | full prefix | 0 | 0 | 32 | 24,832 | 0 |

K/V cat sites are float32 rank-four operations on the sequence axis; output-token
cats are separate int64 rank-two operations. The report preserves every selected
cat event, its layer/key-or-value/start/chunk attribution and observed self bytes.
Each K/V output at length T accounts for 512*T bytes here, four times per append.
Measured sums match the independently fixed expectations, but they are **cumulative
operator outputs**, not the final cache size. Old-prefix logical portions remain
706,560 / 2,983,936 bytes; those portions are calculated, not measured DRAM reads.

Transactional pending state retains the old cache until all layers succeed, so old
and replacement cache storage can coexist. Final payload therefore cannot stand in
for transient live memory, just as cumulative allocation cannot stand in for peak.

### Controls

- Actual float32/int64 cat controls: 40 / 80 positive self bytes; view: 0; clone: 24.
- Disabled scopes still observed a cat, but site attribution was explicitly
  unavailable, never interpreted as an observed zero.
- Full-prefix requests: no K/V-site events; the nonzero token cats remained visible.
- G=1 for both prompts and arms: no K/V cat, one token cat each (272 / 528 bytes).
  Cached prefill retained 32,768 / 65,536 bytes of K/V, demonstrating that zero cat
  events do **not** mean zero cache storage.
- Prefill-only: no K/V or token concat. Reset-only: no clone/cat anywhere in the
  captured scoped or unscoped operator counts, empty cache, unchanged snapshot
  pointers. This is not evidence that reset returns memory to the operating system.

## Tests, installation and historical evidence

239 local tests passed, including 29 new attribution/audit checks. Ruff, format and
compile checks passed. An offline-built wheel was installed into a separate target
and exercised from outside the checkout: actual profiler smoke controls, actual
portable-fixture parity, and the full published-record audit. It did not repeat the
primary full-fixture allocation experiment. CI includes those tiny actual profiler
controls, the existing actual parity command and offline report checks, not merely
a copied success JSON. Remote CI status is for the publisher to verify.

Opt-in source instrumentation changes `inference.py` and `benchmark.py`. All prior
reports, hashes and failure records are preserved, not silently refreshed. The
historical benchmark audit now explicitly checks the two byte-identical archived
sources in [cpu-kv-v1-source](../experiments/cpu-kv-v1-source/README.md). Other source
checks remain in place. Thus historical evidence verifies the old source bytes;
it does **not** claim that the old timings measured today's code. Current default
and enabled inference paths are covered by new/current correctness checks.

## Official references and licensing

PyTorch v2.14.1 resolves to `5c4886908584029761b579af026dcfb627c84070`.
The [official profiler documentation](https://docs.pytorch.org/docs/2.14/profiler.html),
[pinned profiler implementation](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/profiler/profiler.py),
[self-memory accounting](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/autograd/profiler_util.py)
and [Parameter deepcopy](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/nn/parameter.py)
were conceptual references. [Source receipts](../results/operator-allocation-v1/source-references.json)
record fetched byte sizes and SHA256 values, including the
[BSD-style license](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/LICENSE).
No third-party implementation was copied; new scopes, analyzer, tests and CLI are
original MIT code using the existing dependency. No private data or paid resources.
