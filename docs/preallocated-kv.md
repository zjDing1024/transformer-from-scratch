# Bounded preallocated K/V, with a logical transaction

2026-11-21 MYT. Original opt-in CPU inference path, not a default replacement.
The unchanged `CachedByteDecoder` keeps its dynamic concatenation behavior.
This follow-up addresses growth mechanics and failure semantics, not model
quality, historical checkpoint recovery, latency or process memory.

## Use the class or installed checker

```bash
python -m pip install .
# Actual fixed-fixture parity, failure controls and four small CPU profiles:
transformer-preallocated-check --output preallocated-local.json
# Actual fixture checks only, without profiling (also the CI entry point):
transformer-preallocated-check --check-only --output preallocated-parity.json
# Audit the saved evidence without rerunning the model or profiler:
transformer-preallocated-check --audit results/preallocated-kv-v1/report.json \
  --output preallocated-audit.json
```

The equivalent module is `python -m transformer_diagnostics.preallocated`.
Outputs must be new paths; prior evidence is never overwritten. Failed assertions
produce a failure receipt and exit code 2. Run Python normally, not with `-O`:
the diagnostic assertions are intentional fail-closed checks. The offline audit
checks consistency and source fingerprints, not whether observations really ran;
actual inference is the default or `--check-only` path. No external download,
trained checkpoint, optional NumPy, Transformers package or GPU is needed.

```python
import torch
from transformer_core.portable_fixture import verified_fixture
from transformer_core.preallocated import PreallocatedByteDecoder

model, canonical_bytes, identity = verified_fixture()  # untrained fixture
session = PreallocatedByteDecoder(model, max_batch_size=2, max_context=64)
logits = session.prefill(torch.tensor([[65, 66], [67, 68]]))
logits = session.append(torch.tensor([[69], [70]]))
print(session.cache_info())  # active payload and distinct backing bytes
session.reset()  # retains backing storage, not a secure erase
```

The source model must be in eval mode, on CPU, float32 or float64, with finite
parameters. Same-length, nonempty byte-token rows only, no padding or sliding.
CPU must remain the default device; non-CPU global `torch.set_default_device`
settings are outside this contract. Construction deep-copies the validated
model. Batch capacity is explicit and cannot grow. The positive `max_context` is
a hard caller budget, not an architecture limit. The default and frozen fixture
budget are 64; behavior beyond 64 is not validated by this report. Batch size binds
on the first successful prefill and remains fixed until reset. Source
weights modified later do not alter the session snapshot. No public mutable
cache tensors are exposed. Private attributes are used only for white-box tests.

## What is transactional, and what is not

At construction, each layer reserves separate zero-initialized K and V tensors.
Each append writes only the not-yet-committed suffix, computes attention from
`[:batch, :, :prospective_length]`, and publishes the new visible views, length
and batch only after the final output passes a finite check. Thus a late failure
can change inactive backing bytes, but cannot change the already committed
prefix or bind a previously empty session to the failed batch.

A shorter, different retry must overwrite its own range and ignore the rest of
the dirty tail. Masking a full-capacity tensor is not enough: multiplying a zero
attention probability by a poisoned NaN value would still yield NaN. Slicing
excludes inactive sequence positions and inactive batch rows before attention.
Logical length is never inferred from whether K/V values happen to be nonzero.

`reset()` releases logical views and batch binding, but retains the model,
capacity, backing pointers and backing bytes. This is **logical rollback and
logical reset, not whole-buffer rollback or erasure**. Sessions are not concurrent
or thread-safe. No GPU, compilation, serving, beam search, dropout training or
untrusted-hook support is claimed. Snapshot construction can itself fail; an
unconstructed object is not a reusable session.

## Frozen first-run correctness evidence

[Protocol](preallocated-kv-protocol.md) and implementation were frozen in local
commit `1096745a31a48a761c1d1d57061693a44a55891c`, based on public
`70c942b57a155529bc31931cc8cd776848d122d6`, before the one primary report run.
This was a local source freeze, not public preregistration. All first-run checks
passed; no failed primary run or selected rerun was discarded.

The exact untrained `byte-lm-portable-f32-v1` fixture has 99,968 parameters:

- Canonical artifact SHA256:
  `72b3fcbbd1233973e8dadd98c7ab8f3d0d33c8b4b4f272d0e55c24cfb381e597`.
- Tensor SHA256:
  `9ab16b65155a55a2860c6f851f0d69a6ad072885b1342ac0fa4c754d9096813e`.
- No training, model selection or historical-checkpoint identity claim.

Two-row teacher-forced checks cover all 64 positions with one chunk, 64 single
steps and `[5, 13, 1, 17, 28]`, against both dynamic cache and full-prefix forward.
The maximum absolute difference was `9.5367431640625e-7`. Original B2/P16/G16
and B2/P32/G32 greedy outputs had exactly equal IDs; maximum logit differences
were `7.152557373046875e-7` and `1.0728836059570312e-6`, within combined
`atol=rtol=1e-5`. No forward consumes the final generated byte, so final active
lengths are 31 and 63, although returned sequences have lengths 32 and 64.

Four late-fault controls inject either a final-output exception or NaNs, after
both layer value projections: two on a populated B2/P16 session, two on an empty
B2/P4 prefill. Every case preserves committed K/V, metadata and pointers exactly,
and every case demonstrably changes inactive backing bytes. Populated failures
retry a different one-token chunk then two tokens without reset; failed first
prefills retry B1/P2 without reset. Both match clean dynamic/full references,
with maximum retry error `5.960464477539062e-7`. Reset retains exact bytes and
pointers, permits B1/P3, and has zero active payload. NaN versus finite poisoned
inactive tails/rows produce equal valid-prefix values and outputs.

## Measured writes, capacity and allocation are different quantities

Runtime: Python 3.12.14, PyTorch 2.14.1+cpu, one intra-op and one inter-op thread,
CPU float32, deterministic algorithms. An actual constructor profile, two warm
request profiles and one direct-copy calibration were run. They are not timing
samples. Calibration copies 24 B into an existing tensor: one `aten::copy_`,
zero positive self allocation. The source and snapshot tensor digests remain
unchanged after the measured requests.

| Observation | B2/P16/G16 | B2/P32/G32 |
| --- | ---: | ---: |
| Actual distinct backing storage | 131,072 B | 131,072 B |
| Final active-view payload | 63,488 B | 129,024 B |
| Actual K/V `copy_` events, including prefill | 64 | 128 |
| Logical payload written at those sites | 63,488 B | 129,024 B |
| Positive self allocation at K/V write sites | 0 B | 0 B |
| K/V `cat` events | 0 | 0 |
| Output-token `cat` events | 16 | 32 |
| Positive self allocation for output-token cats | 6,272 B | 24,832 B |

The four backing tensors are independently deduplicated by actual untyped
storage identity, each 32,768 B. Their pointers remain stable. Constructor
allocation at `static_reserve` is 131,072 B, matching the actual backing storage.
Active views alias that reserve and are not added to it. Write payload derives
from the observed copy events and their scalar dimensions; it is **not measured
hardware memory traffic**. Zero self allocation does not mean zero writes or
zero memory cost. Projection, attention, finite-check and output allocations
still occur; a fixed reserve can consume more backing bytes than an active
prefix. No latency, RSS, peak footprint or memory-saving comparison is made.
Historical dynamic allocation results are not rerun or presented as a new paired
performance comparison. Profiler overhead makes these traces unsuitable for
comparison with historical latency measurements.

Nearest labeled scope owns each operator's self allocation; inclusive values
are diagnostic only and must not be summed with child self values. Full producer
and copy-event records are in the [report](../results/preallocated-kv-v1/report.json),
SHA256 `71499492bd45906dd0e16f3f045cb6d1b6f078bc4a73f3f715d31a90aadaa6e3`.
[Offline audit](../results/preallocated-kv-v1/audit.json) independently recomputes
the event metadata, counts and byte totals and checks frozen runtime source hashes.

## Tests, packaging and preserved history

`tests/test_preallocated.py` covers float32/64, positions on/off, B1/B2,
three chunk schedules (`[16]`, 16 singles, `[3, 5, 1, 7]`), hard caps, invalid inputs
before writes, snapshot isolation,
late faults, shorter retries, poisoned tails, reset and zero K/V. Separate report
tests mutate evidence to ensure the audit rejects altered identity, source,
capacity, copy events, token cats and transaction controls. CI runs the actual
fixture's `--check-only` in addition to tests. Cross-machine tests require the
runtime tolerance, exact IDs and transaction facts, not bit-identical floating
error maxima from this machine. It does not merely validate a
published JSON, and does not rerun the primary resource profiles.

The full suite passes **304 tests** (239 prior, 46 new cache tests and 19 new
report/CLI tests); independent read-only QA also ran the 46 cache tests. Ruff,
format checks, compileall and diff checks pass. The only test warning is absent
optional NumPy, which this implementation does not use.

The [verification receipt](../results/preallocated-kv-v1/verification.json) records
full-suite/lint, historical-byte checks and an installed wheel run outside the
checkout. That outside run performs actual parity/fault controls plus a separate
offline audit; it does not repeat primary profiling. Historical runtime sources,
benchmark failure records and all earlier result JSON remain unchanged. Only the
old broken HF license citation was corrected from a tag-object ID to its peeled
source commit; no historical numerical evidence was rewritten.

## Design reference and license

Hugging Face's [StaticLayer at v4.57.1 source commit](https://github.com/huggingface/transformers/blob/8cb5963cc22174954e7dca2c0a3320b7dc2f4edc/src/transformers/cache_utils.py#L248)
was a conceptual reference for reserving K/V storage and in-place updates, not
for this model-level transaction contract. Its implementation and reset policy
are not copied. The version's annotated tag object differs from its peeled source
commit; immutable source/license URLs, byte counts and independently checked
SHA256 receipts are [recorded here](../results/preallocated-kv-v1/source-references.json).
Upstream is [Apache-2.0](https://github.com/huggingface/transformers/blob/8cb5963cc22174954e7dca2c0a3320b7dc2f4edc/LICENSE).
All new project code/tests are original MIT; no upstream source, external weights
or documentation body is bundled. Existing PyTorch profiler reference/license
receipts remain in the [allocation guide](operator-allocation.md).
