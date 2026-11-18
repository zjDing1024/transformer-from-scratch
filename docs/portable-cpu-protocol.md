# Portable CPU parity and phase-memory protocol v1

2026-11-19 MYT. Base: `518d9fc09d64e19f37cf245e82574abb5521113f`.
No historical training/latency rerun, GPU, downloaded corpus or paid resource.
Old recovery/plateau reports remain unchanged. This is a separately named
engineering fixture, not an original trained checkpoint or model-quality result.

## Fixture

`byte-lm-portable-f32-v1`: width64/heads4/hidden128/layers2, sinusoidal positions,
99,968 parameters. Every float32 tensor is overwritten from SHAKE256 of the UTF-8
fixture name + NUL + tensor name. Successive little-endian unsigned16 codes minus
32768 are divided by powers of two: embeddings2**15, matrices2**18, biases2**21;
norm weights add1 to the2**21 fraction. All values are exactly float32-representable.
Constructor RNG state is restored and random initial values have no effect.
No weight/seed selection based on outputs and no training. Packaged recipe/identity:
`transformer_diagnostics/data/portable-v1.json`.

Canonical artifact: magic prefix, little-endian32 header length, ASCII sorted-key
compact JSON header (configuration and sorted names/shapes), then sorted
little-endian float32 payloads. NOT PyTorch pickle/zip. BOTH identities must match:
-403,088 canonical bytes; artifact72b3fcbbd1233973e8dadd98c7ab8f3d0d33c8b4b4f272d0e55c24cfb381e597
- tensor9ab16b65155a55a2860c6f851f0d69a6ad072885b1342ac0fa4c754d9096813e

Only recipe/small manifest are published; bytes regenerate without training,
network or unavailable `.pt` files. This does not restore historical trained
weights or promise cross-platform logit/PyTorch-serialization identity.

## Parity before memory

CPU float32, intraop/inter-op1, deterministic algorithms. Compare all logits for
two64-byte formula-generated rows under `[64]`,64 single-token chunks and
`[5,13,1,17,28]` against unchanged full-prefix/full64 oracles. Fixed greedy cases
B2/P16/G16 and B2/P32/G32 use original short phrases (longer repeats each twice).
All used logits must meet combined atol=rtol1e-5 and greedy IDs must match exactly.
No final-byte forward; cache lengths/payloads must match analytical counts.
Abort on mismatch, never change the fixture to obtain parity. No quality claim.

## Memory boundaries

Standard-library launcher imports NO Torch. Four serial memory workers: one per
arm/case, full/cache order reversed for case2. Each independently builds/verifies
the fixture; source model stays alive. Cached arm additionally owns its real
snapshot, with no artificial baseline copy or repeated request/warmup. No timing.

Record absolute VmRSS/VmHWM and ru_maxrss after model construction, after
session/prompt setup, immediately after own-process clear_refs mode5 reset, and
after ONE full request while model/session/output remain alive. Hashing, fixture
construction and snapshot copying are setup, outside the request window. Preserve
setup absolute footprint AND request-window absolute peak, not merely a delta.
The only write is `/proc/self/clear_refs` mode5 for this process's peak accounting,
not another PID, memory protections/data or system-wide/security settings.

Reset must succeed and post-reset VmHWM must be within2MiB of VmRSS. A separate
positive-control window AFTER request recording resets again, touches each page
of a private16MiB anonymous mapping, reads while live, releases, reads again.
Live VmRSS and VmHWM must each rise at least12MiB over their own baseline. This
coarse75% bar tolerates reporting granularity, not precision claims. Never replace
request readings with control readings. Preserve raw snapshots for all phases.

If Linux/proc fields/reset are absent, denied or malformed, or control does not
pass, mark request-memory verification explicitly unavailable with reason, while
retaining available observations/parity. Never fabricate zero or substitute
ru_maxrss for a failed VmHWM window. Metric failure is not parity failure.

Process metrics include Python/imports/libraries and fixture-building residue, not
exclusive model/tensor allocation. RSS reporting can be coarse/non-monotonic;
negative differences remain visible. A16MiB control does NOT establish resolution
of62/126KiB K/V payloads or precise cache-memory savings. ru_maxrss can retain
prior/exec peaks: report it separately, never as the phase-local endpoint. This
run does not establish the cause of the old plateau or equality/savings in memory.

## Runnable boundary and references

Installed wheel includes its manifest and runs outside the checkout.
`transformer-cpu-check --parity-only --output report.json` performs CPU parity;
default also attempts Linux memory with explicit availability/status. Refuse to
overwrite outputs. Replay checks identities/outputs, not identical memory counters.

Official Linux v6.18 conceptual references, no copied code:
- https://github.com/torvalds/linux/blob/v6.18/Documentation/filesystems/proc.rst
- https://github.com/torvalds/linux/blob/v6.18/fs/proc/task_mmu.c
- https://github.com/torvalds/linux/blob/v6.18/fs/exec.c
- https://github.com/torvalds/linux/blob/v6.18/kernel/sys.c
- https://github.com/torvalds/linux/blob/v6.18/COPYING
Linux is GPL-2.0 with its syscall note; none of its implementation is copied or
linked into new MIT code. Reference version is not proof of this worker kernel's
behavior. PyTorch remains the existing BSD-style dependency.
