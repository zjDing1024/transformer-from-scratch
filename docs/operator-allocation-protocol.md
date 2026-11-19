# CPU operator allocation attribution v1: pre-run protocol

2026-11-20 MYT. Base `8f538be45034eb4696c89973937ef1d16a1db178`.
Use the existing `byte-lm-portable-f32-v1` manifest and both locked identities;
no training, corpus download, historical checkpoint reconstruction, latency or RSS run.
Only CPU float32, deterministic algorithms, one intraop/inter-op thread.

## Intervention and boundaries

Opt-in ContextVar scopes annotate the unchanged operations at finite validation,
snapshot deepcopy, prefill/append, K/V concat and output-token concat sites. No
monkeypatch, alternative inference implementation, tensor recording or global switch.
Default ranges are no-ops. Enabled range metadata holds scalars only.
Use torch.profiler CPU activity with profile_memory=True, record_shapes=False,
with_stack=False. Profiler durations are not exported or compared to prior timings.
The actual measured claims are site/event counts and operator self-memory attribution,
not repetition of existing logical formulas. No RSS, physical traffic or savings claim.

Nearest annotated scope owns each event's self bytes exactly once. Preserve positive
self bytes separately from negative self bytes; neither is a live-memory peak.
Inclusive clone bytes are diagnostic only, NEVER added to child self allocations.
Select unaggregated aten::cat events under the exact cat site, retaining per-event
metadata/count/signed self bytes. Range absence or failed byte controls means attribution
is unavailable, not a zero measurement. Counts/outputs/identity mismatch means failure.

## Fixed execution

1. Verify fixture identity and reference parity outside the profiler, scopes disabled.
2. Enable scopes, repeat the same parity checks outside the profiler; compare complete
   records. No fixture/seed changes to obtain a match.
3. One profiled cold constructor: finite validation and deepcopy separately. Inventory
   all 35 parameter tensors, 99,968 float32 elements (399,872 B), no buffers; confirm
   equal values/different storage and unchanged source/snapshot after requests.
4. For each original B2/P16/G16 and B2/P32/G32 case, profile one warm-snapshot cached
   request and one full-prefix request. Reset is part of cached request; output stays
   live through profiler stop. Do not forward the last emitted byte. Compare output
   IDs and final cache metadata to unprofiled parity outside each trace.
5. For each prompt, G=1 cached/full controls; prefill-only and reset-only controls.
   G=1 and prefill-only have no K/V cat. Full-prefix has none for any G. Reset preserves
   snapshot storage and leaves an empty cache, with no snapshot clones.
6. Separate synthetic controls: float32 cat 40 B, int64 cat 80 B, view 0 B and clone
   24 B. Also trace a known cat with scopes disabled to verify attribution cannot
   silently claim zero. Preserve control failures, do not choose nicer reruns.

Expected K/V cat counts: 60 / 124 (four per appended token); output payload sums
737,280 / 3,047,424 B. Old-prefix portions 706,560 / 2,983,936 B are logical only.
Final K/V 63,488 / 129,024 B is not cumulative allocation. Separate int64 output
cats: 16 / 32 calls, 6,272 / 24,832 B. Snapshot clone events are expected to number
35; runtime child allocation decomposition must be observed rather than assumed.
Validation bytes are exploratory, not prespecified. Transactional pending retains
old K/V until the complete append succeeds, so old and replacement storage overlap.

## Reproduction and failure handling

Publish compact JSON event records and disjoint producer aggregates, identities,
source hashes, runtime/configuration and exact control outcomes. Offline audit checks
published observations against frozen expectations; installed CLI also reruns actual
profiling. Replay must match claims, not exact total unrelated allocator activity.
Tests cover ancestry, no inclusive double-counting, absent attribution, controls,
scalar metadata, scope restoration and opt-in parity. Remote CI should run portable
actual allocation smoke checks plus published-record audit, without OS RSS thresholds.
PyTorch v2.14.1 official profiler/Parameter source and BSD-style license are pinned
in source-references.json. References are concepts only; new code is original MIT.
