# Frozen paired position-encoding study v1

Date: 2026-11-14 MYT. This plan is committed before new held-out data are
materialized or any study training/scoring. Configuration:
[`position-stability-v1.json`](../experiments/position-stability-v1.json).
No test-guided hyperparameter choice, retries, seed exclusion or budget extensions.
Implementation correctness tests may use synthetic inputs, never the held-out losses.

## Question and controls

Does adding explicit sinusoidal positions improve this tiny causal byte LM's
held-out loss under a fixed CPU training budget, consistently across three seeds?
Compare sinusoidal vs no explicit positions, not a claim that all positional
information is absent: the causal attention mask still exposes an ordered prefix.
Position encoding has no parameters. Both arms have identical trainable parameter
counts and, within each seed, identical initialization and sampled batch schedule.
Existing post-norm blocks, context64, batch16, AdamW0.003, decay0.01, clipping1.0,
600 updates and validation every100 updates are unchanged. Three fixed new seeds
20261114/15/16, six runs total (3,686,400 target exposures, not unique tokens).
Use PyTorch2.14.1+cpu, one thread and deterministic algorithms; compare arms within
this environment, not numerical changes from yesterday's PyTorch2.15.0 experiment.

## Held-out policy

Keep the existing train documents quickstart/config/errorhandling and validation
logging, upstream Flask commit ab8149664182b662453a563161aa89013c806dc9.
New held-out documents are fixed by path before download:
appcontext.rst, design.rst, reqcontext.rst. Prior testing.rst is neither used for
selection nor evaluated in this study. Manifest byte lengths, SHA256 and upstream
Git blob IDs are recorded and checked. BSD-3-Clause license is preserved locally;
no third-party prose/code bodies are committed. Documentation is never executed.
These are document-disjoint but same-project/domain and not deduplicated.
They are new to this project's training experiment, not a blind third-party benchmark.

Train all six candidates and select each checkpoint using only logging.rst;
then freeze selected checkpoint digests and validate the complete run roster
before scoring any new held-out document. Every document byte after its first is
scored once, with context reset every64 targets and short tails retained, exactly
as v1. Report each document and token-weighted pooled NLL, byte PPL and bits/byte.
Use the same train-only add-0.1 byte bigram as a reference, not compute-matched.

## Analysis fixed before results

Report all six runs, full validation curves, selected steps, maximum pre-clip
norm, clipping frequency, finite-loss checks, initialization and batch-schedule
hashes. Primary endpoint: pooled held-out mean NLL. For each arm report arithmetic
mean and sample SD across the three seeds; report each paired difference
(no_position minus sinusoidal) and its mean/SD. Positive differences favor
sinusoidal. Include mean/SD of byte PPL separately, not exp(mean NLL) mislabeled
as mean PPL. n=3 is descriptive only: no p-value, confidence interval, winner
selection or population/generalization claim. Report negative/mixed results.

Abort on nonfinite loss/gradient, hash mismatch or missing run. Do not silently
drop a failed run. Restart only to fix a demonstrated implementation failure;
retain and disclose failed attempts if any. No tuning after external scoring.
A same-environment replay is a reproducibility check, not another independent seed.

## Primary sources

- [Vaswani et al., §3.5](https://arxiv.org/abs/1706.03762): positional encoding
  motivation. We ablate explicit sinusoidal addition only, not reproduce WMT.
- [PyTorch official word_language_model/main.py](https://github.com/pytorch/examples/blob/acc295dc7b90714f1bf47f06004fc19a7fe235c4/word_language_model/main.py),
  BSD-3-Clause: validation selection, clipped training and separate final test.
  Design reference only; experiment implementation is original.
- [Flask pinned source and license](https://github.com/pallets/flask/tree/ab8149664182b662453a563161aa89013c806dc9),
  BSD-3-Clause. Data manifests retain immutable URLs and checksums.
