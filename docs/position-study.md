# Paired multi-seed position-encoding study

Date: 2026-11-14 MYT. This is a fixed-budget follow-up, **not** a rerun of the
previous one-seed result presented as a new achievement. The new evidence is a
parameter-matched positional-encoding ablation across three paired seeds and three
new document-level holdouts. All code is original and all computation is free CPU.

## What was frozen

The [protocol](stability-protocol.md) and [configuration](../experiments/position-stability-v1.json)
were frozen in local commit `e7e09f5` before new held-out documents were downloaded or
study training began. Only existing train/validation documents determine the six
models and selected checkpoints. Old `testing.rst` is excluded from this experiment.
New appcontext/design/reqcontext are from the same immutable Flask release, with
25,532 total next-byte targets. This is additional same-domain evidence, **not** an
external-domain benchmark, and shared boilerplate was not deduplicated.

Same initialization and sampled byte windows are verified by hashes within each
seed. Sinusoidal encoding is parameter-free, so both arms have99,968 parameters.
Causal masking is retained in the no-position arm: it does **not** remove all order
information. The ablation therefore asks about the value of explicit sinusoidal
addition under this budget, not the necessity of position/order in all Transformers.

## Reproduce from the repository root

```bash
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
python examples/download_text.py --manifest data/stability-manifest.json --output-dir corpus/flask
# Use a fresh directory. Existing experiment evidence is never overwritten.
python examples/study_positions.py train --data-dir corpus/flask --output-dir results/position-local
# Copy the selection_sha256 printed above. It locks ALL six selections, not one seed.
python examples/study_positions.py score --data-dir corpus/flask --output-dir results/position-local \
  --selection-sha256 PRINTED_SHA256
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```

Download is an explicit network step. Training/scoring use only the local corpus;
there is no paid API, GPU, pretrained model, secret or private data. The downloader
checks every SHA256 and retains the upstream BSD-3-Clause license. Public files
contain metadata and derived numerical scores, not copied documentation bodies.
Original references and how their designs informed this implementation are in the
[protocol](stability-protocol.md) and [notices](../THIRD_PARTY_NOTICES.md).

Train and score are separate processes. Selection JSON freezes all seed/arm entries,
curves, checkpoint digests, config/manifest/protocol and entry-point/source hashes.
Scoring checks **every** checkpoint and roster entry before first held-out forward
pass. A changed configuration/source, missing run, corrupted checkpoint or existing
report fails closed. Checkpoints use `weights_only=True` and remain local. This
hash lock prevents accidental pipeline drift, not malicious tamper-proof attestation.

## Observed results

All six runs completed600 updates with finite training/validation losses. Each run
received614,400 byte-target exposures; initial-state and batch-schedule hashes match
within every pair. All three sinusoidal checkpoints were selected at600; no-position
checkpoints were selected at400/600/500 respectively. These are validation choices,
not test choices. Both arms trained the full budget even when earlier states won.

| Seed | Sinusoidal selected step | Sinusoidal held-out NLL | No-position selected step | No-position held-out NLL | Paired delta (no-position minus sinusoidal) |
|---|---:|---:|---:|---:|---:|
| 20261114 | 600 | 2.421483009 | 400 | 2.462787677 | 0.041304668 |
| 20261115 | 600 | 2.414414665 | 600 | 2.432377232 | 0.017962567 |
| 20261116 | 600 | 2.408487501 | 500 | 2.426106172 | 0.017618671 |

Across-seed arithmetic mean ± sample SD, **not** confidence intervals:

| Arm/reference | Pooled byte NLL | Byte PPL | Bits/byte |
|---|---:|---:|---:|
| Sinusoidal | 2.414795 ± 0.006506 | 11.187635 ± 0.072808 | 3.483813 ± 0.009386 |
| No explicit positions | 2.440424 ± 0.019620 | 11.479381 ± 0.226342 | 3.520787 ± 0.028306 |
| Fixed train-only bigram | 2.593534 | 13.376964 | 3.741679 |

Mean paired NLL difference is **+0.025628635**, sample SD0.013576931. Positive
means lower NLL with sinusoidal addition. This direction occurs for all three
pairs, but n=3 and one related corpus do not establish statistical significance
or prove that sinusoidal positions are generally better/more stable.

Per-document NLL means over the same three seeds (not used for selection):

| Held-out document | Target bytes | Sinusoidal mean | No-position mean |
|---|---:|---:|---:|
| appcontext.rst | 5,126 | 2.426367 | 2.448572 |
| design.rst | 11,073 | 2.461165 | 2.479048 |
| reqcontext.rst | 9,333 | 2.353424 | 2.390123 |

The no-position arm remains competitive. Explicit positions yield a small
increment under this budget, not a prerequisite for learning useful local byte
statistics. The causal prefix still carries order-related information. Every
seed/document pair also favored sinusoidal here, but these correlated document
scores must not be counted as nine independent model trials.

The six original training+validation runs took82.73s total on this execution
instance, excluding setup/final scoring. Timings are observations, not benchmarks
or equal-FLOP claims. Maximum pre-clip gradient norms and clipping counts are in
[selection.json](../results/position-v1/selection.json). None had a nonfinite
training or validation loss. No failed full-corpus attempt or test-guided retry.

Implementation was committed as `968eaed` before formal training. The complete
selection lock is `4a12794b6df715539657cbb951a7a52c4423f7102a578eaf7ca52c7739da4b9b`.
[Per-run/per-document report](../results/position-v1/report.json),
[frozen selections and curves](../results/position-v1/selection.json), and
[verification record](../results/position-v1/verification.json) preserve evidence.

## Verification

- 74 CPU pytest cases passed, 0 failed/skipped; Ruff lint/format and compileall passed.
  These include7 independent synthetic-only review cases added before formal training.
- A full six-candidate same-environment replay produced exactly matching sampled
  schedules, initial/selected state digests, validation curves/selected steps and
  every per-document/pooled/aggregate score. Timing and selection JSON file digests
  differ because wall times are recorded. A replay adds **zero** independent seeds.
- Built a wheel and imported it outside the checkout; a separate synthetic
  four-candidate, two-update train→lock→score smoke passed using the installed wheel.
  This smoke is packaging validation, not another real-corpus experiment.
- PyTorch reports an optional NumPy-missing warning. No NumPy operation is used;
  training, scoring and tensor hashes use PyTorch/standard-library operations.
- No raw corpus or generated checkpoint is tracked. CI runs the74 offline unit
  tests, including tiny study fixtures; it does not silently rerun the full corpus
  experiment or assert the exact measured scientific result as a unit-test target.

## Interpretation boundaries

- Three seeds provide descriptive dispersion, not a confidence interval or a
  statistical significance claim. Never count a replay as an additional seed.
- Seed SD measures initialization/minibatch variation under one fixed split, not
  uncertainty across independent documents, projects or domains.
- Byte PPL is not comparable to word/subword PPL. Context resets every64 targets,
  and all short tails are retained; it is not full sliding-window scoring.
- Bigram uses all training pairs with fixed add0.1 smoothing. It is not a
  compute-matched neural-model competitor.
- PyTorch2.14.1 is fixed for both arms and CI. The earlier single-seed text study
  used2.15.0; no cross-version numerical improvement is claimed.
- Lower loss alone does not establish coherent generation, understanding, user
  mastery, translation ability or original Transformer paper reproduction.
