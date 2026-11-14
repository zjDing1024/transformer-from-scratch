# Frozen SQLite transfer evaluation v1

2026-11-15 MYT. Freeze this protocol, configuration and manifest in a local Git
commit before any SQLite model scoring. This is evaluation of the six previously
selected models, not a new seed sweep, parameter search or fine-tuning run.

## Source and representation fixed before scoring

The [official SQLite download guide](https://sqlite.org/getthecode.html) identifies
`https://sqlite.org/docsrc` as the canonical documentation repository. The
[official copyright page](https://sqlite.org/copyright.html) dedicates deliverable
SQLite documentation to the public domain; this does not automatically apply to
unrelated build tools. Only the three documentation bodies and copyright evidence
are downloaded, never generator/build scripts or private extensions.

Pinned Fossil check-in:
`14e2d07cae94f61084a3f289df352d75661ac52c2d9b398fabc26b68b79e84ad`.
The raw manifest's SHA3-256 was checked against this check-in name; selected file
artifact names come from its `F` entries. Each file is checked against its Fossil
artifact hash, SHA256 and length. See `data/sqlite-transfer-manifest.json`.

Chosen by topic before scoring: `pages/datatype3.in` (31,695 bytes),
`pages/isolation.in` (12,133 bytes), `pages/lang_transaction.in` (8,526 bytes).
Score all raw bytes, keeping HTML-like tags, cross-reference syntax and inert Tcl
/template text. Do not render HTML, execute Tcl, strip markup or normalize bytes.
Exactly52,351 next-byte targets, excluding the first byte of each document.

This jointly changes project/topic **and source format** from Flask RST to SQLite
HTML-like/template `.in` files. It is not a controlled domain-only intervention,
not rendered user-facing prose, not a broad generalization benchmark, and not
an independent estimate of separate format and topic effects. No deduplication
beyond checking that whole-file hashes do not match the original training files.

## Model identity and no-tuning rules

Original selection lock must remain byte-for-byte unchanged:
`4a12794b6df715539657cbb951a7a52c4423f7102a578eaf7ca52c7739da4b9b`.
Seeds20261114/15/16; sinusoidal and no-position arms; old selected steps remain
600/400, 600/600, 600/500 respectively. No new validation selection, no model or
hyperparameter change, no seed/document exclusion after seeing SQLite scores.

The prior local checkpoint directory is unavailable on this execution instance.
Reconstruction uses the old fixed training procedure and old Flask train/validation
only, with no new hyperparameter choice and no SQLite input. Check **both** stored
file SHA256 and the old float32 state-tensor digest for all six checkpoints before
any SQLite forward pass. If any tensor digest differs, abort this evaluation:
nearby parameters are not the same checkpoint. If file bytes differ but the tensor
hash and metadata match, report those facts separately. Even identical rebuilt
bytes must be described as rebuilt, not continuously retained original artifacts.
Reconstruction is recovery work, not additional independent scientific trials.

The scorer is separate from the original training/scoring code. Verify the old
model source hashes and selection metadata, preserve the old lock, and save only
a new external report. Hash all new scoring sources/config/protocol. No raw corpus
or checkpoint weights are committed. CPU single thread, deterministic algorithms,
PyTorch2.14.1+cpu; no paid resource, external model or API.

## Objective and analysis

Use the exact previous evaluation objective: context64 reset blocks, every byte
after each document's first scored once, short final blocks retained, no padding
or cross-document target. All six models and train-only add0.1 bigram see the same
bytes and target coverage. The bigram is fitted solely from the original three
Flask training files; no external frequencies/vocabulary fitting or adaptation.
All256 byte IDs are already fixed. Copyright evidence is never scored.

Primary metric: sum NLL divided by52,351 targets. Report each document and pooled
NLL, byte PPL, bits/byte for each of the six models and the same baseline. Across
seeds report arithmetic metric means and sample SD, then three paired differences
(no-position minus sinusoidal). Positive differences favor sinusoidal. Do not
choose an arm based on the result. No significance/CI or universal superiority
claim from n=3; correlated documents are not extra independent training trials.
Report unfavorable results and that a topic+format shift cannot identify its
cause. Scores on different corpora are descriptive, not directly interchangeable
measures of corpus difficulty or an isolated causal degradation effect.

## References informing the original implementation

- [Attention Is All You Need, positional encoding](https://arxiv.org/abs/1706.03762).
- [PyTorch official language-model example](https://github.com/pytorch/examples/blob/acc295dc7b90714f1bf47f06004fc19a7fe235c4/word_language_model/main.py),
  BSD-3-Clause: separate validation selection and final evaluation.
- SQLite canonical docsrc and public-domain dedication above. No copied model code.
