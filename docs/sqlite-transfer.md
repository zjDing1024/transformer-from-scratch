# Same-checkpoint transfer to SQLite documentation

2026-11-15 MYT. Six **existing validation-selected models** are scored on an
external project's documentation, with no fine-tuning, new seed, new selection
or test-guided parameter choice. This extends the prior Flask-only evidence;
it is not another600-step run presented as a new model result.

## Source, format and frozen decisions

[Protocol](transfer-protocol.md), [recipe](../experiments/sqlite-transfer-v1.json)
and [manifest](../data/sqlite-transfer-manifest.json) were committed locally as
`f74675d` before any SQLite model forward. The original model-selection JSON and
all original model/training sources remain unchanged from `8d28a61`.
New scorer implementation was committed as `41e22d1` before real-corpus scoring.

SQLite's [official download guide](https://sqlite.org/getthecode.html) identifies
[docsrc](https://sqlite.org/docsrc) as its canonical documentation repository;
the [official copyright statement](https://sqlite.org/copyright.html) dedicates
its deliverable documentation to public domain. We verified the pinned Fossil
manifest SHA3-256, file membership, each artifact hash, SHA256 and byte length.
The downloader preserves the pinned copyright document locally. No build script,
private extension, paid warranty, API or pretrained model is involved.

Fossil check-in:
`14e2d07cae94f61084a3f289df352d75661ac52c2d9b398fabc26b68b79e84ad`.
Selected full documents: datatype3.in, isolation.in and lang_transaction.in.
The raw`.in` bytes contain HTML-like markup and Tcl/template directives, retained
as **inert text**, never executed or rendered. No trimming, normalization,
markup removal or post-score document filtering. Total52,351 targets.

This is a **joint project/topic and source-format shift**: Flask RST → SQLite
HTML/template source, not a clean domain-only intervention or rendered-prose
benchmark. We cannot assign any loss change uniquely to SQL topics or markup.
Whole-file hashes differ from Flask train files; no broad deduplication claim.

## Checkpoint recovery and identity

The previous local checkpoint directory was unavailable in the new execution
instance. It was **rebuilt once** using the unchanged original recipe and Flask
train/validation data. No SQLite input enters that training/selection path.
All six rebuilt checkpoint files match their original SHA256 **exactly**; all six
legacy float32 tensor digests also match. Original selected steps remain
600/400,600/600,600/500. This means reconstructed byte identity, not uninterrupted
retention of the old physical files. Reconstruction is recovery, not six new
independent trials. Complete expected/observed digests:
[recovery.json](../results/sqlite-transfer-v1/recovery.json).

The original selection lock remains:
`4a12794b6df715539657cbb951a7a52c4423f7102a578eaf7ca52c7739da4b9b`.
The separate scorer checks every model's metadata, original/imported source bytes
and tensor identity **before any external forward**. File-vs-tensor identity is
reported separately. A differently serialized file may pass only when its exact
tensors and metadata match; a tensor mismatch aborts, with no tolerance relaxation.
The new report never overwrites the original selection or prior Flask report.

## Results

Every model and the original-train-only add0.1 bigram score identical targets.
Context64 reset blocks, first byte per document excluded, short tails retained.
NLL is pooled by target counts, not an unweighted average of document means.
Across-seed arithmetic mean ± sample SD, **not** confidence intervals:

| Arm/reference | Byte NLL | Byte PPL | Bits/byte |
|---|---:|---:|---:|
| Sinusoidal positions | 3.064089 ± 0.015219 | 21.416596 ± 0.326707 | 4.420546 ± 0.021956 |
| No explicit positions | 3.074392 ± 0.015358 | 21.638436 ± 0.333650 | 4.435411 ± 0.022156 |
| Fixed Flask-train bigram | 3.089247 | 21.960532 | 4.456841 |

| Seed | Sinusoidal NLL | No-position NLL | Paired delta, no-position minus sinusoidal |
|---|---:|---:|---:|
| 20261114 | 3.080712761 | 3.091953747 | +0.011240986 |
| 20261115 | 3.050843063 | 3.067746248 | +0.016903185 |
| 20261116 | 3.060710614 | 3.063477078 | +0.002766465 |

Mean paired delta **+0.010303545**, sample SD0.007114830. Positive means lower
sinusoidal NLL. All three pooled pairs favor sinusoidal, but n=3 is descriptive;
there is no significance test, universal winner or robust-generalization claim.

| Document | Targets | Sinusoidal mean NLL | No-position mean NLL | Bigram NLL |
|---|---:|---:|---:|---:|
| datatype3.in | 31,694 | 3.207722 | 3.217971 | 3.216598 |
| isolation.in | 12,132 | 2.759414 | 2.762455 | 2.826178 |
| lang_transaction.in | 8,525 | 2.963677 | 2.984522 | 2.990159 |

**Limit surfaced:** seed20261114's no-position model is slightly worse than
bigram on pooled targets; the no-position datatype3 mean is also worse than
bigram. The average advantage over a simple baseline is small, unlike a claim
that the tiny model has learned broadly useful technical-language understanding.
The current corpora are different byte distributions, so comparing prior Flask
NLL2.414795 with SQLite NLL3.064089 does not isolate a causal domain effect.
Byte PPL is not comparable to word/subword perplexities or scores on other corpora.

Full precision per-document/per-seed scores and source fingerprints are in
[report.json](../results/sqlite-transfer-v1/report.json). No samples or user
competence claims are inferred from these likelihood scores.

## Reproduce

Run from the repository root with Python3.12 and the pinned CPU dependency:

```bash
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
python examples/download_text.py --manifest data/stability-manifest.json --output-dir corpus/flask
python examples/download_transfer.py --output-dir corpus/sqlite
# Only when old checkpoints are unavailable. A fresh output directory is required.
# Do not run the old study's score phase or replace its published selection.json.
python examples/study_positions.py train --data-dir corpus/flask --output-dir results/position-rebuilt
python examples/evaluate_transfer.py \
  --checkpoint-dir results/position-rebuilt \
  --train-data-dir corpus/flask \
  --external-data-dir corpus/sqlite \
  --checkpoint-origin rebuilt \
  --output results/sqlite-transfer-local/report.json
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```

With retained originals, skip reconstruction, use their directory and specify
`--checkpoint-origin retained`. Artifact origin is a caller disclosure, not a
cryptographic assertion of continuous retention. Tensor mismatches on another
CPU/library build must stop the study; do not change expected digests to force
success. Cross-hardware bitwise reproducibility is not promised. All training,
scoring and tests after explicit preparation are CPU/offline.

## Verification and boundary

89 unique CPU pytest cases passed,0 failed/skipped, including7 independent new
synthetic-only review cases; Ruff lint/format and compileall passed. Tests cover
all-candidate preflight, metadata/dtype/tensor corruption, unchanged selection,
serialization-vs-tensor identity, train-only bigram fitting, exact target counts,
raw-byte preservation, Fossil manifest membership and refusal to overwrite reports.
A separate-process scoring replay produced a byte-identical JSON report, with
zero additional training/independent seeds. See [verification.json](../results/sqlite-transfer-v1/verification.json).
CI runs offline tests, not a fresh execution of this external-corpus experiment.
Optional NumPy is absent and PyTorch emits a warning; no NumPy operations are used.

No raw third-party bodies or checkpoint weights are tracked in Git. This remains
an original small-model engineering/reproducibility study, not a full Transformer
paper reproduction, production model, SQL reasoning benchmark or proof of mastery.
