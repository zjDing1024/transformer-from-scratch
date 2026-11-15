# Same targets, equal byte histories: SQLite format control

2026-11-16 MYT. This is a new, explicitly scoped context-representation control,
not a rerun of the old full-document SQLite benchmark. Source documents and literal
prose target occurrences are held fixed. Their preceding context representations
vary, but both have exactly64 bytes and only the final target is scored.

## What is controlled, and what is not

The original scorer reset context every64 **targets**. Raw versus cleaned text
could put the same target at different block offsets, creating unequal history.
This control instead evaluates every paired target independently with positions
0..63 in both conditions, using only the immediately preceding64 bytes. The
current target occurrence and future source bytes are excluded from both inputs.

The projected condition uses an original, conservative HTML-like prose rendering:
strip tags/comments and Tcl/script/style/pre/code spans, decode entities, collapse
ASCII whitespace and add block/drop-boundary spaces. SQLite bracket/caret macros
remain. The [Python HTMLParser API](https://docs.python.org/3.12/library/html.parser.html)
was studied for source positions and entity callbacks; no reference implementation
was copied. This is **not** SQLite's full generator, a browser/CSS render, or
execution of document code. See [protocol](format-control-protocol.md) and
[recipe](../experiments/sqlite-format-v1.json).

Equal byte budgets are not equal underlying source/prose spans. Removing markup
can bring earlier prose into the64-byte window and change distances. The measured
intervention is context representation under a fixed byte budget, not formatting
syntax isolated from semantic-context coverage. No topic/domain shift is introduced
here because the documents and target occurrences are identical across conditions.
This does not establish generalization to new sources or causal domain transfer.

## Frozen sampling and checkpoint identity

Local protocol/config commit `d4b6765`; implementation `9059554`; complete target
pair manifest/recovery evidence committed as `a112b9e` before model scoring.
The old SQLite manifest remains pinned to the same canonical public-domain docsrc
artifacts; provenance/license details remain in [the prior transfer report](sqlite-transfer.md).
No source bodies or checkpoint weights are tracked.

Targets are literal source-aligned ASCII letters in paragraph text outside suppressed
spans. Entities and synthesized bytes cannot become targets. Before applying quotas:

| Document | Literal eligible prose letters | Early-window exclusions | Changed-history eligible | Identical-history eligible | Sampled changed / identical |
|---|---:|---:|---:|---:|---:|
| datatype3 | 15,994 | 21 | 15,434 | 539 | 256 / 64 |
| isolation | 9,272 | 20 | 8,706 | 546 | 256 / 64 |
| lang_transaction | 6,064 | 9 | 5,923 | 132 | 256 / 64 |

All50 early exclusions lack64 projected history bytes; none lack raw history.
From30,063 eligible changed and1,217 identical pairs, source-order equal spacing
selects768 changed pairs and192 identical negative controls. This is deterministic
sampling, not a random population estimate; quotas give each document one third
of the pooled sample, unlike the original corpus's byte weighting. Paragraph/ASCII
and changed-history selection excludes most possible target types. Do not compare
these absolute NLL/PPL values with the prior full-document benchmark as an improvement.
The [357,500-byte manifest](../results/sqlite-format-v1/pairs.json) records offsets,
target byte IDs and both history digests without retaining text bodies.

The previous local weight directory was again unavailable. One recovery run using
the unchanged frozen Flask recipe recreated all six selected checkpoints. Every
serialized file SHA256 **and** tensor-state digest matched the original selection;
this is rebuilt byte identity, not uninterrupted artifact retention. It adds no
independent seeds. [Recovery evidence](../results/sqlite-format-v1/recovery.json).
Original model sources, data manifests, selection lock and prior reports are unchanged.

## Measured primary result

768 sampled changed-history targets per model/condition; arithmetic mean ± sample
SD across3 seeds, **not** confidence intervals. Negative delta favors projection.

| Arm/reference | Raw-history NLL | Projected-history NLL | Projected minus raw NLL |
|---|---:|---:|---:|
| Sinusoidal | 2.941931 ± 0.050035 | 2.895443 ± 0.010480 | −0.046488 ± 0.057400 |
| No explicit positions | 2.920568 ± 0.058280 | 2.881337 ± 0.039033 | −0.039231 ± 0.019863 |
| Original-train-only bigram | 2.841660 | 2.827103 | −0.014557 |

| Seed | Sinusoidal projection effect | No-position projection effect |
|---|---:|---:|
| 20261114 | −0.111175355 | −0.061839073 |
| 20261115 | −0.001638474 | −0.024580460 |
| 20261116 | −0.026650082 | −0.031273269 |

All six pooled model cases improve, but only15/18 seed-arm-document cases do.
For seed20261115, projection worsens sinusoidal isolation and lang_transaction,
and no-position isolation. This is not a universal benefit of text cleaning.
The paired difference between arms' projection effects (no-position minus sinusoidal)
is +0.049336283,−0.022941985,−0.004623188 across seeds: mixed direction, mean
+0.007257037, SD0.037575149. No general position-encoding winner or significance claim.

Per-document mean NLL over3 seeds:

| Document | Sinusoidal raw → projected | No-position raw → projected | Bigram raw → projected |
|---|---:|---:|---:|
| datatype3 | 2.884108 → 2.811964 | 2.902247 → 2.833830 | 2.801327 → 2.772328 |
| isolation | 2.917074 → 2.880412 | 2.864318 → 2.839916 | 2.824644 → 2.821727 |
| lang_transaction | 3.024610 → 2.993952 | 2.995139 → 2.970265 | 2.899009 → 2.887253 |

**Important failure boundary:** bigram has lower pooled NLL than every one of the
six neural models in both conditions on this sampled target set. The projection
benefit is not evidence that the tiny neural models outperform the simple baseline
or have robust language understanding. The immediate predecessor changes for12/768
primary targets (6/4/2 by document);756 retain it. Bigram's change comes only from
those12, not from fitting/adapting it to SQLite. It remains fitted on original
Flask train documents only.

The192 identical-history controls have maximum absolute per-target score difference
**0.0** for every neural model and the bigram. Thus the pipeline does not invent
an effect when target and input bytes are actually identical.

## Reproduce and audit

Python3.12.14, PyTorch2.14.1+cpu, single thread, deterministic algorithms. Free CPU,
no paid API/GPU/private data. From repository root after its CPU setup:

```bash
python examples/download_text.py --manifest data/stability-manifest.json --output-dir corpus/flask
python examples/download_transfer.py --output-dir corpus/sqlite
# Only if the six original checkpoints are unavailable. Recovery is not new evidence.
python examples/study_positions.py train --data-dir corpus/flask --output-dir results/position-rebuilt-1116
# Scoring regenerates and checks the entire already-frozen target-pair manifest.
python examples/evaluate_format_control.py score \
  --external-data-dir corpus/sqlite \
  --pairs results/sqlite-format-v1/pairs.json \
  --pairs-sha256 4fe39fcf8a16b122f0f6f1fc9ec1acf4d9e397571983418dae348c7d104563b5 \
  --checkpoint-dir results/position-rebuilt-1116 --checkpoint-origin rebuilt \
  --train-data-dir corpus/flask --output results/sqlite-format-local/report.json
# Model-free preparation audit in a new path; never overwrite the frozen manifest.
python examples/evaluate_format_control.py prepare \
  --external-data-dir corpus/sqlite --pairs results/sqlite-format-local/pairs.json
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```

If using retained originals, change the checkpoint directory and disclose
`--checkpoint-origin retained`. Never relax expected tensor digests when a new
hardware/library build fails reconstruction. No cross-hardware bitwise guarantee.
The prepare phase makes no model forward. Downloading is explicit; subsequent
operations use local files. Only the three immutable documentation files are
processed, never website scripts or the upstream template generator.

[Full per-seed/per-document scores](../results/sqlite-format-v1/report.json),
[pair manifest](../results/sqlite-format-v1/pairs.json) and
[verification](../results/sqlite-format-v1/verification.json) provide audit evidence.
107 CPU tests pass,0 failed/skipped, including7 independent new synthetic review
cases. Ruff lint/format and compileall pass. A fresh-process scoring replay produces
byte-identical report JSON, adding zero independent seeds. Optional NumPy is absent
and causes a warning; no NumPy operation is used. CI runs offline tests, not a fresh
claim of executing this real-corpus experiment. No post-score tuning or new selection.
