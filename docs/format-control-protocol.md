# Frozen equal-byte-history SQLite format control

2026-11-16 MYT. Freeze this protocol/config before model scoring. Prepare and
commit a target-pair manifest from the pinned source before the first model
forward on this control. This is an optional follow-up to completed external
transfer, not a correction or replacement of the original benchmark.

## Unchanged models and source

Use all six original selection entries (3 seeds ×2 position arms), original
selection SHA256 `4a12794b6df715539657cbb951a7a52c4423f7102a578eaf7ca52c7739da4b9b`.
Check serialized file and exact float32 tensor digests separately. If old local
weights are unavailable, rebuild once using the frozen Flask training recipe;
no SQLite data enters training/selection. Abort if any selected tensor differs.
Keep original selection, source modules, datasets and reports byte-identical.

Reuse the three pinned SQLite docsrc files and their immutable Fossil artifact
hashes from `data/sqlite-transfer-manifest.json`. Canonical documentation source:
[SQLite download guide](https://sqlite.org/getthecode.html) and
[docsrc](https://sqlite.org/docsrc). SQLite's
[official copyright statement](https://sqlite.org/copyright.html) dedicates
its deliverable documentation to public domain. No bodies or weights in Git.
No Tcl, JavaScript, SQL, document code or upstream renderer is executed.

## Deterministic conservative prose projection

Parse UTF-8 source using the Python standard-library HTMLParser, with entity
callbacks enabled (`convert_charrefs=False`). Remove tags and comments, suppress
`tcl`, `script`, `style`, `pre`, and `code` spans, decode HTML character references,
collapse ASCII whitespace to one space, and insert spaces at block/drop boundaries.
Do not reflow via a browser. SQLite bracket cross-references and caret annotations
remain literal. This is an explicitly limited **HTML-like prose rendering**, not
the exact public website, a full DOM/CSS render or execution of SQLite's generator.
The parser maps every retained literal byte to its source offset; synthetic,
entity-decoded and normalized whitespace bytes are ineligible as targets.

Eligible targets are literal ASCII letters in paragraph text outside the suppressed
spans. They must survive byte-identically in the projection, with at least64
preceding raw bytes **and**64 preceding projected bytes. Exclude other characters
and paragraph-external text; report counts, early-window exclusions and pair counts.
Keep per-document boundaries; no target or history crosses a document boundary.

## Sampling and intervention

Partition eligible targets by whether their two64-byte histories are different:
changed-history primary and identical-history negative control. For each document
select up to256 changed pairs and64 identical pairs. Use source-order equally
spaced eligible indexes with both endpoints included, floor(i*(N-1)/(K-1)); K=1
uses index0. No random/loss-adaptive sampling, retries or sample replacement.
This deliberately sampled, paragraph/ASCII/changed-history population is not a
representative estimate for all prose, Unicode, source bytes or documents.

At each source-aligned target independently, feed exactly the immediately preceding
64 raw bytes or64 projected bytes. Restart positions0..63 for both inputs. Score
only the same final target via float64 cross-entropy on float32 last-position
logits. Thus old64-target evaluation block offsets do not affect history length.
No target or future byte appears in either input. Equal byte counts do **not**
mean equal semantic content/source spans: stripping markup reallocates the history
budget to earlier prose and changes distances. The intervention is context
representation under a fixed byte budget, not format syntax isolated from content.

Use the same train-only add0.1 Flask byte bigram, evaluating the final predecessor
byte of each history. Report how often that predecessor changes, so a context
change affecting even the baseline is visible. Never fit to projected/SQLite bytes.

## Reporting and limits

Save the pair manifest before scoring: per-document raw/projection digests, counts,
raw/projected target offsets, literal target byte and both history digests. Do not
publish raw prose bodies. The scorer regenerates and matches the entire manifest,
and checks all checkpoint identities before any model scoring. Hash all new code.

For both strata, report each seed/position arm and each document: count, raw and
projected mean NLL, byte PPL, and paired difference (projected minus raw; negative
favors projection). Pool by target counts. Across3 seeds report arithmetic means
and sample SD for each arm/condition and for each arm's projection effect. Also
report each seed's difference between position arms' projection effects. No
confidence intervals, p-values or extra independent trials from3 related documents.
The identical-history control must have zero per-target differences (tolerance
1e-12); fail rather than publish a successful controlled experiment if violated.

Do not compare sampled-target PPL numerically to the old full-document PPL as a
performance gain. Interpret no topic/domain transfer effect: source documents and
target bytes are now held fixed, while preceding encoded histories intentionally
change. No new tuning, fine-tuning, generation quality or mastery claim. CPU only.
