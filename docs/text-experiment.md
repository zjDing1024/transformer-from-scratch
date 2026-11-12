# Real-text byte language modelling, v1

Date: 2026-11-13 UTC+08:00 (2026-11-12 scheduled run).
This extends tested attention primitives into a full train/select/evaluate/save/load/generate
path. It is a small decoder-only language model, **not** the original translation paper,
not a complete encoder-decoder reproduction, and not a usable writing assistant.

## Research and design

Studied [PyTorch official word_language_model/main.py](https://github.com/pytorch/examples/blob/acc295dc7b90714f1bf47f06004fc19a7fe235c4/word_language_model/main.py)
(PyTorch contributors, BSD-3-Clause): separation of train/eval modes, shifted next-token
targets, gradient clipping and validation-based checkpoint selection. This concise official
example is relevant at the current stage because it connects mechanisms to an actual
language-model objective rather than wrapping a pretrained API. It is not selected on
popularity or claimed recent activity. No implementation code was copied.

Our differences: fixed byte vocabulary avoids held-out vocabulary fitting; each document
is isolated before window creation; a smoothed bigram gives a meaningful non-neural baseline;
evaluation sums loss over actual targets, retaining short tails; all attention and norms
use this repository's original explicit implementation. Reusing the existing post-norm
block with an always-on triangular mask makes a causal self-attention decoder; there is
no encoder memory or cross-attention. Original Transformer theory and limitations remain
in `source-study.md`.

## Reproducible protocol

[Protocol frozen before training](text-protocol.md), [data manifest](../data/text-manifest.json).
Flask official technical documentation is real text containing markup/code, not synthetic
repeated strings. BSD-3-Clause permits use; the explicit downloader verifies every SHA256
and preserves the original license. No third-party body is committed in this repository.
Training: 79,535 bytes in 3 documents; validation: 5,927 bytes; test: 10,448 bytes.
Each evaluation excludes its first byte, never scores cross-document transitions.
No test-informed model choice or retry was performed.

## Measured results

[Machine-readable trace](../results/text-v1/text-training.json), Python3.12.14,
PyTorch2.15.0+cpu, single CPU thread. 600 updates, 614,400 byte-target exposures.
Validation checkpoints at100/200/300/400/500/600; selected600 with NLL2.447403.

| Held-out testing.rst, 10,447 byte targets | NLL | byte PPL | bits/byte |
|---|---:|---:|---:|
| Fixed train-only add-0.1 bigram | 2.646989 | 14.111485 | 3.818798 |
| 2-layer original Transformer | 2.419144 | 11.236234 | 3.490087 |

The lower byte NLL is evidence for this fixed corpus/split only. One seed and one test
document cannot establish statistical significance or a general superiority claim.
Baseline has all within-document training pairs, while neural training samples repeated
64-target windows; this is a simple reference, not a compute-matched contest.
Wall time15.66s includes training, validation, final test, checkpoint and generation on
this execution instance; it is not a hardware benchmark or deployment latency.
Checkpoint reload probe has max absolute difference0.0 in the same process. Fresh-process
generation is also exercised. Only locally generated checkpoints are loaded, weights_only=True.

## Failure analysis

Fixed sample starts `The application plis aring_nas ertiche`. It has fragments and markup
but is mostly incoherent. This is expected from a tiny model, tiny domain-specific corpus,
limited context and budget. A lower loss than bigram does **not** mean usable text generation,
understanding Flask, coding competence or production readiness. Byte PPL must not be compared
with word/subword PPL reported elsewhere. Documentation shares terms and boilerplate across
splits; the split is document-disjoint but not deduplicated or out-of-domain.

## Verification and next milestone

59 unique CPU tests passed, including causality under future-token perturbation, finite
backpropagation, no cross-document sampling, full evaluation-tail coverage, identical
baseline target sets, known uniform-PPL and bigram counts, hash corruption rejection,
checkpoint/generation equality and optimizer decrease. Independent reviewer reran59 tests
and separately exercised short-document sampler handling. Repeated runs are not additional
test cases. Ruff check/format and compileall are checked before publication. CI checks
mechanism/unit tests, not a fresh claim that it reran the600-step corpus experiment.

Next: package the train/generate entry points as installed commands, add resume/checkpoint
metadata validation and multiple-seed uncertainty on a predeclared follow-up protocol;
then evaluate longer context or pre-norm without turning the current test set into a tuning
set. Current results demonstrate assistant-built engineering artifacts, not personal mastery.
