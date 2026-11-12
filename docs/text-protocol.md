# Frozen text-training protocol v1

Frozen before the first training run on 2026-11-13 (UTC+08:00).
A single CPU experiment, not a paper benchmark or a hyperparameter search.

- Data: Flask 3.1.0 official reStructuredText, upstream commit
  `ab8149664182b662453a563161aa89013c806dc9`, BSD-3-Clause.
  Exact upstream paths, SHA256 and sizes: `data/text-manifest.json`.
  Raw files are downloaded, not redistributed in this repository. Do not execute
  code contained in documentation. Raw RST includes prose, markup and code.
- Train: quickstart/config/errorhandling; validation: logging; test: testing.
  Split by entire document before windowing, no cross-document target pair.
  Related documentation can share boilerplate: this is document-disjoint, not
  a deduplicated benchmark or a claim of out-of-domain generalization.
- Vocabulary: all 256 byte values, fixed without examining held-out frequencies.
- Model: 2 causal self-attention blocks, width64, 4 heads, FFN128, sinusoidal
  positions, post-norm, ReLU, no dropout, untied embedding and output weights.
  Existing original attention/norm used, not nn.Transformer/SDPA.
- Training: seed20261113, deterministic CPU single thread, context64, batch16,
  600 AdamW updates, LR0.003, weight_decay0.01, clip global norm1.0.
  Uniform over all valid document-local starting offsets, with replacement.
  Total 614400 byte-target exposures, not distinct tokens or epochs.
- Validation every100 updates; lowest mean NLL checkpoint selected, first wins ties.
  Test evaluated only after selection. No test-based retry or parameter tuning.
- Baseline: byte bigram fitted only to training within-document pairs,
  additive smoothing0.1 over256 outcomes. No baseline hyperparameter tuning.
- Evaluation: every document byte after the first exactly once; blocks of64
  targets with shifted inputs, final short block retained, no padding or ignored
  targets. Context resets each block, shared between LM and bigram comparison.
  Report sum NLL/token count, perplexity=exp(mean NLL), bits/byte=NLL/log(2).
  First target in every block has only one preceding byte, so this is not
  full sliding-window context evaluation. PPL is byte-level, not word/token PPL.
- Save checkpoint, fixed prompt sample and machine-readable results. Sample is
  illustrative only, not an independent quality score. No latency/SLA claims.
- Limit: small technical corpus, one seed, no confidence interval, GPU, cross
  attention, translation BLEU, foundation-model capability, or user mastery claim.
