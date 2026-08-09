# Frozen Methodology Protocol V1

Protocol ID: `eclipse-severity-v1`. The machine-readable source of truth is
`configs/protocol_v1.toml`; its SHA-256 fingerprint is recorded whenever the research dataset is
prepared. Source code must load and validate that configuration rather than reproduce mappings as
independent constants.

## Immutable decisions

The raw sources remain immutable and external. Severity classification accepts exactly `blocker`,
`critical`, `major`, `normal`, `minor`, and `trivial`; `enhancement` remains in raw provenance but
is ineligible for S6, S3, and S2. Unknown labels fail closed. The fixed mappings and class orders
are those in the protocol TOML.

Eligibility requires a valid issue ID, accepted severity, valid `Creation time`, and nonblank
content in at least one of `Summary` or `Description`. Missing one text field is allowed when the
other is usable. No status, resolution, priority, assignee, creator, comment, attachment, activity,
last-change, or other post-report field affects eligibility.

Primary predictors are only canonical `Summary` and `Description`. Canonicalization converts null
to empty, normalizes Unicode to NFC and line endings to LF, and joins the fields with two LF
characters. It does not stem, lemmatize, remove stopwords, code, URLs, or HTML, or otherwise clean
semantics. Separate canonical fields are retained for later text ablations.

Each project is sorted by creation time and deterministically by issue ID. The first
`floor(0.8 × eligible rows)` records form DEVELOPMENT; the remainder form
LOCKED_TEST_CANDIDATE. Class labels and model results never affect the boundary. Exact-text and
resolvable `Dupe of` equivalence components are then used to exclude future candidate rows whose
component intersects development. Earlier rows are retained and future rows are never moved
backward. Duplicates wholly inside training are retained.

Development temporal CV uses four deterministic position blocks per project and three
expanding-window folds: block 1 trains fold 1 and block 2 validates it; blocks 1–2 train fold 2 and
block 3 validates it; blocks 1–3 train fold 3 and block 4 validates it. Corresponding project
blocks are pooled only after per-project construction. Validation overlap with training is removed
under the same duplicate-component rule. No shuffling occurs.

The primary selection metric for every task is Macro-F1 evaluated against its complete fixed class
order with zero-division behavior set to zero. Secondary metrics are frozen in configuration. The
S2 precision floor of 0.30 is available only under an explicitly named legacy-reproduction mode.

## Forbidden leakage behavior

- Never use locked-test labels, predictions, metrics, class-specific errors, or examples for model
  selection, preprocessing, features, thresholds, or hyperparameters.
- Never fit preprocessing outside the current training partition.
- Never move a future duplicate into earlier training or adjust chronology to improve balance.
- Never treat `Depends on`, `Blocks`, or `See also` as duplicate equivalence.
- Never use target-derived, post-report, relationship, Product, Component, or source-project fields
  as ordinary predictors in the primary benchmark.
- Never silently change this protocol or regenerate a frozen split under a different protocol
  fingerprint.

## Parameters allowed to vary later

Within the frozen data and leakage boundaries, experiments may vary text ablation (Summary,
Description, or both), vectorizer/model family, controlled class weighting, development-only
hyperparameters, and within-project, pooled, or transfer experiment family. Advanced approaches
remain a later phase. Any semantic preprocessing, metadata features, training deduplication, or
alternative split protocol is a separately labelled sensitivity study and requires an explicit
protocol amendment.

## Locked-test access

Processed development and locked-test artifacts are physically separate and ignored by Git.
Normal loaders reject locked-test access. A future final-evaluation workflow must explicitly set
the configured unlock variable/value and must remain separate from development commands. This
protocol-freezing task records:

```text
TEST_SET_ACCESSED_FOR_MODEL_SELECTION = NO
LOCKED_TEST_MODEL_PERFORMANCE_ACCESSED = NO
```

