# Phase B1.6 — Long-Text-Aware MPNet Semantic Fusion

## Scope and artifact validation

Phase B1.6 evaluates whether replacing B1.5's single truncated MPNet sequence with a complete multi-chunk pooled representation improves the fixed lexical-semantic fusion models. It remains a development-only model-selection experiment under frozen protocol v1.

Finalization validated nine project-level semantic shards and nine competitive task/fold checkpoints. Every artifact has `SUCCESS` status and matches protocol `85faacae1e7f411d68803653388611c37c62730ea9217e9700a0ad6ac41b7cda`, development membership `4f62fdf4164594126c421955804b654cd5d5f8f7b46ada345ae0cffa71460d0f`, B1.6 configuration `646fc17a0e27f2310b6acb488d50d7a594be2e3a8987af0f2a79cd22e0b6254d`, and MPNet revision `e8c3b32edf5434bc2275fc9bab85f82640a19130`. Parquet checksums, dimensions, normalization, identities, project membership, semantic-cache identity, exact CV membership fingerprints, lexical identity, classifier, class weighting, C, and semantic weight 1.0 passed validation.

No semantic embedding was regenerated and no classifier was refitted during finalization. Locked-test text, embeddings, predictions, labels, and performance were not accessed.

## Frozen representation and matrix

Canonical Summary + Description was covered by sequential MPNet windows containing 382 content tokens plus two special tokens, with 64-token overlap and 318-token step. Chunk embeddings were individually L2-normalized, averaged per document, and L2-normalized again. No chunk cap was used. The cache contains one final float32 768-dimensional vector per development document.

The competitive matrix remained exactly S6/S3/S2 × folds 1–3: S6 CHAR + LinearSVC balanced C=0.25, S3 WORD + LogisticRegression balanced C=2.0, and S2 WORD_CHAR + LogisticRegression balanced C=2.0. Semantic weight was fixed at 1.0.

## Development results

| Task | Fold Macro-F1 | Mean ± population SD | Balanced accuracy | Accuracy | Weighted F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| S6 | 0.257173 / 0.274782 / 0.291055 | 0.274337 ± 0.013835 | 0.296269 | 0.669566 | 0.672560 |
| S3 | 0.463887 / 0.485479 / 0.487476 | 0.478947 ± 0.010681 | 0.521555 | 0.647016 | 0.671130 |
| S2 | 0.632911 / 0.641014 / 0.646577 | 0.640167 ± 0.005611 | 0.673601 | 0.777916 | 0.794992 |

## Per-class results

Values below are arithmetic means across the three frozen folds. Support is summed across validation folds; per-fold values are preserved in `per_class_metrics.csv`.

| Task | Class | Precision | Recall | F1 | Total support |
| --- | --- | ---: | ---: | ---: | ---: |
| S6 | blocker | 0.111945 | 0.264149 | 0.157230 | 2,233 |
| S6 | critical | 0.148411 | 0.215880 | 0.175308 | 5,200 |
| S6 | major | 0.240774 | 0.168558 | 0.197609 | 15,361 |
| S6 | normal | 0.822080 | 0.812505 | 0.817117 | 115,963 |
| S6 | minor | 0.153916 | 0.156459 | 0.154117 | 7,932 |
| S6 | trivial | 0.139139 | 0.160059 | 0.144640 | 3,114 |
| S3 | HIGH | 0.337486 | 0.540113 | 0.415135 | 22,794 |
| S3 | MEDIUM | 0.835396 | 0.698188 | 0.760334 | 115,963 |
| S3 | LOW | 0.221354 | 0.326365 | 0.261373 | 11,046 |
| S2 | HIGH_IMPACT | 0.347914 | 0.523796 | 0.417671 | 22,794 |
| S2 | LOWER_IMPACT | 0.906058 | 0.823406 | 0.862664 | 127,009 |

S2 HIGH_IMPACT precision was 0.347914, recall 0.523796, and F1 0.417671. The frozen informational precision guard passed because precision remained above 0.30. Complete labeled confusion matrices for all nine folds are in `confusion_matrices.json`.

## Primary comparison with B1.5

| Task | B1.6 mean | B1.5 mean | Absolute delta | Relative delta | Fold deltas | B1.6 fold wins | New winner? |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| S6 | 0.274337 | 0.277623 | -0.003286 | -1.184% | -0.004225 / -0.003818 / -0.001816 | 0/3 | No |
| S3 | 0.478947 | 0.480109 | -0.001162 | -0.242% | -0.000199 / -0.002381 / -0.000906 | 0/3 | No |
| S2 | 0.640167 | 0.642162 | -0.001995 | -0.311% | -0.002183 / -0.002099 / -0.001703 | 0/3 | No |

B1.6 remained above A2 by +0.003911/+0.009386/+0.002831 for S6/S3/S2, above frozen standalone B1 MPNet by +0.023112/+0.018602/+0.064768, and above B2-Lite by +0.087327/+0.086455/+0.070094. However, B1.5 remains the development winner on every task because B1.6 was lower on all nine paired folds.

## Answer to the research question

Replacing truncated single-sequence MPNet embeddings with complete multi-chunk pooled MPNet document embeddings did **not** improve the fixed lexical-semantic fusion under this protocol. The result is consistent across all three tasks and all nine paired folds, although the losses are small, particularly for S3.

This negative result is informative: 43,152 of 207,575 development documents required multiple chunks; the encoder processed 407,429 chunks, averaging 1.962804 chunks per document, with no chunk cap. Recovering the omitted tail content through simple normalized mean pooling therefore did not translate into improved development Macro-F1. This does not establish that long text lacks useful signal or that all chunk-aware methods would fail; it shows only that this pre-specified frozen encoder and unweighted arithmetic pooling strategy did not outperform B1.5's single-sequence representation.

```text
SEMANTIC_EMBEDDINGS_REGENERATED = NO
COMPETITIVE_MODELS_REFITTED = 0
EXISTING_CHECKPOINTS_REUSED = YES
MODEL_SELECTION_DATA = DEVELOPMENT_ONLY
LOCKED_TEST_TOKENIZED = NO
LOCKED_TEST_EMBEDDED = NO
LOCKED_TEST_MODEL_PERFORMANCE_ACCESSED = NO
LOCKED_TEST_USED_FOR_TUNING = NO
FINAL_LOCKED_TEST_EVALUATION_STARTED = NO
```
