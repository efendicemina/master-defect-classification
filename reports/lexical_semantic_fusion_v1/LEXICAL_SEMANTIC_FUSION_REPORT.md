# Phase B1.5 — Controlled Lexical + Semantic Fusion

## 1. Research question and frozen protocol

Phase B1.5 asks whether adding fixed, frozen MPNet semantic information to each task's already
selected Phase A2 TF-IDF representation improves temporal Eclipse bug-severity classification.
This is a complementarity test, not a hyperparameter search. Protocol v1, canonical `Summary +
"\n\n" + Description` text, targets, class order, duplicate policy, temporal folds, metrics, and
all Phase A2 classifier settings remained unchanged.

This phase is strictly **DEVELOPMENT-only**. Persisted authenticated CV manifests supplied every
TRAIN and VALIDATION membership. TF-IDF was fitted separately on TRAIN text and only transformed
VALIDATION text. Locked-test data, embeddings, labels, predictions, errors, and performance were
not accessed or used for tuning.

## 2. Frozen A2 and B1 references

The Phase A2 report and leaderboard SHA-256 values remained
`4ab436dc330e80a66a1cea5dcfb9c364c8d6f947c4d8f757fd25486c0ecd2a8d` and
`5343448b359fc181f2e341ff6c9a942767f3334d214b91cce7aa41405cdcf54f`. The Phase B1 report and
leaderboard remained `7809d204c0f4b85eb162c071742bf92a6d7d078b355c76537ffb382fc4f09804` and
`5b4e1c7437827dfd547d2aca46e1e5516203e641f1b070970c31431acc3d7268`. Phase A1 references were
also authenticated before modelling. None of these reports was modified.

The reused semantic cache contains normalized float32 embeddings from
`sentence-transformers/all-mpnet-base-v2`, revision
`e8c3b32edf5434bc2275fc9bab85f82640a19130`, dimension 768. Cache reuse failed closed unless its
protocol fingerprint, complete 207,575-row development fingerprint, model identity/revision,
dimension, normalization policy, unique `source_project + issue_id` identities, and exact row
membership matched.

## 3. Fusion design and exact matrix

For each fold, the task-specific A2 vectorizer produced sparse float32 TRAIN and VALIDATION
matrices. The aligned MPNet rows were selected by stable identity, converted to a sparse CSR block,
and appended using SciPy sparse horizontal concatenation. The semantic block weight was the
pre-specified constant 1.0. The lexical matrix was never densified, and no semantic weight, C,
threshold, representation, text view, or classifier was tuned.

| Task | Frozen lexical block | Classifier | C | Hybrid dimensions by fold |
|---|---|---|---:|---|
| S6 | S6-R3: CHAR `char_wb` (2,5), `min_df=2`, cap 150,000 | LinearSVC, balanced | 0.25 | 150,000 + 768 = 150,768 |
| S3 | S3-R5: WORD (1,2), `min_df=2`, cap 250,000 | LogisticRegression, balanced | 2.0 | 250,000 + 768 = 250,768 |
| S2 | S2-R3: WORD (1,2) + CHAR `char_wb` (3,5), `min_df=2`, caps 250,000 each | LogisticRegression, balanced | 2.0 | 500,000 + 768 = 500,768 |

The matrix was exactly three tasks × three frozen folds = **nine competitive fits**. All nine
succeeded, no convergence warnings occurred, and no additional competitive configuration was run.

## 4. S6 result

| Method | Fold Macro-F1 | Mean Macro-F1 | Balanced Accuracy | Accuracy | Weighted F1 |
|---|---|---:|---:|---:|---:|
| A2 lexical | 0.248861, 0.272952, 0.289464 | 0.270426 | 0.288933 | 0.675252 | 0.674010 |
| B1 MPNet | 0.241226, 0.249967, 0.262481 | 0.251225 | 0.305243 | 0.671505 | 0.665411 |
| B1.5 hybrid | 0.261399, 0.278600, 0.292871 | **0.277623** | 0.299459 | 0.671693 | 0.674298 |

Hybrid population SD Macro-F1 was 0.012867. Hybrid minus A2 was **+0.007197** Macro-F1,
+0.010526 Balanced Accuracy, -0.003559 Accuracy, and +0.000288 Weighted F1. Hybrid minus B1 was
+0.026398 Macro-F1.

Hybrid per-class F1 was blocker 0.163596, critical 0.179477, major 0.201481, normal 0.818329,
minor 0.156642, and trivial 0.146212. Versus A2, changes were +0.011705, +0.005384, +0.010750,
-0.002183, +0.005226, and +0.012303 respectively. The hybrid therefore improved Macro-F1 and five
of six class F1 values, while slightly reducing normal-class F1.

## 5. S3 result

| Method | Fold Macro-F1 | Mean Macro-F1 | Balanced Accuracy | Accuracy | Weighted F1 |
|---|---|---:|---:|---:|---:|
| A2 lexical | 0.452463, 0.476688, 0.479532 | 0.469561 | 0.509896 | 0.644051 | 0.667746 |
| B1 MPNet | 0.453555, 0.458937, 0.468545 | 0.460346 | 0.449621 | 0.735205 | 0.718653 |
| B1.5 hybrid | 0.464085, 0.487861, 0.488382 | **0.480109** | 0.525367 | 0.646232 | 0.670866 |

Hybrid population SD Macro-F1 was 0.011333. Hybrid minus A2 was **+0.010548** Macro-F1,
+0.015471 Balanced Accuracy, +0.002181 Accuracy, and +0.003120 Weighted F1. Hybrid minus B1 was
+0.019764 Macro-F1.

Hybrid per-class F1 was HIGH 0.419531, MEDIUM 0.759092, and LOW 0.261705. Versus A2, changes were
+0.012899, -0.000285, and +0.019031 respectively. The complementary gain was concentrated in HIGH
and LOW while majority-class performance was effectively unchanged.

## 6. S2 result

| Method | Fold Macro-F1 | Mean Macro-F1 | Balanced Accuracy | Accuracy | Weighted F1 |
|---|---|---:|---:|---:|---:|
| A2 lexical | 0.629693, 0.639337, 0.642979 | 0.637336 | 0.667543 | 0.778680 | 0.794779 |
| B1 MPNet | 0.573743, 0.577057, 0.575398 | 0.575399 | 0.677785 | 0.659852 | 0.707098 |
| B1.5 hybrid | 0.635093, 0.643113, 0.648280 | **0.642162** | 0.674821 | 0.780204 | 0.796724 |

Hybrid population SD Macro-F1 was 0.005425. Hybrid minus A2 was **+0.004825** Macro-F1,
+0.007278 Balanced Accuracy, +0.001524 Accuracy, and +0.001946 Weighted F1. Hybrid minus B1 was
+0.066763 Macro-F1.

Hybrid HIGH_IMPACT precision, recall, and F1 were 0.351351, 0.523472, and 0.420040; the legacy
precision guard passed. Versus A2, these changed by +0.005387, +0.015540, and +0.008965.
LOWER_IMPACT F1 was 0.864284, +0.000686 versus A2. The hybrid retained lexical precision while
recovering some semantic minority recall, unlike the standalone B1 model's high-recall,
low-precision tradeoff.

## 7. Complementarity interpretation and class-level effects

The fixed hybrid beat the A2 lexical winner on all three development-CV tasks: +0.007197 for S6,
+0.010548 for S3, and +0.004825 for S2 Macro-F1. It also beat standalone B1 MPNet Macro-F1 on all
three tasks. Every hybrid fold exceeded its corresponding A2 fold. This is evidence that the fixed
MPNet block contains complementary development-CV signal, but three folds do not support a claim
of statistical significance and do not guarantee locked-test improvement.

The class-level pattern is more informative than the mean alone. S6 improved most non-majority
classes, S3 improved HIGH and LOW without materially changing MEDIUM, and S2 improved both
HIGH_IMPACT precision and recall. The semantic block therefore helped the optimized lexical models
without reproducing all weaknesses of the standalone semantic classifiers.

## 8. Runtime and resources

Total orchestration runtime was 2,383.821 seconds. Recorded classifier fit/prediction time was
929.918/6.199 seconds for S6, 197.319/0.363 for S3, and 595.533/1.957 for S2. Remaining time was
primarily train-only TF-IDF construction, sparse semantic conversion/concatenation, cache loading,
validation, and reporting. S2 had the widest matrix at 500,768 columns; S6 LinearSVC had the
largest classifier-fit time. Fits ran sequentially and checkpointed individually.

The engineering smoke and competitive run emitted benign PyArrow hardware-cache probe messages
because sandboxed macOS denied `sysctlbyname`; these did not change data or methodology. There
were no fit failures or convergence warnings.

## 9. Limitations and scope statement

This experiment tests only an unscaled weight-1.0 concatenation of the fixed A2 lexical block and
frozen MPNet. It does not determine whether another fusion weight, scaling scheme, nonlinear
classifier, fine-tuned encoder, chunking strategy, or threshold would perform better. Testing any
of those would require a separately pre-specified phase. Sparse storage of the dense-valued MPNet
block preserves compatibility and prevents lexical densification, but adds CSR indexing overhead.

No transformer encoder was trained or fine-tuned. The existing development-only MPNet cache was
reused. Protocol v1 and all earlier reports remained unchanged.

```text
MODELS_FITTED = YES
TRANSFORMER_ENCODERS_FINE_TUNED = NO
MPNET_CACHE_REUSED = YES
MODEL_SELECTION_DATA = DEVELOPMENT_ONLY
FROZEN_CV_MANIFESTS_USED = YES
LOCKED_TEST_EMBEDDED = NO
LOCKED_TEST_MODEL_PERFORMANCE_ACCESSED = NO
LOCKED_TEST_USED_FOR_TUNING = NO
```
