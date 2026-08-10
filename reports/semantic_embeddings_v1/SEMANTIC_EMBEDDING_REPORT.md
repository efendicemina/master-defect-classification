# Phase B1 — Frozen Semantic Embedding Benchmark

## 1. Research question and frozen protocol

Phase B1 asks whether generic, pretrained sentence embeddings improve Eclipse bug-severity
classification over the development-selected Phase A2 classical models. The experiment uses only
the frozen protocol-v1 canonical `Summary + "\n\n" + Description` text and the unchanged S6, S3,
and S2 targets, temporal folds, class order, duplicate policy, and metric policy.

This is a **DEVELOPMENT-only** model-selection experiment. Locked-test text, targets, embeddings,
predictions, and performance were not accessed. Cache membership was required to equal the frozen
207,575-row development membership exactly; consequently its overlap with the disjoint locked
partition is zero. Development-CV results do not guarantee locked-test generalization.

The immutable classical references are Phase A1 report/leaderboard SHA-256
`7a86566aba18d54f028f09781a7cb9a196a6c59347819f1020daeec75591104b` /
`566aef68422d8782bcf2671da825012e35fa9e74840e93cf0ca57ffcd2b4c484`, and Phase A2
report/leaderboard SHA-256
`4ab436dc330e80a66a1cea5dcfb9c364c8d6f947c4d8f757fd25486c0ecd2a8d` /
`5343448b359fc181f2e341ff6c9a942767f3334d214b91cce7aa41405cdcf54f`.

## 2. Encoder provenance and environment

| ID | Public model | Resolved revision | Dimension | Native maximum | Device / dtype |
|---|---|---|---:|---:|---|
| E1 | `sentence-transformers/all-MiniLM-L6-v2` | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` | 384 | 256 | MPS / float32 |
| E2 | `sentence-transformers/all-mpnet-base-v2` | `e8c3b32edf5434bc2275fc9bab85f82640a19130` | 768 | 384 | MPS / float32 |

The run used Python 3.12.13 on Apple Silicon macOS with PyTorch 2.13.0,
sentence-transformers 5.7.0, transformers 5.14.1, NumPy 2.5.2, SciPy 1.18.0,
scikit-learn 1.8.0, and PyArrow 21.0.0. PyTorch reported MPS both built and available.
Both deterministic smoke tests ran on MPS and produced finite, correctly shaped, unit-normalized
embeddings. Remote custom code was disabled. E1 used `BertModel`/`BertTokenizer`; E2 used
`MPNetModel`/`MPNetTokenizer`; both used mean pooling.

## 3. Embedding generation, cache, and truncation

Each encoder was placed in evaluation mode, all parameters remained frozen, inference ran under
no-gradient inference mode, and no labels were supplied. Embeddings were generated once for all
development rows, L2-normalized, and stored as float32. No encoder fine-tuning or classifier-head
training was performed. The initial/final batch size was 64 for both models; there were no batch
reductions, MPS fallbacks, or unsupported-operation failures.

The ignored cache is project-sharded Parquet keyed only by `source_project` and `issue_id`, with a
fixed-size embedding column. It contains no targets or raw text. Metadata binds reuse to protocol,
development membership, text policy, exact model revision, tokenizer/model settings,
normalization, and dimension. Each shard has a content checksum, and reuse fails closed on
membership or provenance drift. Only one complete encoder matrix is loaded during classification.

| Encoder | Development vectors | Cache size | Generation runtime | Truncated | Percentage |
|---|---:|---:|---:|---:|---:|
| E1 | 207,575 | 295,411,578 bytes | 544.643 s | 64,223 | 30.940% |
| E2 | 207,575 | 591,816,188 bytes | 6,216.566 s | 43,152 | 20.789% |

The project-level counts and token-length summaries are in `truncation_audit.csv`. Native tokenizer
truncation was used; sequence lengths were not tuned and no custom chunking was introduced. The
substantial truncation rates, especially E1's 30.9%, mean these representations often omit the tail
of long reports and motivate a separately specified future long-document experiment.

## 4. Fixed classification matrix and selection

The complete matrix was S6/S3/S2 × E1/E2 × LogisticRegression/LinearSVC × none/balanced × three
frozen folds: exactly 72 competitive fits. `C=1.0` was fixed. Supervised classifiers saw only the
training membership for a fold and predicted only its validation membership. There was no tuning
of encoders, pooling, normalization, sequence length, classifier solvers, thresholds, or C.

Selection used mean three-fold Macro-F1, then minimum-fold Macro-F1, mean Balanced Accuracy, and a
deterministic configuration ID. All 72 fits succeeded; none emitted a convergence warning. Total
classification orchestration runtime was 2,312.994 seconds.

## 5. S6 results

The winner was **E2 / LinearSVC / balanced / C=1**.

- Fold Macro-F1: 0.241226, 0.249967, 0.262481
- Mean / population SD Macro-F1: 0.251225 / 0.008723
- Balanced Accuracy / Accuracy / Weighted F1: 0.305243 / 0.671505 / 0.665411
- Mean per-class F1: blocker 0.150669, critical 0.160633, major 0.124619, normal
  0.822288, minor 0.091190, trivial 0.157949

Phase A2 Macro-F1 was 0.270426, so B1 changed it by -0.019201 (-7.10%). Balanced Accuracy changed
by +0.016310 and Weighted F1 by -0.008599. Relative to A2, per-class F1 changed by blocker
-0.001222, critical -0.013460, major -0.066112, normal +0.001775, minor -0.060226, and trivial
+0.024040. The B1 fold values were lower than A2's 0.248861, 0.272952, 0.289464 in every fold.

## 6. S3 results

The winner was **E2 / LinearSVC / balanced / C=1**.

- Fold Macro-F1: 0.453555, 0.458937, 0.468545
- Mean / population SD Macro-F1: 0.460346 / 0.006200
- Balanced Accuracy / Accuracy / Weighted F1: 0.449621 / 0.735205 / 0.718653
- Mean per-class F1: HIGH 0.380138, MEDIUM 0.838094, LOW 0.162805

Phase A2 Macro-F1 was 0.469561, so B1 changed it by -0.009215 (-1.96%). Balanced Accuracy changed
by -0.060275 and Weighted F1 by +0.050907. Per-class F1 changed by HIGH -0.026494, MEDIUM
+0.078716, and LOW -0.079868. B1's folds were lower than A2's 0.452463, 0.476688, 0.479532,
apart from a small gain in fold 1.

## 7. S2 results

The winner was **E2 / LinearSVC / balanced / C=1**.

- Fold Macro-F1: 0.573743, 0.577057, 0.575398
- Mean / population SD Macro-F1: 0.575399 / 0.001353
- Balanced Accuracy / Accuracy / Weighted F1: 0.677785 / 0.659852 / 0.707098
- HIGH_IMPACT precision / recall / F1: 0.266153 / 0.703585 / 0.386141
- LOWER_IMPACT mean F1: 0.764657
- Informational legacy precision guard (`>=0.30`): **FAIL**

Phase A2 Macro-F1 was 0.637336, so B1 changed it by -0.061937 (-9.72%). Balanced Accuracy changed
by +0.010242 and Weighted F1 by -0.087681. HIGH_IMPACT precision changed by -0.079810, recall by
+0.195653, and F1 by -0.024934. Thus the balanced semantic classifier traded many more positive
predictions for much higher minority recall but insufficient precision. All B1 folds were below
A2's 0.629693, 0.639337, 0.642979.

## 8. Cross-configuration findings

MPNet (E2) won all three tasks and averaged 0.375366 Macro-F1 across the 12 task-level
configurations, versus 0.359158 for MiniLM (E1). Its larger representation and longer native
context improved selection scores but cost about twice the cache space and over eleven times the
embedding runtime.

LinearSVC won all tasks and averaged 0.372458 Macro-F1 across configurations, versus 0.362065 for
LogisticRegression. Balanced weighting also won all tasks and averaged 0.399652, versus 0.334871
without weights. These are descriptive comparisons within the fixed matrix, not significance
claims or authorization for further tuning.

At class level, semantic embeddings retained strong majority-class performance but did not
uniformly improve rare severities. The S3 MEDIUM and S6 trivial classes improved over A2, while
several rare S6 classes and S3 LOW declined. S2's recall increase came with a precision collapse and
failed the informational legacy guard.

## 9. Runtime, resources, and limitations

Classifier fit time was 521.804 seconds for E1 and 1,760.234 seconds for E2; prediction time was
0.347 and 0.776 seconds, respectively. E2 embedding generation was the dominant cost. The run used
sequential classifiers, float32 cache matrices, project checkpoints, and one encoder matrix in RAM
at a time. No OOM recovery or CPU fallback was needed.

These frozen generic encoders were not adapted to Eclipse terminology, severity cues, or the target
taxonomy. Mean pooling compresses long, heterogeneous reports into one vector, while native
truncation discards content for 20.8–30.9% of rows. Linear downstream models may also be unable to
recover subtle severity distinctions from generic embedding geometry. Phase B1 intentionally did
not test fine-tuning, chunking, hybrid TF-IDF/semantic features, metadata, thresholds, or additional
encoders. Those require separately frozen future phases. Locked-test performance was not accessed,
and no claim of statistical significance is made from three development folds.
