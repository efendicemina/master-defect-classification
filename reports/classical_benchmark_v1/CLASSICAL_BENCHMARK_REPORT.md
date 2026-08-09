# Classical Benchmark V1

## Objective and frozen evaluation protocol

Phase A1 reproduces controlled classical NLP baselines for S6, S3, and S2 using only frozen
DEVELOPMENT data and the persisted pooled temporal-CV manifests. Vectorizers are fitted on each
fold's TRAIN text only. No locked-test artifact or performance was accessed.

## Models, representations, and matrix

The competitive matrix contains 108 fits: three tasks × WORD, CHAR, and WORD_CHAR TF-IDF ×
Logistic Regression and LinearSVC × no/balanced class weighting × three folds. C is fixed at 1.0.
Sparse float32 matrices were processed one representation/fold at a time. Dummy most-frequent and
stratified fits are non-competitive references. Scikit-learn 1.8 no longer supports multiclass
liblinear, so the S6/S3 Logistic Regression fits use the documented `lbfgs` compatibility solver;
C remains fixed at 1.0. Stage 0 is ENGINEERING_ONLY and excluded here.

## Results

### S6

Best development-CV configuration: `CHAR / LINEARSVC / BALANCED`; mean Macro-F1 0.259631 (population SD 0.012134).

Fold Macro-F1 was 0.244007, 0.261299, and 0.273588. Mean balanced accuracy was 0.270267,
accuracy 0.654808, and weighted F1 0.660952. Mean fold-level class F1 was strongest for
`normal` (0.804509) and weak for blocker (0.140844), critical (0.155710), major (0.199072),
minor (0.147513), and trivial (0.110139), confirming that S6 remains difficult under imbalance.
### S3

Best development-CV configuration: `WORD / LOGREG / BALANCED`; mean Macro-F1 0.461889 (population SD 0.010079).

Fold Macro-F1 was 0.447641, 0.468672, and 0.469353. Mean balanced accuracy was 0.526577,
accuracy 0.606504, and weighted F1 0.640881. Mean fold-level class F1 was 0.409717 for HIGH,
0.723164 for MEDIUM, and 0.252785 for LOW.
### S2

Best development-CV configuration: `WORD_CHAR / LOGREG / BALANCED`; mean Macro-F1 0.627266 (population SD 0.006911).

Fold Macro-F1 was 0.617493, 0.632303, and 0.632001. Mean balanced accuracy was 0.679241,
accuracy 0.750611, and weighted F1 0.776336. HIGH_IMPACT mean precision was 0.322163,
recall 0.576779, and F1 0.413035. The informational `LEGACY_REPRODUCTION_GUARD` is PASS
because mean HIGH_IMPACT precision is at least 0.30; it did not affect ranking.

## Controlled factor comparison

Across the 12 aggregated configurations in each representation family, mean Macro-F1 was
0.424003 for WORD_CHAR, 0.416711 for WORD, and 0.409465 for CHAR. Across 18 configurations per
weighting policy, BALANCED averaged 0.441237 versus 0.392216 for NONE. Across 18 configurations
per classifier, LinearSVC averaged 0.423054 versus 0.410399 for Logistic Regression. These are
descriptive averages over the fixed matrix, not significance tests; interactions are retained in
`leaderboard.csv`.

Detailed rankings, per-class weaknesses, weighting and representation comparisons are retained in
the adjacent CSV/JSON reports. Three temporal folds do not support claims of statistical
significance or final generalization.

## Runtime, warnings, and limitations

The initial matrix attempt took 3176.451 seconds and exposed the scikit-learn 1.8 multiclass
liblinear incompatibility. The checkpointed compatibility rerun took 3302.943 seconds, for a
cumulative wall-clock execution time of 6479.394 seconds. Successful model-fit time summed to
5376.461 seconds, excluding vectorization and orchestration. Peak memory was unavailable because
the macOS `time` utility could not query `kern.clockrate` in the sandbox. Recorded warning count: 0.
Failed competitive fits: 0. This benchmark uses fixed feature caps and fixed baseline
regularization; it is not a hyperparameter search. The S2 0.30 HIGH_IMPACT precision guard is
reported only as `LEGACY_REPRODUCTION_GUARD` information and does not filter the leaderboard.

```text
MODELS_FITTED = YES
MODEL_SELECTION_DATA = DEVELOPMENT_ONLY
FROZEN_CV_MANIFESTS_USED = YES
CV_FOLDS_RECONSTRUCTED_DURING_MODELLING = NO
LOCKED_TEST_MODEL_PERFORMANCE_ACCESSED = NO
LOCKED_TEST_USED_FOR_TUNING = NO
```
