# Classical Optimization V1

## 1. Objective

Phase A2 performs targeted, pre-specified development-CV optimization of the fixed Phase-A1
winning classical model family for S6, S3, and S2. The primary selection metric is mean Macro-F1;
ties are resolved by minimum-fold Macro-F1, mean balanced accuracy, lower feature count, and then
configuration ID.

## 2. Frozen protocol statement

Only frozen DEVELOPMENT Parquets and persisted pooled CV memberships loaded through
`load_cv_membership()` were used. Vectorizers were fitted separately on each fold's TRAIN text,
validation was transformed afterward, and sparse matrices were never densified. Protocol v1,
targets, folds, duplicate handling, and Phase-A1 reports were unchanged. No locked-test artifact
was read.

## 3. Phase-A1 references

| Task | A1 configuration | Fold Macro-F1 | Mean |
|---|---|---|---:|
| S6 | CHAR / LinearSVC / balanced / C=1 | 0.244007, 0.261299, 0.273588 | 0.259631 |
| S3 | WORD / LogisticRegression / balanced / C=1 | 0.447641, 0.468672, 0.469353 | 0.461889 |
| S2 | WORD_CHAR / LogisticRegression / balanced / C=1 | 0.617493, 0.632303, 0.632001 | 0.627266 |

## 4. Search-budget definition

- A2.1: 3 tasks × 5 C values × 3 folds = 45 fits.
- A2.2: 18 representation configurations × 3 folds = 54 fits.
- A2.3: 3 tasks × 2 new text views × 3 folds = 18 new fits. Nine identical
  SUMMARY_DESCRIPTION evaluations were reused from A2.2.
- Total new competitive fits: 117. Successful: 117. Failed: 0.

No unapproved search axis was introduced.

## 5. A2.1 regularization results

| Task | C=0.25 | C=0.50 | C=1.00 | C=2.00 | C=4.00 | Selected C |
|---|---:|---:|---:|---:|---:|---:|
| S6 | **0.269042** | 0.264943 | 0.259631 | 0.253228 | 0.247225 | **0.25** |
| S3 | 0.442301 | 0.453760 | 0.461889 | **0.464983** | 0.463690 | **2.00** |
| S2 | 0.622174 | 0.629227 | 0.632418 | **0.633236** | 0.630222 | **2.00** |

Values are mean three-fold Macro-F1. Complete fold values and secondary metrics are in
`regularization_results.csv`.

## 6. A2.2 representation results

| Task | R1 | R2 | R3 | R4 | R5 | R6 | Selected |
|---|---:|---:|---:|---:|---:|---:|---|
| S6 | 0.269042 | 0.269867 | **0.270426** | 0.268942 | 0.269202 | 0.270115 | S6-R3 |
| S3 | 0.464983 | 0.441519 | 0.459908 | 0.465678 | **0.469561** | 0.466619 | S3-R5 |
| S2 | 0.633236 | 0.636140 | **0.637336** | 0.631278 | 0.632106 | 0.634245 | S2-R3 |

The selected representations are:

- S6-R3: CHAR `char_wb`, n-grams (2,5), `min_df=2`, 150,000 features.
- S3-R5: WORD n-grams (1,2), `min_df=2`, 250,000 features.
- S2-R3: WORD (1,2) and CHAR `char_wb` (3,5), `min_df=2`, 250,000 features each.

All 250,000-feature variants completed without resource failure.

## 7. A2.3 text ablation results

| Task | Summary + Description | Summary only | Description only | Selected view |
|---|---:|---:|---:|---|
| S6 | **0.270426** | 0.247004 | 0.256454 | SUMMARY_DESCRIPTION |
| S3 | **0.469561** | 0.429532 | 0.458607 | SUMMARY_DESCRIPTION |
| S2 | **0.637336** | 0.601375 | 0.629128 | SUMMARY_DESCRIPTION |

The canonical combined view remained preferable for every task. The combined evaluations were
reused from A2.2; the other 18 fits used independently fitted view-specific vectorizers.

## 8. Final development-selected configurations

### S6

CHAR `char_wb` (2,5), `min_df=2`, 150,000 features; LinearSVC; balanced; C=0.25;
SUMMARY_DESCRIPTION.

- Fold Macro-F1: 0.248861, 0.272952, 0.289464
- Mean / population SD Macro-F1: 0.270426 / 0.016672
- Balanced accuracy: 0.288933
- Accuracy: 0.675252
- Weighted F1: 0.674010
- Mean per-class F1: blocker 0.151891, critical 0.174093, major 0.190731,
  normal 0.820513, minor 0.151416, trivial 0.133909

### S3

WORD (1,2), `min_df=2`, 250,000 features; LogisticRegression with the documented scikit-learn
1.8 multiclass `lbfgs` compatibility solver; balanced; C=2.0; SUMMARY_DESCRIPTION.

- Fold Macro-F1: 0.452463, 0.476688, 0.479532
- Mean / population SD Macro-F1: 0.469561 / 0.012146
- Balanced accuracy: 0.509896
- Accuracy: 0.644051
- Weighted F1: 0.667746
- Mean per-class F1: HIGH 0.406632, MEDIUM 0.759378, LOW 0.242673

### S2

WORD (1,2) plus CHAR `char_wb` (3,5), `min_df=2`, 250,000 features each;
LogisticRegression/liblinear; balanced; C=2.0; SUMMARY_DESCRIPTION.

- Fold Macro-F1: 0.629693, 0.639337, 0.642979
- Mean / population SD Macro-F1: 0.637336 / 0.005606
- Balanced accuracy: 0.667543
- Accuracy: 0.778680
- Weighted F1: 0.794779
- HIGH_IMPACT precision: 0.345963
- HIGH_IMPACT recall: 0.507932
- HIGH_IMPACT F1: 0.411075
- Informational `LEGACY_REPRODUCTION_GUARD`: PASS

## 9. Phase-A1 versus Phase-A2 changes

| Task | A1 mean | A2 mean | Absolute delta | Relative change | Fold deltas |
|---|---:|---:|---:|---:|---|
| S6 | 0.259631 | 0.270426 | +0.010794 | +4.158% | +0.004854, +0.011653, +0.015876 |
| S3 | 0.461889 | 0.469561 | +0.007672 | +1.661% | +0.004823, +0.008015, +0.010178 |
| S2 | 0.627266 | 0.637336 | +0.010071 | +1.605% | +0.012199, +0.007034, +0.010978 |

All three folds improved for every task, but three folds do not support a statistical-significance
claim.

## 10. Per-class changes

Relative to the A1 task winners, S6 mean F1 changed by blocker +0.011047, critical +0.018383,
major -0.008341, minor +0.003903, and trivial +0.023770 (normal +0.016004). S3 changed by
HIGH -0.003085, MEDIUM +0.036214, and LOW -0.010112. S2 HIGH_IMPACT precision changed by
+0.023801, recall by -0.068847, and F1 by -0.001960. Macro-F1 improvements therefore do not
imply uniform class-level improvement.

## 11. Fold stability

The A2 population SD was 0.016672 for S6, 0.012146 for S3, and 0.005606 for S2. Scores increased
chronologically across the three folds for each selected configuration. S2 was the most stable;
S6 showed the widest fold spread.

## 12. Runtime and resources

Total orchestration runtime was 9,997.315 seconds. Successful model-fit time was 4,986.487 seconds:
2,669.247 in A2.1, 1,890.672 in A2.2, and 426.568 in the 18 new A2.3 fits. No convergence warning
or competitive failure was recorded. PyArrow emitted benign sandbox `sysctlbyname` capability
messages. Reliable peak RSS was unavailable in this sandbox.

## 13. Search limitations and development-CV caveat

This bounded optimization did not tune class weights, thresholds, preprocessing, solvers, losses,
or any unapproved axis. It performs model selection on the same frozen development CV used in
Phase A1. The selected scores are model-selection estimates, not unbiased final-generalization
estimates. Development improvement does not guarantee locked-test improvement; the locked test is
reserved for a later final evaluation.

```text
MODELS_FITTED = YES
MODEL_SELECTION_DATA = DEVELOPMENT_ONLY
PHASE_A1_REPORTS_MODIFIED = NO
FROZEN_CV_MANIFESTS_USED = YES
CV_FOLDS_RECONSTRUCTED_DURING_MODELLING = NO
LOCKED_TEST_MODEL_PERFORMANCE_ACCESSED = NO
LOCKED_TEST_USED_FOR_TUNING = NO
```
