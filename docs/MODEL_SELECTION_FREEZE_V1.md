# Development Model Selection Freeze V1

Freeze ID: `model-selection-v1`  
Freeze date: 2026-08-14  
Source commit before freeze: `c6a68cf5ab14335e5d7fd55fa04a2d9c6c5cd503`

Protocol ID: `eclipse-severity-v1`  
Protocol SHA-256: `85faacae1e7f411d68803653388611c37c62730ea9217e9700a0ad6ac41b7cda`

## Decision

Development model selection is closed. No additional model family, representation,
hyperparameter, semantic-weight, threshold, class-weight, architecture, or feature search
may be performed before final locked-test evaluation.

The selected final family is **B4-H adapted RTA + lexical fusion**. The direct B4-H
classification heads are not selected for final locked-test scoring.

Selection is based on the frozen primary metric: mean Macro-F1 across the three pooled
chronological development CV folds.

## Development evidence at freeze

| Task | B1.5 Macro-F1 | B4-H fusion Macro-F1 | B4-H accuracy | B4-H balanced accuracy | B4-H weighted F1 | Paired fold wins |
|---|---:|---:|---:|---:|---:|---:|
| S6 | 0.2776230020 | 0.2957116369 | 0.6786597359 | 0.3195588012 | 0.6820881451 | 3/3 |
| S3 | 0.4801093614 | 0.4967305952 | 0.6576278107 | 0.5454309939 | 0.6813260175 | 3/3 |
| S2 | 0.6421618749 | 0.6540443359 | 0.7874565435 | 0.6888012544 | 0.8034381130 | 3/3 |

B4-H adapted lexical fusion wins all nine paired fold/task comparisons against B1.5.
This is evidence of consistency across the frozen development folds; it is not treated as
a formal statistical-significance claim.

## Frozen transformer configuration

- Seed: `20260809`
- Base model: `Colorful/RTA`
- Revision: `56cc614ee7cf17b8c1875e6848037f9e5bafc41a`
- PEFT: AdaLoRA `0.19.1`
- Text: canonical Summary + Description
- Max length: `512`
- Padding/dtype: original fixed-padding, `float32`
- Epochs: `3`
- Optimizer: AdamW
- Learning rate: `1e-4`
- Weight decay: `0.01`
- Warmup fraction: `0.06`
- Linear LR schedule
- Max grad norm: `1.0`
- No early stopping
- Train micro-batch: `8`
- Gradient accumulation: `4`
- Effective batch size: `32`
- Focal gamma: `2.0`
- Train-only balanced class weights normalized to mean one
- Equal S6/S3/S2 task loss weights
- AdaLoRA query/value targets, init rank `12`, target rank `8`
- LoRA alpha `16`, dropout `0.05`
- tinit fraction `0.10`, tfinal fraction `0.20`
- Allocation target updates `100`

## Frozen fusion configuration

Adapted semantic feature: CLS representation, 768 dimensions, L2-normalized,
semantic weight `1.0`.

- S6: char TF-IDF `char_wb`, n-grams `(2,5)`, min_df=2, max_features=150000,
  balanced LinearSVC, C=0.25.
- S3: word TF-IDF n-grams `(1,2)`, min_df=2, max_features=250000,
  balanced LogisticRegression, C=2.0, solver `lbfgs`.
- S2: word TF-IDF `(1,2)` plus char_wb TF-IDF `(3,5)`, min_df=2,
  250000 max features for each branch, balanced LogisticRegression, C=2.0,
  solver `liblinear`.

## Frozen memberships

DEVELOPMENT:
- Rows: `207575`
- Membership SHA-256:
  `4f62fdf4164594126c421955804b654cd5d5f8f7b46ada345ae0cffa71460d0f`

LOCKED TEST:
- Rows: `50675`
- Membership SHA-256:
  `c18f2320c896bf2bdcb94d2447ce9a04d17e6923f03b169f29b8174b8a5cf681`

## Frozen final-training rule

Train exactly one new shared B4-H AdaLoRA model on the complete DEVELOPMENT
membership. It starts from the same pinned RTA revision and uses the frozen B4-H
configuration above for exactly three epochs.

No locked-test data may influence class weights, vocabulary, TF-IDF fitting,
feature fitting, adapter training, early stopping, thresholds, calibration, or any
other learned or selected quantity.

After final adapter training:
1. extract adapted semantic representations for DEVELOPMENT;
2. fit the frozen S6/S3/S2 lexical vectorizers and fusion classifiers on DEVELOPMENT only;
3. freeze/save the fitted final pipeline and provenance;
4. only then unlock the final locked-test workflow;
5. perform exactly one final prediction/scoring pass on LOCKED TEST.

No CV-fold checkpoint is used as the final model.

## Final locked-test scoring rule

Only **B4-H adapted RTA + lexical fusion** is scored on locked test for S6, S3 and S2.
Direct heads are not scored.

Primary metric: Macro-F1 with the complete frozen class order.

Secondary metrics: balanced accuracy, accuracy, weighted F1, per-class
precision/recall/F1/support, confusion matrix, and S2 HIGH_IMPACT
precision/recall/F1.

Unlock:
`DEFECT_CLASSIFIER_UNLOCK_LOCKED_TEST=FINAL_EVALUATION_ONLY`

After locked-test metrics are produced, no model, threshold, feature, mapping, or
training change may be made based on those results.

## Access state at freeze

```text
MODEL_SELECTION_DATA = DEVELOPMENT_ONLY
DEVELOPMENT_MODEL_SELECTION_CLOSED = YES
LOCKED_TEST_TOKENIZED = NO
LOCKED_TEST_EMBEDDED = NO
LOCKED_TEST_MODEL_PERFORMANCE_ACCESSED = NO
LOCKED_TEST_USED_FOR_TUNING = NO
```
