# Phase B2 — Controlled Task-Specific Transformer Fine-Tuning

## 1. Research question and frozen protocol

Phase B2 asks whether supervised task-specific transformer fine-tuning improves Eclipse
bug-severity classification beyond optimized TF-IDF, frozen MPNet, and fixed lexical-semantic
fusion. Protocol v1, canonical `Summary + "\n\n" + Description` text, targets, class order,
duplicate policy, temporal memberships, and metrics remained frozen.

This work remained strictly **DEVELOPMENT-only**. Persisted authenticated CV manifests were loaded;
folds were not reconstructed. Locked-test rows were not loaded, tokenized, trained on, predicted,
or evaluated.

## 2. Previous immutable references

The authenticated comparison references remained unchanged: A2 report/leaderboard SHA-256
`4ab436dc330e80a66a1cea5dcfb9c364c8d6f947c4d8f757fd25486c0ecd2a8d` /
`5343448b359fc181f2e341ff6c9a942767f3334d214b91cce7aa41405cdcf54f`; B1
report/leaderboard `7809d204c0f4b85eb162c071742bf92a6d7d078b355c76537ffb382fc4f09804` /
`5b4e1c7437827dfd547d2aca46e1e5516203e641f1b070970c31431acc3d7268`; and B1.5
report/task summary `e7bcf9afe11ac5b29f0ff0996c5c011351684164048c7afd871062e242bd99c7` /
`0b27212eea2bccd615c228fab6d9d6fd4d7d33a3f5595c41e8a9c656d561ba8f`.

The strongest existing development results remain the B1.5 hybrids: S6 0.277623, S3 0.480109,
and S2 0.642162 mean Macro-F1. S2 HIGH_IMPACT precision/recall/F1 remain
0.351351/0.523472/0.420040.

## 3. Model provenance

The only permitted encoder was `microsoft/deberta-v3-small`, resolved to revision
`a36c739020e01763fe789b4b85e2df55d6180012`. Transformers loaded
`DebertaV2ForSequenceClassification` with `DebertaV2Tokenizer`, remote custom code disabled. The
sequence-classification model contained 141,896,450 parameters, all trainable. The pretrained
checkpoint's language-modelling heads were correctly ignored and a task-specific pooler and
classification head were newly initialized.

## 4. MPS environment and fixed dtype

The environment used Python 3.12.13, PyTorch 2.13.0, Transformers 5.14.1, Accelerate 1.14.0,
SentencePiece 0.2.2, and Protobuf 6.33.6 on Apple Silicon macOS. PyTorch reported MPS built and
available, and all real training operations ran on `mps`.

The pre-specified bf16 probe succeeded: forward pass, balanced weighted loss, backward pass,
finite-gradient check, AdamW optimizer step, evaluation, model/tokenizer save, and checkpoint
reload all passed. Therefore `bfloat16` was fixed for B2. No dtype fallback, MPS fallback, or OOM
event occurred.

## 5. Fixed fine-tuning configuration

The frozen configuration was max length 256, learning rate 4.5e-5, three epochs, TRAIN batch size
8, evaluation batch size 16, gradient accumulation 1, dynamic padding, standard AdamW, and the
project seed 20260809. There was no early stopping. The research model was specified as the state
after epoch 3; validation scores could not select an epoch. No learning-rate, epoch, batch-size,
sequence-length, model, threshold, or class-weight search was allowed.

## 6. Class-weight computation

Balanced cross-entropy weights are computed separately from each frozen fold's TRAIN labels as
`n_train / (n_classes * class_count)`. The implementation fails closed if any frozen class is absent
or an unknown label appears. Validation counts are not accepted by the weight function.

For the engineering-only S2 fold-1 subset, TRAIN-only weights were HIGH_IMPACT 2.387775 and
LOWER_IMPACT 0.632431. These engineering values do not replace the independently computed full-fold
weights that would be used in competitive execution.

## 7. Sequence-length and truncation audit

All tokenization used standard truncation at fixed `max_length=256` and dynamic batch padding.
Across 207,575 development rows, 60,920 required truncation (29.348%). Project-level counts and
safe token-length summaries are in `truncation_audit.csv`; no raw or identifiable text is stored.
No locked-test text was tokenized, and no chunking strategy was introduced.

## 8. Engineering preflight

The development-only preflight completed in 20.295 seconds. Weighted loss was finite
(0.678288), gradients were finite, an optimizer step completed, evaluation completed, and a saved
checkpoint reloaded successfully. MPS and bf16 were used directly.

## 9. Runtime feasibility measurement and stop decision

The mandatory ENGINEERING_ONLY test used exactly 10,000 deterministic S2 fold-1 TRAIN rows, three
epochs, batch size 8, and no validation scoring. It completed 3,750 optimizer steps in 1,547.840
seconds:

- 19.3819 examples/second across epoch exposures
- 2.4227 steps/second
- epoch times: 476.276, 523.031, and 548.532 seconds
- mean epoch losses: 0.636762, 0.533482, and 0.444899
- process RSS before/after: 1,787,559,936 / 1,188,216,832 bytes

The fixed nine-fit matrix contains 2,802,204 TRAIN-example exposures across three epochs. Direct
throughput projection was 144,578.768 seconds, or **40.161 hours**, before full validation and
checkpoint overhead. This exceeds the brief's approximately 24-hour ceiling.

Accordingly, competitive execution was stopped before the first research fit. Hyperparameters were
not changed to improve throughput. This is the required methodological outcome of Stage B2.1, not
a model failure.

## 10. Intended nine-fit matrix and checkpoint design

The frozen competitive matrix remains S6/S3/S2 × folds 1/2/3 = nine fits. The implementation runs
them sequentially, initializes the exact resolved checkpoint independently per fold, computes
TRAIN-only weights, trains exactly three epochs, evaluates the frozen VALIDATION membership, and
saves one final model plus provenance/metrics under a Git-ignored checkpoint root. Deterministic
run IDs and provenance-checked resume are implemented and tested.

Expected competitive fits were nine. Attempted, successful, and failed competitive fits were all
zero because the 24-hour feasibility stop rule fired.

## 11. S6 results

No B2 competitive S6 run was authorized after the feasibility projection. Fold Macro-F1,
aggregate metrics, per-class F1, and deltas versus A2/B1/B1.5 are therefore **not available**. The
current best development model for S6 remains the B1.5 hybrid at 0.277623 Macro-F1.

## 12. S3 results

No B2 competitive S3 run was authorized. Fold metrics and comparisons are **not available**. The
current best development model for S3 remains the B1.5 hybrid at 0.480109 Macro-F1.

## 13. S2 results

No B2 competitive S2 run was authorized. Competitive HIGH_IMPACT precision, recall, F1, legacy
guard, and comparison deltas are **not available**. Engineering training loss is not a research
metric and was not placed on a leaderboard. The current best development model for S2 remains the
B1.5 hybrid at 0.642162 Macro-F1 with HIGH_IMPACT precision/recall/F1
0.351351/0.523472/0.420040.

## 14. Per-class findings and comparison with B1.5

There are no B2 validation predictions or per-class findings because competitive runs did not
start. No claim about DeBERTa versus B1.5 performance can be made. B1.5 remains the strongest
development family for all tasks.

## 15. Training and resource observations

bf16 MPS training was numerically stable and engineering loss decreased across the three fixed
epochs. Throughput, not correctness or memory failure, triggered the stop. The after-run RSS was
lower than the before-run value and is only a process snapshot, not peak unified-memory usage.
Full execution would also add nine validation passes and model checkpoint writes, so 40.161 hours
is a conservative projection.

## 16. Limitations

The 10,000-row measurement uses S2 fold 1 and assumes approximately linear scaling across tasks
and fold sizes. Sequence-length and label distributions can change throughput, but the measured
projection exceeds the ceiling by roughly 16 hours before omitted overhead. The prescribed rule
does not authorize using a smaller model, shorter sequence, fewer epochs, different batch,
gradient accumulation, or altered precision to obtain a faster score. Such a change requires a
separate explicit engineering/methodological decision.

## 17. Development-only and locked-test statement

The model was fine-tuned only for the bounded engineering feasibility measurement. No competitive
research model or validation prediction was produced. The final chronological locked test remained
inaccessible throughout.

```text
MODELS_FITTED = YES
TRANSFORMER_ENCODER_FINE_TUNED = YES
MODEL_SELECTION_DATA = DEVELOPMENT_ONLY
FROZEN_CV_MANIFESTS_USED = YES
CV_FOLDS_RECONSTRUCTED_DURING_MODELLING = NO
LOCKED_TEST_TOKENIZED = NO
LOCKED_TEST_MODEL_PERFORMANCE_ACCESSED = NO
LOCKED_TEST_USED_FOR_TUNING = NO
```
