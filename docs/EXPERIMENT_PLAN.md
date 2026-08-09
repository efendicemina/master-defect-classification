# Experiment Plan

No modelling experiments have been run. Protocol-v1 target mappings, chronological splits,
duplicate controls, and metrics are now frozen in committed configuration. Model/search spaces
and advanced model choices will be specified in reviewed experiment configuration later.

## Phase A — Controlled reproduction of classical NLP experiments

Reproduce the prior word TF-IDF, character TF-IDF, and combined representations with linear SVM
and logistic-regression classifiers. Evaluate the approved S6, S3, and S2 tasks, class weighting,
within-project, pooled-Eclipse, and cross-project settings under chronology-aware development
validation. Macro-F1 is the frozen primary selection metric, accompanied by accuracy,
weighted F1, balanced accuracy, and per-class measures. This phase first freezes data contracts,
target mappings, duplicate policy, split protocol, and provenance requirements.

## Phase B — Advanced Apple Silicon experiments

After Phase A is verified, assess modern document/text embeddings, appropriately sized transformer
representations or fine-tuning, improved hyperparameter optimization, project-aware approaches,
feature ablations, imbalance strategies, and robust statistical comparisons. Choices will account
for the available Apple Silicon memory and runtime while preserving the same leakage controls and
locked-test discipline. This plan makes no claim about future outcomes.
