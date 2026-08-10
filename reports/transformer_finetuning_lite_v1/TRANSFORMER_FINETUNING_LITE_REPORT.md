# Phase B2-Lite MiniLM Fine-Tuning Report

## Scope and safeguards

Phase B2-Lite is a development-only, fixed-protocol comparison using the three frozen chronological development folds. This finalization reused nine completed checkpoints; it fitted no model and did not load, tokenize, evaluate, or tune on the locked test set.

All nine expected S6/S3/S2 × fold 1/2/3 results have `SUCCESS` status. `checkpoint_validation.json` binds them to protocol `85faacae1e7f411d68803653388611c37c62730ea9217e9700a0ad6ac41b7cda`, development membership `4f62fdf4164594126c421955804b654cd5d5f8f7b46ada345ae0cffa71460d0f`, configuration `1900a248c766b85e70ba38f94e243f6f03c55e91f7b96ef1492880ba6e8ce407`, and model revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.

## Frozen method

The model is `sentence-transformers/all-MiniLM-L6-v2` at the pinned revision above, loaded without remote custom code as `BertForSequenceClassification`. It has 6 layers, hidden size 384, and 22,713,986 trainable parameters. Training used MPS/bfloat16, maximum length 256, learning rate 2e-5, batch size 16, three epochs, train-fold-only balanced class weights, deterministic seeds, no early stopping, and the final epoch.

The bounded 10,000-row feasibility fit processed 70.8217 examples/s and projected 10.9908 hours for the matrix, below the frozen 24-hour ceiling. The completed nine-fit runtime was 38,248.830 seconds (10.6247 hours). The original DeBERTa feasibility projection was 40.1608 hours. Across development, 64,223 of 207,575 texts (30.9397%) exceeded 256 tokens.

## Development results

| Task | Fold Macro-F1 | Mean | Population SD | Balanced accuracy | Accuracy | Weighted F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| S6 | 0.195443 / 0.179209 / 0.186376 | 0.187009 | 0.006642 | 0.279379 | 0.413745 | 0.490429 |
| S3 | 0.381452 / 0.394257 / 0.401769 | 0.392493 | 0.008388 | 0.509357 | 0.472177 | 0.516734 |
| S2 | 0.566280 / 0.572625 / 0.571314 | 0.570073 | 0.002735 | 0.668507 | 0.656214 | 0.703913 |

For S2 HIGH_IMPACT, mean precision was 0.260644, recall 0.686196, and F1 0.377733. The frozen legacy precision guard therefore failed (`0.260644 < 0.30`). Per-fold, per-class precision/recall/F1/support are in `per_class_metrics.csv`; all nine labeled confusion matrices are in `confusion_matrices.json`.

## Comparison with prior development phases

Values are mean development-fold Macro-F1. Deltas are B2-Lite minus the named system. Each prior result is read from its fingerprint-verified frozen report.

| Task | B2-Lite | A2 (delta) | B1 MiniLM (delta) | B1 MPNet (delta) | B1.5 hybrid (delta) |
| --- | ---: | ---: | ---: | ---: | ---: |
| S6 | 0.187009 | 0.270426 (-0.083416) | 0.236387 (-0.049377) | 0.251225 (-0.064215) | 0.277623 (-0.090614) |
| S3 | 0.392493 | 0.469561 (-0.077068) | 0.432546 (-0.040054) | 0.460346 (-0.067853) | 0.480109 (-0.087617) |
| S2 | 0.570073 | 0.637336 (-0.067264) | 0.567302 (+0.002771) | 0.575399 (-0.005327) | 0.642162 (-0.072089) |

B2-Lite exceeded frozen MiniLM only on S2. It did not exceed A2, frozen MPNet, or B1.5 on any task, and did not exceed B1.5 on S6 or S3. These are development comparisons only and do not authorize selection using locked-test outcomes.

## Artifact inventory

`training_runs.csv` records nine successful runs; `epoch_diagnostics.csv` records all 27 epochs; `task_summary.csv` contains the three task summaries; `per_class_metrics.csv` contains 33 task/fold/class rows; `confusion_matrices.json` contains nine matrices; `runtime_summary.csv`, `truncation_audit.csv`, `environment.json`, `model_provenance.json`, `comparison.csv`, and `checkpoint_validation.json` preserve runtime, audit, provenance, comparison, and integrity evidence.
