# Permanent Research Instructions

These instructions apply to every agent and contributor working in this repository.

1. Treat the external Eclipse CSV files as immutable. Never modify, rename, move, or copy raw
   data into this repository, and never commit raw data.
2. Preserve chronological evaluation. Any operation that can affect ordering, grouping, target
   construction, deduplication, or splitting must be explicit, documented, and tested.
3. Prevent leakage across every boundary, including duplicates, near-duplicates, temporal
   information, project identity, preprocessing fit, feature selection, and hyperparameter search.
4. The final chronological test set is locked once created. During development, do not load its
   labels, calculate its outcomes, inspect its errors, or use its metrics to make feature, model,
   threshold, mapping, or hyperparameter decisions.
5. Fit every learned transformation on the relevant training partition only. Model selection and
   tuning use development data only.
6. Use deterministic seeds wherever applicable. Record seeds, package/runtime versions, source
   revision, dataset manifest, configuration, and other experiment provenance.
7. Add or update tests for all new data and split logic, including leakage and boundary cases.
   Tests must use fixtures or synthetic data rather than the external raw dataset.
8. Run the relevant tests and quality checks before declaring a task complete.
9. Do not silently change a frozen protocol, split, target mapping, evaluation metric, or
   catalogue. Explain material methodology changes and preserve an auditable record.
10. Prefer reproducibility, clarity, and research correctness over clever shortcuts.
11. Keep small configs, manifests, metrics, and reports Git-trackable. Keep generated datasets,
    caches, downloaded models, checkpoints, and large artifacts outside Git.
12. Do not preprocess, summarize, or otherwise inspect raw dataset contents unless the active task
    explicitly authorizes that research phase.

