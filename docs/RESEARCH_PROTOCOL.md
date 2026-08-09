# Research Protocol

## Purpose and governing principle

This project evaluates bug-severity classifiers under a reproducible, chronology-aware protocol.
The final test evaluation is confirmatory, not another development iteration. Protocol changes
must be documented, justified, reviewed, and versioned before affected results are produced.

## Data states and permitted use

### Raw data

The nine source CSV files live under an externally configured `ECLIPSE_DATA_ROOT`. They are
immutable inputs and must never be copied into Git or modified. A versioned catalogue identifies
each project by an explicit relative path. Future ingestion should record a manifest (including
safe file metadata and cryptographic hashes where practical) without altering the sources.

### Development data

Development data is the only material available for exploration, preprocessing design, target
mapping decisions, feature engineering, ablations, model selection, threshold selection, and
error analysis. Any learned preprocessing is fitted within the appropriate training partition.
Duplicates and derived variants must not cross evaluation boundaries.

### Cross-validation and model selection

Cross-validation or chronological validation operates solely within development data. Its folds
must respect the declared temporal and project-aware design. Hyperparameters, feature choices,
imbalance handling, and model families are selected using development results, with Macro-F1 as
the planned primary metric unless a protocol revision is explicitly approved. Seeds, folds,
configs, software versions, dataset identity, and source revision must be recorded.

### Final locked test evaluation

Protocol v1 and its chronological membership are frozen in `docs/FROZEN_PROTOCOL_V1.md` and
`configs/protocol_v1.toml`. Locked-test text-bearing artifacts are physically separate from
development artifacts and guarded from normal research workflows. Their labels, predictions,
metrics, and error cases remain unavailable for model selection. Locked-test results must not
guide hyperparameter, feature, preprocessing, target-mapping, threshold, or model-family
decisions. A future explicit final-evaluation action must write results separately and audibly;
repeated evaluation is not a substitute for validation.

## Leakage controls

Future implementations must test temporal ordering, partition disjointness, duplicate grouping,
project boundaries, and train-only fitting. No global vocabulary, statistics, resampling, feature
selection, or calibration may be learned before a split. Identifiers and post-report information
must be assessed for target leakage before use.

## Reproducibility and provenance

Every experiment should be recoverable from committed configuration plus an external data
manifest. Record deterministic seeds, environment and dependency versions, code revision, input
identity, split identity, command/config, metrics, and artifact locations. Small metadata and
reports belong in Git; large derived data and models do not.
