# Eclipse Bug Severity Classification

This repository is the reproducible implementation for a Master's thesis on classifying
software defect severity from Eclipse bug-report text. The present milestone establishes the
research foundation only: it does not preprocess data, define targets or splits, or train models.

## Repository organization

- `src/defect_classifier/`: installable Python package and command-line interface
- `configs/`: committed, reviewable dataset and future experiment configuration
- `tests/`: isolated tests that use temporary fake datasets
- `docs/`: research protocol and phased experiment plan
- `scripts/`: thin entry-point scripts, if future workflows require them
- `reports/`: small, Git-trackable metrics, manifests, and research reports
- `notebooks/`: exploratory notebooks; production research logic belongs in `src/`

Raw and generated datasets, model files, and experiment artifacts are intentionally excluded
from version control.

## Environment and installation

The project requires Python 3.12 or newer. An existing local environment can be activated and
the package installed with development tools as follows:

```bash
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

The Eclipse CSV files stay outside the repository. Configure their parent directory with an
environment variable; `.env.example` documents the expected setting without committing a
machine-specific path:

```bash
export ECLIPSE_DATA_ROOT=/absolute/path/to/Eclipse
defect-classifier verify-dataset
```

The command reads only filesystem metadata. It checks the nine explicit relative paths in
`configs/datasets.toml`, reports their sizes, and exits non-zero if configuration or files are
missing. A different catalogue can be supplied with `--catalogue PATH` for testing or auditing.
The application deliberately does not load `.env` implicitly; export it in the shell or use an
environment manager of your choice.

## Forensic dataset audit

The restartable audit processes one project at a time with Python's streaming CSV reader and
writes small aggregate reports without creating a derived dataset:

```bash
defect-classifier audit-data
```

Completed project aggregates are cached under the ignored
`reports/dataset_audit/.work/` directory. Use `--no-resume` to deliberately recompute every
project. The audit preserves raw severity labels and does not define eligibility, mappings,
preprocessing, splits, or models.

## Current status

The repository currently provides configuration, a canonical project catalogue, path and
environment validation, deterministic seed setup, a CLI, tests, and protocol documentation.
Classical reproduction and advanced modelling are planned but not yet implemented.

## Reproducibility principles

Research decisions are configuration-driven and reviewable. Raw inputs are immutable and
external, data logic is tested, random seeds are explicit, and experiment provenance must be
recorded. Chronology and project boundaries must be preserved where required. Model selection
uses development data only; the final chronological test set will be locked, and its outcomes
must never influence feature, model, or hyperparameter decisions.

See `docs/RESEARCH_PROTOCOL.md` before adding data or modelling code.
