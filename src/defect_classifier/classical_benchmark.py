"""Development-only Phase A1 classical benchmark engine."""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
import platform
import statistics
import time
import tomllib
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

from defect_classifier.classical_features import build_sparse_features
from defect_classifier.classical_metrics import classification_metrics
from defect_classifier.cv_manifests import load_cv_membership
from defect_classifier.preparation import _membership_fingerprint
from defect_classifier.protocol import FrozenProtocol

TASKS = ("S6", "S3", "S2")
REPRESENTATIONS = ("WORD", "CHAR", "WORD_CHAR")
CLASSIFIERS = ("LOGREG", "LINEARSVC")
CLASS_WEIGHTS = ("NONE", "BALANCED")


class BenchmarkError(RuntimeError):
    """Raised when benchmark provenance or execution fails closed."""


@dataclass(frozen=True)
class DevelopmentRow:
    stable_id: str
    text: str
    targets: dict[str, str]
    summary: str = ""
    description: str = ""


def default_benchmark_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "classical_benchmark_v1.toml"


def load_benchmark_config(path: Path | None = None) -> tuple[dict[str, Any], str]:
    config_path = path or default_benchmark_config_path()
    raw = config_path.read_bytes()
    return tomllib.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()


def experiment_id(
    stage: str,
    task: str,
    representation: str,
    classifier: str,
    class_weight: str,
    fold: int,
    benchmark_fingerprint: str,
) -> str:
    payload = "|".join(
        (stage, task, representation, classifier, class_weight, str(fold), benchmark_fingerprint)
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


def _load_development_rows(development_dir: Path) -> dict[str, DevelopmentRow]:
    import pyarrow.parquet as pq

    rows = {}
    columns = [
        "source_project",
        "issue_id",
        "summary",
        "description",
        "text_combined",
        "target_s6",
        "target_s3",
        "target_s2",
    ]
    for path in sorted(development_dir.glob("*.parquet")):
        for record in pq.read_table(path, columns=columns).to_pylist():
            stable_id = f"{record['source_project']}:{record['issue_id']}"
            if stable_id in rows:
                raise BenchmarkError(f"duplicate development stable ID: {stable_id}")
            rows[stable_id] = DevelopmentRow(
                stable_id=stable_id,
                text=record["text_combined"],
                targets={
                    "S6": record["target_s6"],
                    "S3": record["target_s3"],
                    "S2": record["target_s2"],
                },
                summary=record["summary"],
                description=record["description"],
            )
    return rows


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _checkpoint(path: Path, result: dict[str, Any]) -> None:
    _atomic_json(path, result)


def _load_checkpoint(path: Path, provenance: dict[str, str]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if any(value.get(key) != expected for key, expected in provenance.items()):
        raise BenchmarkError(f"checkpoint provenance drift: {path}")
    return value


def _make_model(
    classifier: str, class_weight: str, config: dict[str, Any], seed: int, task: str
) -> Any:
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC

    weight = None if class_weight == "NONE" else "balanced"
    if classifier == "LOGREG":
        # scikit-learn 1.8 removed liblinear's implicit multiclass OvR support.
        # Use its current general-purpose solver for the frozen multiclass tasks.
        solver = "lbfgs" if task in ("S6", "S3") else config["models"]["logreg_solver"]
        return LogisticRegression(
            C=config["models"]["c"],
            class_weight=weight,
            solver=solver,
            max_iter=config["models"]["logreg_max_iter"],
            random_state=seed,
        )
    if classifier == "LINEARSVC":
        return LinearSVC(
            C=config["models"]["c"],
            class_weight=weight,
            max_iter=config["models"]["linearsvc_max_iter"],
            random_state=seed,
            dual="auto",
        )
    raise ValueError(f"unknown classifier: {classifier}")


def _fit_one(
    *,
    stage: str,
    task: str,
    representation: str,
    classifier: str,
    class_weight: str,
    fold: int,
    feature_count: int,
    training_matrix: Any,
    validation_matrix: Any,
    training_labels: list[str],
    validation_labels: list[str],
    protocol: FrozenProtocol,
    config: dict[str, Any],
    benchmark_fingerprint: str,
    provenance: dict[str, str],
) -> dict[str, Any]:
    identifier = experiment_id(
        stage, task, representation, classifier, class_weight, fold, benchmark_fingerprint
    )
    result = {
        "experiment_id": identifier,
        "stage": stage,
        "task": task,
        "representation": representation,
        "classifier": classifier,
        "class_weight": class_weight,
        "fold": fold,
        "training_rows": len(training_labels),
        "validation_rows": len(validation_labels),
        "feature_count": feature_count,
        "random_seed": protocol.seed,
        **provenance,
    }
    try:
        model = _make_model(classifier, class_weight, config, protocol.seed, task)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            started = time.monotonic()
            model.fit(training_matrix, training_labels)
            result["fit_runtime_seconds"] = time.monotonic() - started
            started = time.monotonic()
            predictions = model.predict(validation_matrix).tolist()
            result["prediction_runtime_seconds"] = time.monotonic() - started
        result["warnings"] = [f"{type(item.message).__name__}: {item.message}" for item in caught]
        labels = protocol.targets[task.casefold()].order
        result["metrics"] = classification_metrics(validation_labels, predictions, labels)
        result["status"] = "SUCCESS"
    except Exception as exc:  # recorded per-fit so the matrix remains resumable
        result.update(
            {
                "status": "FAILED",
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            }
        )
    return result


def _load_fold(
    rows: dict[str, DevelopmentRow],
    manifest_dir: Path,
    fingerprint_path: Path,
    protocol: FrozenProtocol,
    fold: int,
) -> tuple[list[str], list[str]]:
    training = load_cv_membership(
        manifest_dir, fingerprint_path, protocol, family="pooled", fold=fold, role="TRAIN"
    )
    validation = load_cv_membership(
        manifest_dir,
        fingerprint_path,
        protocol,
        family="pooled",
        fold=fold,
        role="VALIDATION",
    )
    training_ids = [row.stable_id for row in training]
    validation_ids = [row.stable_id for row in validation]
    if set(training_ids) & set(validation_ids):
        raise BenchmarkError(f"frozen fold {fold} has membership overlap")
    if any(value not in rows for value in training_ids + validation_ids):
        raise BenchmarkError(f"frozen fold {fold} references missing development rows")
    return training_ids, validation_ids


def _dummy_results(
    rows: dict[str, DevelopmentRow],
    folds: dict[int, tuple[list[str], list[str]]],
    protocol: FrozenProtocol,
    config: dict[str, Any],
    benchmark_fingerprint: str,
    checkpoint_dir: Path,
    provenance: dict[str, str],
    resume: bool,
) -> list[dict[str, Any]]:
    from sklearn.dummy import DummyClassifier

    results = []
    for task in TASKS:
        labels = protocol.targets[task.casefold()].order
        for strategy in config["dummy"]["strategies"]:
            for fold, (training_ids, validation_ids) in folds.items():
                identifier = experiment_id(
                    "REFERENCE",
                    task,
                    "NONE",
                    f"DUMMY_{strategy.upper()}",
                    "NONE",
                    fold,
                    benchmark_fingerprint,
                )
                path = checkpoint_dir / f"{identifier}.json"
                if resume and (cached := _load_checkpoint(path, provenance)) is not None:
                    results.append(cached)
                    continue
                y_train = [rows[value].targets[task] for value in training_ids]
                y_validation = [rows[value].targets[task] for value in validation_ids]
                started = time.monotonic()
                model = DummyClassifier(strategy=strategy, random_state=protocol.seed)
                model.fit([[0]] * len(y_train), y_train)
                predictions = model.predict([[0]] * len(y_validation)).tolist()
                result = {
                    "experiment_id": identifier,
                    "stage": "REFERENCE",
                    "task": task,
                    "representation": "NONE",
                    "classifier": f"DUMMY_{strategy.upper()}",
                    "class_weight": "NONE",
                    "fold": fold,
                    "training_rows": len(y_train),
                    "validation_rows": len(y_validation),
                    "feature_count": 0,
                    "fit_runtime_seconds": time.monotonic() - started,
                    "prediction_runtime_seconds": 0.0,
                    "warnings": [],
                    "metrics": classification_metrics(y_validation, predictions, labels),
                    "status": "SUCCESS",
                    "random_seed": protocol.seed,
                    **provenance,
                }
                _checkpoint(path, result)
                results.append(result)
    return results


def aggregate_leaderboard(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        if result["stage"] != "COMPETITIVE" or result["status"] != "SUCCESS":
            continue
        groups[
            (
                result["task"],
                result["representation"],
                result["classifier"],
                result["class_weight"],
            )
        ].append(result)
    leaderboard = []
    for key, folds in groups.items():
        if len(folds) != 3:
            continue
        ordered = sorted(folds, key=lambda value: value["fold"])
        macro = [value["metrics"]["macro_f1"] for value in ordered]
        row = {
            "task": key[0],
            "representation": key[1],
            "classifier": key[2],
            "class_weight": key[3],
            "fold_macro_f1": "|".join(f"{value:.10f}" for value in macro),
            "mean_macro_f1": statistics.fmean(macro),
            "std_macro_f1": statistics.pstdev(macro),
            "mean_balanced_accuracy": statistics.fmean(
                value["metrics"]["balanced_accuracy"] for value in ordered
            ),
            "mean_accuracy": statistics.fmean(value["metrics"]["accuracy"] for value in ordered),
            "mean_weighted_f1": statistics.fmean(
                value["metrics"]["weighted_f1"] for value in ordered
            ),
        }
        if key[0] == "S2":
            high = [value["metrics"]["per_class"]["HIGH_IMPACT"] for value in ordered]
            row.update(
                {
                    "mean_high_impact_precision": statistics.fmean(
                        value["precision"] for value in high
                    ),
                    "mean_high_impact_recall": statistics.fmean(value["recall"] for value in high),
                    "mean_high_impact_f1": statistics.fmean(value["f1"] for value in high),
                }
            )
            row["legacy_reproduction_guard"] = (
                "PASS" if row["mean_high_impact_precision"] >= 0.30 else "FAIL"
            )
        leaderboard.append(row)
    leaderboard.sort(
        key=lambda row: (
            row["task"],
            -row["mean_macro_f1"],
            -row["mean_balanced_accuracy"],
            -row["mean_weighted_f1"],
            row["representation"],
            row["classifier"],
            row["class_weight"],
        )
    )
    ranks = Counter()
    for row in leaderboard:
        ranks[row["task"]] += 1
        row["rank_within_task"] = ranks[row["task"]]
    return leaderboard


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = fields or list(rows[0])
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=selected, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_reports(
    report_dir: Path,
    results: list[dict[str, Any]],
    stage0: dict[str, Any],
    total_runtime: float,
    environment: dict[str, Any],
    protocol: FrozenProtocol,
) -> None:
    detailed = [row for row in results if row["stage"] in ("COMPETITIVE", "REFERENCE")]
    competitive = [row for row in detailed if row["stage"] == "COMPETITIVE"]
    flat = []
    per_class = []
    matrices = {}
    for result in detailed:
        metrics = result.get("metrics", {})
        flat.append(
            {
                key: value
                for key, value in {
                    **{k: v for k, v in result.items() if k not in ("metrics", "warnings")},
                    "macro_f1": metrics.get("macro_f1"),
                    "balanced_accuracy": metrics.get("balanced_accuracy"),
                    "accuracy": metrics.get("accuracy"),
                    "weighted_f1": metrics.get("weighted_f1"),
                    "warning_count": len(result.get("warnings", [])),
                    "warning_messages": " | ".join(result.get("warnings", [])),
                }.items()
                if not isinstance(value, (dict, list))
            }
        )
        if result.get("status") == "SUCCESS":
            for label, values in metrics["per_class"].items():
                per_class.append(
                    {
                        "experiment_id": result["experiment_id"],
                        "task": result["task"],
                        "class": label,
                        **values,
                    }
                )
            matrices[result["experiment_id"]] = {
                "task": result["task"],
                "labels": list(protocol.targets[result["task"].casefold()].order),
                "matrix": metrics["confusion_matrix"],
            }
    fit_fields = sorted({key for row in flat for key in row})
    _write_csv(report_dir / "fit_results.csv", flat, fit_fields)
    leaderboard = aggregate_leaderboard(results)
    _write_csv(report_dir / "leaderboard.csv", leaderboard)
    _write_csv(report_dir / "per_class_metrics.csv", per_class)
    _atomic_json(report_dir / "confusion_matrices.json", matrices)
    best = [row for row in leaderboard if row["rank_within_task"] == 1]
    _write_csv(report_dir / "task_summary.csv", best)
    runtime_rows = []
    for representation in REPRESENTATIONS:
        selected = [
            r
            for r in competitive
            if r["representation"] == representation and r["status"] == "SUCCESS"
        ]
        runtime_rows.append(
            {
                "representation": representation,
                "successful_fits": len(selected),
                "total_fit_seconds": sum(r["fit_runtime_seconds"] for r in selected),
                "total_prediction_seconds": sum(r["prediction_runtime_seconds"] for r in selected),
            }
        )
    runtime_rows.append(
        {
            "representation": "TOTAL",
            "successful_fits": sum(r["status"] == "SUCCESS" for r in competitive),
            "total_fit_seconds": sum(r.get("fit_runtime_seconds", 0) for r in competitive),
            "total_prediction_seconds": sum(
                r.get("prediction_runtime_seconds", 0) for r in competitive
            ),
        }
    )
    _write_csv(report_dir / "runtime_summary.csv", runtime_rows)
    environment["total_runtime_seconds"] = total_runtime
    environment["stage0"] = stage0
    _atomic_json(report_dir / "environment.json", environment)
    _write_research_report(report_dir, results, leaderboard, total_runtime)


def _write_research_report(
    report_dir: Path,
    results: list[dict[str, Any]],
    leaderboard: list[dict[str, Any]],
    total_runtime: float,
) -> None:
    best = {row["task"]: row for row in leaderboard if row["rank_within_task"] == 1}
    sections = []
    for task in TASKS:
        row = best.get(task)
        if row:
            sections.append(
                f"### {task}\n\nBest development-CV configuration: `{row['representation']} / "
                f"{row['classifier']} / {row['class_weight']}`; mean Macro-F1 "
                f"{row['mean_macro_f1']:.6f} (population SD {row['std_macro_f1']:.6f})."
            )
    failures = [row for row in results if row.get("status") == "FAILED"]
    warnings_count = sum(len(row.get("warnings", [])) for row in results)
    report = f"""# Classical Benchmark V1

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

{chr(10).join(sections)}

Detailed rankings, per-class weaknesses, weighting and representation comparisons are retained in
the adjacent CSV/JSON reports. Three temporal folds do not support claims of statistical
significance or final generalization.

## Runtime, warnings, and limitations

Total orchestration runtime was {total_runtime:.3f} seconds. Recorded warning count:
{warnings_count}.
Failed competitive fits: {len(failures)}. This benchmark uses fixed feature caps and fixed baseline
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
"""
    path = report_dir / "CLASSICAL_BENCHMARK_REPORT.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def run_classical_benchmark(
    development_dir: Path,
    manifest_dir: Path,
    protocol_report_dir: Path,
    report_dir: Path,
    protocol: FrozenProtocol,
    *,
    benchmark_config_path: Path | None = None,
    stage: str = "all",
    task_filter: str | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    started_total = time.monotonic()
    config, benchmark_fingerprint = load_benchmark_config(benchmark_config_path)
    frozen = json.loads((protocol_report_dir / "fingerprints.json").read_text(encoding="utf-8"))
    if frozen["protocol_sha256"] != protocol.fingerprint:
        raise BenchmarkError("protocol fingerprint drift")
    rows = _load_development_rows(development_dir)
    if _membership_fingerprint(rows) != frozen["development_membership_sha256"]:
        raise BenchmarkError("development membership fingerprint drift")
    folds = {
        fold: _load_fold(
            rows, manifest_dir, protocol_report_dir / "fingerprints.json", protocol, fold
        )
        for fold in range(1, 4)
    }
    provenance = {
        "protocol_sha256": protocol.fingerprint,
        "development_membership_sha256": frozen["development_membership_sha256"],
        "benchmark_config_sha256": benchmark_fingerprint,
    }
    checkpoint_dir = report_dir / ".work" / "checkpoints"
    stage0_path = report_dir / ".work" / "stage0.json"
    stage0: dict[str, Any] = {}
    if stage in ("all", "smoke"):
        smoke = config["smoke"]
        train_ids, validation_ids = folds[smoke["fold"]]
        train_ids = train_ids[: smoke["max_training_rows"]]
        validation_ids = validation_ids[: smoke["max_validation_rows"]]
        stage0 = _load_checkpoint(stage0_path, provenance) if resume else None
        if stage0 is None or stage0.get("status") != "SUCCESS":
            features = build_sparse_features(
                smoke["representation"],
                [rows[value].text for value in train_ids],
                [rows[value].text for value in validation_ids],
                config,
                smoke_max_features=smoke["max_features"],
            )
            stage0 = _fit_one(
                stage="ENGINEERING_ONLY",
                task=smoke["task"],
                representation=smoke["representation"],
                classifier=smoke["classifier"],
                class_weight=smoke["class_weight"],
                fold=smoke["fold"],
                feature_count=features.feature_count,
                training_matrix=features.training,
                validation_matrix=features.validation,
                training_labels=[rows[value].targets[smoke["task"]] for value in train_ids],
                validation_labels=[rows[value].targets[smoke["task"]] for value in validation_ids],
                protocol=protocol,
                config=config,
                benchmark_fingerprint=benchmark_fingerprint,
                provenance=provenance,
            )
            _checkpoint(stage0_path, stage0)
        if stage0["status"] != "SUCCESS":
            raise BenchmarkError("Stage 0 failed; Stage 1 was not started")
        if stage == "smoke":
            return {
                "stage0": stage0,
                "competitive": [],
                "total_runtime_seconds": time.monotonic() - started_total,
            }

    results = _dummy_results(
        rows, folds, protocol, config, benchmark_fingerprint, checkpoint_dir, provenance, resume
    )
    selected_tasks = (task_filter,) if task_filter else TASKS
    for fold, (training_ids, validation_ids) in folds.items():
        training_text = [rows[value].text for value in training_ids]
        validation_text = [rows[value].text for value in validation_ids]
        for representation in REPRESENTATIONS:
            combinations = [
                (task, classifier, class_weight)
                for task in selected_tasks
                for classifier in CLASSIFIERS
                for class_weight in CLASS_WEIGHTS
            ]
            pending = []
            for task, classifier, class_weight in combinations:
                identifier = experiment_id(
                    "COMPETITIVE",
                    task,
                    representation,
                    classifier,
                    class_weight,
                    fold,
                    benchmark_fingerprint,
                )
                path = checkpoint_dir / f"{identifier}.json"
                cached = _load_checkpoint(path, provenance) if resume else None
                if cached is not None and cached.get("status") == "SUCCESS":
                    results.append(cached)
                else:
                    pending.append((task, classifier, class_weight, path))
            if not pending:
                continue
            features = build_sparse_features(representation, training_text, validation_text, config)
            for task, classifier, class_weight, path in pending:
                result = _fit_one(
                    stage="COMPETITIVE",
                    task=task,
                    representation=representation,
                    classifier=classifier,
                    class_weight=class_weight,
                    fold=fold,
                    feature_count=features.feature_count,
                    training_matrix=features.training,
                    validation_matrix=features.validation,
                    training_labels=[rows[value].targets[task] for value in training_ids],
                    validation_labels=[rows[value].targets[task] for value in validation_ids],
                    protocol=protocol,
                    config=config,
                    benchmark_fingerprint=benchmark_fingerprint,
                    provenance=provenance,
                )
                _checkpoint(path, result)
                results.append(result)
            del features
            gc.collect()
        del training_text, validation_text
        gc.collect()
    total_runtime = time.monotonic() - started_total
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            package: version(package) for package in ("numpy", "scipy", "scikit-learn", "pyarrow")
        },
        "protocol_sha256": protocol.fingerprint,
        "development_membership_sha256": frozen["development_membership_sha256"],
        "benchmark_config_sha256": benchmark_fingerprint,
        "random_seed": protocol.seed,
        "locked_test_accessed": False,
        "engineering_adjustments": {
            "multiclass_logistic_regression_solver": (
                "lbfgs because scikit-learn 1.8 no longer supports multiclass liblinear"
            )
        },
    }
    _write_reports(report_dir, results, stage0, total_runtime, environment, protocol)
    competitive = [row for row in results if row["stage"] == "COMPETITIVE"]
    return {
        "stage0": stage0,
        "competitive": competitive,
        "successful": sum(row["status"] == "SUCCESS" for row in competitive),
        "failed": sum(row["status"] == "FAILED" for row in competitive),
        "total_runtime_seconds": total_runtime,
    }
