"""Development-only classification over frozen semantic embedding caches."""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
import platform
import statistics
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from defect_classifier.classical_benchmark import (
    BenchmarkError,
    _checkpoint,
    _fit_one,
    _load_checkpoint,
    _load_development_rows,
    _load_fold,
)
from defect_classifier.embedding_cache import atomic_json, read_shard, validate_membership
from defect_classifier.preparation import _membership_fingerprint
from defect_classifier.protocol import FrozenProtocol
from defect_classifier.semantic_embeddings import load_semantic_config

TASKS = ("S6", "S3", "S2")
ENCODERS = ("E1", "E2")
CLASSIFIERS = ("LOGREG", "LINEARSVC")
WEIGHTS = ("NONE", "BALANCED")


def semantic_experiment_id(
    task: str,
    encoder: str,
    classifier: str,
    weight: str,
    fold: int,
    fingerprint: str,
) -> str:
    value = f"B1|{task}|{encoder}|{classifier}|{weight}|C1|{fold}|{fingerprint}"
    return hashlib.sha256(value.encode()).hexdigest()[:20]


def aggregate_semantic_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in results:
        if row["status"] == "SUCCESS":
            key = (row["task"], row["encoder_id"], row["classifier"], row["class_weight"])
            groups.setdefault(key, []).append(row)
    output = []
    for key, folds in groups.items():
        if len(folds) != 3:
            continue
        ordered = sorted(folds, key=lambda row: row["fold"])
        macro = [row["metrics"]["macro_f1"] for row in ordered]
        item = {
            "task": key[0],
            "encoder_id": key[1],
            "classifier": key[2],
            "class_weight": key[3],
            "configuration_id": "-".join(key),
            "fold_macro_f1": "|".join(f"{value:.10f}" for value in macro),
            "mean_macro_f1": statistics.fmean(macro),
            "min_fold_macro_f1": min(macro),
            "std_macro_f1": statistics.pstdev(macro),
            "mean_balanced_accuracy": statistics.fmean(
                row["metrics"]["balanced_accuracy"] for row in ordered
            ),
            "mean_accuracy": statistics.fmean(row["metrics"]["accuracy"] for row in ordered),
            "mean_weighted_f1": statistics.fmean(row["metrics"]["weighted_f1"] for row in ordered),
        }
        if key[0] == "S2":
            high = [row["metrics"]["per_class"]["HIGH_IMPACT"] for row in ordered]
            item.update(
                {
                    "high_impact_precision": statistics.fmean(x["precision"] for x in high),
                    "high_impact_recall": statistics.fmean(x["recall"] for x in high),
                    "high_impact_f1": statistics.fmean(x["f1"] for x in high),
                }
            )
            item["legacy_reproduction_guard"] = (
                "PASS" if item["high_impact_precision"] >= 0.30 else "FAIL"
            )
        output.append(item)
    output.sort(
        key=lambda row: (
            row["task"],
            -row["mean_macro_f1"],
            -row["min_fold_macro_f1"],
            -row["mean_balanced_accuracy"],
            row["configuration_id"],
        )
    )
    counts: dict[str, int] = {}
    for row in output:
        counts[row["task"]] = counts.get(row["task"], 0) + 1
        row["rank_within_task"] = counts[row["task"]]
    return output


def _load_encoder_cache(
    cache_root: Path, encoder_id: str, development_ids: set[str]
) -> tuple[list[str], np.ndarray, dict[str, Any]]:
    metadata = json.loads((cache_root / encoder_id / "metadata.json").read_text())
    ids, arrays = [], []
    for path in sorted((cache_root / encoder_id / "shards").glob("*.parquet")):
        shard_ids, values = read_shard(path, metadata["embedding_dimension"])
        ids.extend(shard_ids)
        arrays.append(values)
    validate_membership(set(ids), development_ids)
    return ids, np.concatenate(arrays).astype(np.float32, copy=False), metadata


def _model_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "models": {
            "c": 1.0,
            "logreg_solver": config["models"]["logreg_solver"],
            "logreg_max_iter": config["models"]["logreg_max_iter"],
            "linearsvc_max_iter": config["models"]["linearsvc_max_iter"],
        }
    }


def run_semantic_benchmark(
    development_dir: Path,
    manifest_dir: Path,
    protocol_report_dir: Path,
    cache_root: Path,
    report_dir: Path,
    protocol: FrozenProtocol,
    *,
    config_path: Path | None = None,
    task_filter: str | None = None,
    encoder_filter: str | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    started = time.monotonic()
    config, fingerprint = load_semantic_config(config_path)
    frozen = json.loads((protocol_report_dir / "fingerprints.json").read_text())
    if frozen["protocol_sha256"] != protocol.fingerprint:
        raise BenchmarkError("protocol fingerprint drift")
    rows = _load_development_rows(development_dir)
    development_ids = set(rows)
    if _membership_fingerprint(development_ids) != frozen["development_membership_sha256"]:
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
        "semantic_config_sha256": fingerprint,
    }
    tasks = (task_filter,) if task_filter else TASKS
    encoders = (encoder_filter,) if encoder_filter else ENCODERS
    results = []
    checkpoint_dir = report_dir / ".work" / "checkpoints"
    for encoder_id in encoders:
        ids, embeddings, metadata = _load_encoder_cache(cache_root, encoder_id, development_ids)
        index = {stable_id: position for position, stable_id in enumerate(ids)}
        for fold, (train_ids, validation_ids) in folds.items():
            train_matrix = embeddings[[index[value] for value in train_ids]]
            validation_matrix = embeddings[[index[value] for value in validation_ids]]
            for task in tasks:
                for classifier in CLASSIFIERS:
                    for weight in WEIGHTS:
                        identifier = semantic_experiment_id(
                            task, encoder_id, classifier, weight, fold, fingerprint
                        )
                        path = checkpoint_dir / f"{identifier}.json"
                        cached = _load_checkpoint(path, provenance) if resume else None
                        if cached is not None and cached.get("status") == "SUCCESS":
                            results.append(cached)
                            continue
                        result = _fit_one(
                            stage="COMPETITIVE",
                            task=task,
                            representation=encoder_id,
                            classifier=classifier,
                            class_weight=weight,
                            fold=fold,
                            feature_count=metadata["embedding_dimension"],
                            training_matrix=train_matrix,
                            validation_matrix=validation_matrix,
                            training_labels=[rows[value].targets[task] for value in train_ids],
                            validation_labels=[
                                rows[value].targets[task] for value in validation_ids
                            ],
                            protocol=protocol,
                            config=_model_config(config),
                            benchmark_fingerprint=fingerprint,
                            provenance=provenance,
                        )
                        result.update(
                            {
                                "experiment_id": identifier,
                                "encoder_id": encoder_id,
                                "model_id": metadata["model_id"],
                                "resolved_revision": metadata["resolved_revision"],
                                "c": 1.0,
                            }
                        )
                        _checkpoint(path, result)
                        results.append(result)
            del train_matrix, validation_matrix
            gc.collect()
        del embeddings
        gc.collect()
    elapsed = time.monotonic() - started
    leaderboard = aggregate_semantic_results(results)
    _write_reports(report_dir, results, leaderboard, elapsed, provenance)
    return {
        "successful": sum(row["status"] == "SUCCESS" for row in results),
        "failed": sum(row["status"] == "FAILED" for row in results),
        "runtime_seconds": elapsed,
    }


def _write_reports(
    report_dir: Path,
    results: list[dict[str, Any]],
    leaderboard: list[dict[str, Any]],
    elapsed: float,
    provenance: dict[str, str],
) -> None:
    flat, per_class, matrices = [], [], {}
    for result in results:
        metrics = result.get("metrics", {})
        flat.append(
            {
                **{k: v for k, v in result.items() if k not in ("metrics", "warnings")},
                "macro_f1": metrics.get("macro_f1"),
                "balanced_accuracy": metrics.get("balanced_accuracy"),
                "accuracy": metrics.get("accuracy"),
                "weighted_f1": metrics.get("weighted_f1"),
                "warning_count": len(result.get("warnings", [])),
                "warning_messages": " | ".join(result.get("warnings", [])),
            }
        )
        for label, values in metrics.get("per_class", {}).items():
            per_class.append(
                {
                    "experiment_id": result["experiment_id"],
                    "task": result["task"],
                    "class": label,
                    **values,
                }
            )
        if metrics:
            matrices[result["experiment_id"]] = {
                "task": result["task"],
                "labels": list(metrics["per_class"]),
                "matrix": metrics["confusion_matrix"],
            }
    _csv(report_dir / "fit_results.csv", flat)
    _csv(report_dir / "leaderboard.csv", leaderboard)
    _csv(
        report_dir / "task_summary.csv",
        [row for row in leaderboard if row["rank_within_task"] == 1],
    )
    _csv(report_dir / "per_class_metrics.csv", per_class)
    atomic_json(report_dir / "confusion_matrices.json", matrices)
    runtime = []
    for encoder_id in ENCODERS:
        selected = [row for row in results if row["encoder_id"] == encoder_id]
        runtime.append(
            {
                "encoder_id": encoder_id,
                "fit_count": len(selected),
                "fit_seconds": sum(row.get("fit_runtime_seconds", 0) for row in selected),
                "prediction_seconds": sum(
                    row.get("prediction_runtime_seconds", 0) for row in selected
                ),
            }
        )
    _csv(report_dir / "runtime_summary.csv", runtime)
    environment_path = report_dir / "environment.json"
    environment = (
        json.loads(environment_path.read_text(encoding="utf-8"))
        if environment_path.exists()
        else {}
    )
    environment.update(
        {
            **provenance,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                package: version(package)
                for package in (
                    "numpy",
                    "scipy",
                    "scikit-learn",
                    "pyarrow",
                    "torch",
                    "transformers",
                    "sentence-transformers",
                )
            },
            "classification_runtime_seconds": max(
                elapsed, environment.get("classification_runtime_seconds", 0.0)
            ),
            "locked_test_accessed": False,
        }
    )
    atomic_json(environment_path, environment)


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)
