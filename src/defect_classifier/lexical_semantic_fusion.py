"""Controlled development-only Phase B1.5 lexical and MPNet feature fusion."""

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
from defect_classifier.classical_features import FeatureResult, build_sparse_features
from defect_classifier.classical_optimization import _feature_config, load_optimization_config
from defect_classifier.embedding_cache import (
    EmbeddingCacheError,
    atomic_json,
    read_shard,
    validate_membership,
)
from defect_classifier.preparation import _membership_fingerprint
from defect_classifier.protocol import FrozenProtocol

TASKS = ("S6", "S3", "S2")
EXPECTED_TASKS = {
    "S6": ("S6-R3", "CHAR", "LINEARSVC", "BALANCED", 0.25),
    "S3": ("S3-R5", "WORD", "LOGREG", "BALANCED", 2.0),
    "S2": ("S2-R3", "WORD_CHAR", "LOGREG", "BALANCED", 2.0),
}
EXPECTED_ENCODER = (
    "E2",
    "sentence-transformers/all-mpnet-base-v2",
    "e8c3b32edf5434bc2275fc9bab85f82640a19130",
    768,
    True,
    1.0,
)
REFERENCE_HASHES = {
    "A1_REPORT": (
        "classical_benchmark_v1/CLASSICAL_BENCHMARK_REPORT.md",
        "7a86566aba18d54f028f09781a7cb9a196a6c59347819f1020daeec75591104b",
    ),
    "A1_LEADERBOARD": (
        "classical_benchmark_v1/leaderboard.csv",
        "566aef68422d8782bcf2671da825012e35fa9e74840e93cf0ca57ffcd2b4c484",
    ),
    "A2_REPORT": (
        "classical_optimization_v1/CLASSICAL_OPTIMIZATION_REPORT.md",
        "4ab436dc330e80a66a1cea5dcfb9c364c8d6f947c4d8f757fd25486c0ecd2a8d",
    ),
    "A2_LEADERBOARD": (
        "classical_optimization_v1/leaderboard.csv",
        "5343448b359fc181f2e341ff6c9a942767f3334d214b91cce7aa41405cdcf54f",
    ),
    "B1_REPORT": (
        "semantic_embeddings_v1/SEMANTIC_EMBEDDING_REPORT.md",
        "7809d204c0f4b85eb162c071742bf92a6d7d078b355c76537ffb382fc4f09804",
    ),
    "B1_LEADERBOARD": (
        "semantic_embeddings_v1/leaderboard.csv",
        "5b4e1c7437827dfd547d2aca46e1e5516203e641f1b070970c31431acc3d7268",
    ),
}


def default_fusion_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "lexical_semantic_fusion_v1.toml"


def load_fusion_config(path: Path | None = None) -> tuple[dict[str, Any], str]:
    raw = (path or default_fusion_config_path()).read_bytes()
    config = tomllib.loads(raw.decode())
    validate_fusion_config(config)
    return config, hashlib.sha256(raw).hexdigest()


def validate_fusion_config(config: dict[str, Any]) -> None:
    encoder = tuple(
        config[key]
        for key in (
            "semantic_encoder_id",
            "semantic_model_id",
            "semantic_revision",
            "semantic_dimension",
            "semantic_normalized",
            "semantic_weight",
        )
    )
    if encoder != EXPECTED_ENCODER:
        raise BenchmarkError("Phase B1.5 semantic method drift")
    if tuple(config["tasks"]) != TASKS:
        raise BenchmarkError("Phase B1.5 task matrix drift")
    for task, expected in EXPECTED_TASKS.items():
        actual = config["tasks"][task]
        signature = tuple(
            actual[key]
            for key in ("representation_id", "representation", "classifier", "class_weight", "c")
        )
        if signature != expected:
            raise BenchmarkError(f"Phase B1.5 method drift for {task}")
    if set(config) != {
        "experiment_id",
        "seed",
        "semantic_encoder_id",
        "semantic_model_id",
        "semantic_revision",
        "semantic_dimension",
        "semantic_normalized",
        "semantic_weight",
        "models",
        "tasks",
        "smoke",
    }:
        raise BenchmarkError("unapproved Phase B1.5 search axis")


def search_space_size(config: dict[str, Any]) -> int:
    validate_fusion_config(config)
    return len(config["tasks"]) * 3


def fusion_experiment_id(task: str, fold: int, fingerprint: str) -> str:
    return hashlib.sha256(f"B1.5|{task}|HYBRID|{fold}|{fingerprint}".encode()).hexdigest()[:20]


def selected_variant(
    a2_config: dict[str, Any], fusion_config: dict[str, Any], task: str
) -> dict[str, Any]:
    wanted = fusion_config["tasks"][task]["representation_id"]
    variants = [value for value in a2_config["representations"][task] if value["id"] == wanted]
    if len(variants) != 1:
        raise BenchmarkError(f"missing exact A2 representation for {task}")
    return variants[0]


def fuse_sparse_features(
    lexical: FeatureResult,
    train_semantic: np.ndarray,
    validation_semantic: np.ndarray,
    weight: float,
) -> FeatureResult:
    from scipy import sparse

    if weight != 1.0:
        raise BenchmarkError("semantic block weight must remain 1.0")
    if (
        lexical.training.shape[0] != train_semantic.shape[0]
        or lexical.validation.shape[0] != validation_semantic.shape[0]
    ):
        raise BenchmarkError("lexical and semantic row alignment mismatch")
    training = sparse.hstack(
        (lexical.training, sparse.csr_matrix(train_semantic * weight)),
        format="csr",
        dtype=np.float32,
    )
    validation = sparse.hstack(
        (lexical.validation, sparse.csr_matrix(validation_semantic * weight)),
        format="csr",
        dtype=np.float32,
    )
    if not sparse.issparse(training) or not sparse.issparse(validation):
        raise BenchmarkError("hybrid representation unexpectedly became dense")
    return FeatureResult(lexical.transformer, training, validation, training.shape[1])


def _verify_references(reports_root: Path) -> dict[str, str]:
    values = {}
    for key, (relative, expected) in REFERENCE_HASHES.items():
        path = reports_root / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "MISSING"
        if actual != expected:
            raise BenchmarkError(f"frozen reference drift: {key}")
        values[key] = actual
    return values


def load_mpnet_cache(
    cache_root: Path,
    development_ids: set[str],
    protocol: FrozenProtocol,
    development_fingerprint: str,
    config: dict[str, Any],
) -> tuple[dict[str, int], np.ndarray, dict[str, Any]]:
    root = cache_root / "E2"
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    expected = {
        "encoder_id": config["semantic_encoder_id"],
        "model_id": config["semantic_model_id"],
        "resolved_revision": config["semantic_revision"],
        "embedding_dimension": config["semantic_dimension"],
        "normalize_embeddings": config["semantic_normalized"],
        "protocol_sha256": protocol.fingerprint,
        "development_membership_sha256": development_fingerprint,
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise EmbeddingCacheError("MPNet cache provenance mismatch")
    ids: list[str] = []
    arrays = []
    for path in sorted((root / "shards").glob("*.parquet")):
        shard_ids, embeddings = read_shard(path, config["semantic_dimension"])
        ids.extend(shard_ids)
        arrays.append(embeddings)
    if len(ids) != len(set(ids)):
        raise EmbeddingCacheError("duplicate MPNet embedding identity")
    validate_membership(set(ids), development_ids)
    return {value: index for index, value in enumerate(ids)}, np.concatenate(arrays), metadata


def _model_config(config: dict[str, Any], task: str) -> dict[str, Any]:
    return {"models": {"c": config["tasks"][task]["c"], **config["models"]}}


def _run_fit(
    *,
    task: str,
    fold: int,
    features: FeatureResult,
    train_ids: list[str],
    validation_ids: list[str],
    rows: dict[str, Any],
    protocol: FrozenProtocol,
    config: dict[str, Any],
    fingerprint: str,
    provenance: dict[str, str],
    checkpoint_dir: Path,
    resume: bool,
    stage: str = "COMPETITIVE",
) -> dict[str, Any]:
    identifier = (
        fusion_experiment_id(task, fold, fingerprint)
        if stage == "COMPETITIVE"
        else "B15-ENGINEERING-SMOKE"
    )
    path = checkpoint_dir / f"{identifier}.json"
    cached = _load_checkpoint(path, provenance) if resume else None
    if cached is not None and cached.get("status") == "SUCCESS":
        return cached
    method = config["tasks"][task]
    result = _fit_one(
        stage=stage,
        task=task,
        representation="LEXICAL_MPNET",
        classifier=method["classifier"],
        class_weight=method["class_weight"],
        fold=fold,
        feature_count=features.feature_count,
        training_matrix=features.training,
        validation_matrix=features.validation,
        training_labels=[rows[value].targets[task] for value in train_ids],
        validation_labels=[rows[value].targets[task] for value in validation_ids],
        protocol=protocol,
        config=_model_config(config, task),
        benchmark_fingerprint=fingerprint,
        provenance=provenance,
    )
    result.update(
        {
            "experiment_id": identifier,
            "configuration_id": f"{task}-A2-MPNET",
            "representation_id": method["representation_id"],
            "lexical_feature_count": features.feature_count - config["semantic_dimension"],
            "semantic_feature_count": config["semantic_dimension"],
            "semantic_weight": 1.0,
            "c": method["c"],
            "encoder_id": "E2",
            "resolved_revision": config["semantic_revision"],
        }
    )
    _checkpoint(path, result)
    return result


def _aggregate(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for task in TASKS:
        rows = sorted(
            (row for row in results if row["task"] == task and row["status"] == "SUCCESS"),
            key=lambda row: row["fold"],
        )
        if len(rows) != 3:
            continue
        macro = [row["metrics"]["macro_f1"] for row in rows]
        item = {
            "task": task,
            "configuration_id": rows[0]["configuration_id"],
            "representation_id": rows[0]["representation_id"],
            "classifier": rows[0]["classifier"],
            "c": rows[0]["c"],
            "class_weight": rows[0]["class_weight"],
            "lexical_feature_count_by_fold": "|".join(
                str(row["lexical_feature_count"]) for row in rows
            ),
            "semantic_feature_count": 768,
            "semantic_weight": 1.0,
            "fold_macro_f1": "|".join(f"{value:.10f}" for value in macro),
            "mean_macro_f1": statistics.fmean(macro),
            "std_macro_f1": statistics.pstdev(macro),
            "mean_balanced_accuracy": statistics.fmean(
                row["metrics"]["balanced_accuracy"] for row in rows
            ),
            "mean_accuracy": statistics.fmean(row["metrics"]["accuracy"] for row in rows),
            "mean_weighted_f1": statistics.fmean(row["metrics"]["weighted_f1"] for row in rows),
        }
        if task == "S2":
            high = [row["metrics"]["per_class"]["HIGH_IMPACT"] for row in rows]
            item.update(
                {
                    "high_impact_precision": statistics.fmean(value["precision"] for value in high),
                    "high_impact_recall": statistics.fmean(value["recall"] for value in high),
                    "high_impact_f1": statistics.fmean(value["f1"] for value in high),
                }
            )
            item["legacy_reproduction_guard"] = (
                "PASS" if item["high_impact_precision"] >= 0.30 else "FAIL"
            )
        output.append(item)
    return output


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


def _write_reports(
    report_dir: Path,
    results: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    elapsed: float,
    provenance: dict[str, str],
    reference_hashes: dict[str, str],
) -> None:
    flat, per_class, matrices = [], [], {}
    for result in results:
        metrics = result.get("metrics", {})
        flat.append(
            {
                **{
                    key: value
                    for key, value in result.items()
                    if key not in ("metrics", "warnings")
                },
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
    _csv(report_dir / "task_summary.csv", summary)
    _csv(report_dir / "per_class_metrics.csv", per_class)
    atomic_json(report_dir / "confusion_matrices.json", matrices)
    _csv(
        report_dir / "runtime_summary.csv",
        [
            {
                "task": task,
                "fit_count": sum(row["task"] == task for row in results),
                "fit_seconds": sum(
                    row.get("fit_runtime_seconds", 0) for row in results if row["task"] == task
                ),
                "prediction_seconds": sum(
                    row.get("prediction_runtime_seconds", 0)
                    for row in results
                    if row["task"] == task
                ),
            }
            for task in TASKS
        ],
    )
    atomic_json(
        report_dir / "environment.json",
        {
            **provenance,
            "reference_sha256": reference_hashes,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: version(name) for name in ("numpy", "scipy", "scikit-learn", "pyarrow")
            },
            "runtime_seconds": elapsed,
            "locked_test_accessed": False,
            "locked_test_used_for_tuning": False,
        },
    )


def run_fusion_benchmark(
    development_dir: Path,
    manifest_dir: Path,
    protocol_report_dir: Path,
    cache_root: Path,
    reports_root: Path,
    report_dir: Path,
    protocol: FrozenProtocol,
    *,
    config_path: Path | None = None,
    stage: str = "all",
    task_filter: str | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    started = time.monotonic()
    config, fingerprint = load_fusion_config(config_path)
    a2_config, a2_fingerprint = load_optimization_config()
    reference_hashes = _verify_references(reports_root)
    frozen = json.loads((protocol_report_dir / "fingerprints.json").read_text(encoding="utf-8"))
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
    index, embeddings, metadata = load_mpnet_cache(
        cache_root, development_ids, protocol, frozen["development_membership_sha256"], config
    )
    provenance = {
        "protocol_sha256": protocol.fingerprint,
        "development_membership_sha256": frozen["development_membership_sha256"],
        "fusion_config_sha256": fingerprint,
        "a2_config_sha256": a2_fingerprint,
        "mpnet_revision": metadata["resolved_revision"],
    }
    if stage in ("all", "smoke"):
        smoke = config["smoke"]
        task = smoke["task"]
        train_ids, validation_ids = folds[smoke["fold"]]
        train_ids = train_ids[: smoke["max_training_rows"]]
        validation_ids = validation_ids[: smoke["max_validation_rows"]]
        variant = selected_variant(a2_config, config, task)
        lexical = build_sparse_features(
            variant["representation"],
            [rows[x].text for x in train_ids],
            [rows[x].text for x in validation_ids],
            _feature_config(a2_config, variant),
            smoke_max_features=smoke["max_features"],
        )
        features = fuse_sparse_features(
            lexical,
            embeddings[[index[x] for x in train_ids]],
            embeddings[[index[x] for x in validation_ids]],
            1.0,
        )
        smoke_result = _run_fit(
            task=task,
            fold=smoke["fold"],
            features=features,
            train_ids=train_ids,
            validation_ids=validation_ids,
            rows=rows,
            protocol=protocol,
            config=config,
            fingerprint=fingerprint,
            provenance=provenance,
            checkpoint_dir=report_dir / ".work",
            resume=resume,
            stage="ENGINEERING_ONLY",
        )
        if smoke_result["status"] != "SUCCESS":
            raise BenchmarkError("Phase B1.5 smoke failed")
        if stage == "smoke":
            return {
                "smoke": smoke_result,
                "successful": 0,
                "failed": 0,
                "runtime_seconds": time.monotonic() - started,
            }
    results = []
    for task in (task_filter,) if task_filter else TASKS:
        variant = selected_variant(a2_config, config, task)
        for fold, (train_ids, validation_ids) in folds.items():
            lexical = build_sparse_features(
                variant["representation"],
                [rows[x].text for x in train_ids],
                [rows[x].text for x in validation_ids],
                _feature_config(a2_config, variant),
            )
            features = fuse_sparse_features(
                lexical,
                embeddings[[index[x] for x in train_ids]],
                embeddings[[index[x] for x in validation_ids]],
                1.0,
            )
            results.append(
                _run_fit(
                    task=task,
                    fold=fold,
                    features=features,
                    train_ids=train_ids,
                    validation_ids=validation_ids,
                    rows=rows,
                    protocol=protocol,
                    config=config,
                    fingerprint=fingerprint,
                    provenance=provenance,
                    checkpoint_dir=report_dir / ".work" / "checkpoints",
                    resume=resume,
                )
            )
            del lexical, features
            gc.collect()
    elapsed = time.monotonic() - started
    _write_reports(report_dir, results, _aggregate(results), elapsed, provenance, reference_hashes)
    return {
        "successful": sum(row["status"] == "SUCCESS" for row in results),
        "failed": sum(row["status"] == "FAILED" for row in results),
        "runtime_seconds": elapsed,
    }
