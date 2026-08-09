"""Controlled development-only Phase A2 classical optimization."""

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

from defect_classifier.classical_benchmark import (
    BenchmarkError,
    DevelopmentRow,
    _atomic_json,
    _checkpoint,
    _fit_one,
    _load_checkpoint,
    _load_development_rows,
    _load_fold,
)
from defect_classifier.classical_features import build_sparse_features
from defect_classifier.preparation import _membership_fingerprint
from defect_classifier.protocol import FrozenProtocol

TASKS = ("S6", "S3", "S2")
STAGES = ("A2.1", "A2.2", "A2.3")
ALLOWED_C_GRID = (0.25, 0.5, 1.0, 2.0, 4.0)
ALLOWED_TEXT_VIEWS = ("SUMMARY_DESCRIPTION", "SUMMARY_ONLY", "DESCRIPTION_ONLY")
EXPECTED_VARIANT_SIGNATURES = {
    "S6": (
        ("S6-R1", None, ("char_wb", 3, 5, 2, 150000)),
        ("S6-R2", None, ("char_wb", 3, 6, 2, 150000)),
        ("S6-R3", None, ("char_wb", 2, 5, 2, 150000)),
        ("S6-R4", None, ("char_wb", 3, 5, 5, 150000)),
        ("S6-R5", None, ("char_wb", 3, 5, 2, 250000)),
        ("S6-R6", None, ("char_wb", 3, 6, 2, 250000)),
    ),
    "S3": (
        ("S3-R1", ("word", 1, 2, 2, 150000), None),
        ("S3-R2", ("word", 1, 1, 2, 150000), None),
        ("S3-R3", ("word", 1, 3, 2, 150000), None),
        ("S3-R4", ("word", 1, 2, 5, 150000), None),
        ("S3-R5", ("word", 1, 2, 2, 250000), None),
        ("S3-R6", ("word", 1, 3, 2, 250000), None),
    ),
    "S2": (
        ("S2-R1", ("word", 1, 2, 2, 150000), ("char_wb", 3, 5, 2, 150000)),
        ("S2-R2", ("word", 1, 2, 2, 200000), ("char_wb", 3, 5, 2, 200000)),
        ("S2-R3", ("word", 1, 2, 2, 250000), ("char_wb", 3, 5, 2, 250000)),
        ("S2-R4", ("word", 1, 3, 2, 150000), ("char_wb", 3, 5, 2, 150000)),
        ("S2-R5", ("word", 1, 2, 2, 150000), ("char_wb", 3, 6, 2, 150000)),
        ("S2-R6", ("word", 1, 2, 2, 150000), ("char_wb", 3, 5, 2, 250000)),
    ),
}
EXPECTED_A1_HASHES = {
    "CLASSICAL_BENCHMARK_REPORT.md": "7a86566aba18d54f028f09781a7cb9a196a6c59347819f1020daeec75591104b",  # noqa: E501
    "confusion_matrices.json": "121776acaf17dc27889d564f5954900611853b594b679251dc1a4c7f1d2ead6a",
    "environment.json": "6b4dc9fa7d83a3f424333ec31221316aeb19f4d1f73f46b816fd0cf05de30a86",
    "fit_results.csv": "9d0ebc9eabd871d4eec107393fcadf332f6c6bd511e692cc6b6a0d762eadd394",
    "leaderboard.csv": "566aef68422d8782bcf2671da825012e35fa9e74840e93cf0ca57ffcd2b4c484",
    "per_class_metrics.csv": "b4d81de17777baff16b7fba60d8f0a8244f8a28c37adbd38a32b0cd97857e5cb",
    "runtime_summary.csv": "b5e97897ace433080e8aa6a609c400199a7b9018757671fc64e30b49e8613666",
    "task_summary.csv": "761bfea475d243d15981d3f4ffe055e27b3f4f5e31a0e849f81027b6e24fe48b",
}


def default_optimization_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "classical_optimization_v1.toml"


def load_optimization_config(path: Path | None = None) -> tuple[dict[str, Any], str]:
    raw = (path or default_optimization_config_path()).read_bytes()
    config = tomllib.loads(raw.decode())
    validate_search_space(config)
    return config, hashlib.sha256(raw).hexdigest()


def validate_search_space(config: dict[str, Any]) -> None:
    if tuple(config["c_grid"]) != ALLOWED_C_GRID:
        raise BenchmarkError("Phase A2 C grid drift")
    if tuple(config["text_views"]) != ALLOWED_TEXT_VIEWS:
        raise BenchmarkError("Phase A2 text-view grid drift")
    expected = {
        "S6": ("LINEARSVC", "CHAR"),
        "S3": ("LOGREG", "WORD"),
        "S2": ("LOGREG", "WORD_CHAR"),
    }
    for task, family in expected.items():
        winner = config["winners"][task]
        if (winner["classifier"], winner["representation"]) != family:
            raise BenchmarkError(f"Phase A1 winning family drift for {task}")
        variants = config["representations"][task]
        if len(variants) != 6 or [v["id"] for v in variants] != [
            f"{task}-R{i}" for i in range(1, 7)
        ]:
            raise BenchmarkError(f"Phase A2 representation grid drift for {task}")
        if any(v["representation"] != family[1] for v in variants):
            raise BenchmarkError(f"unapproved representation family for {task}")
        signatures = tuple(
            (
                variant["id"],
                _section_signature(variant.get("word")),
                _section_signature(variant.get("char")),
            )
            for variant in variants
        )
        if signatures != EXPECTED_VARIANT_SIGNATURES[task]:
            raise BenchmarkError(f"Phase A2 representation parameters drift for {task}")
    allowed_top = {
        "optimization_id",
        "seed",
        "c_grid",
        "text_views",
        "models",
        "fixed_tfidf",
        "winners",
        "representations",
        "smoke",
    }
    if set(config) != allowed_top or config["models"]["class_weight"] != "balanced":
        raise BenchmarkError("unapproved Phase A2 search axis")


def _section_signature(values: dict[str, Any] | None) -> tuple[Any, ...] | None:
    if values is None:
        return None
    if set(values) != {"analyzer", "ngram_min", "ngram_max", "min_df", "max_features"}:
        raise BenchmarkError("unapproved TF-IDF parameter axis")
    return tuple(
        values[key] for key in ("analyzer", "ngram_min", "ngram_max", "min_df", "max_features")
    )


def search_space_size(config: dict[str, Any]) -> dict[str, int]:
    return {
        "A2.1": len(TASKS) * len(config["c_grid"]) * 3,
        "A2.2": sum(len(config["representations"][t]) for t in TASKS) * 3,
        "A2.3": len(TASKS) * 2 * 3,
    }


def text_for_view(row: DevelopmentRow, view: str) -> str:
    if view == "SUMMARY_DESCRIPTION":
        return row.text
    if view == "SUMMARY_ONLY":
        return row.summary
    if view == "DESCRIPTION_ONLY":
        return row.description
    raise BenchmarkError(f"unknown text view: {view}")


def _feature_config(config: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    fixed = config["fixed_tfidf"]
    result: dict[str, Any] = {}
    for section in ("word", "char"):
        values = dict(variant.get(section, {}))
        if values:
            values.update(fixed)
            result[section] = values
    if variant["representation"] == "WORD_CHAR":
        result["word_char"] = {
            "word_max_features": result["word"]["max_features"],
            "char_max_features": result["char"]["max_features"],
        }
    return result


def _model_config(config: dict[str, Any], c_value: float) -> dict[str, Any]:
    return {
        "models": {
            "c": c_value,
            "logreg_solver": config["models"]["logreg_solver"],
            "logreg_max_iter": config["models"]["logreg_max_iter"],
            "linearsvc_max_iter": config["models"]["linearsvc_max_iter"],
        }
    }


def optimization_experiment_id(
    stage: str, task: str, configuration_id: str, fold: int, fingerprint: str
) -> str:
    return hashlib.sha256(
        f"A2|{stage}|{task}|{configuration_id}|{fold}|{fingerprint}".encode()
    ).hexdigest()[:20]


def aggregate_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in results:
        if row["status"] == "SUCCESS" and row["stage"] in STAGES:
            groups.setdefault((row["stage"], row["task"], row["configuration_id"]), []).append(row)
    output = []
    for (stage, task, configuration_id), folds in groups.items():
        if len(folds) != 3:
            continue
        ordered = sorted(folds, key=lambda r: r["fold"])
        macro = [r["metrics"]["macro_f1"] for r in ordered]
        item = {
            "stage": stage,
            "task": task,
            "configuration_id": configuration_id,
            "c": ordered[0]["c"],
            "representation_id": ordered[0]["representation_id"],
            "text_view": ordered[0]["text_view"],
            "classifier": ordered[0]["classifier"],
            "fold_macro_f1": "|".join(f"{v:.10f}" for v in macro),
            "mean_macro_f1": statistics.fmean(macro),
            "min_fold_macro_f1": min(macro),
            "std_macro_f1": statistics.pstdev(macro),
            "mean_balanced_accuracy": statistics.fmean(
                r["metrics"]["balanced_accuracy"] for r in ordered
            ),
            "mean_accuracy": statistics.fmean(r["metrics"]["accuracy"] for r in ordered),
            "mean_weighted_f1": statistics.fmean(r["metrics"]["weighted_f1"] for r in ordered),
            "mean_feature_count": statistics.fmean(r["feature_count"] for r in ordered),
        }
        if task == "S2":
            high = [r["metrics"]["per_class"]["HIGH_IMPACT"] for r in ordered]
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
        key=lambda r: (
            r["stage"],
            r["task"],
            -r["mean_macro_f1"],
            -r["min_fold_macro_f1"],
            -r["mean_balanced_accuracy"],
            r["mean_feature_count"],
            r["configuration_id"],
        )
    )
    counters: dict[tuple[str, str], int] = {}
    for row in output:
        key = (row["stage"], row["task"])
        counters[key] = counters.get(key, 0) + 1
        row["rank"] = counters[key]
    return output


def select_best(rows: list[dict[str, Any]], stage: str, task: str) -> dict[str, Any]:
    return next(r for r in rows if r["stage"] == stage and r["task"] == task and r["rank"] == 1)


def comparison_delta(a1_folds: list[float], a2_folds: list[float]) -> dict[str, Any]:
    a1_mean, a2_mean = statistics.fmean(a1_folds), statistics.fmean(a2_folds)
    return {
        "a1_mean_macro_f1": a1_mean,
        "a2_mean_macro_f1": a2_mean,
        "absolute_delta": a2_mean - a1_mean,
        "relative_percent_change": ((a2_mean - a1_mean) / a1_mean) * 100,
        "fold_deltas": [b - a for a, b in zip(a1_folds, a2_folds, strict=True)],
    }


def _verify_a1(report_dir: Path) -> None:
    for name, expected in EXPECTED_A1_HASHES.items():
        path = report_dir / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise BenchmarkError(f"Phase A1 report drift: {name}")


def _run_fit(
    *,
    stage: str,
    task: str,
    configuration_id: str,
    representation_id: str,
    text_view: str,
    c_value: float,
    fold: int,
    features: Any,
    train_ids: list[str],
    validation_ids: list[str],
    rows: dict[str, DevelopmentRow],
    protocol: FrozenProtocol,
    config: dict[str, Any],
    fingerprint: str,
    provenance: dict[str, str],
    checkpoint_dir: Path,
    resume: bool,
) -> dict[str, Any]:
    identifier = optimization_experiment_id(stage, task, configuration_id, fold, fingerprint)
    path = checkpoint_dir / f"{identifier}.json"
    cached = _load_checkpoint(path, provenance) if resume else None
    if cached is not None and cached.get("status") == "SUCCESS":
        return cached
    winner = config["winners"][task]
    result = _fit_one(
        stage=stage,
        task=task,
        representation=winner["representation"],
        classifier=winner["classifier"],
        class_weight="BALANCED",
        fold=fold,
        feature_count=features.feature_count,
        training_matrix=features.training,
        validation_matrix=features.validation,
        training_labels=[rows[x].targets[task] for x in train_ids],
        validation_labels=[rows[x].targets[task] for x in validation_ids],
        protocol=protocol,
        config=_model_config(config, c_value),
        benchmark_fingerprint=fingerprint,
        provenance=provenance,
    )
    result.update(
        {
            "experiment_id": identifier,
            "configuration_id": configuration_id,
            "representation_id": representation_id,
            "text_view": text_view,
            "c": c_value,
        }
    )
    _checkpoint(path, result)
    return result


def _texts(rows: dict[str, DevelopmentRow], ids: list[str], view: str) -> list[str]:
    return [text_for_view(rows[x], view) for x in ids]


def _base_variant(config: dict[str, Any], task: str) -> dict[str, Any]:
    return config["representations"][task][0]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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
    aggregate: list[dict[str, Any]],
    config: dict[str, Any],
    environment: dict[str, Any],
) -> None:
    flat, per_class = [], []
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
                    "stage": result["stage"],
                    "task": result["task"],
                    "configuration_id": result["configuration_id"],
                    "class": label,
                    **values,
                }
            )
    _write_csv(report_dir / "a2_fit_results.csv", flat)
    _write_csv(
        report_dir / "regularization_results.csv", [r for r in aggregate if r["stage"] == "A2.1"]
    )
    _write_csv(
        report_dir / "representation_results.csv", [r for r in aggregate if r["stage"] == "A2.2"]
    )
    _write_csv(
        report_dir / "text_ablation_results.csv", [r for r in aggregate if r["stage"] == "A2.3"]
    )
    _write_csv(report_dir / "leaderboard.csv", aggregate)
    _write_csv(report_dir / "per_class_metrics.csv", per_class)
    executed = [row for row in results if row.get("fit_execution") != "REUSED_FROM_A2.2"]
    runtime = [
        {
            "stage": stage,
            "fit_count": sum(r["stage"] == stage for r in executed),
            "fit_seconds": sum(
                r.get("fit_runtime_seconds", 0) for r in executed if r["stage"] == stage
            ),
            "prediction_seconds": sum(
                r.get("prediction_runtime_seconds", 0) for r in executed if r["stage"] == stage
            ),
        }
        for stage in STAGES
    ]
    _write_csv(report_dir / "runtime_summary.csv", runtime)
    _atomic_json(report_dir / "environment.json", environment)
    available_tasks = tuple(task for task in TASKS if any(r["task"] == task for r in aggregate))
    lines = [
        "# Classical Optimization V1",
        "",
        "## Objective",
        "",
        "Phase A2 performs targeted development-CV optimization of the fixed Phase-A1 "
        "winning model families.",
        "",
        "## Frozen protocol and search budget",
        "",
        "Only DEVELOPMENT data and persisted pooled frozen CV memberships were used. The "
        "bounded stages contain 45 regularization, 54 representation, and 18 new text-view "
        "fits (117 maximum).",
        "",
        "## Results",
    ]
    for task in available_tasks:
        final_stage = (
            "A2.3"
            if any(r["stage"] == "A2.3" and r["task"] == task for r in aggregate)
            else (
                "A2.2"
                if any(r["stage"] == "A2.2" and r["task"] == task for r in aggregate)
                else "A2.1"
            )
        )
        c = select_best(aggregate, "A2.1", task)
        rep = (
            select_best(aggregate, "A2.2", task)
            if any(r["stage"] == "A2.2" and r["task"] == task for r in aggregate)
            else c
        )
        final = select_best(aggregate, final_stage, task)
        a1 = config["winners"][task]
        folds = [float(x) for x in final["fold_macro_f1"].split("|")]
        delta = comparison_delta(a1["a1_fold_macro_f1"], folds)
        lines += [
            "",
            f"### {task}",
            "",
            f"Selected C: {c['c']}; selected representation: {rep['representation_id']}; "
            f"selected text view: {final['text_view']}.",
            f"Final fold Macro-F1: {final['fold_macro_f1']}; mean "
            f"{final['mean_macro_f1']:.6f}, SD {final['std_macro_f1']:.6f}.",
            f"A1 mean {delta['a1_mean_macro_f1']:.6f}; absolute delta "
            f"{delta['absolute_delta']:+.6f}; relative change "
            f"{delta['relative_percent_change']:+.3f}%.",
        ]
    lines += [
        "",
        "## Limitations and interpretation",
        "",
        "This is targeted model selection on the same frozen development CV used in Phase A1. "
        "The selected score is not an unbiased final-generalization estimate, and improvement "
        "does not guarantee locked-test improvement. Three folds do not justify significance "
        "claims.",
        "",
        "```text",
        "MODELS_FITTED = YES",
        "MODEL_SELECTION_DATA = DEVELOPMENT_ONLY",
        "PHASE_A1_REPORTS_MODIFIED = NO",
        "FROZEN_CV_MANIFESTS_USED = YES",
        "CV_FOLDS_RECONSTRUCTED_DURING_MODELLING = NO",
        "LOCKED_TEST_MODEL_PERFORMANCE_ACCESSED = NO",
        "LOCKED_TEST_USED_FOR_TUNING = NO",
        "```",
        "",
    ]
    (report_dir / "CLASSICAL_OPTIMIZATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run_classical_optimization(
    development_dir: Path,
    manifest_dir: Path,
    protocol_report_dir: Path,
    a1_report_dir: Path,
    report_dir: Path,
    protocol: FrozenProtocol,
    *,
    config_path: Path | None = None,
    stage: str = "all",
    task_filter: str | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    started = time.monotonic()
    config, fingerprint = load_optimization_config(config_path)
    _verify_a1(a1_report_dir)
    frozen = json.loads((protocol_report_dir / "fingerprints.json").read_text())
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
        "optimization_config_sha256": fingerprint,
    }
    checkpoint_dir = report_dir / ".work" / "checkpoints"
    tasks = (task_filter,) if task_filter else TASKS
    if stage in ("all", "smoke"):
        smoke = config["smoke"]
        task = smoke["task"]
        train, validation = folds[smoke["fold"]]
        train = train[: smoke["max_training_rows"]]
        validation = validation[: smoke["max_validation_rows"]]
        variant = _base_variant(config, task)
        features = build_sparse_features(
            variant["representation"],
            _texts(rows, train, "SUMMARY_DESCRIPTION"),
            _texts(rows, validation, "SUMMARY_DESCRIPTION"),
            _feature_config(config, variant),
            smoke_max_features=smoke["max_features"],
        )
        smoke_result = _run_fit(
            stage="ENGINEERING_ONLY",
            task=task,
            configuration_id="A2-SMOKE",
            representation_id=variant["id"],
            text_view="SUMMARY_DESCRIPTION",
            c_value=smoke["c"],
            fold=smoke["fold"],
            features=features,
            train_ids=train,
            validation_ids=validation,
            rows=rows,
            protocol=protocol,
            config=config,
            fingerprint=fingerprint,
            provenance=provenance,
            checkpoint_dir=report_dir / ".work",
            resume=resume,
        )
        if smoke_result["status"] != "SUCCESS":
            raise BenchmarkError("Phase A2 smoke failed")
        if stage == "smoke":
            return {
                "smoke": smoke_result,
                "successful": 0,
                "failed": 0,
                "runtime_seconds": time.monotonic() - started,
            }
    results: list[dict[str, Any]] = []
    if stage in ("all", "a2.1", "a2.2", "a2.3"):
        for task in tasks:
            variant = _base_variant(config, task)
            for fold, (train, validation) in folds.items():
                features = build_sparse_features(
                    variant["representation"],
                    _texts(rows, train, "SUMMARY_DESCRIPTION"),
                    _texts(rows, validation, "SUMMARY_DESCRIPTION"),
                    _feature_config(config, variant),
                )
                for c_value in config["c_grid"]:
                    results.append(
                        _run_fit(
                            stage="A2.1",
                            task=task,
                            configuration_id=f"{task}-C{c_value:g}",
                            representation_id=variant["id"],
                            text_view="SUMMARY_DESCRIPTION",
                            c_value=c_value,
                            fold=fold,
                            features=features,
                            train_ids=train,
                            validation_ids=validation,
                            rows=rows,
                            protocol=protocol,
                            config=config,
                            fingerprint=fingerprint,
                            provenance=provenance,
                            checkpoint_dir=checkpoint_dir,
                            resume=resume,
                        )
                    )
                del features
                gc.collect()
        aggregate = aggregate_results(results)
        if stage == "a2.1":
            return _finish(report_dir, results, aggregate, config, provenance, fingerprint, started)
        selected_c = {task: select_best(aggregate, "A2.1", task)["c"] for task in tasks}
        for task in tasks:
            for variant in config["representations"][task]:
                for fold, (train, validation) in folds.items():
                    features = build_sparse_features(
                        variant["representation"],
                        _texts(rows, train, "SUMMARY_DESCRIPTION"),
                        _texts(rows, validation, "SUMMARY_DESCRIPTION"),
                        _feature_config(config, variant),
                    )
                    results.append(
                        _run_fit(
                            stage="A2.2",
                            task=task,
                            configuration_id=variant["id"],
                            representation_id=variant["id"],
                            text_view="SUMMARY_DESCRIPTION",
                            c_value=selected_c[task],
                            fold=fold,
                            features=features,
                            train_ids=train,
                            validation_ids=validation,
                            rows=rows,
                            protocol=protocol,
                            config=config,
                            fingerprint=fingerprint,
                            provenance=provenance,
                            checkpoint_dir=checkpoint_dir,
                            resume=resume,
                        )
                    )
                    del features
                    gc.collect()
        aggregate = aggregate_results(results)
        if stage == "a2.2":
            return _finish(report_dir, results, aggregate, config, provenance, fingerprint, started)
        selected_rep = {
            task: select_best(aggregate, "A2.2", task)["representation_id"] for task in tasks
        }
        for task in tasks:
            variant = next(
                v for v in config["representations"][task] if v["id"] == selected_rep[task]
            )
            # The combined view is methodologically identical to the selected A2.2 result.
            # Reuse its metrics rather than fitting nine unnecessary duplicate models.
            for original in results:
                if (
                    original["stage"] == "A2.2"
                    and original["task"] == task
                    and original["representation_id"] == variant["id"]
                ):
                    reused = dict(original)
                    reused.update(
                        {
                            "stage": "A2.3",
                            "configuration_id": f"{variant['id']}-SUMMARY_DESCRIPTION",
                            "text_view": "SUMMARY_DESCRIPTION",
                            "fit_execution": "REUSED_FROM_A2.2",
                            "reused_experiment_id": original["experiment_id"],
                            "experiment_id": optimization_experiment_id(
                                "A2.3",
                                task,
                                f"{variant['id']}-SUMMARY_DESCRIPTION",
                                original["fold"],
                                fingerprint,
                            ),
                        }
                    )
                    results.append(reused)
            for view in ("SUMMARY_ONLY", "DESCRIPTION_ONLY"):
                for fold, (train, validation) in folds.items():
                    features = build_sparse_features(
                        variant["representation"],
                        _texts(rows, train, view),
                        _texts(rows, validation, view),
                        _feature_config(config, variant),
                    )
                    results.append(
                        _run_fit(
                            stage="A2.3",
                            task=task,
                            configuration_id=f"{variant['id']}-{view}",
                            representation_id=variant["id"],
                            text_view=view,
                            c_value=selected_c[task],
                            fold=fold,
                            features=features,
                            train_ids=train,
                            validation_ids=validation,
                            rows=rows,
                            protocol=protocol,
                            config=config,
                            fingerprint=fingerprint,
                            provenance=provenance,
                            checkpoint_dir=checkpoint_dir,
                            resume=resume,
                        )
                    )
                    del features
                    gc.collect()
    return _finish(
        report_dir, results, aggregate_results(results), config, provenance, fingerprint, started
    )


def _finish(
    report_dir: Path,
    results: list[dict[str, Any]],
    aggregate: list[dict[str, Any]],
    config: dict[str, Any],
    provenance: dict[str, str],
    fingerprint: str,
    started: float,
) -> dict[str, Any]:
    elapsed = time.monotonic() - started
    environment = {
        **provenance,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {p: version(p) for p in ("numpy", "scipy", "scikit-learn", "pyarrow")},
        "optimization_config_sha256": fingerprint,
        "runtime_seconds": elapsed,
        "locked_test_accessed": False,
    }
    _write_reports(report_dir, results, aggregate, config, environment)
    executed = [r for r in results if r.get("fit_execution") != "REUSED_FROM_A2.2"]
    return {
        "successful": sum(r["status"] == "SUCCESS" for r in executed),
        "failed": sum(r["status"] == "FAILED" for r in executed),
        "runtime_seconds": elapsed,
        "aggregate": aggregate,
    }
