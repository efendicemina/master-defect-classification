"""Resource-aware development-only Phase B2-LITE MiniLM fine-tuning."""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import platform
import time
import tomllib
from importlib.metadata import version
from pathlib import Path
from typing import Any

from defect_classifier.classical_benchmark import BenchmarkError, _load_development_rows, _load_fold
from defect_classifier.embedding_cache import atomic_json
from defect_classifier.preparation import _membership_fingerprint
from defect_classifier.protocol import FrozenProtocol
from defect_classifier.transformer_finetuning import (
    TASKS,
    _bounded_ids_with_class_coverage,
    _competitive_projection,
    _csv,
    _load_model,
    _seed_everything,
    _train,
    _truncation_audit,
    _write_competitive_reports,
    balanced_class_weights,
    mps_training_preflight,
)

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
EXPECTED = {
    "model_id": MODEL_ID,
    "resolved_revision": MODEL_REVISION,
    "max_length": 256,
    "learning_rate": 2e-5,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 32,
    "gradient_accumulation_steps": 1,
    "class_weighting": "balanced_train_only",
    "early_stopping": False,
    "final_epoch_selection": 3,
    "trust_remote_code": False,
}
REFERENCE_HASHES = {
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
    "B15_REPORT": (
        "lexical_semantic_fusion_v1/LEXICAL_SEMANTIC_FUSION_REPORT.md",
        "e7bcf9afe11ac5b29f0ff0996c5c011351684164048c7afd871062e242bd99c7",
    ),
    "B15_SUMMARY": (
        "lexical_semantic_fusion_v1/task_summary.csv",
        "0b27212eea2bccd615c228fab6d9d6fd4d7d33a3f5595c41e8a9c656d561ba8f",
    ),
    "B2_REPORT": (
        "transformer_finetuning_v1/TRANSFORMER_FINETUNING_REPORT.md",
        "c6f2162e7a1525394883d2948947f97ed3b2efe06481497124e19673581c7c03",
    ),
    "B2_ENVIRONMENT": (
        "transformer_finetuning_v1/environment.json",
        "150179731e7708b82c847fd60d31a5a3291faff4a8a6c3aeb454160a87cf2f1d",
    ),
}


def default_lite_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "transformer_finetuning_lite_v1.toml"


def load_lite_config(path: Path | None = None) -> tuple[dict[str, Any], str]:
    raw = (path or default_lite_config_path()).read_bytes()
    config = tomllib.loads(raw.decode())
    validate_lite_config(config)
    return config, hashlib.sha256(raw).hexdigest()


def validate_lite_config(config: dict[str, Any]) -> None:
    if any(config.get(key) != value for key, value in EXPECTED.items()):
        raise BenchmarkError("Phase B2-LITE fixed training method drift")
    if set(config) != {"experiment_id", "seed", *EXPECTED, "feasibility", "smoke"}:
        raise BenchmarkError("unapproved Phase B2-LITE search axis")
    if config["feasibility"] != {
        "task": "S2",
        "fold": 1,
        "training_rows": 10000,
        "maximum_projected_hours": 24.0,
    }:
        raise BenchmarkError("Phase B2-LITE feasibility design drift")


def search_space_size(config: dict[str, Any]) -> int:
    validate_lite_config(config)
    return 9


def lite_run_id(task: str, fold: int, revision: str, fingerprint: str) -> str:
    return hashlib.sha256(f"B2-LITE|{task}|{fold}|{revision}|{fingerprint}".encode()).hexdigest()[
        :20
    ]


def _verify_references(root: Path) -> dict[str, str]:
    output = {}
    for name, (relative, expected) in REFERENCE_HASHES.items():
        path = root / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "MISSING"
        if actual != expected:
            raise BenchmarkError(f"previous report fingerprint drift: {name}")
        output[name] = actual
    return output


def _release_model(model: Any) -> None:
    import torch

    del model
    gc.collect()
    torch.mps.empty_cache()


def _run_competitive_lite(
    *,
    rows: dict[str, Any],
    folds: dict[int, tuple[list[str], list[str]]],
    protocol: FrozenProtocol,
    tokenizer: Any,
    config: dict[str, Any],
    fingerprint: str,
    dtype: str,
    checkpoint_root: Path,
    resume: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    runs, epoch_rows = [], []
    for task in TASKS:
        order = protocol.targets[task.casefold()].order
        for fold, (train_ids, validation_ids) in folds.items():
            identifier = lite_run_id(task, fold, MODEL_REVISION, fingerprint)
            root = checkpoint_root / identifier
            result_path = root / "result.json"
            if resume and result_path.is_file():
                cached = json.loads(result_path.read_text())
                if (
                    cached.get("config_sha256") != fingerprint
                    or cached.get("resolved_revision") != MODEL_REVISION
                ):
                    raise BenchmarkError("B2-LITE checkpoint provenance drift")
                diagnostics = cached.pop("epoch_diagnostics", [])
                runs.append(cached)
                epoch_rows.extend(diagnostics)
                continue
            _seed_everything(config["seed"] + fold)
            model = _load_model(MODEL_ID, MODEL_REVISION, order, dtype, "mps")
            train_labels = [rows[value].targets[task] for value in train_ids]
            validation_labels = [rows[value].targets[task] for value in validation_ids]
            diagnostics, metrics, runtime = _train(
                model,
                tokenizer,
                [rows[value].text for value in train_ids],
                train_labels,
                order,
                config,
                "mps",
                dtype,
                ([rows[value].text for value in validation_ids], validation_labels),
            )
            root.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(root / "model")
            tokenizer.save_pretrained(root / "model")
            run = {
                "run_id": identifier,
                "task": task,
                "fold": fold,
                "status": "SUCCESS",
                "training_rows": len(train_ids),
                "validation_rows": len(validation_ids),
                "training_seconds": runtime["elapsed_seconds"],
                "evaluation_seconds": 0.0,
                "steps": runtime["steps"],
                "class_weights": json.dumps(
                    dict(zip(order, balanced_class_weights(train_labels, order), strict=True)),
                    sort_keys=True,
                ),
                "dtype": dtype,
                "final_epoch": 3,
                "resolved_revision": MODEL_REVISION,
                "config_sha256": fingerprint,
                "metrics": metrics or {},
            }
            tagged = [
                {"run_id": identifier, "task": task, "fold": fold, **row} for row in diagnostics
            ]
            atomic_json(result_path, {**run, "epoch_diagnostics": tagged})
            runs.append(run)
            epoch_rows.extend(tagged)
            _release_model(model)
    return runs, epoch_rows


def _write_stopped(
    report_dir: Path,
    environment: dict[str, Any],
    provenance: dict[str, Any],
    truncation: list[dict[str, Any]],
    feasibility: dict[str, Any],
) -> None:
    identifier = "B2-LITE-ENGINEERING-FEASIBILITY-S2-FOLD1"
    _csv(
        report_dir / "training_runs.csv",
        [
            {
                "run_id": identifier,
                "task": "S2",
                "fold": 1,
                "status": "ENGINEERING_ONLY",
                "training_rows": 10000,
                "epochs": 3,
                "steps": feasibility["steps"],
                "elapsed_seconds": feasibility["elapsed_seconds"],
                "examples_per_second": feasibility["examples_per_second"],
                "steps_per_second": feasibility["steps_per_second"],
            }
        ],
    )
    _csv(
        report_dir / "epoch_diagnostics.csv",
        [
            {"run_id": identifier, "task": "S2", "fold": 1, **row}
            for row in feasibility["epoch_diagnostics"]
        ],
    )
    _csv(report_dir / "task_summary.csv", [], ["task", "fold_macro_f1", "mean_macro_f1"])
    _csv(
        report_dir / "per_class_metrics.csv",
        [],
        ["run_id", "task", "class", "precision", "recall", "f1", "support"],
    )
    atomic_json(report_dir / "confusion_matrices.json", {})
    _csv(
        report_dir / "runtime_summary.csv",
        [
            {
                "stage": "ENGINEERING_ONLY",
                "training_seconds": feasibility["elapsed_seconds"],
                "projected_competitive_hours": feasibility["projected_hours"],
            }
        ],
    )
    _csv(report_dir / "truncation_audit.csv", truncation)
    atomic_json(report_dir / "environment.json", environment)
    atomic_json(report_dir / "model_provenance.json", provenance)


def finalize_lite_checkpoints(
    *,
    checkpoint_root: Path,
    protocol_report_dir: Path,
    reports_root: Path,
    report_dir: Path,
    protocol: FrozenProtocol,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Aggregate nine already-successful checkpoints without loading or fitting a model."""
    config, fingerprint = load_lite_config(config_path)
    references = _verify_references(reports_root)
    frozen = json.loads((protocol_report_dir / "fingerprints.json").read_text())
    if frozen["protocol_sha256"] != protocol.fingerprint:
        raise BenchmarkError("protocol fingerprint drift")
    provenance = json.loads((report_dir / "model_provenance.json").read_text())
    if (
        provenance.get("protocol_sha256") != protocol.fingerprint
        or provenance.get("development_membership_sha256")
        != frozen["development_membership_sha256"]
        or provenance.get("config_sha256") != fingerprint
        or provenance.get("resolved_revision") != MODEL_REVISION
    ):
        raise BenchmarkError("B2-LITE model provenance drift")
    expected = {(task, fold) for task in TASKS for fold in range(1, 4)}
    runs, diagnostics, validation = [], [], []
    paths = sorted(checkpoint_root.glob("*/result.json"))
    if len(paths) != 9:
        raise BenchmarkError(f"expected 9 B2-LITE checkpoints, found {len(paths)}")
    seen = set()
    for path in paths:
        stored = json.loads(path.read_text())
        key = (stored.get("task"), stored.get("fold"))
        model_config = path.parent / "model" / "config.json"
        checks = {
            "task_fold_expected": key in expected,
            "task_fold_unique": key not in seen,
            "status_success": stored.get("status") == "SUCCESS",
            "config_matches": stored.get("config_sha256") == fingerprint,
            "revision_matches": stored.get("resolved_revision") == MODEL_REVISION,
            "run_id_matches": stored.get("run_id")
            == lite_run_id(key[0], key[1], MODEL_REVISION, fingerprint),
            "final_epoch_is_three": stored.get("final_epoch") == 3,
            "metrics_present": bool(stored.get("metrics")),
            "final_model_config_present": model_config.is_file(),
        }
        if not all(checks.values()):
            raise BenchmarkError(f"invalid B2-LITE checkpoint: {path}")
        seen.add(key)
        epoch_rows = stored.pop("epoch_diagnostics", [])
        if [row.get("epoch") for row in epoch_rows] != [1, 2, 3]:
            raise BenchmarkError(f"incomplete epoch diagnostics: {path}")
        runs.append(stored)
        diagnostics.extend(epoch_rows)
        validation.append(
            {
                "checkpoint": str(path),
                "task": key[0],
                "fold": key[1],
                **checks,
                "protocol_sha256": protocol.fingerprint,
                "development_membership_sha256": frozen["development_membership_sha256"],
            }
        )
    if seen != expected:
        raise BenchmarkError("B2-LITE task/fold checkpoint matrix is incomplete")
    environment = json.loads((report_dir / "environment.json").read_text())
    if environment.get("reference_sha256") != references:
        raise BenchmarkError("B2-LITE reference fingerprint drift")
    environment.update(
        {
            "competitive_runtime_seconds": sum(run["training_seconds"] for run in runs),
            "competitive_models_refitted_during_finalization": 0,
            "existing_checkpoints_reused": True,
            "locked_test_tokenized": False,
            "locked_test_model_performance_accessed": False,
            "locked_test_used_for_tuning": False,
        }
    )
    with (report_dir / "truncation_audit.csv").open(encoding="utf-8", newline="") as handle:
        truncation = list(csv.DictReader(handle))
    _write_competitive_reports(report_dir, runs, diagnostics, environment, provenance, truncation)
    atomic_json(
        report_dir / "checkpoint_validation.json",
        {
            "status": "PASS",
            "checkpoint_count": 9,
            "expected_task_folds": sorted(f"{task}:{fold}" for task, fold in expected),
            "protocol_binding_source": "pre-training authenticated model_provenance.json",
            "competitive_models_refitted": 0,
            "checkpoints": validation,
        },
    )
    return {
        "successful": 9,
        "failed": 0,
        "competitive_models_refitted": 0,
        "existing_checkpoints_reused": True,
        "competitive_runtime_seconds": environment["competitive_runtime_seconds"],
    }


def run_lite_pipeline(
    *,
    stage: str,
    development_dir: Path,
    manifest_dir: Path,
    protocol_report_dir: Path,
    reports_root: Path,
    report_dir: Path,
    checkpoint_root: Path,
    protocol: FrozenProtocol,
    config_path: Path | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    config, fingerprint = load_lite_config(config_path)
    references = _verify_references(reports_root)
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
    _seed_everything(config["seed"])
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / ".work").mkdir(parents=True, exist_ok=True)
    if stage == "benchmark":
        preflight = json.loads((report_dir / ".work" / "preflight.json").read_text())
        feasibility = json.loads((report_dir / ".work" / "feasibility.json").read_text())
        provenance = json.loads((report_dir / "model_provenance.json").read_text())
        environment = json.loads((report_dir / "environment.json").read_text())
        if (
            preflight["resolved_revision"] != MODEL_REVISION
            or provenance["config_sha256"] != fingerprint
            or not feasibility["competitive_execution_allowed"]
        ):
            raise BenchmarkError("B2-LITE feasibility handoff provenance mismatch")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False, use_fast=True
        )
        with (report_dir / "truncation_audit.csv").open(encoding="utf-8", newline="") as handle:
            truncation = list(csv.DictReader(handle))
        started = time.monotonic()
        runs, epochs = _run_competitive_lite(
            rows=rows,
            folds=folds,
            protocol=protocol,
            tokenizer=tokenizer,
            config=config,
            fingerprint=fingerprint,
            dtype=preflight["dtype"],
            checkpoint_root=checkpoint_root,
            resume=resume,
        )
        environment["competitive_runtime_seconds"] = time.monotonic() - started
        _write_competitive_reports(report_dir, runs, epochs, environment, provenance, truncation)
        return {
            "preflight": preflight,
            "feasibility": feasibility,
            "stopped_before_competitive": False,
            "successful": len(runs),
            "failed": 0,
            "competitive_runtime_seconds": environment["competitive_runtime_seconds"],
        }
    smoke = config["smoke"]
    train_ids, _ = folds[1]
    order = protocol.targets["s2"].order
    sample_ids = _bounded_ids_with_class_coverage(
        train_ids, rows, "S2", smoke["training_rows"], order
    )
    preflight = mps_training_preflight(
        config,
        [rows[value].text for value in sample_ids],
        [rows[value].targets["S2"] for value in sample_ids],
        order,
        report_dir,
    )
    if preflight["resolved_revision"] != MODEL_REVISION:
        raise BenchmarkError("MiniLM Hub revision differs from frozen B1 revision")
    model_config = __import__("transformers").AutoConfig.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False
    )
    provenance = {
        "model_id": MODEL_ID,
        "resolved_revision": MODEL_REVISION,
        "architecture": preflight["architecture"],
        "parameter_count": preflight["parameter_count"],
        "trainable_parameter_count": preflight["trainable_parameter_count"],
        "tokenizer_class": preflight["tokenizer_class"],
        "hidden_size": model_config.hidden_size,
        "layer_count": model_config.num_hidden_layers,
        "max_length": 256,
        "device": "mps",
        "dtype": preflight["dtype"],
        "trust_remote_code": False,
        "config_sha256": fingerprint,
        "protocol_sha256": protocol.fingerprint,
        "development_membership_sha256": frozen["development_membership_sha256"],
    }
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: version(name)
            for name in ("torch", "transformers", "accelerate", "sentencepiece", "protobuf")
        },
        "mps_is_built": True,
        "mps_is_available": True,
        "device": "mps",
        "dtype": preflight["dtype"],
        "bf16_error": preflight["bf16_error"],
        "reference_sha256": references,
        "locked_test_tokenized": False,
        "locked_test_model_performance_accessed": False,
        "locked_test_used_for_tuning": False,
    }
    if stage == "preflight":
        atomic_json(report_dir / "environment.json", environment)
        atomic_json(report_dir / "model_provenance.json", provenance)
        return {"preflight": preflight}
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False, use_fast=True
    )
    truncation = _truncation_audit(tokenizer, rows, 256)
    feasibility_ids = folds[1][0][:10000]
    _seed_everything(config["seed"])
    import psutil

    rss_before = psutil.Process().memory_info().rss
    model = _load_model(MODEL_ID, MODEL_REVISION, order, preflight["dtype"], "mps")
    diagnostics, _, feasibility = _train(
        model,
        tokenizer,
        [rows[value].text for value in feasibility_ids],
        [rows[value].targets["S2"] for value in feasibility_ids],
        order,
        config,
        "mps",
        preflight["dtype"],
    )
    feasibility.update(
        {
            "stage": "ENGINEERING_ONLY",
            "task": "S2",
            "fold": 1,
            "bounded_training_rows": 10000,
            "epoch_diagnostics": diagnostics,
            "process_rss_before_bytes": rss_before,
            "process_rss_after_bytes": psutil.Process().memory_info().rss,
        }
    )
    feasibility.update(_competitive_projection(feasibility, folds, 3))
    feasibility["maximum_projected_hours"] = 24.0
    feasibility["competitive_execution_allowed"] = feasibility["projected_hours"] <= 24.0
    atomic_json(report_dir / ".work" / "feasibility.json", feasibility)
    environment["engineering_feasibility"] = feasibility
    _write_stopped(report_dir, environment, provenance, truncation, feasibility)
    _release_model(model)
    if not feasibility["competitive_execution_allowed"]:
        return {
            "preflight": preflight,
            "feasibility": feasibility,
            "stopped_before_competitive": True,
            "successful": 0,
            "failed": 0,
        }
    if stage == "feasibility":
        return {
            "preflight": preflight,
            "feasibility": feasibility,
            "stopped_before_competitive": False,
            "successful": 0,
            "failed": 0,
        }
    started = time.monotonic()
    runs, epochs = _run_competitive_lite(
        rows=rows,
        folds=folds,
        protocol=protocol,
        tokenizer=tokenizer,
        config=config,
        fingerprint=fingerprint,
        dtype=preflight["dtype"],
        checkpoint_root=checkpoint_root,
        resume=resume,
    )
    environment["competitive_runtime_seconds"] = time.monotonic() - started
    _write_competitive_reports(report_dir, runs, epochs, environment, provenance, truncation)
    return {
        "preflight": preflight,
        "feasibility": feasibility,
        "stopped_before_competitive": False,
        "successful": len(runs),
        "failed": 0,
        "competitive_runtime_seconds": environment["competitive_runtime_seconds"],
    }
