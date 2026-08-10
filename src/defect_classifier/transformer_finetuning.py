"""Development-only Phase B2 controlled DeBERTa fine-tuning."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random
import statistics
import tempfile
import time
import tomllib
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from defect_classifier.classical_benchmark import (
    BenchmarkError,
    _load_development_rows,
    _load_fold,
)
from defect_classifier.classical_metrics import classification_metrics
from defect_classifier.embedding_cache import atomic_json
from defect_classifier.preparation import _membership_fingerprint
from defect_classifier.protocol import FrozenProtocol

TASKS = ("S6", "S3", "S2")
MODEL_ID = "microsoft/deberta-v3-small"
EXPECTED = {
    "model_id": MODEL_ID,
    "max_length": 256,
    "learning_rate": 4.5e-5,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 16,
    "gradient_accumulation_steps": 1,
    "class_weighting": "balanced_train_only",
    "early_stopping": False,
    "final_epoch_selection": 3,
    "trust_remote_code": False,
}
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
    "B15_REPORT": (
        "lexical_semantic_fusion_v1/LEXICAL_SEMANTIC_FUSION_REPORT.md",
        "e7bcf9afe11ac5b29f0ff0996c5c011351684164048c7afd871062e242bd99c7",
    ),
    "B15_SUMMARY": (
        "lexical_semantic_fusion_v1/task_summary.csv",
        "0b27212eea2bccd615c228fab6d9d6fd4d7d33a3f5595c41e8a9c656d561ba8f",
    ),
}


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "transformer_finetuning_v1.toml"


def load_finetuning_config(path: Path | None = None) -> tuple[dict[str, Any], str]:
    raw = (path or default_config_path()).read_bytes()
    config = tomllib.loads(raw.decode())
    validate_finetuning_config(config)
    return config, hashlib.sha256(raw).hexdigest()


def validate_finetuning_config(config: dict[str, Any]) -> None:
    if any(config.get(key) != value for key, value in EXPECTED.items()):
        raise BenchmarkError("Phase B2 fixed training method drift")
    if set(config) != {"experiment_id", "seed", *EXPECTED, "feasibility", "smoke"}:
        raise BenchmarkError("unapproved Phase B2 search axis")
    if config["feasibility"] != {
        "task": "S2",
        "fold": 1,
        "training_rows": 10000,
        "maximum_projected_hours": 24.0,
    }:
        raise BenchmarkError("Phase B2 feasibility design drift")


def search_space_size(config: dict[str, Any]) -> int:
    validate_finetuning_config(config)
    return len(TASKS) * 3


def run_id(task: str, fold: int, revision: str, config_fingerprint: str) -> str:
    payload = f"B2|{task}|{fold}|{revision}|{config_fingerprint}"
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


def balanced_class_weights(train_labels: list[str], class_order: tuple[str, ...]) -> list[float]:
    counts = Counter(train_labels)
    if set(counts) - set(class_order) or any(counts[label] == 0 for label in class_order):
        raise BenchmarkError("TRAIN labels do not cover the frozen task class set")
    total = len(train_labels)
    return [total / (len(class_order) * counts[label]) for label in class_order]


def _bounded_ids_with_class_coverage(
    ids: list[str], rows: dict[str, Any], task: str, limit: int, class_order: tuple[str, ...]
) -> list[str]:
    selected = []
    for label in class_order:
        selected.append(next(value for value in ids if rows[value].targets[task] == label))
    selected_set = set(selected)
    selected.extend(value for value in ids if value not in selected_set)
    return selected[:limit]


def _verify_references(reports_root: Path) -> dict[str, str]:
    actual = {}
    for name, (relative, expected) in REFERENCE_HASHES.items():
        path = reports_root / relative
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "MISSING"
        if digest != expected:
            raise BenchmarkError(f"previous report fingerprint drift: {name}")
        actual[name] = digest
    return actual


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class _TokenDataset:
    def __init__(self, encodings: dict[str, list[list[int]]], labels: list[int]) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            **{key: value[index] for key, value in self.encodings.items()},
            "labels": self.labels[index],
        }


def _tokenize(tokenizer: Any, texts: list[str], max_length: int) -> dict[str, list[list[int]]]:
    return tokenizer(texts, truncation=True, max_length=max_length, padding=False)


def _loader(
    tokenizer: Any, texts: list[str], labels: list[int], config: dict[str, Any], train: bool
) -> Any:
    from torch.utils.data import DataLoader
    from transformers import DataCollatorWithPadding

    dataset = _TokenDataset(_tokenize(tokenizer, texts, config["max_length"]), labels)
    generator = __import__("torch").Generator().manual_seed(config["seed"])
    return DataLoader(
        dataset,
        batch_size=config["per_device_train_batch_size"]
        if train
        else config["per_device_eval_batch_size"],
        shuffle=train,
        generator=generator,
        collate_fn=DataCollatorWithPadding(tokenizer, return_tensors="pt"),
        num_workers=0,
    )


def _move(batch: dict[str, Any], device: str) -> tuple[dict[str, Any], Any]:
    labels = batch.pop("labels").to(device)
    return {key: value.to(device) for key, value in batch.items()}, labels


def _weighted_loss(logits: Any, labels: Any, weights: Any) -> Any:
    import torch.nn.functional as functional

    return functional.cross_entropy(logits.float(), labels, weight=weights.float())


def _evaluate(model: Any, loader: Any, device: str, class_order: tuple[str, ...]) -> dict[str, Any]:
    import torch

    model.eval()
    true, predicted = [], []
    with torch.inference_mode():
        for batch in loader:
            inputs, labels = _move(batch, device)
            logits = model(**inputs).logits
            true.extend(labels.cpu().tolist())
            predicted.extend(logits.argmax(dim=-1).cpu().tolist())
    return classification_metrics(
        [class_order[index] for index in true],
        [class_order[index] for index in predicted],
        class_order,
    )


def _load_model(
    model_id: str, revision: str, labels: tuple[str, ...], dtype: str, device: str
) -> Any:
    import torch
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        revision=revision,
        num_labels=len(labels),
        id2label={index: label for index, label in enumerate(labels)},
        label2id={label: index for index, label in enumerate(labels)},
        trust_remote_code=False,
    )
    model.to(device=device, dtype=torch.bfloat16 if dtype == "bfloat16" else torch.float32)
    return model


def mps_training_preflight(
    config: dict[str, Any],
    texts: list[str],
    labels: list[str],
    class_order: tuple[str, ...],
    report_dir: Path,
) -> dict[str, Any]:
    import torch
    from huggingface_hub import model_info
    from transformers import AutoTokenizer

    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise BenchmarkError("Apple MPS training is unavailable")
    revision = model_info(config["model_id"]).sha
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"], revision=revision, trust_remote_code=False, use_fast=True
    )
    label_ids = [class_order.index(value) for value in labels]
    weights = torch.tensor(balanced_class_weights(labels, class_order), device="mps")
    dtype, bf16_error = "bfloat16", None
    started = time.monotonic()
    try:
        model = _load_model(config["model_id"], revision, class_order, dtype, "mps")
        loader = _loader(tokenizer, texts, label_ids, config, True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
        model.train()
        batch = next(iter(loader))
        inputs, target = _move(batch, "mps")
        loss = _weighted_loss(model(**inputs).logits, target, weights)
        loss.backward()
        if not all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ):
            raise RuntimeError("non-finite gradient")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    except Exception as exc:
        bf16_error = f"{type(exc).__name__}: {exc}"
        if hasattr(torch, "mps"):
            torch.mps.empty_cache()
        dtype = "float32"
        model = _load_model(config["model_id"], revision, class_order, dtype, "mps")
        loader = _loader(tokenizer, texts, label_ids, config, True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
        model.train()
        batch = next(iter(loader))
        inputs, target = _move(batch, "mps")
        loss = _weighted_loss(model(**inputs).logits, target, weights)
        loss.backward()
        if not all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ):
            raise BenchmarkError("float32 preflight produced non-finite gradients") from exc
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    evaluation = _evaluate(
        model, _loader(tokenizer, texts, label_ids, config, False), "mps", class_order
    )
    with tempfile.TemporaryDirectory(dir=report_dir / ".work") as temporary:
        model.save_pretrained(temporary)
        tokenizer.save_pretrained(temporary)
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        restored_model = AutoModelForSequenceClassification.from_pretrained(
            temporary, trust_remote_code=False
        )
        restored_tokenizer = AutoTokenizer.from_pretrained(temporary, trust_remote_code=False)
        if (
            restored_model.config.num_labels != len(class_order)
            or restored_tokenizer.model_max_length != tokenizer.model_max_length
        ):
            raise BenchmarkError("checkpoint save/load smoke validation failed")
    parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    result = {
        "mps_is_built": True,
        "mps_is_available": True,
        "device": "mps",
        "dtype": dtype,
        "bf16_error": bf16_error,
        "resolved_revision": revision,
        "architecture": type(model).__name__,
        "parameter_count": parameters,
        "trainable_parameter_count": trainable,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_model_max_length": tokenizer.model_max_length,
        "configured_max_length": 256,
        "weighted_loss": float(loss.detach().cpu()),
        "gradients_finite": True,
        "optimizer_step_completed": True,
        "evaluation_completed": bool(evaluation),
        "checkpoint_save_load_completed": True,
        "runtime_seconds": time.monotonic() - started,
    }
    atomic_json(report_dir / ".work" / "preflight.json", result)
    return result


def _train(
    model: Any,
    tokenizer: Any,
    train_texts: list[str],
    train_labels: list[str],
    class_order: tuple[str, ...],
    config: dict[str, Any],
    device: str,
    dtype: str,
    validation: tuple[list[str], list[str]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, Any]]:
    import torch

    label_ids = [class_order.index(value) for value in train_labels]
    weights_values = balanced_class_weights(train_labels, class_order)
    weights = torch.tensor(weights_values, device=device)
    loader = _loader(tokenizer, train_texts, label_ids, config, True)
    evaluation_loader = None
    if validation:
        evaluation_loader = _loader(
            tokenizer,
            validation[0],
            [class_order.index(value) for value in validation[1]],
            config,
            False,
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    diagnostics = []
    steps = 0
    started = time.monotonic()
    for epoch in range(1, config["num_train_epochs"] + 1):
        model.train()
        losses = []
        epoch_started = time.monotonic()
        for batch in loader:
            inputs, labels = _move(batch, device)
            loss = _weighted_loss(model(**inputs).logits, labels, weights)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))
            steps += 1
        row = {
            "epoch": epoch,
            "training_loss": statistics.fmean(losses),
            "training_seconds": time.monotonic() - epoch_started,
            "steps": len(losses),
        }
        if evaluation_loader is not None:
            metrics = _evaluate(model, evaluation_loader, device, class_order)
            row.update(
                {
                    f"validation_{key}": metrics[key]
                    for key in ("macro_f1", "balanced_accuracy", "accuracy", "weighted_f1")
                }
            )
        diagnostics.append(row)
    metrics = (
        _evaluate(model, evaluation_loader, device, class_order)
        if evaluation_loader is not None
        else None
    )
    elapsed = time.monotonic() - started
    return (
        diagnostics,
        metrics,
        {
            "elapsed_seconds": elapsed,
            "steps": steps,
            "examples_per_second": len(train_texts) * config["num_train_epochs"] / elapsed,
            "steps_per_second": steps / elapsed,
            "class_weights": dict(zip(class_order, weights_values, strict=True)),
            "dtype": dtype,
        },
    )


def _truncation_audit(
    tokenizer: Any, rows: dict[str, Any], max_length: int
) -> list[dict[str, Any]]:
    by_project: dict[str, list[str]] = {}
    for stable_id, row in rows.items():
        by_project.setdefault(stable_id.split(":", 1)[0], []).append(row.text)
    output = []
    for project, texts in sorted(by_project.items()):
        lengths = []
        for start in range(0, len(texts), 256):
            lengths.extend(
                len(ids)
                for ids in tokenizer(texts[start : start + 256], truncation=False, padding=False)[
                    "input_ids"
                ]
            )
        truncated = sum(length > max_length for length in lengths)
        output.append(
            {
                "project": project,
                "development_records": len(lengths),
                "fits_without_truncation": len(lengths) - truncated,
                "requires_truncation": truncated,
                "truncation_percentage": truncated / len(lengths) * 100,
                "token_length_min": min(lengths),
                "token_length_mean": statistics.fmean(lengths),
                "token_length_max": max(lengths),
                "max_length": max_length,
            }
        )
    return output


def _csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    columns = fields or sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_stopped_reports(
    report_dir: Path,
    environment: dict[str, Any],
    provenance: dict[str, Any],
    truncation: list[dict[str, Any]],
    feasibility: dict[str, Any],
) -> None:
    engineering_id = "B2-ENGINEERING-FEASIBILITY-S2-FOLD1"
    _csv(
        report_dir / "training_runs.csv",
        [
            {
                "run_id": engineering_id,
                "task": "S2",
                "fold": 1,
                "status": "ENGINEERING_ONLY",
                "training_rows": feasibility["bounded_training_rows"],
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
            {"run_id": engineering_id, "task": "S2", "fold": 1, **row}
            for row in feasibility["epoch_diagnostics"]
        ],
    )
    _csv(report_dir / "task_summary.csv", [], ["task", "fold_macro_f1", "mean_macro_f1"])
    _csv(
        report_dir / "per_class_metrics.csv",
        [],
        ["run_id", "task", "class", "precision", "recall", "f1", "support"],
    )
    _csv(
        report_dir / "runtime_summary.csv",
        [
            {
                "stage": "ENGINEERING_ONLY",
                "task": "S2",
                "training_seconds": feasibility["elapsed_seconds"],
                "projected_competitive_seconds": feasibility["projected_seconds"],
                "projected_competitive_hours": feasibility["projected_hours"],
            }
        ],
    )
    atomic_json(report_dir / "confusion_matrices.json", {})
    _csv(report_dir / "truncation_audit.csv", truncation)
    atomic_json(report_dir / "environment.json", environment)
    atomic_json(report_dir / "model_provenance.json", provenance)


def _write_competitive_reports(
    report_dir: Path,
    runs: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
    environment: dict[str, Any],
    provenance: dict[str, Any],
    truncation: list[dict[str, Any]],
) -> None:
    flat, per_class, matrices, summary = [], [], {}, []
    for run in runs:
        metrics = run["metrics"]
        flat.append(
            {
                **{key: value for key, value in run.items() if key != "metrics"},
                **{
                    key: metrics[key]
                    for key in ("macro_f1", "balanced_accuracy", "accuracy", "weighted_f1")
                },
            }
        )
        for label, values in metrics["per_class"].items():
            per_class.append(
                {
                    "run_id": run["run_id"],
                    "task": run["task"],
                    "fold": run["fold"],
                    "class": label,
                    **values,
                }
            )
        matrices[run["run_id"]] = {
            "task": run["task"],
            "fold": run["fold"],
            "labels": list(metrics["per_class"]),
            "matrix": metrics["confusion_matrix"],
        }
    for task in TASKS:
        selected = sorted((run for run in runs if run["task"] == task), key=lambda run: run["fold"])
        macro = [run["metrics"]["macro_f1"] for run in selected]
        row = {
            "task": task,
            "fold_macro_f1": "|".join(f"{value:.10f}" for value in macro),
            "mean_macro_f1": statistics.fmean(macro),
            "std_macro_f1": statistics.pstdev(macro),
            "mean_balanced_accuracy": statistics.fmean(
                run["metrics"]["balanced_accuracy"] for run in selected
            ),
            "mean_accuracy": statistics.fmean(run["metrics"]["accuracy"] for run in selected),
            "mean_weighted_f1": statistics.fmean(run["metrics"]["weighted_f1"] for run in selected),
        }
        if task == "S2":
            high = [run["metrics"]["per_class"]["HIGH_IMPACT"] for run in selected]
            row.update(
                {
                    "high_impact_precision": statistics.fmean(value["precision"] for value in high),
                    "high_impact_recall": statistics.fmean(value["recall"] for value in high),
                    "high_impact_f1": statistics.fmean(value["f1"] for value in high),
                }
            )
            row["legacy_reproduction_guard"] = (
                "PASS" if row["high_impact_precision"] >= 0.30 else "FAIL"
            )
        summary.append(row)
    _csv(report_dir / "training_runs.csv", flat)
    _csv(report_dir / "epoch_diagnostics.csv", diagnostics)
    _csv(report_dir / "task_summary.csv", summary)
    _csv(report_dir / "per_class_metrics.csv", per_class)
    atomic_json(report_dir / "confusion_matrices.json", matrices)
    _csv(
        report_dir / "runtime_summary.csv",
        [
            {
                "task": task,
                "training_seconds": sum(
                    run["training_seconds"] for run in runs if run["task"] == task
                ),
                "evaluation_seconds": sum(
                    run["evaluation_seconds"] for run in runs if run["task"] == task
                ),
            }
            for task in TASKS
        ],
    )
    _csv(report_dir / "truncation_audit.csv", truncation)
    atomic_json(report_dir / "environment.json", environment)
    atomic_json(report_dir / "model_provenance.json", provenance)


def _run_competitive(
    *,
    rows: dict[str, Any],
    folds: dict[int, tuple[list[str], list[str]]],
    protocol: FrozenProtocol,
    tokenizer: Any,
    config: dict[str, Any],
    fingerprint: str,
    revision: str,
    dtype: str,
    checkpoint_root: Path,
    resume: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    runs, all_diagnostics = [], []
    for task in TASKS:
        order = protocol.targets[task.casefold()].order
        for fold, (train_ids, validation_ids) in folds.items():
            identifier = run_id(task, fold, revision, fingerprint)
            root = checkpoint_root / identifier
            metrics_path = root / "result.json"
            if resume and metrics_path.is_file():
                cached = json.loads(metrics_path.read_text())
                if (
                    cached.get("config_sha256") != fingerprint
                    or cached.get("resolved_revision") != revision
                ):
                    raise BenchmarkError("B2 checkpoint provenance drift")
                runs.append(cached)
                all_diagnostics.extend(cached.pop("epoch_diagnostics", []))
                continue
            _seed_everything(config["seed"] + fold)
            model = _load_model(MODEL_ID, revision, order, dtype, "mps")
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
            evaluation_started = time.monotonic()
            metrics = metrics or {}
            evaluation_seconds = time.monotonic() - evaluation_started
            root.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(root / "model")
            tokenizer.save_pretrained(root / "model")
            weights = balanced_class_weights(train_labels, order)
            run = {
                "run_id": identifier,
                "task": task,
                "fold": fold,
                "status": "SUCCESS",
                "training_rows": len(train_ids),
                "validation_rows": len(validation_ids),
                "training_seconds": runtime["elapsed_seconds"],
                "evaluation_seconds": evaluation_seconds,
                "steps": runtime["steps"],
                "class_weights": json.dumps(dict(zip(order, weights, strict=True)), sort_keys=True),
                "dtype": dtype,
                "final_epoch": 3,
                "resolved_revision": revision,
                "config_sha256": fingerprint,
                "metrics": metrics,
            }
            tagged = [
                {"run_id": identifier, "task": task, "fold": fold, **row} for row in diagnostics
            ]
            checkpoint = {**run, "epoch_diagnostics": tagged}
            atomic_json(metrics_path, checkpoint)
            runs.append(run)
            all_diagnostics.extend(tagged)
    return runs, all_diagnostics


def _competitive_projection(
    feasibility: dict[str, Any], folds: dict[int, tuple[list[str], list[str]]], epochs: int
) -> dict[str, Any]:
    total_training_examples = len(TASKS) * sum(len(train) for train, _ in folds.values()) * epochs
    seconds = total_training_examples / feasibility["examples_per_second"]
    return {
        "competitive_training_examples_across_epochs": total_training_examples,
        "projected_seconds": seconds,
        "projected_hours": seconds / 3600,
    }


def run_b2_pipeline(
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
    config, fingerprint = load_finetuning_config(config_path)
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
    smoke = config["smoke"]
    train_ids, validation_ids = folds[smoke["fold"]]
    class_order = protocol.targets[smoke["task"].casefold()].order
    sample_ids = _bounded_ids_with_class_coverage(
        train_ids, rows, smoke["task"], smoke["training_rows"], class_order
    )
    preflight = mps_training_preflight(
        config,
        [rows[value].text for value in sample_ids],
        [rows[value].targets[smoke["task"]] for value in sample_ids],
        class_order,
        report_dir,
    )
    provenance = {
        "model_id": MODEL_ID,
        "resolved_revision": preflight["resolved_revision"],
        "architecture": preflight["architecture"],
        "parameter_count": preflight["parameter_count"],
        "trainable_parameter_count": preflight["trainable_parameter_count"],
        "tokenizer_class": preflight["tokenizer_class"],
        "tokenizer_model_max_length": preflight["tokenizer_model_max_length"],
        "max_length": 256,
        "device": preflight["device"],
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
        **{
            key: preflight[key]
            for key in ("mps_is_built", "mps_is_available", "device", "dtype", "bf16_error")
        },
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
        MODEL_ID, revision=preflight["resolved_revision"], trust_remote_code=False, use_fast=True
    )
    truncation = _truncation_audit(tokenizer, rows, 256)
    feasibility_design = config["feasibility"]
    feasibility_train = folds[1][0][: feasibility_design["training_rows"]]
    task = "S2"
    order = protocol.targets["s2"].order
    _seed_everything(config["seed"])
    import psutil

    resident_before = psutil.Process().memory_info().rss
    model = _load_model(MODEL_ID, preflight["resolved_revision"], order, preflight["dtype"], "mps")
    diagnostics, _, feasibility = _train(
        model,
        tokenizer,
        [rows[value].text for value in feasibility_train],
        [rows[value].targets[task] for value in feasibility_train],
        order,
        config,
        "mps",
        preflight["dtype"],
    )
    feasibility.update(
        {
            "stage": "ENGINEERING_ONLY",
            "task": task,
            "fold": 1,
            "bounded_training_rows": len(feasibility_train),
            "epoch_diagnostics": diagnostics,
            "process_rss_before_bytes": resident_before,
            "process_rss_after_bytes": psutil.Process().memory_info().rss,
        }
    )
    feasibility.update(_competitive_projection(feasibility, folds, config["num_train_epochs"]))
    feasibility["maximum_projected_hours"] = config["feasibility"]["maximum_projected_hours"]
    feasibility["competitive_execution_allowed"] = (
        feasibility["projected_hours"] <= feasibility["maximum_projected_hours"]
    )
    atomic_json(report_dir / ".work" / "feasibility.json", feasibility)
    environment["engineering_feasibility"] = feasibility
    _write_stopped_reports(report_dir, environment, provenance, truncation, feasibility)
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
    runs, epoch_rows = _run_competitive(
        rows=rows,
        folds=folds,
        protocol=protocol,
        tokenizer=tokenizer,
        config=config,
        fingerprint=fingerprint,
        revision=preflight["resolved_revision"],
        dtype=preflight["dtype"],
        checkpoint_root=checkpoint_root,
        resume=resume,
    )
    environment["competitive_runtime_seconds"] = sum(
        run["training_seconds"] + run["evaluation_seconds"] for run in runs
    )
    _write_competitive_reports(report_dir, runs, epoch_rows, environment, provenance, truncation)
    return {
        "preflight": preflight,
        "feasibility": feasibility,
        "stopped_before_competitive": False,
        "successful": len(runs),
        "failed": 0,
    }
