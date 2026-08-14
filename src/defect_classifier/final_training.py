"""Final full-DEVELOPMENT training for the frozen model-selection-v1 pipeline."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import pickle
import platform
import resource
import shutil
import subprocess
import time
import warnings
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from defect_classifier.classical_benchmark import (
    BenchmarkError,
    _load_development_rows,
    _make_model,
)
from defect_classifier.classical_features import build_sparse_features
from defect_classifier.classical_optimization import _feature_config, load_optimization_config
from defect_classifier.lexical_semantic_fusion import fuse_sparse_features
from defect_classifier.preparation import _membership_fingerprint
from defect_classifier.protocol import FrozenProtocol
from defect_classifier.rta_adalora import (
    EFFECTIVE_BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    PEFT_VERSION,
    _atomic_json,
    _clear_mps,
    _seed_everything,
    _status,
    adalora_schedule,
    balanced_mean_one_weights,
)
from defect_classifier.rta_adalora_multitask import (
    TASKS,
    _build_multitask_model,
    _embed_only,
    _fixed_tokenize,
    _labels_for_ids,
    _loader,
    _train,
    load_multitask_config,
)
from defect_classifier.rta_fusion import DIMENSION, MODEL_ID, MODEL_REVISION

FINAL_TRAINING_TAG = "final-training-runner-v1"
FREEZE_ID = "model-selection-v1"
SELECTED_FAMILY = "B4H_ADAPTED_RTA_LEXICAL_FUSION"
EXPECTED_DEVELOPMENT_ROWS = 207575
EXPECTED_DEVELOPMENT_SHA256 = "4f62fdf4164594126c421955804b654cd5d5f8f7b46ada345ae0cffa71460d0f"
EXPECTED_PROTOCOL_SHA256 = "85faacae1e7f411d68803653388611c37c62730ea9217e9700a0ad6ac41b7cda"
TRAIN_MICRO_BATCH = 8


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _git_provenance(repo_root: Path) -> dict[str, Any]:
    status = _git(repo_root, "status", "--porcelain")
    if status:
        raise BenchmarkError("final training requires a clean Git working tree")
    head = _git(repo_root, "rev-parse", "HEAD")
    tags = [value for value in _git(repo_root, "tag", "--points-at", "HEAD").splitlines() if value]
    if FINAL_TRAINING_TAG not in tags:
        raise BenchmarkError(
            f"final training requires tag {FINAL_TRAINING_TAG!r} on the current clean HEAD"
        )
    return {"git_head": head, "git_tags_at_head": tags, "git_worktree_clean": True}


def _load_and_validate_freeze(path: Path, protocol: FrozenProtocol) -> tuple[dict[str, Any], str]:
    document = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "freeze_id": FREEZE_ID,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "selection_metric": "macro_f1",
        "selection_scope": "DEVELOPMENT_ONLY",
        "development_model_selection_closed": True,
        "additional_model_selection_experiments_allowed": False,
        "selected_family": SELECTED_FAMILY,
        "direct_heads_selected": False,
        "development_rows": EXPECTED_DEVELOPMENT_ROWS,
        "development_membership_sha256": EXPECTED_DEVELOPMENT_SHA256,
        "locked_test_accessed_for_model_selection": False,
    }
    if any(document.get(key) != value for key, value in expected.items()):
        raise BenchmarkError("model-selection freeze artifact drift")
    final_training = document.get("final_training", {})
    if final_training != {
        "training_partition": "FULL_DEVELOPMENT",
        "training_rows": EXPECTED_DEVELOPMENT_ROWS,
        "shared_transformer_training_runs": 1,
        "epochs": 3,
        "use_locked_test_during_training": False,
        "use_cv_checkpoint_as_final_model": False,
    }:
        raise BenchmarkError("frozen final-training rule drift")
    if protocol.fingerprint != EXPECTED_PROTOCOL_SHA256:
        raise BenchmarkError("protocol fingerprint differs from the frozen final-training protocol")
    return document, _sha256_file(path)


def _pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def _artifact_hashes(root: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        output[str(path.relative_to(root))] = {
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    return output


def mark_final_training_failed(report_dir: Path, error: BaseException) -> None:
    _status(
        report_dir / ".work" / "final_training.status",
        state="FAILED",
        current_phase="failed",
        error_type=type(error).__name__,
        error_message=str(error),
    )


def run_final_training(
    *,
    development_dir: Path,
    protocol_report_dir: Path,
    report_dir: Path,
    artifact_root: Path,
    freeze_file: Path,
    protocol: FrozenProtocol,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Train and freeze one final B4-H fusion pipeline using DEVELOPMENT only."""
    import psutil
    import torch
    from transformers import AutoTokenizer

    repo_root = Path(__file__).resolve().parents[2]
    git_provenance = _git_provenance(repo_root)

    if os.environ.get(protocol.unlock_environment_variable) is not None:
        raise BenchmarkError(
            "locked-test unlock variable must be completely unset during final model training"
        )
    if version("peft") != PEFT_VERSION:
        raise BenchmarkError("installed PEFT version differs from the frozen B4-H version")
    if not torch.backends.mps.is_available():
        raise BenchmarkError("final B4-H training requires MPS")

    freeze, freeze_sha256 = _load_and_validate_freeze(freeze_file, protocol)
    frozen = json.loads((protocol_report_dir / "fingerprints.json").read_text(encoding="utf-8"))
    if (
        frozen.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256
        or frozen.get("development_membership_sha256") != EXPECTED_DEVELOPMENT_SHA256
    ):
        raise BenchmarkError("frozen protocol/development fingerprints drift")

    config, config_sha256 = load_multitask_config(config_path)
    a2_config, a2_config_sha256 = load_optimization_config()

    report_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = report_dir / "final_training_manifest.json"
    if manifest_path.exists():
        raise BenchmarkError("final training manifest already exists; refusing a second final fit")
    work = report_dir / ".work"
    work.mkdir(parents=True, exist_ok=True)
    status_path = work / "final_training.status"
    if status_path.exists():
        status_path.unlink()

    if artifact_root.exists():
        shutil.rmtree(artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)

    _status(
        status_path,
        state="RUNNING",
        current_phase="loading_development",
        stage="FINAL_TRAINING_DEVELOPMENT_ONLY",
        selected_family=SELECTED_FAMILY,
        source_membership="FULL_DEVELOPMENT",
        expected_rows=EXPECTED_DEVELOPMENT_ROWS,
        expected_development_sha256=EXPECTED_DEVELOPMENT_SHA256,
        model_id=MODEL_ID,
        resolved_revision=MODEL_REVISION,
        max_length=512,
        dtype="float32",
        train_micro_batch=TRAIN_MICRO_BATCH,
        gradient_accumulation=4,
        effective_batch_size=EFFECTIVE_BATCH_SIZE,
        epochs=EPOCHS,
        locked_test_tokenized=False,
        locked_test_embedded=False,
        locked_test_model_performance_accessed=False,
        locked_test_used_for_tuning=False,
        **git_provenance,
    )

    started = time.monotonic()
    rows = _load_development_rows(development_dir)
    train_ids = list(rows)
    if len(train_ids) != EXPECTED_DEVELOPMENT_ROWS:
        raise BenchmarkError("final training DEVELOPMENT row-count drift")
    if _membership_fingerprint(train_ids) != EXPECTED_DEVELOPMENT_SHA256:
        raise BenchmarkError("final training DEVELOPMENT membership drift")

    labels, label_ids = _labels_for_ids(rows, train_ids, protocol)
    orders = {task: protocol.targets[task.casefold()].order for task in TASKS}
    class_weight_values = {
        task: balanced_mean_one_weights(labels[task], orders[task]) for task in TASKS
    }
    texts = [rows[stable_id].text for stable_id in train_ids]

    _seed_everything(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=False,
        local_files_only=True,
    )
    _status(status_path, current_phase="tokenizing_full_development")
    token_started = time.monotonic()
    encoded = _fixed_tokenize(tokenizer, texts)
    tokenization_seconds = time.monotonic() - token_started

    accumulation = EFFECTIVE_BATCH_SIZE // TRAIN_MICRO_BATCH
    if accumulation != 4:
        raise BenchmarkError("frozen final-training gradient accumulation drift")
    steps_per_epoch = math.ceil(math.ceil(len(train_ids) / TRAIN_MICRO_BATCH) / accumulation)
    schedule = adalora_schedule(steps_per_epoch * EPOCHS)
    resolved = {
        **config,
        "resolved_train_micro_batch": TRAIN_MICRO_BATCH,
        "resolved_gradient_accumulation": accumulation,
        "resolved_effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "steps_per_epoch": steps_per_epoch,
        "adalora_schedule": schedule,
        "source_membership": "FULL_DEVELOPMENT",
        "development_rows": len(train_ids),
        "development_membership_sha256": EXPECTED_DEVELOPMENT_SHA256,
    }
    _atomic_json(work / "resolved_training_configuration.json", resolved)

    model, model_details = _build_multitask_model(schedule, "mps")
    weight_tensors = {
        task: torch.tensor(values, device="mps") for task, values in class_weight_values.items()
    }
    loader = _loader(
        tokenizer,
        encoded,
        label_ids,
        TRAIN_MICRO_BATCH,
        config["seed"],
        True,
    )
    _status(
        status_path,
        current_phase="training",
        total_optimizer_steps=schedule["total_step"],
        steps_per_epoch=steps_per_epoch,
    )
    rss_before = psutil.Process().memory_info().rss
    diagnostics, training = _train(
        model,
        loader,
        weight_tensors,
        resolved,
        schedule,
        status_path,
    )
    training.update(
        {
            "epoch_diagnostics": diagnostics,
            "tokenization_runtime_seconds": tokenization_seconds,
            "class_weights": {
                task: dict(zip(orders[task], values, strict=True))
                for task, values in class_weight_values.items()
            },
            "selected_micro_batch": TRAIN_MICRO_BATCH,
            "gradient_accumulation": accumulation,
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "process_rss_before_training_bytes": rss_before,
            "process_rss_after_training_bytes": psutil.Process().memory_info().rss,
            "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "mps_current_allocated_bytes": torch.mps.current_allocated_memory(),
            "mps_driver_allocated_bytes": torch.mps.driver_allocated_memory(),
        }
    )

    adapter_dir = artifact_root / "adapter"
    tokenizer_dir = artifact_root / "tokenizer"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(tokenizer_dir)

    _status(status_path, current_phase="extracting_development_embeddings")
    development_embeddings, embedding_diagnostics = _embed_only(
        model,
        tokenizer,
        encoded,
        label_ids,
        config["inference_batch_size"],
        config["seed"],
    )
    if development_embeddings.shape != (EXPECTED_DEVELOPMENT_ROWS, DIMENSION):
        raise BenchmarkError("final DEVELOPMENT adapted-embedding shape drift")
    development_embedding_sha256 = hashlib.sha256(
        memoryview(np.ascontiguousarray(development_embeddings))
    ).hexdigest()

    del loader, weight_tensors, model, encoded, label_ids
    _clear_mps()
    gc.collect()

    task_artifacts: dict[str, Any] = {}
    for task in TASKS:
        _status(status_path, current_phase=f"fitting_{task.lower()}_fusion")
        lexical = config["lexical"][task]
        variant = next(
            item
            for item in a2_config["representations"][task]
            if item["id"] == lexical["representation_id"]
        )
        feature_config = _feature_config(a2_config, variant)
        lexical_features = build_sparse_features(
            lexical["representation"],
            texts,
            [texts[0]],
            feature_config,
        )
        fused = fuse_sparse_features(
            lexical_features,
            development_embeddings,
            np.zeros((1, DIMENSION), dtype=np.float32),
            config["semantic_weight"],
        )
        fit_config = {
            "models": {
                "c": lexical["c"],
                "logreg_solver": a2_config["models"]["logreg_solver"],
                "logreg_max_iter": a2_config["models"]["logreg_max_iter"],
                "linearsvc_max_iter": a2_config["models"]["linearsvc_max_iter"],
            }
        }
        classifier = _make_model(
            lexical["classifier"],
            lexical["class_weight"],
            fit_config,
            protocol.seed,
            task,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit_started = time.monotonic()
            classifier.fit(fused.training, labels[task])
            fit_seconds = time.monotonic() - fit_started
        vectorizer_path = artifact_root / f"{task.lower()}_vectorizer.pkl"
        classifier_path = artifact_root / f"{task.lower()}_classifier.pkl"
        _pickle(vectorizer_path, lexical_features.transformer)
        _pickle(classifier_path, classifier)
        task_artifacts[task] = {
            "representation_id": lexical["representation_id"],
            "representation": lexical["representation"],
            "classifier": lexical["classifier"],
            "class_weight": lexical["class_weight"],
            "c": lexical["c"],
            "lexical_feature_count": lexical_features.training.shape[1],
            "final_fused_feature_count": fused.training.shape[1],
            "fit_runtime_seconds": fit_seconds,
            "warnings": [f"{type(item.message).__name__}: {item.message}" for item in caught],
            "vectorizer_artifact": vectorizer_path.name,
            "classifier_artifact": classifier_path.name,
        }
        del lexical_features, fused, classifier
        gc.collect()

    del development_embeddings
    gc.collect()

    artifact_hashes = _artifact_hashes(artifact_root)
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: version(name)
            for name in (
                "numpy",
                "scikit-learn",
                "torch",
                "transformers",
                "peft",
                "accelerate",
            )
        },
        "device": "mps",
        "dtype": "float32",
    }
    result = {
        "status": "SUCCESS",
        "stage": "FINAL_TRAINING_DEVELOPMENT_ONLY",
        "selected_family": SELECTED_FAMILY,
        "shared_transformer_training_runs": 1,
        "development_rows": len(train_ids),
        "development_membership_sha256": EXPECTED_DEVELOPMENT_SHA256,
        "protocol_sha256": protocol.fingerprint,
        "model_selection_freeze_id": freeze["freeze_id"],
        "model_selection_freeze_sha256": freeze_sha256,
        "b4h_config_sha256": config_sha256,
        "a2_config_sha256": a2_config_sha256,
        "model_id": MODEL_ID,
        "resolved_revision": MODEL_REVISION,
        "semantic_dimension": DIMENSION,
        "semantic_weight": config["semantic_weight"],
        "development_embedding_sha256": development_embedding_sha256,
        "training": training,
        "embedding_extraction": embedding_diagnostics,
        "model_details": model_details,
        "optimizer_scheduler": {
            "adamw_lr": LEARNING_RATE,
            "warmup_fraction": config["warmup_fraction"],
            **schedule,
        },
        "task_artifacts": task_artifacts,
        "artifact_root": str(artifact_root),
        "artifact_hashes": artifact_hashes,
        "environment": environment,
        "runtime_seconds": time.monotonic() - started,
        "locked_test_accessed": False,
        "locked_test_tokenized": False,
        "locked_test_embedded": False,
        "locked_test_model_performance_accessed": False,
        "locked_test_used_for_tuning": False,
        **git_provenance,
    }
    _atomic_json(manifest_path, result)
    report = (
        "# Final B4-H Model Training V1\\n\\n"
        "One frozen B4-H adapted-RTA + lexical-fusion pipeline was fitted on the complete "
        "DEVELOPMENT membership only.\\n\\n"
        f"- DEVELOPMENT rows: `{len(train_ids)}`\\n"
        f"- DEVELOPMENT membership SHA-256: `{EXPECTED_DEVELOPMENT_SHA256}`\\n"
        f"- RTA revision: `{MODEL_REVISION}`\\n"
        f"- Git HEAD: `{git_provenance['git_head']}`\\n"
        f"- Runner tag: `{FINAL_TRAINING_TAG}`\\n"
        f"- Runtime: `{result['runtime_seconds'] / 3600:.3f}` hours\\n"
        "- Locked test accessed: **no**\\n"
        "- Predictive metrics calculated: **no**\\n"
    )
    (report_dir / "FINAL_TRAINING_REPORT.md").write_text(report, encoding="utf-8")
    _status(
        status_path,
        state="SUCCESS",
        current_phase="complete",
        completed_tasks=list(TASKS),
        final_manifest=str(manifest_path.resolve()),
        artifact_root=str(artifact_root.resolve()),
        locked_test_tokenized=False,
        locked_test_embedded=False,
        locked_test_model_performance_accessed=False,
        locked_test_used_for_tuning=False,
    )
    return result
