"""One-shot locked-test evaluation for the frozen final B4-H fusion pipeline."""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
import pickle
import platform
import subprocess
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from defect_classifier.classical_benchmark import BenchmarkError
from defect_classifier.classical_metrics import classification_metrics
from defect_classifier.locked_test import require_locked_test_unlock
from defect_classifier.preparation import _membership_fingerprint
from defect_classifier.protocol import FrozenProtocol
from defect_classifier.rta_adalora_multitask import HEAD_DIMENSIONS, TASKS
from defect_classifier.rta_fusion import ARCHITECTURE, DIMENSION, MODEL_ID, MODEL_REVISION

FINAL_MODEL_TAG = "final-model-v1"
FINAL_MODEL_COMMIT = "0f124923ac23234c6e9953f4434f9ad9b90584af"
FINAL_TRAINING_RUNNER_COMMIT = "ed2f1fc0534c29b3dc9654a14789339c87584506"
FINAL_EVALUATION_RUNNER_TAG = "final-evaluation-runner-v1"
SELECTED_FAMILY = "B4H_ADAPTED_RTA_LEXICAL_FUSION"
EXPECTED_PROTOCOL_SHA256 = "85faacae1e7f411d68803653388611c37c62730ea9217e9700a0ad6ac41b7cda"
EXPECTED_DEVELOPMENT_ROWS = 207575
EXPECTED_DEVELOPMENT_SHA256 = "4f62fdf4164594126c421955804b654cd5d5f8f7b46ada345ae0cffa71460d0f"
EXPECTED_LOCKED_ROWS = 50675
EXPECTED_LOCKED_SHA256 = "c18f2320c896bf2bdcb94d2447ce9a04d17e6923f03b169f29b8174b8a5cf681"
EXPECTED_SEED = 20260809
EXPECTED_INFERENCE_BATCH = 16
EXPECTED_ARTIFACTS = {
    "adapter/README.md",
    "adapter/adapter_config.json",
    "adapter/adapter_model.safetensors",
    "s2_classifier.pkl",
    "s2_vectorizer.pkl",
    "s3_classifier.pkl",
    "s3_vectorizer.pkl",
    "s6_classifier.pkl",
    "s6_vectorizer.pkl",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
}


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


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


def _verify_git_boundary(repo_root: Path) -> dict[str, Any]:
    if _git(repo_root, "status", "--porcelain"):
        raise BenchmarkError("final evaluation requires a clean Git working tree")
    head = _git(repo_root, "rev-parse", "HEAD")
    tags = [item for item in _git(repo_root, "tag", "--points-at", "HEAD").splitlines() if item]
    if FINAL_EVALUATION_RUNNER_TAG not in tags:
        raise BenchmarkError(
            f"final evaluation requires tag {FINAL_EVALUATION_RUNNER_TAG!r} on current HEAD"
        )
    final_model_commit = _git(repo_root, "rev-parse", f"{FINAL_MODEL_TAG}^{{}}")
    if final_model_commit != FINAL_MODEL_COMMIT:
        raise BenchmarkError("final-model-v1 tag drift")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", FINAL_MODEL_COMMIT, head],
        cwd=repo_root,
        check=False,
    )
    if ancestor.returncode != 0:
        raise BenchmarkError("final evaluation runner does not descend from final-model-v1")
    for tracked in (
        "reports/final_training_v1/final_training_manifest.json",
        "reports/final_training_v1/FINAL_TRAINING_REPORT.md",
    ):
        changed = subprocess.run(
            ["git", "diff", "--quiet", FINAL_MODEL_TAG, "--", tracked],
            cwd=repo_root,
            check=False,
        )
        if changed.returncode != 0:
            raise BenchmarkError(
                f"frozen final-model tracked artifact changed after tag: {tracked}"
            )
    return {
        "git_head": head,
        "git_tags_at_head": tags,
        "final_model_tag": FINAL_MODEL_TAG,
        "final_model_commit": final_model_commit,
        "git_worktree_clean": True,
    }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"cannot read frozen JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BenchmarkError(f"expected JSON object: {path}")
    return value


def _verify_freeze(
    freeze_file: Path,
    final_manifest: dict[str, Any],
    protocol: FrozenProtocol,
) -> dict[str, Any]:
    freeze = _load_json(freeze_file)
    expected = {
        "freeze_id": "model-selection-v1",
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "selection_metric": "macro_f1",
        "selection_scope": "DEVELOPMENT_ONLY",
        "development_model_selection_closed": True,
        "additional_model_selection_experiments_allowed": False,
        "selected_family": SELECTED_FAMILY,
        "direct_heads_selected": False,
        "development_rows": EXPECTED_DEVELOPMENT_ROWS,
        "development_membership_sha256": EXPECTED_DEVELOPMENT_SHA256,
        "locked_test_rows": EXPECTED_LOCKED_ROWS,
        "locked_test_membership_sha256": EXPECTED_LOCKED_SHA256,
        "locked_test_accessed_for_model_selection": False,
    }
    if any(freeze.get(key) != value for key, value in expected.items()):
        raise BenchmarkError("model-selection freeze drift before final evaluation")
    if freeze.get("transformer", {}).get("seed") != EXPECTED_SEED:
        raise BenchmarkError("frozen transformer seed drift")
    evaluation = freeze.get("final_evaluation", {})
    if evaluation != {
        "evaluate_selected_family_only": True,
        "evaluate_direct_heads_on_locked_test": False,
        "tasks": ["S6", "S3", "S2"],
        "primary_metric": "macro_f1",
        "unlock_environment_variable": protocol.unlock_environment_variable,
        "unlock_value": protocol.unlock_value,
        "post_test_tuning_allowed": False,
    }:
        raise BenchmarkError("frozen final-evaluation policy drift")
    if protocol.fingerprint != EXPECTED_PROTOCOL_SHA256:
        raise BenchmarkError("protocol fingerprint drift")
    required_s2 = {
        "high_impact_precision",
        "high_impact_recall",
        "high_impact_f1",
        "pr_auc",
        "roc_auc",
    }
    if set(protocol.document["evaluation"]["s2_score_metrics"]) != required_s2:
        raise BenchmarkError("frozen S2 score-metric policy drift")
    if (
        final_manifest.get("model_selection_freeze_id") != freeze["freeze_id"]
        or final_manifest.get("protocol_sha256") != protocol.fingerprint
        or final_manifest.get("selected_family") != SELECTED_FAMILY
    ):
        raise BenchmarkError("final model manifest does not match model-selection freeze")
    return freeze


def _verify_final_manifest(
    manifest_path: Path,
    artifact_root: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest = _load_json(manifest_path)
    expected = {
        "status": "SUCCESS",
        "stage": "FINAL_TRAINING_DEVELOPMENT_ONLY",
        "selected_family": SELECTED_FAMILY,
        "shared_transformer_training_runs": 1,
        "development_rows": EXPECTED_DEVELOPMENT_ROWS,
        "development_membership_sha256": EXPECTED_DEVELOPMENT_SHA256,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "model_id": MODEL_ID,
        "resolved_revision": MODEL_REVISION,
        "semantic_dimension": DIMENSION,
        "semantic_weight": 1.0,
        "git_head": FINAL_TRAINING_RUNNER_COMMIT,
        "locked_test_accessed": False,
        "locked_test_tokenized": False,
        "locked_test_embedded": False,
        "locked_test_model_performance_accessed": False,
        "locked_test_used_for_tuning": False,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise BenchmarkError("frozen final model manifest drift")
    if set(manifest.get("task_artifacts", {})) != set(TASKS):
        raise BenchmarkError("final model manifest task set drift")
    hashes = manifest.get("artifact_hashes")
    if not isinstance(hashes, dict) or set(hashes) != EXPECTED_ARTIFACTS:
        raise BenchmarkError("final model artifact inventory drift")
    actual_files = {
        str(path.relative_to(artifact_root)) for path in artifact_root.rglob("*") if path.is_file()
    }
    if actual_files != EXPECTED_ARTIFACTS:
        raise BenchmarkError("local final model artifact inventory differs from frozen manifest")
    verified: dict[str, dict[str, Any]] = {}
    for relative in sorted(EXPECTED_ARTIFACTS):
        path = artifact_root / relative
        expected_artifact = hashes[relative]
        actual_hash = _sha256_file(path)
        actual_size = path.stat().st_size
        if actual_hash != expected_artifact.get("sha256") or actual_size != expected_artifact.get(
            "size_bytes"
        ):
            raise BenchmarkError(f"final model artifact hash/size mismatch: {relative}")
        verified[relative] = {
            "sha256": actual_hash,
            "size_bytes": actual_size,
        }
    packages = manifest.get("environment", {}).get("packages", {})
    for package, expected_version in packages.items():
        actual_version = version(package)
        if actual_version != expected_version:
            raise BenchmarkError(
                f"package version drift for {package}: {actual_version} != {expected_version}"
            )
    return manifest, verified


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _build_base_multitask_model() -> Any:
    import copy

    import torch
    from transformers import AutoModelForSequenceClassification
    from transformers.models.roberta.modeling_roberta import RobertaClassificationHead

    base = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        num_labels=6,
        trust_remote_code=False,
        use_safetensors=False,
        local_files_only=True,
    )
    if type(base).__name__ != ARCHITECTURE:
        raise BenchmarkError("RTA architecture drift during final evaluation")

    class MultiTaskHead(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            heads = {}
            for task, outputs in HEAD_DIMENSIONS.items():
                task_config = copy.deepcopy(base.config)
                task_config.num_labels = outputs
                heads[task] = RobertaClassificationHead(task_config)
            self.heads = torch.nn.ModuleDict(heads)

        def forward(self, features: Any) -> dict[str, Any]:
            return {task: head(features) for task, head in self.heads.items()}

    base.classifier = MultiTaskHead()
    return base


def _load_pipeline(
    artifact_root: Path,
    protocol: FrozenProtocol,
    manifest: dict[str, Any],
) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer

    if not torch.backends.mps.is_available():
        raise BenchmarkError("final locked evaluation requires MPS")
    tokenizer = AutoTokenizer.from_pretrained(
        artifact_root / "tokenizer",
        local_files_only=True,
        trust_remote_code=False,
    )
    base = _build_base_multitask_model()
    model = PeftModel.from_pretrained(
        base,
        artifact_root / "adapter",
        is_trainable=False,
        local_files_only=True,
    )
    model.to("mps")
    model.eval()

    vectorizers = {}
    classifiers = {}
    freeze_types = {
        "S6": ("LinearSVC", 0.25, None),
        "S3": ("LogisticRegression", 2.0, "lbfgs"),
        "S2": ("LogisticRegression", 2.0, "liblinear"),
    }
    for task in TASKS:
        vectorizer = _load_pickle(artifact_root / f"{task.casefold()}_vectorizer.pkl")
        classifier = _load_pickle(artifact_root / f"{task.casefold()}_classifier.pkl")
        expected_type, expected_c, expected_solver = freeze_types[task]
        if type(classifier).__name__ != expected_type:
            raise BenchmarkError(f"{task} classifier type drift")
        if float(classifier.C) != expected_c or classifier.class_weight != "balanced":
            raise BenchmarkError(f"{task} classifier frozen parameter drift")
        if expected_solver is not None and classifier.solver != expected_solver:
            raise BenchmarkError(f"{task} logistic solver drift")
        if set(classifier.classes_) != set(protocol.targets[task.casefold()].order):
            raise BenchmarkError(f"{task} classifier class inventory drift")
        expected_features = manifest["task_artifacts"][task]["final_fused_feature_count"]
        if int(classifier.n_features_in_) != int(expected_features):
            raise BenchmarkError(f"{task} classifier feature-count drift")
        vectorizers[task] = vectorizer
        classifiers[task] = classifier
    return model, tokenizer, vectorizers, classifiers


def _embed_texts(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    *,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch

    if batch_size != EXPECTED_INFERENCE_BATCH:
        raise BenchmarkError("frozen inference batch-size drift")
    encoded = tokenizer(
        texts,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_attention_mask=True,
    )
    arrays: list[np.ndarray] = []
    documents = 0
    token_count = 0
    started = time.monotonic()
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            stop = min(start + batch_size, len(texts))
            inputs = {
                key: torch.tensor(values[start:stop], device="mps")
                for key, values in encoded.items()
                if key in {"input_ids", "attention_mask"}
            }
            output = model(**inputs, output_hidden_states=True)
            embeddings = torch.nn.functional.normalize(
                output.hidden_states[-1][:, 0, :].float(), p=2, dim=1
            )
            if embeddings.shape[1] != DIMENSION or not torch.isfinite(embeddings).all():
                raise BenchmarkError("invalid adapted RTA representation during final evaluation")
            arrays.append(embeddings.cpu().numpy().astype(np.float32, copy=False))
            documents += embeddings.shape[0]
            token_count += int(inputs["attention_mask"].sum().cpu())
    torch.mps.synchronize()
    elapsed = time.monotonic() - started
    matrix = np.concatenate(arrays, axis=0) if arrays else np.empty((0, DIMENSION), np.float32)
    return matrix, {
        "rows": documents,
        "runtime_seconds": elapsed,
        "documents_per_second": documents / elapsed if elapsed else None,
        "tokens_per_second": token_count / elapsed if elapsed else None,
        "embedding_dimension": DIMENSION,
        "inference_batch_size": batch_size,
        "direct_head_predictions_read": False,
        "direct_head_metrics_calculated": False,
    }


def _fused_matrix(vectorizer: Any, texts: list[str], embeddings: np.ndarray) -> Any:
    from scipy import sparse

    lexical = vectorizer.transform(texts).tocsr()
    if lexical.shape[0] != embeddings.shape[0]:
        raise BenchmarkError("locked lexical/semantic row alignment mismatch")
    return sparse.hstack(
        (lexical, sparse.csr_matrix(embeddings)),
        format="csr",
        dtype=np.float32,
    )


def _synthetic_preflight(
    model: Any,
    tokenizer: Any,
    vectorizers: dict[str, Any],
    classifiers: dict[str, Any],
    protocol: FrozenProtocol,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    texts = [
        "Synthetic summary A\n\nSynthetic defect description used only for pipeline validation.",
        "Synthetic summary B\n\nAnother synthetic defect description for frozen inference checks.",
    ]
    embeddings, diagnostics = _embed_texts(
        model,
        tokenizer,
        texts,
        batch_size=EXPECTED_INFERENCE_BATCH,
    )
    if embeddings.shape != (2, DIMENSION):
        raise BenchmarkError("synthetic embedding shape drift")
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise BenchmarkError("synthetic adapted embeddings are not L2-normalized")

    predictions: dict[str, list[str]] = {}
    for task in TASKS:
        fused = _fused_matrix(vectorizers[task], texts, embeddings)
        expected_features = manifest["task_artifacts"][task]["final_fused_feature_count"]
        if fused.shape != (2, expected_features):
            raise BenchmarkError(f"{task} synthetic fused feature shape drift")
        values = classifiers[task].predict(fused).tolist()
        if any(value not in protocol.targets[task.casefold()].order for value in values):
            raise BenchmarkError(f"{task} synthetic prediction outside frozen labels")
        predictions[task] = values
        if task == "S2":
            probability = classifiers[task].predict_proba(fused)
            index = list(classifiers[task].classes_).index("HIGH_IMPACT")
            high = probability[:, index]
            if not np.isfinite(high).all() or np.any((high < 0) | (high > 1)):
                raise BenchmarkError("invalid synthetic S2 HIGH_IMPACT probabilities")
        del fused
    return {
        "status": "PASS",
        "synthetic_rows": 2,
        "embedding_diagnostics": diagnostics,
        "prediction_labels": predictions,
        "locked_test_accessed": False,
    }


def _load_locked_texts(locked_dir: Path) -> tuple[list[str], list[str]]:
    import pyarrow.parquet as pq

    stable_ids: list[str] = []
    texts: list[str] = []
    columns = ["source_project", "issue_id", "text_combined"]
    for path in sorted(locked_dir.glob("*.parquet")):
        for record in pq.read_table(path, columns=columns).to_pylist():
            stable_ids.append(f"{record['source_project']}:{record['issue_id']}")
            texts.append(record["text_combined"])
    if len(stable_ids) != EXPECTED_LOCKED_ROWS:
        raise BenchmarkError("locked-test row-count drift")
    if len(stable_ids) != len(set(stable_ids)):
        raise BenchmarkError("duplicate stable IDs in locked test")
    if _membership_fingerprint(stable_ids) != EXPECTED_LOCKED_SHA256:
        raise BenchmarkError("locked-test membership fingerprint drift")
    return stable_ids, texts


def _load_locked_labels(
    locked_dir: Path,
    expected_ids: list[str],
) -> dict[str, list[str]]:
    import pyarrow.parquet as pq

    stable_ids: list[str] = []
    labels: dict[str, list[str]] = {task: [] for task in TASKS}
    columns = ["source_project", "issue_id", "target_s6", "target_s3", "target_s2"]
    for path in sorted(locked_dir.glob("*.parquet")):
        for record in pq.read_table(path, columns=columns).to_pylist():
            stable_ids.append(f"{record['source_project']}:{record['issue_id']}")
            labels["S6"].append(record["target_s6"])
            labels["S3"].append(record["target_s3"])
            labels["S2"].append(record["target_s2"])
    if stable_ids != expected_ids:
        raise BenchmarkError("locked label rows do not align exactly with frozen prediction rows")
    return labels


def _prediction_fingerprint(stable_ids: list[str], predictions: list[str]) -> str:
    if len(stable_ids) != len(predictions):
        raise BenchmarkError("prediction fingerprint row alignment mismatch")
    digest = hashlib.sha256()
    for stable_id, prediction in zip(stable_ids, predictions, strict=True):
        digest.update(stable_id.encode("utf-8"))
        digest.update(b"\t")
        digest.update(prediction.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _s2_score_metrics(
    y_true: list[str],
    high_probabilities: np.ndarray,
) -> dict[str, float]:
    from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

    binary = np.asarray([1 if value == "HIGH_IMPACT" else 0 for value in y_true], dtype=np.int8)
    if high_probabilities.shape != binary.shape:
        raise BenchmarkError("S2 score/label shape mismatch")
    precision, recall, _ = precision_recall_curve(binary, high_probabilities)
    return {
        "pr_auc": float(auc(recall, precision)),
        "roc_auc": float(roc_auc_score(binary, high_probabilities)),
    }


def _write_per_class_csv(path: Path, metrics: dict[str, Any]) -> None:
    rows = []
    for task in TASKS:
        for label, values in metrics[task]["per_class"].items():
            rows.append({"task": task, "class": label, **values})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["task", "class", "precision", "recall", "f1", "support"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Final Locked-Test Evaluation V1",
        "",
        "This is the single frozen locked-test evaluation of the selected "
        "B4-H adapted-RTA + lexical-fusion family.",
        "",
        f"- Locked-test rows: `{result['locked_test_rows']}`",
        f"- Locked-test membership SHA-256: `{result['locked_test_membership_sha256']}`",
        f"- Final model tag: `{result['final_model_tag']}`",
        f"- Evaluation runner tag: `{FINAL_EVALUATION_RUNNER_TAG}`",
        "- Direct B4-H heads evaluated: **no**",
        "- Post-test tuning allowed: **no**",
        "",
        "## Metrics",
        "",
    ]
    for task in TASKS:
        metric = result["metrics"][task]
        lines.extend(
            [
                f"### {task}",
                "",
                f"- Macro-F1: `{metric['macro_f1']:.10f}`",
                f"- Balanced accuracy: `{metric['balanced_accuracy']:.10f}`",
                f"- Accuracy: `{metric['accuracy']:.10f}`",
                f"- Weighted F1: `{metric['weighted_f1']:.10f}`",
            ]
        )
        if task == "S2":
            high = metric["per_class"]["HIGH_IMPACT"]
            lines.extend(
                [
                    f"- HIGH_IMPACT precision: `{high['precision']:.10f}`",
                    f"- HIGH_IMPACT recall: `{high['recall']:.10f}`",
                    f"- HIGH_IMPACT F1: `{high['f1']:.10f}`",
                    f"- PR-AUC: `{metric['pr_auc']:.10f}`",
                    f"- ROC-AUC: `{metric['roc_auc']:.10f}`",
                    f"- Legacy reproduction guard: `{metric['legacy_reproduction_guard']}`",
                ]
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def mark_final_evaluation_failed(report_dir: Path, error: BaseException) -> None:
    _atomic_json(
        report_dir / ".work" / "final_evaluation.status",
        {
            "state": "FAILED",
            "current_phase": "failed",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )


def run_final_evaluation(
    *,
    stage: str,
    locked_dir: Path,
    artifact_root: Path,
    final_manifest_path: Path,
    freeze_file: Path,
    report_dir: Path,
    protocol: FrozenProtocol,
) -> dict[str, Any]:
    """Run safe preflight, or the one-shot locked evaluation after explicit unlock."""
    if stage not in {"preflight", "locked"}:
        raise BenchmarkError("final evaluation stage must be preflight or locked")

    repo_root = Path(__file__).resolve().parents[2]
    git_provenance = _verify_git_boundary(repo_root)
    manifest, verified_artifacts = _verify_final_manifest(final_manifest_path, artifact_root)
    freeze = _verify_freeze(freeze_file, manifest, protocol)

    report_dir.mkdir(parents=True, exist_ok=True)
    work = report_dir / ".work"
    work.mkdir(parents=True, exist_ok=True)
    status_path = work / "final_evaluation.status"
    final_metrics_path = report_dir / "final_locked_metrics.json"
    receipt_path = work / "locked_access_receipt.json"

    if final_metrics_path.exists():
        raise BenchmarkError("final locked metrics already exist; refusing another evaluation")
    if receipt_path.exists() and stage == "locked":
        raise BenchmarkError("locked-test access receipt already exists; refusing a second access")

    _atomic_json(
        status_path,
        {
            "state": "RUNNING",
            "current_phase": "preflight",
            "stage": stage.upper(),
            "selected_family": SELECTED_FAMILY,
            "locked_test_accessed": False,
            "locked_test_model_performance_accessed": False,
            "direct_heads_selected": False,
            "direct_head_metrics_calculated": False,
            "post_test_tuning_allowed": False,
            **git_provenance,
        },
    )

    model, tokenizer, vectorizers, classifiers = _load_pipeline(
        artifact_root,
        protocol,
        manifest,
    )
    preflight = _synthetic_preflight(
        model,
        tokenizer,
        vectorizers,
        classifiers,
        protocol,
        manifest,
    )
    _atomic_json(
        work / "preflight.json",
        {
            **preflight,
            "artifact_count": len(verified_artifacts),
            "artifact_hashes_verified": True,
            **git_provenance,
        },
    )
    if stage == "preflight":
        _atomic_json(
            status_path,
            {
                "state": "SUCCESS",
                "current_phase": "preflight_complete",
                "stage": "PREFLIGHT_ONLY",
                "artifact_hashes_verified": True,
                "synthetic_pipeline": "PASS",
                "locked_test_accessed": False,
                "locked_test_model_performance_accessed": False,
                "direct_head_metrics_calculated": False,
                **git_provenance,
            },
        )
        return {
            "status": "SUCCESS",
            "stage": "PREFLIGHT_ONLY",
            "locked_test_accessed": False,
            "artifact_count": len(verified_artifacts),
            "synthetic_pipeline": "PASS",
        }

    require_locked_test_unlock(protocol)
    _atomic_json(
        receipt_path,
        {
            "state": "STARTED",
            "purpose": "ONE_SHOT_FINAL_EVALUATION",
            "locked_test_membership_sha256": EXPECTED_LOCKED_SHA256,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **git_provenance,
        },
    )
    _atomic_json(
        status_path,
        {
            "state": "RUNNING",
            "current_phase": "reading_locked_text",
            "stage": "FINAL_LOCKED_EVALUATION",
            "locked_test_accessed": True,
            "locked_test_model_performance_accessed": False,
            "direct_head_metrics_calculated": False,
            "post_test_tuning_allowed": False,
            **git_provenance,
        },
    )

    started = time.monotonic()
    stable_ids, texts = _load_locked_texts(locked_dir)
    _atomic_json(
        status_path,
        {
            "state": "RUNNING",
            "current_phase": "embedding_locked_text",
            "locked_test_accessed": True,
            "locked_test_rows": len(stable_ids),
            "locked_test_membership_sha256": EXPECTED_LOCKED_SHA256,
            "locked_test_model_performance_accessed": False,
            "direct_head_metrics_calculated": False,
            **git_provenance,
        },
    )
    embeddings, embedding_diagnostics = _embed_texts(
        model,
        tokenizer,
        texts,
        batch_size=EXPECTED_INFERENCE_BATCH,
    )
    if embeddings.shape != (EXPECTED_LOCKED_ROWS, DIMENSION):
        raise BenchmarkError("locked adapted-embedding shape drift")

    predictions: dict[str, list[str]] = {}
    prediction_hashes: dict[str, str] = {}
    s2_high_probabilities: np.ndarray | None = None
    for task in TASKS:
        _atomic_json(
            status_path,
            {
                "state": "RUNNING",
                "current_phase": f"predicting_{task.casefold()}_fusion",
                "locked_test_accessed": True,
                "locked_test_model_performance_accessed": False,
                "direct_head_metrics_calculated": False,
                **git_provenance,
            },
        )
        fused = _fused_matrix(vectorizers[task], texts, embeddings)
        expected_features = manifest["task_artifacts"][task]["final_fused_feature_count"]
        if fused.shape != (EXPECTED_LOCKED_ROWS, expected_features):
            raise BenchmarkError(f"{task} locked fused feature shape drift")
        values = classifiers[task].predict(fused).tolist()
        if any(value not in protocol.targets[task.casefold()].order for value in values):
            raise BenchmarkError(f"{task} locked prediction outside frozen labels")
        predictions[task] = values
        prediction_hashes[task] = _prediction_fingerprint(stable_ids, values)
        if task == "S2":
            probability = classifiers[task].predict_proba(fused)
            positive_index = list(classifiers[task].classes_).index("HIGH_IMPACT")
            s2_high_probabilities = probability[:, positive_index].astype(np.float64, copy=False)
            if (
                s2_high_probabilities.shape != (EXPECTED_LOCKED_ROWS,)
                or not np.isfinite(s2_high_probabilities).all()
            ):
                raise BenchmarkError("invalid S2 HIGH_IMPACT locked probabilities")
        del fused
        gc.collect()

    _atomic_json(
        status_path,
        {
            "state": "RUNNING",
            "current_phase": "scoring_locked_predictions",
            "locked_test_accessed": True,
            "locked_test_model_performance_accessed": True,
            "direct_head_metrics_calculated": False,
            **git_provenance,
        },
    )
    labels = _load_locked_labels(locked_dir, stable_ids)
    metrics = {}
    for task in TASKS:
        order = protocol.targets[task.casefold()].order
        metrics[task] = classification_metrics(labels[task], predictions[task], order)
    if s2_high_probabilities is None:
        raise BenchmarkError("missing S2 probability scores")
    metrics["S2"].update(_s2_score_metrics(labels["S2"], s2_high_probabilities))
    high_precision = metrics["S2"]["per_class"]["HIGH_IMPACT"]["precision"]
    threshold = float(protocol.document["legacy_reproduction"]["s2_minimum_high_impact_precision"])
    metrics["S2"]["legacy_reproduction_guard"] = "PASS" if high_precision >= threshold else "FAIL"
    metrics["S2"]["legacy_reproduction_guard_reproduction_only"] = True

    result = {
        "status": "SUCCESS",
        "stage": "FINAL_LOCKED_EVALUATION",
        "selected_family": SELECTED_FAMILY,
        "final_model_tag": FINAL_MODEL_TAG,
        "final_model_commit": FINAL_MODEL_COMMIT,
        "evaluation_runner_tag": FINAL_EVALUATION_RUNNER_TAG,
        "protocol_sha256": protocol.fingerprint,
        "model_selection_freeze_id": freeze["freeze_id"],
        "development_rows": EXPECTED_DEVELOPMENT_ROWS,
        "development_membership_sha256": EXPECTED_DEVELOPMENT_SHA256,
        "locked_test_rows": EXPECTED_LOCKED_ROWS,
        "locked_test_membership_sha256": EXPECTED_LOCKED_SHA256,
        "artifact_hashes_verified": True,
        "artifact_count": len(verified_artifacts),
        "prediction_sha256": prediction_hashes,
        "embedding_diagnostics": embedding_diagnostics,
        "metrics": metrics,
        "direct_heads_selected": False,
        "direct_head_predictions_read": False,
        "direct_head_metrics_calculated": False,
        "locked_test_accessed": True,
        "locked_test_tokenized": True,
        "locked_test_embedded": True,
        "locked_test_model_performance_accessed": True,
        "locked_test_used_for_tuning": False,
        "post_test_tuning_allowed": False,
        "runtime_seconds": time.monotonic() - started,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": manifest["environment"]["packages"],
            "device": "mps",
            "dtype": "float32",
        },
        **git_provenance,
    }
    _atomic_json(final_metrics_path, result)
    _atomic_json(
        report_dir / "confusion_matrices.json",
        {
            task: {
                "labels": list(protocol.targets[task.casefold()].order),
                "matrix": metrics[task]["confusion_matrix"],
            }
            for task in TASKS
        },
    )
    _write_per_class_csv(report_dir / "per_class_metrics.csv", metrics)
    _write_markdown(report_dir / "FINAL_LOCKED_EVALUATION.md", result)
    _atomic_json(
        receipt_path,
        {
            "state": "COMPLETED",
            "purpose": "ONE_SHOT_FINAL_EVALUATION",
            "locked_test_membership_sha256": EXPECTED_LOCKED_SHA256,
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "final_metrics": str(final_metrics_path.resolve()),
            **git_provenance,
        },
    )
    _atomic_json(
        status_path,
        {
            "state": "SUCCESS",
            "current_phase": "complete",
            "stage": "FINAL_LOCKED_EVALUATION",
            "locked_test_accessed": True,
            "locked_test_model_performance_accessed": True,
            "direct_head_metrics_calculated": False,
            "post_test_tuning_allowed": False,
            "final_metrics": str(final_metrics_path.resolve()),
            **git_provenance,
        },
    )
    return result
