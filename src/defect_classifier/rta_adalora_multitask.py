"""Development-TRAIN-only B4-H hierarchical multi-task AdaLoRA feasibility."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import platform
import resource
import statistics
import time
import tomllib
from importlib.metadata import version
from pathlib import Path
from typing import Any

from defect_classifier.classical_benchmark import BenchmarkError
from defect_classifier.preparation import _membership_fingerprint
from defect_classifier.protocol import FrozenProtocol
from defect_classifier.rta_adalora import (
    EFFECTIVE_BATCH_SIZE,
    EPOCHS,
    FOCAL_GAMMA,
    INIT_R,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_DROPOUT,
    PEFT_METHOD,
    PEFT_VERSION,
    TARGET_MODULES,
    _atomic_json,
    _clear_mps,
    _seed_everything,
    _status,
    _tokenize,
    adalora_schedule,
    balanced_mean_one_weights,
    deterministic_stratified_subset,
    weighted_focal_cross_entropy,
)
from defect_classifier.rta_fusion import (
    ARCHITECTURE,
    DIMENSION,
    MAX_LENGTH,
    MODEL_ID,
    MODEL_REVISION,
)

TASKS = ("S6", "S3", "S2")
HEAD_DIMENSIONS = {"S6": 6, "S3": 3, "S2": 2}
TASK_LOSS_WEIGHT = 1.0 / 3.0
B4_SUBSET_SHA256 = "d3b5a5f646e66d2c05fc26f4051b6abca6e15a0af80fce8baf604516f5aebad4"


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "rta_adalora_multitask_v1.toml"


def load_multitask_config(path: Path | None = None) -> tuple[dict[str, Any], str]:
    raw = (path or default_config_path()).read_bytes()
    config = tomllib.loads(raw.decode())
    validate_multitask_config(config)
    return config, hashlib.sha256(raw).hexdigest()


def validate_multitask_config(config: dict[str, Any]) -> None:
    fixed = {
        "model_id": MODEL_ID,
        "resolved_revision": MODEL_REVISION,
        "architecture": ARCHITECTURE,
        "peft_method": PEFT_METHOD,
        "peft_version": PEFT_VERSION,
        "max_length": MAX_LENGTH,
        "text_view": "SUMMARY_DESCRIPTION",
        "dtype": "float32",
        "epochs": EPOCHS,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": 0.01,
        "warmup_fraction": 0.06,
        "lr_schedule": "linear",
        "max_grad_norm": 1.0,
        "early_stopping": False,
        "final_epoch_selection": EPOCHS,
        "focal_gamma": FOCAL_GAMMA,
        "class_weighting": "balanced_train_only_mean_one",
        "task_loss_weights": [TASK_LOSS_WEIGHT] * 3,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "train_micro_batch_candidates": [8, 4, 2],
        "init_r": INIT_R,
        "target_r": 8,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "bias": "none",
        "target_modules": list(TARGET_MODULES),
        "modules_to_save": ["classifier"],
        "tinit_fraction": 0.10,
        "tfinal_fraction": 0.20,
        "allocation_target_updates": 100,
        "semantic_weight": 1.0,
        "future_direct_outputs": list(TASKS),
        "future_fusion_outputs": list(TASKS),
        "shared_embedding_extractions_per_fold": 1,
    }
    if any(config.get(key) != expected for key, expected in fixed.items()):
        raise BenchmarkError("Phase B4-H configuration drift")
    if config.get("heads") != HEAD_DIMENSIONS:
        raise BenchmarkError("Phase B4-H task-head drift")
    if config.get("future_matrix") != {"folds": [1, 2, 3], "shared_adapter_count": 3}:
        raise BenchmarkError("Phase B4-H shared-adapter matrix drift")
    if config.get("feasibility") != {
        "stage": "ENGINEERING_ONLY",
        "fold": 1,
        "training_rows": 10000,
        "inference_rows": 1000,
        "reuse_b4_subset": True,
        "b4_subset_sha256": B4_SUBSET_SHA256,
    }:
        raise BenchmarkError("Phase B4-H feasibility boundary drift")
    expected_lexical = {
        "S6": ("S6-R3", "CHAR", "LINEARSVC", "BALANCED", 0.25),
        "S3": ("S3-R5", "WORD", "LOGREG", "BALANCED", 2.0),
        "S2": ("S2-R3", "WORD_CHAR", "LOGREG", "BALANCED", 2.0),
    }
    for task, expected in expected_lexical.items():
        actual = config.get("lexical", {}).get(task, {})
        values = tuple(
            actual.get(key)
            for key in ("representation_id", "representation", "classifier", "class_weight", "c")
        )
        if values != expected:
            raise BenchmarkError(f"Phase B4-H lexical configuration drift: {task}")


def hierarchical_targets(s6_label: str, protocol: FrozenProtocol) -> tuple[str, str, str]:
    if s6_label not in protocol.accepted_labels:
        raise BenchmarkError(f"unknown frozen S6 label: {s6_label}")
    return (
        s6_label,
        protocol.targets["s3"].mapping[s6_label],
        protocol.targets["s2"].mapping[s6_label],
    )


def equal_joint_loss(loss_s6: Any, loss_s3: Any, loss_s2: Any) -> Any:
    return (loss_s6 + loss_s3 + loss_s2) / 3.0


def future_shared_adapter_folds() -> tuple[int, int, int]:
    return (1, 2, 3)


def mark_feasibility_failed(report_dir: Path, error: BaseException) -> None:
    _status(
        report_dir / ".work" / "feasibility.status",
        state="FAILED",
        current_phase="failed",
        error_type=type(error).__name__,
        error_message=str(error),
    )


def _build_multitask_model(schedule: dict[str, int], device: str) -> tuple[Any, dict[str, Any]]:
    import torch
    from peft import AdaLoraConfig, TaskType, get_peft_model
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
        raise BenchmarkError("RTA architecture drift")
    target_names = [
        name
        for name, module in base.named_modules()
        if name.endswith((".query", ".value")) and type(module).__name__ == "Linear"
    ]
    expected_names = [
        f"roberta.encoder.layer.{layer}.attention.self.{projection}"
        for layer in range(12)
        for projection in TARGET_MODULES
    ]
    if sorted(target_names) != sorted(expected_names):
        raise BenchmarkError("RTA query/value targets are not the expected attention projections")

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
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    peft_config = AdaLoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        init_r=INIT_R,
        target_r=8,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=list(TARGET_MODULES),
        modules_to_save=["classifier"],
        bias="none",
        total_step=schedule["total_step"],
        tinit=schedule["tinit"],
        tfinal=schedule["tfinal"],
        deltaT=schedule["deltaT"],
    )
    model = get_peft_model(base, peft_config)
    model.to(device=device)
    trainable = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    forbidden = [
        name for name, _ in trainable if "lora_" not in name and "modules_to_save" not in name
    ]
    for task in TASKS:
        if not any("modules_to_save" in name and f"heads.{task}" in name for name, _ in trainable):
            raise BenchmarkError(f"Phase B4-H {task} head is not trainable/checkpointed")
    if forbidden:
        raise BenchmarkError("Phase B4-H trainable-parameter boundary drift")
    adapter_parameters = sum(value.numel() for name, value in trainable if "lora_" in name)
    head_parameters = sum(value.numel() for name, value in trainable if "modules_to_save" in name)
    dropout = (
        base.config.classifier_dropout
        if base.config.classifier_dropout is not None
        else base.config.hidden_dropout_prob
    )
    return model, {
        "target_module_names": target_names,
        "task_head_output_dimensions": HEAD_DIMENSIONS,
        "classification_head_architecture": (
            "first_token->dropout->dense(768,768)->tanh->dropout->output"
        ),
        "classification_head_dropout": dropout,
        "adalora_parameter_count": adapter_parameters,
        "three_head_parameter_count": head_parameters,
        "trainable_parameter_count": sum(value.numel() for _, value in trainable),
        "total_parameter_count": sum(value.numel() for value in model.parameters()),
        "trainable_parameter_names": [name for name, _ in trainable],
    }


class _MultiDataset:
    def __init__(self, encoded: dict[str, Any], label_ids: dict[str, list[int]]) -> None:
        self.encoded = encoded
        self.label_ids = label_ids

    def __len__(self) -> int:
        return len(self.label_ids["S6"])

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            **{key: value[index] for key, value in self.encoded.items()},
            **{
                f"labels_{task.casefold()}": values[index]
                for task, values in self.label_ids.items()
            },
        }


def _loader(
    tokenizer: Any,
    encoded: dict[str, Any],
    label_ids: dict[str, list[int]],
    batch_size: int,
    seed: int,
    shuffle: bool,
) -> Any:
    import torch
    from torch.utils.data import DataLoader
    from transformers import DataCollatorWithPadding

    return DataLoader(
        _MultiDataset(encoded, label_ids),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=DataCollatorWithPadding(tokenizer, return_tensors="pt"),
        num_workers=0,
    )


def _move(batch: dict[str, Any], device: str) -> tuple[dict[str, Any], dict[str, Any]]:
    labels = {task: batch.pop(f"labels_{task.casefold()}").to(device) for task in TASKS}
    return {key: value.to(device) for key, value in batch.items()}, labels


def _losses(
    logits: dict[str, Any], labels: dict[str, Any], weights: dict[str, Any]
) -> dict[str, Any]:
    output = {
        task: weighted_focal_cross_entropy(logits[task], labels[task], weights[task], FOCAL_GAMMA)
        for task in TASKS
    }
    output["TOTAL"] = equal_joint_loss(output["S6"], output["S3"], output["S2"])
    return output


def _select_micro_batch(
    tokenizer: Any,
    encoded: dict[str, Any],
    label_ids: dict[str, list[int]],
    class_weights: dict[str, list[float]],
    config: dict[str, Any],
) -> tuple[int, list[dict[str, Any]]]:
    import torch

    longest = sorted(
        range(len(label_ids["S6"])),
        key=lambda index: len(encoded["input_ids"][index]),
        reverse=True,
    )
    oom_events = []
    for micro_batch in config["train_micro_batch_candidates"]:
        accumulation = EFFECTIVE_BATCH_SIZE // micro_batch
        steps_per_epoch = math.ceil(math.ceil(len(longest) / micro_batch) / accumulation)
        schedule = adalora_schedule(steps_per_epoch * EPOCHS)
        try:
            model, _ = _build_multitask_model(schedule, "mps")
            indices = longest[:micro_batch]
            probe = {key: [values[index] for index in indices] for key, values in encoded.items()}
            probe_labels = {
                task: [values[index] for index in indices] for task, values in label_ids.items()
            }
            batch = next(
                iter(_loader(tokenizer, probe, probe_labels, micro_batch, config["seed"], False))
            )
            inputs, labels = _move(batch, "mps")
            tensors = {
                task: torch.tensor(values, device="mps") for task, values in class_weights.items()
            }
            losses = _losses(model(**inputs).logits, labels, tensors)
            losses["TOTAL"].backward()
            if not all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                raise BenchmarkError("non-finite B4-H preflight gradient")
            del batch, inputs, labels, losses, tensors, model
            _clear_mps()
            return micro_batch, oom_events
        except RuntimeError as exc:
            if "memory" not in str(exc).lower():
                raise
            oom_events.append({"micro_batch": micro_batch, "error": str(exc)})
            _clear_mps()
    raise BenchmarkError("B4-H does not fit MPS at micro-batch 2")


def _train(
    model: Any,
    loader: Any,
    weights: dict[str, Any],
    config: dict[str, Any],
    schedule: dict[str, int],
    status_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch
    from transformers import get_linear_schedule_with_warmup

    optimizer = torch.optim.AdamW(
        (value for value in model.parameters() if value.requires_grad),
        lr=LEARNING_RATE,
        weight_decay=config["weight_decay"],
    )
    warmup = round(config["warmup_fraction"] * schedule["total_step"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, schedule["total_step"])
    accumulation = config["resolved_gradient_accumulation"]
    diagnostics, optimizer_step, nan_inf_events = [], 0, 0
    started = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        collected = {task: [] for task in (*TASKS, "TOTAL")}
        epoch_started = time.monotonic()
        for batch_index, batch in enumerate(loader, 1):
            inputs, labels = _move(batch, "mps")
            losses = _losses(model(**inputs).logits, labels, weights)
            if not all(torch.isfinite(value) for value in losses.values()):
                nan_inf_events += 1
                raise BenchmarkError("non-finite B4-H training loss")
            (losses["TOTAL"] / accumulation).backward()
            for task, value in losses.items():
                collected[task].append(float(value.detach().cpu()))
            if batch_index % accumulation == 0 or batch_index == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer_step += 1
                model.base_model.update_and_allocate(optimizer_step)
                optimizer.zero_grad(set_to_none=True)
        elapsed = time.monotonic() - epoch_started
        row = {
            "epoch": epoch,
            "training_seconds": elapsed,
            "optimizer_steps_completed": optimizer_step,
            **{
                f"{task.casefold()}_training_loss": statistics.fmean(values)
                for task, values in collected.items()
            },
        }
        diagnostics.append(row)
        _status(
            status_path,
            state="RUNNING",
            current_phase="training",
            epoch_completed=epoch,
            total_epochs=EPOCHS,
        )
        print(
            f"[B4-H] epoch={epoch}/{EPOCHS} s6_loss={row['s6_training_loss']:.8f} "
            f"s3_loss={row['s3_training_loss']:.8f} s2_loss={row['s2_training_loss']:.8f} "
            f"joint_loss={row['total_training_loss']:.8f} seconds={elapsed:.3f}",
            flush=True,
        )
    elapsed = time.monotonic() - started
    if optimizer_step != schedule["total_step"]:
        raise BenchmarkError("B4-H optimizer-step schedule mismatch")
    return diagnostics, {
        "training_runtime_seconds": elapsed,
        "optimizer_steps": optimizer_step,
        "optimizer_steps_per_second": optimizer_step / elapsed,
        "warmup_steps": warmup,
        "nan_inf_events": nan_inf_events,
    }


def _inference(
    model: Any,
    tokenizer: Any,
    encoded: dict[str, Any],
    label_ids: dict[str, list[int]],
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    import torch

    loader = _loader(tokenizer, encoded, label_ids, batch_size, seed, False)
    model.eval()
    documents = token_count = 0
    started = time.monotonic()
    with torch.inference_mode():
        for batch in loader:
            inputs, _ = _move(batch, "mps")
            output = model(**inputs, output_hidden_states=True)
            if {task: output.logits[task].shape[1] for task in TASKS} != HEAD_DIMENSIONS:
                raise BenchmarkError("B4-H inference head-output drift")
            embeddings = torch.nn.functional.normalize(
                output.hidden_states[-1][:, 0, :].float(), p=2, dim=1
            )
            if embeddings.shape[1] != DIMENSION or not torch.isfinite(embeddings).all():
                raise BenchmarkError("invalid B4-H adapted representation")
            documents += embeddings.shape[0]
            token_count += int(inputs["attention_mask"].sum().cpu())
    torch.mps.synchronize()
    elapsed = time.monotonic() - started
    return {
        "inference_rows": documents,
        "inference_batch_size": batch_size,
        "runtime_seconds": elapsed,
        "documents_per_second": documents / elapsed,
        "tokens_per_second": token_count / elapsed,
        "embedding_dimension": DIMENSION,
        "outputs_per_shared_forward": ["embedding_768", "S6_logits", "S3_logits", "S2_logits"],
        "predictive_metrics_calculated": False,
        "mps_fallback_events": 0,
    }


def _projection(
    training: dict[str, Any],
    inference: dict[str, Any],
    folds: dict[int, tuple[list[str], list[str]]],
) -> dict[str, Any]:
    exposure_rate = training["example_exposures"] / training["training_runtime_seconds"]
    inference_rate = inference["documents_per_second"]
    lexical_by_fold = {fold: 0.0 for fold in (1, 2, 3)}
    with Path("reports/lexical_semantic_fusion_v1/fit_results.csv").open() as handle:
        for row in csv.DictReader(handle):
            if row["stage"] == "COMPETITIVE":
                lexical_by_fold[int(row["fold"])] += float(row["fit_runtime_seconds"])
                lexical_by_fold[int(row["fold"])] += float(row["prediction_runtime_seconds"])
    fold_rows = []
    for fold, (train_ids, validation_ids) in folds.items():
        training_seconds = len(train_ids) * EPOCHS / exposure_rate
        train_embedding_seconds = len(train_ids) / inference_rate
        combined_validation_seconds = len(validation_ids) / inference_rate
        checkpoint_reporting = 100.0
        total = (
            training_seconds
            + train_embedding_seconds
            + combined_validation_seconds
            + lexical_by_fold[fold]
            + checkpoint_reporting
        )
        fold_rows.append(
            {
                "fold": fold,
                "training_rows": len(train_ids),
                "validation_rows": len(validation_ids),
                "shared_adalora_training_seconds": training_seconds,
                "train_embedding_extraction_seconds": train_embedding_seconds,
                "combined_validation_forward_seconds": combined_validation_seconds,
                "lexical_fusion_fit_prediction_seconds": lexical_by_fold[fold],
                "checkpoint_reporting_seconds": checkpoint_reporting,
                "central_fold_wall_seconds": total,
            }
        )
    central = sum(row["central_fold_wall_seconds"] for row in fold_rows)
    return {
        "transformer_training_runs": 3,
        "shared_embedding_extractions_per_fold": 1,
        "folds": fold_rows,
        "total_transformer_training_seconds": sum(
            row["shared_adalora_training_seconds"] for row in fold_rows
        ),
        "total_adapted_embedding_inference_seconds": sum(
            row["train_embedding_extraction_seconds"] + row["combined_validation_forward_seconds"]
            for row in fold_rows
        ),
        "total_lexical_fusion_seconds": sum(
            row["lexical_fusion_fit_prediction_seconds"] for row in fold_rows
        ),
        "total_checkpoint_reporting_seconds": 300.0,
        "central_total_seconds": central,
        "conservative_total_seconds": central * 1.35,
        "safety_margin": "35% over measured-throughput central projection",
    }


def run_multitask_feasibility(
    *,
    development_dir: Path,
    manifest_dir: Path,
    protocol_report_dir: Path,
    report_dir: Path,
    protocol: FrozenProtocol,
    config_path: Path | None = None,
    stage: str = "feasibility",
) -> dict[str, Any]:
    """Run one B4-H engineering adapter; competitive/full execution is disabled."""
    if stage != "feasibility":
        raise BenchmarkError("Phase B4-H full competitive execution is not authorized")
    import psutil
    import torch
    from transformers import AutoTokenizer

    from defect_classifier.classical_benchmark import _load_development_rows, _load_fold

    config, fingerprint = load_multitask_config(config_path)
    if version("peft") != PEFT_VERSION:
        raise BenchmarkError("installed PEFT version differs from frozen B4-H version")
    b3 = json.loads((report_dir.parent / "rta_fusion_v1" / "model_provenance.json").read_text())
    b4 = json.loads((report_dir.parent / "rta_adalora_v1" / "model_provenance.json").read_text())
    b4_result = json.loads(
        (report_dir.parent / "rta_adalora_v1" / ".work" / "feasibility_result.json").read_text()
    )
    if (
        any(
            artifact.get("model_id") != MODEL_ID
            or artifact.get("resolved_revision") != MODEL_REVISION
            for artifact in (b3, b4)
        )
        or b4.get("peft_version") != PEFT_VERSION
    ):
        raise BenchmarkError("B3/B4 RTA provenance mismatch")
    if (
        b4_result.get("status") != "SUCCESS"
        or b4_result.get("competitive_metrics") is not None
        or b4_result.get("subset_membership_sha256") != B4_SUBSET_SHA256
    ):
        raise BenchmarkError("B4 feasibility handoff mismatch")
    frozen = json.loads((protocol_report_dir / "fingerprints.json").read_text())
    if frozen["protocol_sha256"] != protocol.fingerprint:
        raise BenchmarkError("protocol fingerprint drift")
    if not torch.backends.mps.is_available():
        raise BenchmarkError("Phase B4-H feasibility requires MPS")
    report_dir.mkdir(parents=True, exist_ok=True)
    work = report_dir / ".work"
    work.mkdir(parents=True, exist_ok=True)
    status_path = work / "feasibility.status"
    result_path = work / "feasibility_result.json"
    _status(
        status_path,
        state="RUNNING",
        current_phase="loading_development_train_membership",
        epoch_completed=0,
        total_epochs=EPOCHS,
        model_id=MODEL_ID,
        resolved_revision=MODEL_REVISION,
        peft_method=PEFT_METHOD,
        task_heads=list(TASKS),
        task_loss_weights=[TASK_LOSS_WEIGHT] * 3,
        config_sha256=fingerprint,
        source_membership="DEVELOPMENT_TRAIN_ONLY",
        competitive_models_fitted=0,
        result_artifact_path=str(result_path.resolve()),
    )
    (work / "feasibility.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
    print(
        f"[B4-H] starting model={MODEL_ID} revision={MODEL_REVISION} peft=ADALORA "
        f"peft_version={PEFT_VERSION} config_sha256={fingerprint} heads=S6:6,S3:3,S2:2 "
        "joint_objective=(L_S6+L_S3+L_S2)/3 source=DEVELOPMENT_TRAIN_ONLY "
        "locked_test=NOT_ACCESSED",
        flush=True,
    )
    rows = _load_development_rows(development_dir)
    if _membership_fingerprint(rows) != frozen["development_membership_sha256"]:
        raise BenchmarkError("development membership fingerprint drift")
    train_ids, validation_ids = _load_fold(
        rows, manifest_dir, protocol_report_dir / "fingerprints.json", protocol, 1
    )
    del validation_ids
    s2_labels = {stable_id: rows[stable_id].targets["S2"] for stable_id in train_ids}
    subset_ids = deterministic_stratified_subset(train_ids, s2_labels, 10000, config["seed"])
    subset_hash = _membership_fingerprint(subset_ids)
    if subset_hash != B4_SUBSET_SHA256:
        raise BenchmarkError("B4-H did not reproduce the controlled B4 subset")
    labels: dict[str, list[str]] = {task: [] for task in TASKS}
    for stable_id in subset_ids:
        expected = hierarchical_targets(rows[stable_id].targets["S6"], protocol)
        actual = tuple(rows[stable_id].targets[task] for task in TASKS)
        if actual != expected:
            raise BenchmarkError("frozen hierarchical target mapping drift")
        for task, value in zip(TASKS, actual, strict=True):
            labels[task].append(value)
    orders = {task: protocol.targets[task.casefold()].order for task in TASKS}
    label_ids = {
        task: [orders[task].index(value) for value in values] for task, values in labels.items()
    }
    class_weight_values = {
        task: balanced_mean_one_weights(labels[task], orders[task]) for task in TASKS
    }
    _seed_everything(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False, local_files_only=True
    )
    token_started = time.monotonic()
    encoded = _tokenize(tokenizer, [rows[stable_id].text for stable_id in subset_ids])
    tokenization_seconds = time.monotonic() - token_started
    _status(
        status_path, current_phase="mps_micro_batch_preflight", subset_membership_sha256=subset_hash
    )
    micro_batch, oom_events = _select_micro_batch(
        tokenizer, encoded, label_ids, class_weight_values, config
    )
    print(
        f"[B4-H] AdaLoRA active targets=query,value target_count=24 heads=S6:6,S3:3,S2:2 "
        f"joint_objective=equal_one_third micro_batch={micro_batch}",
        flush=True,
    )
    accumulation = EFFECTIVE_BATCH_SIZE // micro_batch
    steps_per_epoch = math.ceil(math.ceil(len(subset_ids) / micro_batch) / accumulation)
    schedule = adalora_schedule(steps_per_epoch * EPOCHS)
    resolved = {
        **config,
        "resolved_train_micro_batch": micro_batch,
        "resolved_gradient_accumulation": accumulation,
        "resolved_effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "steps_per_epoch": steps_per_epoch,
        "adalora_schedule": schedule,
    }
    _atomic_json(report_dir / "training_configuration.json", resolved)
    _status(
        status_path,
        current_phase="training",
        selected_micro_batch=micro_batch,
        gradient_accumulation=accumulation,
        effective_batch_size=EFFECTIVE_BATCH_SIZE,
    )
    print(
        f"[B4-H] subset_rows=10000 subset_sha256={subset_hash} source=DEVELOPMENT_TRAIN "
        f"micro_batch={micro_batch} accumulation={accumulation} effective_batch=32 "
        f"total_step={schedule['total_step']} tinit={schedule['tinit']} "
        f"tfinal={schedule['tfinal']} deltaT={schedule['deltaT']} locked_test=NOT_ACCESSED",
        flush=True,
    )
    model, model_details = _build_multitask_model(schedule, "mps")
    weight_tensors = {
        task: torch.tensor(values, device="mps") for task, values in class_weight_values.items()
    }
    loader = _loader(tokenizer, encoded, label_ids, micro_batch, config["seed"], True)
    rss_before = psutil.Process().memory_info().rss
    diagnostics, training = _train(model, loader, weight_tensors, resolved, schedule, status_path)
    exposure_count = len(subset_ids) * EPOCHS
    training.update(
        {
            "example_exposures": exposure_count,
            "example_exposures_per_second": exposure_count / training["training_runtime_seconds"],
            "tokens_per_second": sum(len(values) for values in encoded["input_ids"])
            * EPOCHS
            / training["training_runtime_seconds"],
            "tokenization_runtime_seconds": tokenization_seconds,
            "epoch_diagnostics": diagnostics,
            "class_weights": {
                task: dict(zip(orders[task], values, strict=True))
                for task, values in class_weight_values.items()
            },
            "task_loss_weights": {task: TASK_LOSS_WEIGHT for task in TASKS},
            "selected_micro_batch": micro_batch,
            "gradient_accumulation": accumulation,
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "oom_events": oom_events,
            "mps_fallback_events": 0,
            "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "process_rss_before_training_bytes": rss_before,
            "process_rss_after_training_bytes": psutil.Process().memory_info().rss,
            "mps_current_allocated_bytes": torch.mps.current_allocated_memory(),
            "mps_driver_allocated_bytes": torch.mps.driver_allocated_memory(),
        }
    )
    adapter_dir = work / "engineering_adapter"
    model.save_pretrained(adapter_dir)
    checkpoint_size = sum(path.stat().st_size for path in adapter_dir.rglob("*") if path.is_file())
    training["adapter_and_three_head_checkpoint_size_bytes"] = checkpoint_size
    inference_count = min(config["feasibility"]["inference_rows"], len(subset_ids))
    inference_encoded = {key: values[:inference_count] for key, values in encoded.items()}
    inference_labels = {task: values[:inference_count] for task, values in label_ids.items()}
    _status(status_path, current_phase="train_only_shared_inference_throughput")
    inference = _inference(
        model,
        tokenizer,
        inference_encoded,
        inference_labels,
        config["inference_batch_size"],
        config["seed"],
    )
    folds = {
        fold: _load_fold(
            rows, manifest_dir, protocol_report_dir / "fingerprints.json", protocol, fold
        )
        for fold in (1, 2, 3)
    }
    projection = _projection(training, inference, folds)
    projection.update(
        {
            "checkpoint_size_per_fold_bytes": checkpoint_size,
            "three_fold_checkpoint_storage_bytes": checkpoint_size * 3,
            "temporary_embedding_storage_by_fold_bytes": {
                str(fold): (len(train_ids) + len(validation_ids)) * DIMENSION * 4
                for fold, (train_ids, validation_ids) in folds.items()
            },
        }
    )
    model_details["trainable_percentage"] = (
        model_details["trainable_parameter_count"] / model_details["total_parameter_count"] * 100
    )
    provenance = {
        "model_id": MODEL_ID,
        "resolved_revision": MODEL_REVISION,
        "architecture": ARCHITECTURE,
        "b3_b4_provenance_verified": True,
        "peft_method": PEFT_METHOD,
        "peft_version": PEFT_VERSION,
        "target_modules": list(TARGET_MODULES),
        "modules_to_save": ["classifier"],
        "base_model_otherwise_frozen": True,
        "shared_encoder_single_forward": True,
        **model_details,
        "protocol_sha256": protocol.fingerprint,
        "development_membership_sha256": frozen["development_membership_sha256"],
        "config_sha256": fingerprint,
    }
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: version(name) for name in ("torch", "transformers", "peft", "accelerate")
        },
        "device": "mps",
        "dtype": "float32",
        "locked_test_tokenized": False,
        "locked_test_embedded": False,
        "locked_test_model_performance_accessed": False,
        "locked_test_used_for_tuning": False,
        "competitive_models_fitted": 0,
        "competitive_metrics_calculated": False,
    }
    result = {
        "status": "SUCCESS",
        "stage": "ENGINEERING_ONLY",
        "source_membership": "DEVELOPMENT_TRAIN_ONLY",
        "subset_rows": len(subset_ids),
        "subset_membership_sha256": subset_hash,
        "training": training,
        "inference": inference,
        "runtime_projection": projection,
        "competitive_metrics": None,
        "locked_test_accessed": False,
    }
    _atomic_json(report_dir / "model_provenance.json", provenance)
    _atomic_json(
        report_dir / "throughput_estimate.json", {"training": training, "inference": inference}
    )
    _atomic_json(report_dir / "runtime_projection.json", projection)
    _atomic_json(report_dir / "environment.json", environment)
    _atomic_json(result_path, result)
    report = (
        "# Phase B4-H Hierarchical Multi-Task RTA AdaLoRA Feasibility\n\n"
        "Engineering-only DEVELOPMENT-TRAIN runtime study; no validation performance was "
        "calculated. Equal one-third loss weighting is frozen to avoid a performance-tuned "
        "task-priority hyperparameter.\n\n"
        f"- Configuration: `{fingerprint}`\n"
        f"- Training runtime: {training['training_runtime_seconds']:.3f} seconds\n"
        "- Central three-adapter projection: "
        f"{projection['central_total_seconds'] / 3600:.3f} hours\n"
        f"- Conservative projection: {projection['conservative_total_seconds'] / 3600:.3f} hours\n"
        "- Locked test accessed: no\n"
    )
    (report_dir / "RTA_ADALORA_MULTITASK_FEASIBILITY.md").write_text(report, encoding="utf-8")
    _status(
        status_path,
        state="SUCCESS",
        current_phase="complete",
        epoch_completed=EPOCHS,
        result_artifact_path=str(result_path.resolve()),
    )
    print(f"[B4-H] SUCCESS result={result_path.resolve()}", flush=True)
    return result
