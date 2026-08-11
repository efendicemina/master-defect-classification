"""Development-TRAIN-only Phase B4 RTA AdaLoRA engineering feasibility."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import resource
import statistics
import time
import tomllib
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from typing import Any

from defect_classifier.classical_benchmark import BenchmarkError
from defect_classifier.preparation import _membership_fingerprint
from defect_classifier.protocol import FrozenProtocol
from defect_classifier.rta_fusion import (
    ARCHITECTURE,
    DIMENSION,
    MAX_LENGTH,
    MODEL_ID,
    MODEL_REVISION,
)

PEFT_METHOD = "ADALORA"
PEFT_VERSION = "0.19.1"
INIT_R = 12
TARGET_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ("query", "value")
MODULES_TO_SAVE = ("classifier",)
EPOCHS = 3
LEARNING_RATE = 1e-4
FOCAL_GAMMA = 2.0
EFFECTIVE_BATCH_SIZE = 32
TASKS = ("S6", "S3", "S2")
FOLDS = (1, 2, 3)


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "rta_adalora_v1.toml"


def load_adalora_config(path: Path | None = None) -> tuple[dict[str, Any], str]:
    raw = (path or default_config_path()).read_bytes()
    config = tomllib.loads(raw.decode())
    validate_adalora_config(config)
    return config, hashlib.sha256(raw).hexdigest()


def validate_adalora_config(config: dict[str, Any]) -> None:
    fixed = {
        "model_id": MODEL_ID,
        "resolved_revision": MODEL_REVISION,
        "architecture": ARCHITECTURE,
        "peft_method": PEFT_METHOD,
        "peft_version": PEFT_VERSION,
        "max_length": MAX_LENGTH,
        "text_view": "SUMMARY_DESCRIPTION",
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
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "train_micro_batch_candidates": [8, 4, 2],
        "init_r": INIT_R,
        "target_r": TARGET_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "bias": "none",
        "target_modules": list(TARGET_MODULES),
        "modules_to_save": list(MODULES_TO_SAVE),
        "tinit_fraction": 0.10,
        "tfinal_fraction": 0.20,
        "allocation_target_updates": 100,
        "semantic_weight": 1.0,
        "future_outputs": ["DIRECT_CLASSIFIER", "LEXICAL_FUSION"],
    }
    if any(config.get(key) != expected for key, expected in fixed.items()):
        raise BenchmarkError("Phase B4 AdaLoRA configuration drift")
    feasibility = config.get("feasibility", {})
    if feasibility != {
        "stage": "ENGINEERING_ONLY",
        "task": "S2",
        "fold": 1,
        "training_rows": 10000,
        "inference_rows": 1000,
    }:
        raise BenchmarkError("Phase B4 feasibility boundary drift")
    matrix = config.get("future_matrix", {})
    if matrix != {"tasks": list(TASKS), "folds": list(FOLDS), "adapter_count": 9}:
        raise BenchmarkError("Phase B4 future adapter matrix drift")
    expected_lexical = {
        "S6": ("S6-R3", "CHAR", "LINEARSVC", "BALANCED", 0.25),
        "S3": ("S3-R5", "WORD", "LOGREG", "BALANCED", 2.0),
        "S2": ("S2-R3", "WORD_CHAR", "LOGREG", "BALANCED", 2.0),
    }
    for task, expected in expected_lexical.items():
        actual = config.get("lexical", {}).get(task, {})
        if (
            tuple(
                actual.get(key)
                for key in (
                    "representation_id",
                    "representation",
                    "classifier",
                    "class_weight",
                    "c",
                )
            )
            != expected
        ):
            raise BenchmarkError(f"Phase B4 lexical configuration drift: {task}")


def adalora_schedule(total_step: int) -> dict[str, int]:
    if total_step < 1:
        raise BenchmarkError("AdaLoRA total_step must be positive")
    tinit = round(0.10 * total_step)
    tfinal = round(0.20 * total_step)
    allocation_steps = max(1, total_step - tinit - tfinal)
    delta_t = max(1, round(allocation_steps / 100))
    return {
        "total_step": total_step,
        "tinit": tinit,
        "tfinal": tfinal,
        "deltaT": delta_t,
        "formula": (
            "tinit=round(0.10*total_step); tfinal=round(0.20*total_step); "
            "deltaT=max(1,round(max(1,total_step-tinit-tfinal)/100))"
        ),
    }


def balanced_mean_one_weights(labels: list[str], class_order: tuple[str, ...]) -> list[float]:
    counts = Counter(labels)
    if set(counts) != set(class_order):
        raise BenchmarkError("TRAIN-only class weights require every task class")
    raw = [len(labels) / (len(class_order) * counts[label]) for label in class_order]
    mean = statistics.fmean(raw)
    return [value / mean for value in raw]


def weighted_focal_cross_entropy(
    logits: Any, labels: Any, class_weights: Any, gamma: float = FOCAL_GAMMA
) -> Any:
    import torch
    import torch.nn.functional as functional

    log_probabilities = functional.log_softmax(logits.float(), dim=-1)
    selected = log_probabilities.gather(1, labels.unsqueeze(1)).squeeze(1)
    probabilities = selected.exp()
    weights = class_weights.float().gather(0, labels)
    losses = -weights * (1.0 - probabilities).pow(gamma) * selected
    if not torch.isfinite(losses).all():
        raise BenchmarkError("non-finite weighted focal loss")
    return losses.mean()


def deterministic_stratified_subset(
    training_ids: list[str], labels: dict[str, str], size: int, seed: int
) -> list[str]:
    if size > len(training_ids) or size < 1:
        raise BenchmarkError("invalid engineering subset size")
    groups: dict[str, list[str]] = {}
    for stable_id in training_ids:
        groups.setdefault(labels[stable_id], []).append(stable_id)
    exact = {label: size * len(ids) / len(training_ids) for label, ids in groups.items()}
    allocations = {label: math.floor(value) for label, value in exact.items()}
    remainder = size - sum(allocations.values())
    order = sorted(groups, key=lambda label: (-(exact[label] - allocations[label]), label))
    for label in order[:remainder]:
        allocations[label] += 1
    selected = []
    for label, ids in groups.items():
        ranked = sorted(
            ids,
            key=lambda stable_id: hashlib.sha256(
                f"{seed}|B4|S2|fold-1|{stable_id}".encode()
            ).digest(),
        )
        selected.extend(ranked[: allocations[label]])
    return sorted(selected, key=training_ids.index)


def future_adapter_matrix() -> tuple[tuple[str, int], ...]:
    return tuple((task, fold) for task in TASKS for fold in FOLDS)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _status(path: Path, **updates: Any) -> None:
    current = json.loads(path.read_text()) if path.is_file() else {}
    _atomic_json(
        path,
        {
            **current,
            **updates,
            "pid": os.getpid(),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "locked_test_accessed": False,
        },
    )


def mark_feasibility_failed(report_dir: Path, error: BaseException) -> None:
    """Persist a terminal failure for a detached feasibility process."""
    _status(
        report_dir / ".work" / "feasibility.status",
        state="FAILED",
        current_phase="failed",
        error_type=type(error).__name__,
        error_message=str(error),
    )


def _seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    __import__("random").seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "mps"):
        torch.mps.manual_seed(seed)


def _load_adalora_model(
    class_order: tuple[str, ...], schedule: dict[str, int], device: str
) -> tuple[Any, dict[str, Any]]:
    from peft import AdaLoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification

    base = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        num_labels=len(class_order),
        id2label={index: label for index, label in enumerate(class_order)},
        label2id={label: index for index, label in enumerate(class_order)},
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
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    peft_config = AdaLoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        init_r=INIT_R,
        target_r=TARGET_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=list(TARGET_MODULES),
        modules_to_save=list(MODULES_TO_SAVE),
        bias="none",
        total_step=schedule["total_step"],
        tinit=schedule["tinit"],
        tfinal=schedule["tfinal"],
        deltaT=schedule["deltaT"],
    )
    model = get_peft_model(base, peft_config)
    model.to(device=device)
    trainable_names = [name for name, value in model.named_parameters() if value.requires_grad]
    forbidden = [
        name for name in trainable_names if "lora_" not in name and "modules_to_save" not in name
    ]
    if forbidden or not any(
        "modules_to_save" in name and "classifier" in name for name in trainable_names
    ):
        raise BenchmarkError("AdaLoRA trainable-parameter boundary drift")
    return model, {
        "target_module_names": target_names,
        "trainable_parameter_names": trainable_names,
        "total_parameter_count": sum(value.numel() for value in model.parameters()),
        "trainable_parameter_count": sum(
            value.numel() for value in model.parameters() if value.requires_grad
        ),
    }


def _tokenize(tokenizer: Any, texts: list[str]) -> dict[str, Any]:
    return tokenizer(
        texts,
        max_length=MAX_LENGTH,
        truncation=True,
        padding=False,
        return_attention_mask=True,
    )


class _Dataset:
    def __init__(self, encoded: dict[str, Any], labels: list[int]) -> None:
        self.encoded = encoded
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            **{key: value[index] for key, value in self.encoded.items()},
            "labels": self.labels[index],
        }


def _loader(
    tokenizer: Any,
    encoded: dict[str, Any],
    labels: list[int],
    batch_size: int,
    seed: int,
    shuffle: bool,
) -> Any:
    import torch
    from torch.utils.data import DataLoader
    from transformers import DataCollatorWithPadding

    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        _Dataset(encoded, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        collate_fn=DataCollatorWithPadding(tokenizer, return_tensors="pt"),
        num_workers=0,
    )


def _move(batch: dict[str, Any], device: str) -> tuple[dict[str, Any], Any]:
    labels = batch.pop("labels").to(device)
    return {key: value.to(device) for key, value in batch.items()}, labels


def _clear_mps() -> None:
    import gc

    import torch

    gc.collect()
    torch.mps.empty_cache()


def _select_micro_batch(
    tokenizer: Any,
    encoded: dict[str, Any],
    label_ids: list[int],
    class_order: tuple[str, ...],
    config: dict[str, Any],
) -> tuple[int, list[dict[str, Any]]]:
    import torch

    oom_events = []
    longest = sorted(
        range(len(label_ids)), key=lambda index: len(encoded["input_ids"][index]), reverse=True
    )
    for micro_batch in config["train_micro_batch_candidates"]:
        accumulation = EFFECTIVE_BATCH_SIZE // micro_batch
        steps_per_epoch = math.ceil(math.ceil(len(label_ids) / micro_batch) / accumulation)
        schedule = adalora_schedule(steps_per_epoch * EPOCHS)
        model = None
        try:
            model, _ = _load_adalora_model(class_order, schedule, "mps")
            indices = longest[:micro_batch]
            probe = {key: [values[index] for index in indices] for key, values in encoded.items()}
            probe_labels = [label_ids[index] for index in indices]
            batch = next(
                iter(_loader(tokenizer, probe, probe_labels, micro_batch, config["seed"], False))
            )
            inputs, labels = _move(batch, "mps")
            weights = torch.tensor(
                balanced_mean_one_weights([class_order[index] for index in label_ids], class_order),
                device="mps",
            )
            loss = weighted_focal_cross_entropy(model(**inputs).logits, labels, weights)
            loss.backward()
            if not all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                raise BenchmarkError("non-finite AdaLoRA preflight gradient")
            del batch, inputs, labels, loss, weights, model
            _clear_mps()
            return micro_batch, oom_events
        except RuntimeError as exc:
            if "memory" not in str(exc).lower():
                raise
            oom_events.append({"micro_batch": micro_batch, "error": str(exc)})
            _clear_mps()
    raise BenchmarkError("RTA AdaLoRA does not fit MPS at micro-batch 2")


def _train_engineering(
    model: Any,
    loader: Any,
    class_weights: Any,
    config: dict[str, Any],
    schedule_values: dict[str, int],
    status_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch
    from transformers import get_linear_schedule_with_warmup

    optimizer = torch.optim.AdamW(
        (value for value in model.parameters() if value.requires_grad),
        lr=LEARNING_RATE,
        weight_decay=config["weight_decay"],
    )
    warmup_steps = round(config["warmup_fraction"] * schedule_values["total_step"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, schedule_values["total_step"]
    )
    accumulation = config["resolved_gradient_accumulation"]
    diagnostics, optimizer_step, nan_inf_events = [], 0, 0
    started = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        epoch_started = time.monotonic()
        for batch_index, batch in enumerate(loader, 1):
            inputs, labels = _move(batch, "mps")
            loss = weighted_focal_cross_entropy(
                model(**inputs).logits, labels, class_weights, FOCAL_GAMMA
            )
            if not torch.isfinite(loss):
                nan_inf_events += 1
                raise BenchmarkError("non-finite AdaLoRA training loss")
            (loss / accumulation).backward()
            losses.append(float(loss.detach().cpu()))
            if batch_index % accumulation == 0 or batch_index == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer_step += 1
                model.base_model.update_and_allocate(optimizer_step)
                optimizer.zero_grad(set_to_none=True)
        elapsed = time.monotonic() - epoch_started
        diagnostics.append(
            {
                "epoch": epoch,
                "training_loss": statistics.fmean(losses),
                "training_seconds": elapsed,
                "optimizer_steps_completed": optimizer_step,
            }
        )
        _status(
            status_path,
            state="RUNNING",
            current_phase="training",
            epoch_completed=epoch,
            total_epochs=EPOCHS,
        )
        print(
            f"[B4] epoch={epoch}/{EPOCHS} loss={diagnostics[-1]['training_loss']:.8f} "
            f"seconds={elapsed:.3f} optimizer_steps={optimizer_step}",
            flush=True,
        )
    elapsed = time.monotonic() - started
    if optimizer_step != schedule_values["total_step"]:
        raise BenchmarkError("AdaLoRA optimizer-step schedule mismatch")
    return diagnostics, {
        "training_runtime_seconds": elapsed,
        "optimizer_steps": optimizer_step,
        "optimizer_steps_per_second": optimizer_step / elapsed,
        "nan_inf_events": nan_inf_events,
        "warmup_steps": warmup_steps,
    }


def _inference_benchmark(
    model: Any, tokenizer: Any, encoded: dict[str, Any], batch_size: int, seed: int
) -> dict[str, Any]:
    import torch

    labels = [0] * len(encoded["input_ids"])
    loader = _loader(tokenizer, encoded, labels, batch_size, seed, False)
    model.eval()
    documents, token_count = 0, 0
    started = time.monotonic()
    with torch.inference_mode():
        for batch in loader:
            inputs, _ = _move(batch, "mps")
            output = model(**inputs, output_hidden_states=True)
            embeddings = output.hidden_states[-1][:, 0, :].float()
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            if embeddings.shape[1] != DIMENSION or not torch.isfinite(embeddings).all():
                raise BenchmarkError("invalid adapted RTA representation")
            documents += embeddings.shape[0]
            token_count += int(inputs["attention_mask"].sum().cpu())
            del output, embeddings
    torch.mps.synchronize()
    elapsed = time.monotonic() - started
    return {
        "inference_rows": documents,
        "inference_batch_size": batch_size,
        "runtime_seconds": elapsed,
        "documents_per_second": documents / elapsed,
        "tokens_per_second": token_count / elapsed,
        "representation": "adapted_encoder_final_layer_first_token_l2",
        "embedding_dimension": DIMENSION,
    }


def _runtime_projection(
    training: dict[str, Any],
    inference: dict[str, Any],
    folds: dict[int, tuple[list[str], list[str]]],
) -> dict[str, Any]:
    exposure_rate = training["example_exposures"] / training["training_runtime_seconds"]
    inference_rate = inference["documents_per_second"]
    rows = []
    lexical_seconds = 0.0
    try:
        with Path("reports/lexical_semantic_fusion_v1/fit_results.csv").open() as handle:
            import csv

            lexical_seconds = sum(
                float(row["fit_runtime_seconds"]) + float(row["prediction_runtime_seconds"])
                for row in csv.DictReader(handle)
                if row["stage"] == "COMPETITIVE"
            )
    except OSError:
        lexical_seconds = 0.0
    total_training = total_direct = total_embeddings = 0.0
    for task in TASKS:
        for fold, (train_ids, validation_ids) in folds.items():
            train_seconds = len(train_ids) * EPOCHS / exposure_rate
            direct_seconds = len(validation_ids) / inference_rate
            embedding_seconds = (len(train_ids) + len(validation_ids)) / inference_rate
            rows.append(
                {
                    "task": task,
                    "fold": fold,
                    "training_rows": len(train_ids),
                    "validation_rows": len(validation_ids),
                    "projected_training_seconds": train_seconds,
                    "projected_direct_inference_seconds": direct_seconds,
                    "projected_embedding_seconds": embedding_seconds,
                }
            )
            total_training += train_seconds
            total_direct += direct_seconds
            total_embeddings += embedding_seconds
    reporting_overhead = 300.0
    central = (
        total_training + total_direct + total_embeddings + lexical_seconds + reporting_overhead
    )
    return {
        "scaling_method": (
            "measured exposure and inference throughput applied to exact frozen CV sizes"
        ),
        "folds": rows,
        "adalora_training_seconds": total_training,
        "direct_classifier_validation_inference_seconds": total_direct,
        "adapted_embedding_extraction_seconds": total_embeddings,
        "lexical_fusion_fit_prediction_seconds": lexical_seconds,
        "checkpoint_reporting_overhead_seconds": reporting_overhead,
        "central_total_seconds": central,
        "conservative_total_seconds": central * 1.35,
        "safety_margin": "35% over central projection",
    }


def run_adalora_feasibility(
    *,
    development_dir: Path,
    manifest_dir: Path,
    protocol_report_dir: Path,
    report_dir: Path,
    protocol: FrozenProtocol,
    config_path: Path | None = None,
    stage: str = "feasibility",
) -> dict[str, Any]:
    """Run one engineering adapter on S2 fold-1 TRAIN only; competitive stages fail closed."""
    if stage != "feasibility":
        raise BenchmarkError("Phase B4 competitive execution is not authorized")
    import psutil
    import torch
    from transformers import AutoTokenizer

    from defect_classifier.classical_benchmark import _load_development_rows, _load_fold

    config, fingerprint = load_adalora_config(config_path)
    if version("peft") != PEFT_VERSION:
        raise BenchmarkError("installed PEFT version differs from frozen B4 version")
    b3 = json.loads((report_dir.parent / "rta_fusion_v1" / "model_provenance.json").read_text())
    if (
        b3.get("model_id") != MODEL_ID
        or b3.get("resolved_revision") != MODEL_REVISION
        or b3.get("architecture") != ARCHITECTURE
    ):
        raise BenchmarkError("B3 RTA provenance mismatch")
    frozen = json.loads((protocol_report_dir / "fingerprints.json").read_text())
    if frozen["protocol_sha256"] != protocol.fingerprint:
        raise BenchmarkError("protocol fingerprint drift")
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise BenchmarkError("Phase B4 feasibility requires MPS")
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
        result_artifact_path=str(result_path.resolve()),
        config_sha256=fingerprint,
        model_id=MODEL_ID,
        resolved_revision=MODEL_REVISION,
        peft_method=PEFT_METHOD,
        source_membership="DEVELOPMENT_TRAIN_ONLY",
        competitive_models_fitted=0,
    )
    (work / "feasibility.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
    print(
        f"[B4] starting model={MODEL_ID} revision={MODEL_REVISION} peft=ADALORA "
        f"peft_version={PEFT_VERSION} config_sha256={fingerprint} "
        "source=DEVELOPMENT_TRAIN_ONLY locked_test=NOT_ACCESSED",
        flush=True,
    )
    rows = _load_development_rows(development_dir)
    if _membership_fingerprint(rows) != frozen["development_membership_sha256"]:
        raise BenchmarkError("development membership fingerprint drift")
    train_ids, _validation_ids = _load_fold(
        rows, manifest_dir, protocol_report_dir / "fingerprints.json", protocol, 1
    )
    del _validation_ids
    labels_by_id = {stable_id: rows[stable_id].targets["S2"] for stable_id in train_ids}
    subset_ids = deterministic_stratified_subset(
        train_ids, labels_by_id, config["feasibility"]["training_rows"], config["seed"]
    )
    subset_membership = _membership_fingerprint(subset_ids)
    class_order = protocol.targets["s2"].order
    labels = [labels_by_id[stable_id] for stable_id in subset_ids]
    label_ids = [class_order.index(label) for label in labels]
    _seed_everything(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=False,
        local_files_only=True,
    )
    token_started = time.monotonic()
    encoded = _tokenize(tokenizer, [rows[stable_id].text for stable_id in subset_ids])
    tokenization_seconds = time.monotonic() - token_started
    _status(status_path, current_phase="mps_micro_batch_preflight")
    micro_batch, oom_events = _select_micro_batch(
        tokenizer, encoded, label_ids, class_order, config
    )
    print(f"[B4] AdaLoRA MPS preflight active micro_batch={micro_batch}", flush=True)
    accumulation = EFFECTIVE_BATCH_SIZE // micro_batch
    batches = math.ceil(len(subset_ids) / micro_batch)
    steps_per_epoch = math.ceil(batches / accumulation)
    schedule_values = adalora_schedule(steps_per_epoch * EPOCHS)
    resolved = {
        **config,
        "resolved_train_micro_batch": micro_batch,
        "resolved_gradient_accumulation": accumulation,
        "resolved_effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "steps_per_epoch": steps_per_epoch,
        "adalora_schedule": schedule_values,
    }
    _atomic_json(report_dir / "training_configuration.json", resolved)
    print(
        f"[B4] model={MODEL_ID} revision={MODEL_REVISION} peft=ADALORA peft_version={PEFT_VERSION}",
        flush=True,
    )
    print(
        f"[B4] config_sha256={fingerprint} stage=ENGINEERING_ONLY task=S2 fold=1 "
        f"subset_rows={len(subset_ids)} subset_sha256={subset_membership} source=DEVELOPMENT_TRAIN",
        flush=True,
    )
    print(
        f"[B4] micro_batch={micro_batch} accumulation={accumulation} effective_batch=32 "
        f"total_step={schedule_values['total_step']} tinit={schedule_values['tinit']} "
        f"tfinal={schedule_values['tfinal']} deltaT={schedule_values['deltaT']} "
        "locked_test=NOT_ACCESSED",
        flush=True,
    )
    _status(
        status_path,
        current_phase="training",
        selected_micro_batch=micro_batch,
        gradient_accumulation=accumulation,
        subset_membership_sha256=subset_membership,
    )
    model, model_details = _load_adalora_model(class_order, schedule_values, "mps")
    class_weight_values = balanced_mean_one_weights(labels, class_order)
    class_weights = torch.tensor(class_weight_values, device="mps")
    loader = _loader(tokenizer, encoded, label_ids, micro_batch, config["seed"], True)
    rss_before = psutil.Process().memory_info().rss
    diagnostics, training = _train_engineering(
        model, loader, class_weights, resolved, schedule_values, status_path
    )
    training.update(
        {
            "example_exposures": len(subset_ids) * EPOCHS,
            "example_exposures_per_second": len(subset_ids)
            * EPOCHS
            / training["training_runtime_seconds"],
            "tokens_per_second": sum(len(values) for values in encoded["input_ids"])
            * EPOCHS
            / training["training_runtime_seconds"],
            "epoch_diagnostics": diagnostics,
            "class_weights": dict(zip(class_order, class_weight_values, strict=True)),
            "tokenization_runtime_seconds": tokenization_seconds,
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
    adapter_size = sum(path.stat().st_size for path in adapter_dir.rglob("*") if path.is_file())
    training["adapter_checkpoint_size_bytes"] = adapter_size
    inference_count = min(config["feasibility"]["inference_rows"], len(subset_ids))
    inference_encoded = {key: values[:inference_count] for key, values in encoded.items()}
    _status(status_path, current_phase="train_only_inference_throughput")
    inference = _inference_benchmark(
        model, tokenizer, inference_encoded, config["inference_batch_size"], config["seed"]
    )
    folds = {
        fold: _load_fold(
            rows, manifest_dir, protocol_report_dir / "fingerprints.json", protocol, fold
        )
        for fold in FOLDS
    }
    projection = _runtime_projection(training, inference, folds)
    projection["adapter_checkpoint_size_bytes"] = adapter_size
    projection["nine_adapter_storage_bytes"] = adapter_size * 9
    model_details["trainable_percentage"] = (
        model_details["trainable_parameter_count"] / model_details["total_parameter_count"] * 100
    )
    provenance = {
        "model_id": MODEL_ID,
        "resolved_revision": MODEL_REVISION,
        "architecture": ARCHITECTURE,
        "b3_provenance_verified": True,
        "peft_method": PEFT_METHOD,
        "peft_version": PEFT_VERSION,
        "target_modules": list(TARGET_MODULES),
        "modules_to_save": list(MODULES_TO_SAVE),
        "base_model_otherwise_frozen": True,
        "new_task_classification_head": True,
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
        "mps_is_built": True,
        "mps_is_available": True,
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
        "task": "S2",
        "fold": 1,
        "source_membership": "DEVELOPMENT_TRAIN_ONLY",
        "subset_rows": len(subset_ids),
        "subset_membership_sha256": subset_membership,
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
        "# Phase B4 RTA AdaLoRA Feasibility\n\n"
        "This is an engineering-only DEVELOPMENT-TRAIN benchmark. It contains no predictive "
        "validation metrics and is excluded from competitive research tables.\n\n"
        f"- Configuration: `{fingerprint}`\n"
        f"- Training runtime: {training['training_runtime_seconds']:.3f} seconds\n"
        "- Central nine-adapter projection: "
        f"{projection['central_total_seconds'] / 3600:.3f} hours\n"
        f"- Conservative projection: {projection['conservative_total_seconds'] / 3600:.3f} hours\n"
        "- Locked test accessed: no\n"
    )
    (report_dir / "RTA_ADALORA_FEASIBILITY.md").write_text(report, encoding="utf-8")
    _status(
        status_path,
        state="SUCCESS",
        current_phase="complete",
        epoch_completed=EPOCHS,
        result_artifact_path=str(result_path.resolve()),
    )
    print(f"[B4] SUCCESS result={result_path.resolve()}", flush=True)
    return result
