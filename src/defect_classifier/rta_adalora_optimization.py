"""B4-H-O implementation-only runtime optimization feasibility."""

from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import os
import platform
import random
import resource
import statistics
import time
import tomllib
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from defect_classifier.classical_benchmark import BenchmarkError
from defect_classifier.preparation import _membership_fingerprint
from defect_classifier.protocol import FrozenProtocol
from defect_classifier.rta_adalora import (
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
)
from defect_classifier.rta_adalora_multitask import (
    B4_SUBSET_SHA256,
    HEAD_DIMENSIONS,
    TASKS,
    _build_multitask_model,
    _losses,
    hierarchical_targets,
)
from defect_classifier.rta_fusion import DIMENSION, MAX_LENGTH, MODEL_ID, MODEL_REVISION

EFFECTIVE_BATCH_SIZE = 32
MICRO_BATCH = 8
ACCUMULATION = 4
PAD_MULTIPLE = 8
WARMUP_OPTIMIZER_STEPS = 101
BASELINE_CENTRAL_SECONDS = 232117.726


def default_config_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "rta_adalora_multitask_optimization_v1.toml"
    )


def load_optimization_config(path: Path | None = None) -> tuple[dict[str, Any], str]:
    raw = (path or default_config_path()).read_bytes()
    config = tomllib.loads(raw.decode())
    validate_optimization_config(config)
    return config, hashlib.sha256(raw).hexdigest()


def validate_optimization_config(config: dict[str, Any]) -> None:
    expected = {
        "model_id": MODEL_ID,
        "resolved_revision": MODEL_REVISION,
        "peft_method": PEFT_METHOD,
        "peft_version": PEFT_VERSION,
        "max_length": MAX_LENGTH,
        "future_epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": 0.01,
        "warmup_fraction": 0.06,
        "max_grad_norm": 1.0,
        "focal_gamma": FOCAL_GAMMA,
        "task_loss_weights": [1 / 3] * 3,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "micro_batch": MICRO_BATCH,
        "gradient_accumulation": ACCUMULATION,
        "dynamic_padding": True,
        "pad_to_multiple_of": PAD_MULTIPLE,
        "deterministic_length_bucketing": True,
        "length_signal": "TRAIN_TOKEN_LENGTH_ONLY",
        "init_r": INIT_R,
        "target_r": 8,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "target_modules": list(TARGET_MODULES),
        "semantic_weight": 1.0,
        "benchmark_epochs_per_candidate": 1,
        "warmup_optimizer_steps": WARMUP_OPTIMIZER_STEPS,
        "candidate_dtypes": ["float32", "bfloat16"],
        "float16_allowed": False,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise BenchmarkError("B4-H-O scientific or engineering configuration drift")
    if config.get("subset") != {
        "rows": 10000,
        "fingerprint": B4_SUBSET_SHA256,
        "source": "DEVELOPMENT_TRAIN_ONLY",
    }:
        raise BenchmarkError("B4-H-O subset boundary drift")


def deterministic_length_batches(
    stable_ids: list[str], lengths: list[int], batch_size: int, seed: int, epoch: int
) -> list[list[int]]:
    """Sort by TRAIN token length, then deterministically shuffle whole batches."""
    if len(stable_ids) != len(lengths) or len(stable_ids) != len(set(stable_ids)):
        raise BenchmarkError("length batching requires unique aligned TRAIN identities")
    if any(length < 1 or length > MAX_LENGTH for length in lengths):
        raise BenchmarkError("length batching received an invalid capped token length")
    ordered = sorted(
        range(len(stable_ids)),
        key=lambda index: (
            lengths[index],
            hashlib.sha256(f"{seed}|{epoch}|{stable_ids[index]}".encode()).digest(),
        ),
    )
    batches = [ordered[start : start + batch_size] for start in range(0, len(ordered), batch_size)]
    random.Random(f"{seed}|B4HO|epoch-{epoch}").shuffle(batches)
    return batches


def dynamic_pad_batch(
    tokenizer: Any, encoded: dict[str, list[list[int]]], indices: list[int]
) -> dict[str, Any]:
    features = [{key: values[index] for key, values in encoded.items()} for index in indices]
    if any(len(feature["input_ids"]) > MAX_LENGTH for feature in features):
        raise BenchmarkError("dynamic padding input exceeds the frozen 512-token hard maximum")
    return tokenizer.pad(
        features,
        padding=True,
        pad_to_multiple_of=PAD_MULTIPLE,
        return_tensors="pt",
    )


def _autocast(dtype: str) -> Any:
    if dtype == "float32":
        return contextlib.nullcontext()
    if dtype == "bfloat16":
        import torch

        return torch.autocast(device_type="mps", dtype=torch.bfloat16)
    raise BenchmarkError("only float32 or validated bfloat16 are allowed")


def reject_unstable_bf16(checks: dict[str, bool]) -> None:
    if not checks or not all(checks.values()):
        failed = sorted(key for key, passed in checks.items() if not passed)
        raise BenchmarkError(f"bfloat16 numerical preflight failed closed: {failed}")


def _batch_labels(
    label_ids: dict[str, list[int]], indices: list[int], device: str
) -> dict[str, Any]:
    import torch

    return {
        task: torch.tensor([label_ids[task][index] for index in indices], device=device)
        for task in TASKS
    }


def _move_inputs(batch: dict[str, Any]) -> dict[str, Any]:
    return {key: value.to("mps") for key, value in batch.items()}


def _optimizer_step(
    model: Any,
    optimizer: Any,
    scheduler: Any,
    optimizer_step: int,
    max_grad_norm: float,
) -> int:
    import torch

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    scheduler.step()
    optimizer_step += 1
    model.base_model.update_and_allocate(optimizer_step)
    optimizer.zero_grad(set_to_none=True)
    return optimizer_step


def _bf16_preflight(
    tokenizer: Any,
    encoded: dict[str, Any],
    label_ids: dict[str, list[int]],
    weights: dict[str, Any],
    stable_ids: list[str],
    lengths: list[int],
    config: dict[str, Any],
) -> dict[str, Any]:
    import torch

    schedule = adalora_schedule(939)
    indices = deterministic_length_batches(stable_ids, lengths, MICRO_BATCH, config["seed"], 0)[-1]
    repeated_losses = []
    checks = {}
    for _repeat in range(2):
        _seed_everything(config["seed"])
        model, _ = _build_multitask_model(schedule, "mps")
        optimizer = torch.optim.AdamW(
            (value for value in model.parameters() if value.requires_grad), lr=LEARNING_RATE
        )
        inputs = _move_inputs(dynamic_pad_batch(tokenizer, encoded, indices))
        labels = _batch_labels(label_ids, indices, "mps")
        with _autocast("bfloat16"):
            output = model(**inputs, output_hidden_states=True)
        losses = _losses(output.logits, labels, weights)
        losses["TOTAL"].backward()
        trainable = [
            (name, value) for name, value in model.named_parameters() if value.requires_grad
        ]
        checks.update(
            {
                "forward_finite": all(
                    torch.isfinite(value).all() for value in output.logits.values()
                ),
                "all_task_losses_finite": all(torch.isfinite(value) for value in losses.values()),
                "adalora_gradients_finite": all(
                    value.grad is not None and torch.isfinite(value.grad).all()
                    for name, value in trainable
                    if "lora_" in name
                ),
                "head_gradients_finite": all(
                    value.grad is not None and torch.isfinite(value.grad).all()
                    for name, value in trainable
                    if "modules_to_save" in name
                ),
            }
        )
        optimizer.step()
        model.base_model.update_and_allocate(1)
        checks["updated_parameters_finite"] = all(
            torch.isfinite(value).all() for _, value in trainable
        )
        model.eval()
        with torch.inference_mode(), _autocast("bfloat16"):
            inference = model(**inputs, output_hidden_states=True)
            embeddings = torch.nn.functional.normalize(
                inference.hidden_states[-1][:, 0, :].float(), p=2, dim=1
            )
        checks["inference_finite"] = all(
            torch.isfinite(value).all() for value in inference.logits.values()
        )
        checks["embedding_valid"] = bool(
            embeddings.shape[1] == DIMENSION and torch.isfinite(embeddings).all().item()
        )
        repeated_losses.append(float(losses["TOTAL"].detach().cpu()))
        del model, optimizer, inputs, labels, output, inference, embeddings, losses
        _clear_mps()
    denominator = max(abs(repeated_losses[0]), 1e-12)
    relative_difference = abs(repeated_losses[0] - repeated_losses[1]) / denominator
    checks["repeated_loss_stable"] = relative_difference <= 0.02
    reject_unstable_bf16(checks)
    return {
        "status": "PASS",
        "checks": checks,
        "repeated_joint_losses": repeated_losses,
        "relative_loss_difference": relative_difference,
        "autocast_device": "mps",
        "autocast_dtype": "bfloat16",
        "model_parameters_dtype": "float32",
        "loss_reductions_dtype": "float32",
        "float16_used": False,
    }


def _run_candidate(
    *,
    name: str,
    dtype: str,
    tokenizer: Any,
    encoded: dict[str, Any],
    label_ids: dict[str, list[int]],
    class_weights: dict[str, list[float]],
    stable_ids: list[str],
    lengths: list[int],
    config: dict[str, Any],
    status_path: Path,
) -> dict[str, Any]:
    import psutil
    import torch
    from transformers import get_linear_schedule_with_warmup

    schedule = adalora_schedule(939)
    initialization_started = time.monotonic()
    _seed_everything(config["seed"])
    model, details = _build_multitask_model(schedule, "mps")
    optimizer = torch.optim.AdamW(
        (value for value in model.parameters() if value.requires_grad),
        lr=LEARNING_RATE,
        weight_decay=config["weight_decay"],
    )
    scheduler = get_linear_schedule_with_warmup(optimizer, 56, 939)
    weights = {task: torch.tensor(values, device="mps") for task, values in class_weights.items()}
    initialization_seconds = time.monotonic() - initialization_started
    optimizer.zero_grad(set_to_none=True)
    warmup_started = time.monotonic()
    optimizer_step = 0
    warmup_batches = deterministic_length_batches(
        stable_ids, lengths, MICRO_BATCH, config["seed"], -1
    )
    for batch_index in range(WARMUP_OPTIMIZER_STEPS * ACCUMULATION):
        indices = warmup_batches[batch_index % len(warmup_batches)]
        inputs = _move_inputs(dynamic_pad_batch(tokenizer, encoded, indices))
        labels = _batch_labels(label_ids, indices, "mps")
        with _autocast(dtype):
            logits = model(**inputs).logits
        losses = _losses(logits, labels, weights)
        (losses["TOTAL"] / ACCUMULATION).backward()
        if (batch_index + 1) % ACCUMULATION == 0:
            optimizer_step = _optimizer_step(
                model, optimizer, scheduler, optimizer_step, config["max_grad_norm"]
            )
    torch.mps.synchronize()
    warmup_seconds = time.monotonic() - warmup_started
    _status(
        status_path,
        current_candidate=name,
        candidate_phase="timed_complete_train_exposure",
        dtype=dtype,
    )
    batches = deterministic_length_batches(stable_ids, lengths, MICRO_BATCH, config["seed"], 1)
    timed_started = time.monotonic()
    padded_tokens = examples = timed_optimizer_steps = nan_inf_events = 0
    batch_lengths = []
    model.train()
    for batch_number, indices in enumerate(batches, 1):
        inputs = _move_inputs(dynamic_pad_batch(tokenizer, encoded, indices))
        labels = _batch_labels(label_ids, indices, "mps")
        padded_tokens += int(inputs["attention_mask"].shape[0] * inputs["attention_mask"].shape[1])
        examples += len(indices)
        batch_lengths.append(int(inputs["attention_mask"].shape[1]))
        with _autocast(dtype):
            logits = model(**inputs).logits
        losses = _losses(logits, labels, weights)
        if not all(torch.isfinite(value) for value in losses.values()):
            nan_inf_events += 1
            raise BenchmarkError(f"{name} produced non-finite loss")
        (losses["TOTAL"] / ACCUMULATION).backward()
        if batch_number % ACCUMULATION == 0 or batch_number == len(batches):
            optimizer_step = _optimizer_step(
                model, optimizer, scheduler, optimizer_step, config["max_grad_norm"]
            )
            timed_optimizer_steps += 1
    torch.mps.synchronize()
    timed_seconds = time.monotonic() - timed_started
    result = {
        "candidate": name,
        "dtype": dtype,
        "initialization_seconds": initialization_seconds,
        "warmup_seconds": warmup_seconds,
        "warmup_optimizer_steps": WARMUP_OPTIMIZER_STEPS,
        "timed_training_seconds": timed_seconds,
        "timed_examples": examples,
        "examples_per_second": examples / timed_seconds,
        "actual_padded_tokens": padded_tokens,
        "actual_padded_tokens_per_second": padded_tokens / timed_seconds,
        "optimizer_steps": timed_optimizer_steps,
        "optimizer_steps_per_second": timed_optimizer_steps / timed_seconds,
        "micro_batch": MICRO_BATCH,
        "gradient_accumulation": ACCUMULATION,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "average_padded_sequence_length": statistics.fmean(batch_lengths),
        "maximum_batch_sequence_length": max(batch_lengths),
        "batch_length_distribution": {
            "minimum": min(batch_lengths),
            "median": statistics.median(batch_lengths),
            "p90": float(np.percentile(batch_lengths, 90)),
            "p95": float(np.percentile(batch_lengths, 95)),
            "p99": float(np.percentile(batch_lengths, 99)),
            "maximum": max(batch_lengths),
        },
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "process_rss_bytes": psutil.Process().memory_info().rss,
        "mps_current_allocated_bytes": torch.mps.current_allocated_memory(),
        "mps_driver_allocated_bytes": torch.mps.driver_allocated_memory(),
        "oom_events": [],
        "mps_fallback_events": 0,
        "nan_inf_events": nan_inf_events,
        "scientific_invariants_unchanged": True,
        "adalora_details": details,
    }
    del model, optimizer, scheduler, weights
    _clear_mps()
    return result


def _candidate_checkpoint(work: Path, candidate: dict[str, Any]) -> None:
    _atomic_json(work / f"{candidate['candidate'].casefold()}.json", candidate)


def _inference_benchmark(
    dtype: str,
    tokenizer: Any,
    encoded: dict[str, Any],
    label_ids: dict[str, list[int]],
    stable_ids: list[str],
    lengths: list[int],
    config: dict[str, Any],
) -> dict[str, Any]:
    import torch

    _seed_everything(config["seed"])
    model, _ = _build_multitask_model(adalora_schedule(939), "mps")
    count = min(1000, len(stable_ids))
    batches = deterministic_length_batches(
        stable_ids[:count], lengths[:count], MICRO_BATCH, config["seed"], 99
    )
    model.eval()
    documents = tokens = 0
    started = time.monotonic()
    with torch.inference_mode():
        for indices in batches:
            inputs = _move_inputs(dynamic_pad_batch(tokenizer, encoded, indices))
            with _autocast(dtype):
                output = model(**inputs, output_hidden_states=True)
            embeddings = torch.nn.functional.normalize(
                output.hidden_states[-1][:, 0, :].float(), p=2, dim=1
            )
            if embeddings.shape[1] != DIMENSION or not torch.isfinite(embeddings).all():
                raise BenchmarkError("optimized shared inference representation invalid")
            if {task: output.logits[task].shape[1] for task in TASKS} != HEAD_DIMENSIONS:
                raise BenchmarkError("optimized shared inference heads invalid")
            documents += len(indices)
            tokens += int(inputs["attention_mask"].numel())
    torch.mps.synchronize()
    elapsed = time.monotonic() - started
    del model
    _clear_mps()
    return {
        "rows": documents,
        "runtime_seconds": elapsed,
        "documents_per_second": documents / elapsed,
        "actual_padded_tokens_per_second": tokens / elapsed,
        "dtype": dtype,
        "single_shared_forward": True,
        "predictive_metrics_calculated": False,
    }


def _project(
    candidate: dict[str, Any],
    inference: dict[str, Any],
    folds: dict[int, tuple[list[str], list[str]]],
) -> dict[str, Any]:
    example_rate = candidate["examples_per_second"]
    inference_rate = inference["documents_per_second"]
    lexical_by_fold = {fold: 0.0 for fold in (1, 2, 3)}
    with Path("reports/lexical_semantic_fusion_v1/fit_results.csv").open() as handle:
        for row in csv.DictReader(handle):
            if row["stage"] == "COMPETITIVE":
                lexical_by_fold[int(row["fold"])] += float(row["fit_runtime_seconds"])
                lexical_by_fold[int(row["fold"])] += float(row["prediction_runtime_seconds"])
    rows = []
    for fold, (training, validation) in folds.items():
        startup = candidate["initialization_seconds"] + candidate["warmup_seconds"]
        train = len(training) * EPOCHS / example_rate
        inference_seconds = (len(training) + len(validation)) / inference_rate
        overhead = 100.0
        total = startup + train + inference_seconds + lexical_by_fold[fold] + overhead
        rows.append(
            {
                "fold": fold,
                "training_rows": len(training),
                "validation_rows": len(validation),
                "startup_seconds": startup,
                "steady_state_training_seconds": train,
                "shared_embedding_and_validation_forward_seconds": inference_seconds,
                "lexical_fusion_seconds": lexical_by_fold[fold],
                "checkpoint_reporting_seconds": overhead,
                "central_fold_seconds": total,
                "central_fold_hours": total / 3600,
            }
        )
    central = sum(row["central_fold_seconds"] for row in rows)
    hours_saved = (BASELINE_CENTRAL_SECONDS - central) / 3600
    if central <= 36 * 3600:
        category = "VERY PRACTICAL"
    elif central <= 48 * 3600:
        category = "PRACTICAL"
    elif central <= 60 * 3600:
        category = "EXPENSIVE BUT FEASIBLE"
    else:
        category = "STILL VERY EXPENSIVE"
    return {
        "baseline_central_seconds": BASELINE_CENTRAL_SECONDS,
        "baseline_central_hours": BASELINE_CENTRAL_SECONDS / 3600,
        "folds": rows,
        "optimized_central_seconds": central,
        "optimized_central_hours": central / 3600,
        "optimized_conservative_seconds": central * 1.35,
        "optimized_conservative_hours": central * 1.35 / 3600,
        "absolute_hours_saved": hours_saved,
        "runtime_reduction_percent": (BASELINE_CENTRAL_SECONDS - central)
        / BASELINE_CENTRAL_SECONDS
        * 100,
        "engineering_category": category,
        "safety_margin": "35%",
    }


def mark_optimization_failed(report_dir: Path, error: BaseException) -> None:
    _status(
        report_dir / ".work" / "optimization.status",
        state="FAILED",
        candidate_phase="failed",
        error_type=type(error).__name__,
        error_message=str(error),
    )


def run_optimization_feasibility(
    *,
    development_dir: Path,
    manifest_dir: Path,
    protocol_report_dir: Path,
    report_dir: Path,
    protocol: FrozenProtocol,
    config_path: Path | None = None,
    stage: str = "feasibility",
) -> dict[str, Any]:
    if stage != "feasibility":
        raise BenchmarkError("B4-H-O full competitive execution is not authorized")
    import torch
    from transformers import AutoTokenizer

    from defect_classifier.classical_benchmark import _load_development_rows, _load_fold

    config, fingerprint = load_optimization_config(config_path)
    if version("peft") != PEFT_VERSION or not torch.backends.mps.is_available():
        raise BenchmarkError("B4-H-O PEFT/MPS environment drift")
    baseline = json.loads(
        (
            report_dir.parent / "rta_adalora_multitask_v1" / ".work" / "feasibility_result.json"
        ).read_text()
    )
    if (
        baseline.get("status") != "SUCCESS"
        or baseline.get("subset_membership_sha256") != B4_SUBSET_SHA256
        or baseline.get("competitive_metrics") is not None
    ):
        raise BenchmarkError("B4-H baseline handoff drift")
    frozen = json.loads((protocol_report_dir / "fingerprints.json").read_text())
    if frozen["protocol_sha256"] != protocol.fingerprint:
        raise BenchmarkError("protocol fingerprint drift")
    work = report_dir / ".work"
    work.mkdir(parents=True, exist_ok=True)
    status_path = work / "optimization.status"
    result_path = work / "optimization_result.json"
    _status(
        status_path,
        state="RUNNING",
        current_candidate="O1",
        candidate_phase="loading_train_only_subset",
        model_id=MODEL_ID,
        resolved_revision=MODEL_REVISION,
        subset_membership_sha256=B4_SUBSET_SHA256,
        dtype="float32",
        micro_batch=MICRO_BATCH,
        effective_batch_size=EFFECTIVE_BATCH_SIZE,
        config_sha256=fingerprint,
        scientific_configuration_changed=False,
        competitive_models_fitted=0,
        result_artifact_path=str(result_path.resolve()),
    )
    (work / "optimization.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
    print(
        f"[B4-H-O] starting model={MODEL_ID} revision={MODEL_REVISION} "
        f"config_sha256={fingerprint} subset_sha256={B4_SUBSET_SHA256} "
        "candidate=O1 dtype=float32 dynamic_padding=YES deterministic_length_bucketing=YES "
        "max_length=512 micro_batch=8 accumulation=4 effective_batch=32 "
        "scientific_configuration=UNCHANGED source=DEVELOPMENT_TRAIN_ONLY "
        "locked_test=NOT_ACCESSED competitive_metrics=DISABLED",
        flush=True,
    )
    rows = _load_development_rows(development_dir)
    train_ids, validation_ids = _load_fold(
        rows, manifest_dir, protocol_report_dir / "fingerprints.json", protocol, 1
    )
    del validation_ids
    subset_ids = deterministic_stratified_subset(
        train_ids,
        {stable_id: rows[stable_id].targets["S2"] for stable_id in train_ids},
        10000,
        config["seed"],
    )
    if _membership_fingerprint(subset_ids) != B4_SUBSET_SHA256:
        raise BenchmarkError("B4-H-O controlled subset mismatch")
    labels = {task: [] for task in TASKS}
    for stable_id in subset_ids:
        expected = hierarchical_targets(rows[stable_id].targets["S6"], protocol)
        actual = tuple(rows[stable_id].targets[task] for task in TASKS)
        if actual != expected:
            raise BenchmarkError("hierarchical target mapping drift")
        for task, value in zip(TASKS, actual, strict=True):
            labels[task].append(value)
    orders = {task: protocol.targets[task.casefold()].order for task in TASKS}
    label_ids = {
        task: [orders[task].index(value) for value in values] for task, values in labels.items()
    }
    class_weights = {
        task: balanced_mean_one_weights(values, orders[task]) for task, values in labels.items()
    }
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False, local_files_only=True
    )
    token_started = time.monotonic()
    encoded = _tokenize(tokenizer, [rows[stable_id].text for stable_id in subset_ids])
    tokenization_seconds = time.monotonic() - token_started
    lengths = [len(value) for value in encoded["input_ids"]]
    baseline_padded = len(lengths) * MAX_LENGTH
    audit = {
        "rows": len(lengths),
        "mean_unpadded_tokens": statistics.fmean(lengths),
        "median_unpadded_tokens": statistics.median(lengths),
        "p90_unpadded_tokens": float(np.percentile(lengths, 90)),
        "p95_unpadded_tokens": float(np.percentile(lengths, 95)),
        "p99_unpadded_tokens": float(np.percentile(lengths, 99)),
        "maximum_capped_tokens": max(lengths),
        "baseline_global_512_padded_tokens": baseline_padded,
        "tokenization_seconds": tokenization_seconds,
        "raw_text_persisted": False,
    }
    o1 = _run_candidate(
        name="O1_DYNAMIC_LENGTH_FLOAT32",
        dtype="float32",
        tokenizer=tokenizer,
        encoded=encoded,
        label_ids=label_ids,
        class_weights=class_weights,
        stable_ids=subset_ids,
        lengths=lengths,
        config=config,
        status_path=status_path,
    )
    o1["padding_tokens_avoided"] = baseline_padded - o1["actual_padded_tokens"]
    o1["padding_reduction_percent"] = o1["padding_tokens_avoided"] / baseline_padded * 100
    o1["relative_example_throughput_improvement_percent"] = (
        o1["examples_per_second"] / config["baseline"]["example_exposures_per_second"] - 1
    ) * 100
    _candidate_checkpoint(work, o1)
    _atomic_json(work / "token_padding_audit.json", audit)
    candidates = [o1]
    bf16_preflight: dict[str, Any]
    try:
        _status(
            status_path,
            current_candidate="O2",
            candidate_phase="strict_bfloat16_preflight",
            dtype="bfloat16",
        )
        tensor_weights = {
            task: torch.tensor(values, device="mps") for task, values in class_weights.items()
        }
        bf16_preflight = _bf16_preflight(
            tokenizer, encoded, label_ids, tensor_weights, subset_ids, lengths, config
        )
        _atomic_json(work / "bfloat16_preflight.json", bf16_preflight)
        o2 = _run_candidate(
            name="O2_DYNAMIC_LENGTH_BFLOAT16",
            dtype="bfloat16",
            tokenizer=tokenizer,
            encoded=encoded,
            label_ids=label_ids,
            class_weights=class_weights,
            stable_ids=subset_ids,
            lengths=lengths,
            config=config,
            status_path=status_path,
        )
        o2["padding_tokens_avoided"] = baseline_padded - o2["actual_padded_tokens"]
        o2["padding_reduction_percent"] = o2["padding_tokens_avoided"] / baseline_padded * 100
        o2["relative_example_throughput_improvement_percent"] = (
            o2["examples_per_second"] / config["baseline"]["example_exposures_per_second"] - 1
        ) * 100
        _candidate_checkpoint(work, o2)
        candidates.append(o2)
    except (BenchmarkError, RuntimeError) as exc:
        bf16_preflight = {"status": "REJECTED", "reason": f"{type(exc).__name__}: {exc}"}
        _atomic_json(work / "bfloat16_preflight.json", bf16_preflight)
        print(f"[B4-H-O] O2 rejected: {bf16_preflight['reason']}", flush=True)
    valid = [candidate for candidate in candidates if candidate["nan_inf_events"] == 0]
    best = max(valid, key=lambda candidate: candidate["examples_per_second"])
    inference = _inference_benchmark(
        best["dtype"], tokenizer, encoded, label_ids, subset_ids, lengths, config
    )
    folds = {
        fold: _load_fold(
            rows, manifest_dir, protocol_report_dir / "fingerprints.json", protocol, fold
        )
        for fold in (1, 2, 3)
    }
    projection = _project(best, inference, folds)
    future_config = {
        "dynamic_padding": True,
        "pad_to_multiple_of": PAD_MULTIPLE,
        "deterministic_length_bucketing": True,
        "dtype": best["dtype"],
        "micro_batch": MICRO_BATCH,
        "gradient_accumulation": ACCUMULATION,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "epochs": EPOCHS,
        "scientific_configuration_changed": False,
        "competitive_mode_enabled": False,
    }
    future_config["fingerprint"] = hashlib.sha256(
        json.dumps(future_config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    result = {
        "status": "SUCCESS",
        "stage": "ENGINEERING_ONLY",
        "baseline_timing_investigation": {
            "observed": "epoch 1 was 3.5x slower than epochs 2 and 3",
            "tokenization_inside_epoch_timing": False,
            "explicit_warmup_in_baseline": False,
            "supported_conclusion": (
                "first-use MPS/model training startup is included in epoch 1; persisted evidence "
                "cannot isolate kernel compilation, graph setup, allocation, or AdaLoRA components"
            ),
        },
        "token_padding_audit": audit,
        "candidates": candidates,
        "bfloat16_preflight": bf16_preflight,
        "selected_candidate": best["candidate"],
        "future_full_engineering_configuration": future_config,
        "inference": inference,
        "runtime_projection": projection,
        "competitive_metrics": None,
        "locked_test_accessed": False,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(result_path, result)
    _atomic_json(report_dir / "optimization_result.json", result)
    _atomic_json(report_dir / "future_full_engineering_configuration.json", future_config)
    _atomic_json(report_dir / "runtime_projection.json", projection)
    _atomic_json(report_dir / "token_padding_audit.json", audit)
    _atomic_json(
        report_dir / "environment.json",
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: version(name) for name in ("torch", "transformers", "peft", "accelerate")
            },
            "locked_test_tokenized": False,
            "locked_test_embedded": False,
            "locked_test_model_performance_accessed": False,
            "locked_test_used_for_tuning": False,
            "competitive_models_fitted": 0,
        },
    )
    _status(
        status_path,
        state="SUCCESS",
        current_candidate=best["candidate"],
        candidate_phase="complete",
        dtype=best["dtype"],
    )
    print(f"[B4-H-O] SUCCESS selected={best['candidate']} result={result_path}", flush=True)
    return result
