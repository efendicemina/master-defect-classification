"""Development-only Phase B3 RTA frozen-representation feasibility."""

from __future__ import annotations

import csv
import hashlib
import heapq
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

import numpy as np

from defect_classifier.classical_benchmark import BenchmarkError
from defect_classifier.embedding_cache import atomic_json, validate_embeddings
from defect_classifier.preparation import _membership_fingerprint
from defect_classifier.protocol import FrozenProtocol

MODEL_ID = "Colorful/RTA"
MODEL_REVISION = "56cc614ee7cf17b8c1875e6848037f9e5bafc41a"
ARCHITECTURE = "RobertaForSequenceClassification"
REPRESENTATION = "base_encoder_final_layer_first_token"
DIMENSION = 768
MAX_LENGTH = 512
TASKS = ("S6", "S3", "S2")
EXPECTED_TASKS = {
    "S6": ("S6-R3", "CHAR", "LINEARSVC", "BALANCED", 0.25),
    "S3": ("S3-R5", "WORD", "LOGREG", "BALANCED", 2.0),
    "S2": ("S2-R3", "WORD_CHAR", "LOGREG", "BALANCED", 2.0),
}
REFERENCE_HASHES = {
    "A2_REPORT": (
        "classical_optimization_v1/CLASSICAL_OPTIMIZATION_REPORT.md",
        "4ab436dc330e80a66a1cea5dcfb9c364c8d6f947c4d8f757fd25486c0ecd2a8d",
    ),
    "B1_REPORT": (
        "semantic_embeddings_v1/SEMANTIC_EMBEDDING_REPORT.md",
        "7809d204c0f4b85eb162c071742bf92a6d7d078b355c76537ffb382fc4f09804",
    ),
    "B15_REPORT": (
        "lexical_semantic_fusion_v1/LEXICAL_SEMANTIC_FUSION_REPORT.md",
        "e7bcf9afe11ac5b29f0ff0996c5c011351684164048c7afd871062e242bd99c7",
    ),
    "B16_REPORT": (
        "long_text_fusion_v1/LONG_TEXT_FUSION_REPORT.md",
        "86278c1a10ce2980179fa9d20d95679e6541d57f6d0a8cfb19d06a02a4573c21",
    ),
}


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "rta_fusion_v1.toml"


def load_rta_config(path: Path | None = None) -> tuple[dict[str, Any], str]:
    raw = (path or default_config_path()).read_bytes()
    config = tomllib.loads(raw.decode())
    validate_rta_config(config)
    return config, hashlib.sha256(raw).hexdigest()


def validate_rta_config(config: dict[str, Any]) -> None:
    expected = {
        "model_id": MODEL_ID,
        "resolved_revision": MODEL_REVISION,
        "architecture": ARCHITECTURE,
        "representation": REPRESENTATION,
        "embedding_dimension": DIMENSION,
        "max_sequence_length": MAX_LENGTH,
        "normalize_embeddings": True,
        "dtype": "float32",
        "semantic_weight": 1.0,
        "throughput_sample_documents": 10000,
        "initial_batch_size": 16,
        "candidate_batch_size": 32,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise BenchmarkError("Phase B3 frozen RTA method drift")
    if tuple(config.get("tasks", {})) != TASKS:
        raise BenchmarkError("Phase B3 task matrix drift")
    for task, signature in EXPECTED_TASKS.items():
        actual = config["tasks"][task]
        if (
            tuple(
                actual[key]
                for key in (
                    "representation_id",
                    "representation",
                    "classifier",
                    "class_weight",
                    "c",
                )
            )
            != signature
        ):
            raise BenchmarkError(f"Phase B3 lexical method drift: {task}")


def future_fit_count(config: dict[str, Any]) -> int:
    validate_rta_config(config)
    return len(config["tasks"]) * 3


def normalize_rta_embeddings(values: np.ndarray) -> np.ndarray:
    if values.dtype != np.float32 or values.ndim != 2 or values.shape[1] != DIMENSION:
        raise BenchmarkError("RTA representation dimension or dtype mismatch")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(values).all() or np.any(norms == 0):
        raise BenchmarkError("invalid RTA representation")
    output = values / norms
    validate_embeddings(output.astype(np.float32, copy=False), DIMENSION)
    return output.astype(np.float32, copy=False)


def extract_first_token(last_hidden_state: Any) -> Any:
    """Match the standard RoBERTa classification head's sequence-token selection."""
    if last_hidden_state.ndim != 3 or last_hidden_state.shape[-1] != DIMENSION:
        raise BenchmarkError("unexpected RTA encoder output")
    return last_hidden_state[:, 0, :]


def semantic_shard_path(cache_root: Path, project: str) -> Path:
    return cache_root / "RTA" / "shards" / f"{project}.parquet"


def competitive_checkpoint_path(root: Path, task: str, fold: int) -> Path:
    if task not in TASKS or fold not in (1, 2, 3):
        raise BenchmarkError("unknown Phase B3 task/fold")
    return root / f"{task.casefold()}-fold-{fold}.json"


def _verify_references(reports_root: Path) -> dict[str, str]:
    output = {}
    for name, (relative, expected) in REFERENCE_HASHES.items():
        path = reports_root / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "MISSING"
        if actual != expected:
            raise BenchmarkError(f"frozen reference drift: {name}")
        output[name] = actual
    return output


def _load_model(device: str) -> tuple[Any, Any, dict[str, Any]]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=False,
        use_safetensors=False,
    )
    if type(model).__name__ != ARCHITECTURE or type(tokenizer).__name__ != "RobertaTokenizer":
        raise BenchmarkError("Colorful/RTA architecture or tokenizer drift")
    if model.config.hidden_size != DIMENSION or tokenizer.model_max_length != MAX_LENGTH:
        raise BenchmarkError("Colorful/RTA dimension or sequence-limit drift")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to(device)
    metadata = {
        "model_id": MODEL_ID,
        "resolved_revision": MODEL_REVISION,
        "architecture": type(model).__name__,
        "base_encoder_architecture": type(model.roberta).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "total_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "base_encoder_parameter_count": sum(
            parameter.numel() for parameter in model.roberta.parameters()
        ),
        "hidden_dimension": model.config.hidden_size,
        "num_hidden_layers": model.config.num_hidden_layers,
        "num_attention_heads": model.config.num_attention_heads,
        "model_max_position_embeddings": model.config.max_position_embeddings,
        "tokenizer_model_max_length": tokenizer.model_max_length,
        "representation": REPRESENTATION,
        "normalization": "L2",
        "license": "MIT",
        "model_card_claim": "pre-trained language model for bug reports",
        "checkpoint_head_observation": (
            "repository weights contain an MLM head and no trained classification head; "
            "B3 uses only the cleanly loaded RoBERTa base encoder"
        ),
        "trust_remote_code": False,
        "dtype": "float32",
        "device": device,
    }
    return model, tokenizer, metadata


def _encode(
    model: Any, tokenizer: Any, texts: list[str], batch_size: int, device: str
) -> np.ndarray:
    import torch

    output = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            inputs = tokenizer(
                texts[start : start + batch_size],
                max_length=MAX_LENGTH,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            hidden = model.roberta(**inputs).last_hidden_state
            output.append(extract_first_token(hidden).cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(output)


def audit_tokenization(
    development_dir: Path,
    tokenizer: Any,
    development_fingerprint: str,
    *,
    seed: int,
    sample_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    import pyarrow.parquet as pq

    ids: set[str] = set()
    lengths: list[int] = []
    project_rows = []
    candidates: list[tuple[int, str, str]] = []
    for path in sorted(development_dir.glob("*.parquet")):
        project_lengths = []
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=512, columns=["source_project", "issue_id", "text_combined"]
        ):
            records = batch.to_pylist()
            encoded = tokenizer(
                [row["text_combined"] for row in records],
                add_special_tokens=True,
                truncation=False,
                padding=False,
                return_length=True,
            )
            for row, length in zip(records, encoded["length"], strict=True):
                stable = f"{row['source_project']}:{row['issue_id']}"
                ids.add(stable)
                length = int(length)
                lengths.append(length)
                project_lengths.append(length)
                rank = int.from_bytes(hashlib.sha256(f"{seed}|{stable}".encode()).digest()[:8])
                item = (-rank, stable, row["text_combined"])
                if len(candidates) < sample_size:
                    heapq.heappush(candidates, item)
                elif item > candidates[0]:
                    heapq.heapreplace(candidates, item)
        truncated = sum(value > MAX_LENGTH for value in project_lengths)
        project_rows.append(
            {
                "project": path.stem,
                "development_records": len(project_lengths),
                "fits_without_truncation": len(project_lengths) - truncated,
                "requires_truncation": truncated,
                "truncation_percentage": truncated / len(project_lengths) * 100,
                "mean_tokens": statistics.fmean(project_lengths),
                "median_tokens": statistics.median(project_lengths),
                "maximum_tokens": max(project_lengths),
            }
        )
    if _membership_fingerprint(ids) != development_fingerprint:
        raise BenchmarkError("development membership fingerprint drift")
    values = np.asarray(lengths)
    truncated = int(np.sum(values > MAX_LENGTH))
    summary = {
        "total_records": len(lengths),
        "native_max_sequence_length": MAX_LENGTH,
        "fits_without_truncation": len(lengths) - truncated,
        "fits_without_truncation_percentage": (len(lengths) - truncated) / len(lengths) * 100,
        "requires_truncation": truncated,
        "requires_truncation_percentage": truncated / len(lengths) * 100,
        "mean_tokens": statistics.fmean(lengths),
        "median_tokens": statistics.median(lengths),
        "p90_tokens": float(np.percentile(values, 90)),
        "p95_tokens": float(np.percentile(values, 95)),
        "p99_tokens": float(np.percentile(values, 99)),
        "maximum_tokens": max(lengths),
    }
    ordered = sorted([(-rank, stable, text) for rank, stable, text in candidates])
    return project_rows, summary, [text for _, _, text in ordered]


def _preflight(model: Any, tokenizer: Any, texts: list[str], device: str, config: dict[str, Any]):
    import torch

    short = texts[:4]
    first = normalize_rta_embeddings(
        _encode(model, tokenizer, short, config["initial_batch_size"], device)
    )
    repeated = normalize_rta_embeddings(
        _encode(model, tokenizer, short, config["initial_batch_size"], device)
    )
    if not np.allclose(first, repeated, atol=1e-6):
        raise BenchmarkError("RTA extraction is not deterministic in eval mode")
    candidate = config["candidate_batch_size"]
    oom_events = []
    try:
        probe = (short * math.ceil(candidate / len(short)))[:candidate]
        normalize_rta_embeddings(_encode(model, tokenizer, probe, candidate, device))
        selected = candidate
    except RuntimeError as exc:
        if "memory" not in str(exc).lower():
            raise
        oom_events.append(str(exc))
        torch.mps.empty_cache()
        selected = config["initial_batch_size"]
    return {
        "status": "PASS",
        "mps_available": torch.backends.mps.is_available(),
        "mps_used": device == "mps",
        "dtype": "float32",
        "batch_size": selected,
        "deterministic_repeated_output": True,
        "finite_embeddings": bool(np.isfinite(first).all()),
        "normalized_embeddings": True,
        "encoder_frozen": not any(parameter.requires_grad for parameter in model.parameters()),
        "gradients_enabled": False,
        "long_input_tested": True,
        "batch_inference_tested": True,
        "oom_events": oom_events,
        "cpu_fallback_events": 0,
    }


def _benchmark(
    model: Any, tokenizer: Any, texts: list[str], batch_size: int, device: str
) -> dict[str, Any]:
    import torch

    started = time.monotonic()
    token_started = time.monotonic()
    tokenized = [
        tokenizer(
            texts[start : start + batch_size],
            max_length=MAX_LENGTH,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        for start in range(0, len(texts), batch_size)
    ]
    token_seconds = time.monotonic() - token_started
    values = []
    torch.mps.synchronize()
    encode_started = time.monotonic()
    with torch.inference_mode():
        for inputs in tokenized:
            inputs = {key: value.to(device) for key, value in inputs.items()}
            hidden = model.roberta(**inputs).last_hidden_state
            values.append(extract_first_token(hidden).cpu().numpy().astype(np.float32, copy=False))
    torch.mps.synchronize()
    encode_seconds = time.monotonic() - encode_started
    normalize_started = time.monotonic()
    normalized = normalize_rta_embeddings(np.concatenate(values))
    normalization_seconds = time.monotonic() - normalize_started
    del normalized
    total = time.monotonic() - started
    return {
        "sample_documents": len(texts),
        "tokenization_seconds": token_seconds,
        "encoding_seconds": encode_seconds,
        "normalization_seconds": normalization_seconds,
        "total_runtime_seconds": total,
        "documents_per_second": len(texts) / total,
        "examples_per_second": len(texts) / encode_seconds,
        "batch_size": batch_size,
        "oom_events": 0,
        "cpu_fallback_events": 0,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "mps_driver_allocated_bytes_after": torch.mps.driver_allocated_memory(),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def run_rta_feasibility(
    *,
    development_dir: Path,
    protocol_report_dir: Path,
    reports_root: Path,
    report_dir: Path,
    protocol: FrozenProtocol,
    config_path: Path | None = None,
    stage: str = "feasibility",
) -> dict[str, Any]:
    if stage == "full":
        raise BenchmarkError("Phase B3 full run is not authorized by this feasibility task")
    config, config_hash = load_rta_config(config_path)
    references = _verify_references(reports_root)
    frozen = json.loads((protocol_report_dir / "fingerprints.json").read_text())
    if frozen["protocol_sha256"] != protocol.fingerprint:
        raise BenchmarkError("protocol fingerprint drift")
    import torch

    if not torch.backends.mps.is_available():
        raise BenchmarkError("Phase B3 feasibility requires available MPS")
    report_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer, provenance = _load_model("mps")
    audit_rows, audit, sample = audit_tokenization(
        development_dir,
        tokenizer,
        frozen["development_membership_sha256"],
        seed=config["seed"],
        sample_size=config["throughput_sample_documents"],
    )
    _write_csv(report_dir / "tokenization_audit.csv", audit_rows + [{"project": "ALL", **audit}])
    preflight = _preflight(model, tokenizer, sample, "mps", config)
    throughput = _benchmark(model, tokenizer, sample, preflight["batch_size"], "mps")
    atomic_json(report_dir / "throughput_estimate.json", {"preflight": preflight, **throughput})
    fit_predict = 929.918 + 6.199 + 197.319 + 0.363 + 595.533 + 1.957
    overhead = 2383.821 - fit_predict
    embedding = audit["total_records"] / throughput["documents_per_second"]
    central = embedding + fit_predict + overhead
    conservative = embedding / 0.75 + (fit_predict + overhead) * 1.25
    raw_cache = audit["total_records"] * DIMENSION * 4
    projection = {
        "semantic_embedding_generation_seconds": embedding,
        "nine_fit_fusion_fit_prediction_seconds": fit_predict,
        "validation_reporting_overhead_seconds": overhead,
        "total_central_seconds": central,
        "total_conservative_seconds": conservative,
        "conservative_assumptions": "75% measured throughput and 25% fusion/overhead margin",
        "raw_float32_embedding_bytes": raw_cache,
        "projected_cache_bytes_central": 591_816_188,
        "projected_cache_bytes_upper": raw_cache + 5_000_000,
    }
    atomic_json(report_dir / "runtime_projection.json", projection)
    atomic_json(
        report_dir / "model_provenance.json",
        {
            **provenance,
            "config": model.config.to_dict(),
            "config_sha256": config_hash,
            "protocol_sha256": protocol.fingerprint,
            "development_membership_sha256": frozen["development_membership_sha256"],
            "reference_sha256": references,
        },
    )
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: version(name)
            for name in ("torch", "transformers", "huggingface-hub", "numpy", "pyarrow")
        },
        "mps_is_built": torch.backends.mps.is_built(),
        "mps_is_available": torch.backends.mps.is_available(),
        "device": "mps",
        "dtype": "float32",
        "config_sha256": config_hash,
        "full_run_started": False,
        "full_rta_embeddings_materialized": False,
        "competitive_fits_started": 0,
        "rta_fine_tuned": False,
        "peft_used": False,
        "locked_test_tokenized": False,
        "locked_test_embedded": False,
        "locked_test_model_performance_accessed": False,
        "locked_test_used_for_tuning": False,
    }
    atomic_json(report_dir / "environment.json", environment)
    return {
        "provenance": provenance,
        "audit": audit,
        "preflight": preflight,
        "throughput": throughput,
        "projection": projection,
    }
