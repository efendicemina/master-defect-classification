"""Development-only Phase B1.6 long-text MPNet feasibility utilities."""

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
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from defect_classifier.classical_benchmark import BenchmarkError
from defect_classifier.embedding_cache import atomic_json, validate_embeddings
from defect_classifier.preparation import _membership_fingerprint
from defect_classifier.protocol import FrozenProtocol

MODEL_ID = "sentence-transformers/all-mpnet-base-v2"
MODEL_REVISION = "e8c3b32edf5434bc2275fc9bab85f82640a19130"
EMBEDDING_DIMENSION = 768
CHUNK_LENGTH = 384
CHUNK_OVERLAP = 64
SPECIAL_TOKENS = 2
CONTENT_LENGTH = CHUNK_LENGTH - SPECIAL_TOKENS
CHUNK_STEP = CONTENT_LENGTH - CHUNK_OVERLAP
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
}


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "long_text_fusion_v1.toml"


def load_long_text_config(path: Path | None = None) -> tuple[dict[str, Any], str]:
    raw = (path or default_config_path()).read_bytes()
    config = tomllib.loads(raw.decode())
    validate_long_text_config(config)
    return config, hashlib.sha256(raw).hexdigest()


def validate_long_text_config(config: dict[str, Any]) -> None:
    fixed = {
        "model_id": MODEL_ID,
        "resolved_revision": MODEL_REVISION,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "chunk_length": CHUNK_LENGTH,
        "chunk_overlap": CHUNK_OVERLAP,
        "batch_size": 64,
        "normalize_chunks": True,
        "document_pooling": "arithmetic_mean_then_l2",
        "semantic_weight": 1.0,
        "target_benchmark_chunks": 10000,
    }
    if any(config.get(key) != value for key, value in fixed.items()):
        raise BenchmarkError("Phase B1.6 fixed long-text method drift")
    if tuple(config.get("tasks", {})) != TASKS:
        raise BenchmarkError("Phase B1.6 task matrix drift")
    for task, expected in EXPECTED_TASKS.items():
        actual = config["tasks"][task]
        signature = tuple(
            actual[key]
            for key in ("representation_id", "representation", "classifier", "class_weight", "c")
        )
        if signature != expected:
            raise BenchmarkError(f"Phase B1.6 task method drift: {task}")


def future_fit_count(config: dict[str, Any]) -> int:
    validate_long_text_config(config)
    return len(config["tasks"]) * 3


def chunk_token_ids(token_ids: list[int]) -> list[list[int]]:
    """Create complete deterministic content windows; model special tokens are added later."""
    if len(token_ids) <= CONTENT_LENGTH:
        return [token_ids]
    return [
        token_ids[index * CHUNK_STEP : index * CHUNK_STEP + CONTENT_LENGTH]
        for index in range(chunk_count(len(token_ids)))
    ]


def chunk_count(token_count: int) -> int:
    if token_count <= CONTENT_LENGTH:
        return 1
    return 1 + math.ceil((token_count - CONTENT_LENGTH) / CHUNK_STEP)


def mean_pool_documents(chunk_embeddings: np.ndarray, owners: list[int], count: int) -> np.ndarray:
    validate_embeddings(chunk_embeddings, EMBEDDING_DIMENSION)
    pooled = np.zeros((count, EMBEDDING_DIMENSION), dtype=np.float32)
    counts = np.zeros(count, dtype=np.int64)
    for embedding, owner in zip(chunk_embeddings, owners, strict=True):
        pooled[owner] += embedding
        counts[owner] += 1
    if np.any(counts == 0):
        raise BenchmarkError("document without a chunk")
    pooled /= counts[:, None]
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise BenchmarkError("zero pooled document embedding")
    pooled /= norms
    validate_embeddings(pooled, EMBEDDING_DIMENSION)
    return pooled


def semantic_shard_path(cache_root: Path, project: str) -> Path:
    return cache_root / "B16-MPNET" / "shards" / f"{project}.parquet"


def competitive_checkpoint_path(checkpoint_root: Path, task: str, fold: int) -> Path:
    if task not in TASKS or fold not in (1, 2, 3):
        raise BenchmarkError("unknown Phase B1.6 task/fold")
    return checkpoint_root / f"{task.casefold()}-fold-{fold}.json"


def _verify_references(reports_root: Path) -> dict[str, str]:
    output = {}
    for name, (relative, expected) in REFERENCE_HASHES.items():
        path = reports_root / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "MISSING"
        if actual != expected:
            raise BenchmarkError(f"frozen reference drift: {name}")
        output[name] = actual
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        fields = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _bucket(count: int) -> str:
    return str(count) if count <= 5 else ">5"


def audit_development_chunks(
    development_dir: Path,
    tokenizer: Any,
    expected_membership: str,
    *,
    seed: int,
    candidate_limit: int = 12000,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[tuple[str, str, int]]]:
    import pyarrow.parquet as pq

    totals: Counter[str] = Counter()
    project_counts: dict[str, Counter[str]] = defaultdict(Counter)
    project_chunks: Counter[str] = Counter()
    project_documents: Counter[str] = Counter()
    all_counts: list[int] = []
    development_ids: set[str] = set()
    candidates: list[tuple[int, str, str, int]] = []
    for path in sorted(development_dir.glob("*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=512, columns=["source_project", "issue_id", "text_combined"]
        ):
            records = batch.to_pylist()
            encoded = tokenizer(
                [row["text_combined"] for row in records],
                add_special_tokens=False,
                truncation=False,
                padding=False,
            )["input_ids"]
            for row, token_ids in zip(records, encoded, strict=True):
                stable = f"{row['source_project']}:{row['issue_id']}"
                development_ids.add(stable)
                count = chunk_count(len(token_ids))
                bucket = _bucket(count)
                totals[bucket] += 1
                project_counts[row["source_project"]][bucket] += 1
                project_chunks[row["source_project"]] += count
                project_documents[row["source_project"]] += 1
                all_counts.append(count)
                rank = int.from_bytes(hashlib.sha256(f"{seed}|{stable}".encode()).digest()[:8])
                item = (-rank, stable, row["text_combined"], count)
                if len(candidates) < candidate_limit:
                    heapq.heappush(candidates, item)
                elif item > candidates[0]:
                    heapq.heapreplace(candidates, item)
    if _membership_fingerprint(development_ids) != expected_membership:
        raise BenchmarkError("development membership fingerprint drift")
    rows = []
    for project in sorted(project_documents):
        counts = project_counts[project]
        rows.append(
            {
                "project": project,
                "total_records": project_documents[project],
                "one_chunk": counts["1"],
                "two_chunks": counts["2"],
                "three_chunks": counts["3"],
                "four_chunks": counts["4"],
                "five_chunks": counts["5"],
                "over_five_chunks": counts[">5"],
                "total_chunks": project_chunks[project],
                "mean_chunks": project_chunks[project] / project_documents[project],
            }
        )
    summary = {
        "total_records": len(all_counts),
        "one_chunk": totals["1"],
        "two_chunks": totals["2"],
        "three_chunks": totals["3"],
        "four_chunks": totals["4"],
        "five_chunks": totals["5"],
        "over_five_chunks": totals[">5"],
        "maximum_chunks": max(all_counts),
        "mean_chunks": statistics.fmean(all_counts),
        "median_chunks": statistics.median(all_counts),
        "total_chunks": sum(all_counts),
        "additional_chunks_vs_b1": sum(all_counts) - len(all_counts),
    }
    ordered = sorted([(-rank, stable, text, count) for rank, stable, text, count in candidates])
    return rows, summary, [(stable, text, count) for _, stable, text, count in ordered]


def _load_frozen_model(device: str) -> tuple[Any, Any]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        MODEL_ID,
        revision=MODEL_REVISION,
        device=device,
        trust_remote_code=False,
        local_files_only=True,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if model.get_embedding_dimension() != EMBEDDING_DIMENSION:
        raise BenchmarkError("MPNet embedding dimension drift")
    return model[0].auto_model, model.tokenizer


def benchmark_chunks(
    model: Any,
    tokenizer: Any,
    candidates: list[tuple[str, str, int]],
    target_chunks: int,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    import torch

    selected = []
    planned = 0
    for row in candidates:
        selected.append(row)
        planned += row[2]
        if planned >= target_chunks:
            break
    construction_started = time.monotonic()
    chunks: list[list[int]] = []
    owners: list[int] = []
    for owner, (_, text, expected_count) in enumerate(selected):
        ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
        windows = chunk_token_ids(ids)
        if len(windows) != expected_count:
            raise BenchmarkError("chunk audit/benchmark disagreement")
        chunks.extend(windows)
        owners.extend([owner] * len(windows))
    construction_seconds = time.monotonic() - construction_started
    initial_batch_size = batch_size
    reductions = []
    while True:
        values = []
        try:
            if device == "mps":
                torch.mps.synchronize()
            encoding_started = time.monotonic()
            with torch.inference_mode():
                for start in range(0, len(chunks), batch_size):
                    prepared = [
                        {
                            "input_ids": [tokenizer.bos_token_id, *ids, tokenizer.eos_token_id],
                            "attention_mask": [1] * (len(ids) + SPECIAL_TOKENS),
                        }
                        for ids in chunks[start : start + batch_size]
                    ]
                    inputs = tokenizer.pad(prepared, padding=True, return_tensors="pt")
                    inputs = {key: value.to(device) for key, value in inputs.items()}
                    output = model(**inputs).last_hidden_state
                    mask = inputs["attention_mask"].unsqueeze(-1).expand(output.size()).float()
                    pooled = (output * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                    values.append(pooled.cpu().numpy().astype(np.float32, copy=False))
            if device == "mps":
                torch.mps.synchronize()
            encoding_seconds = time.monotonic() - encoding_started
            break
        except RuntimeError as exc:
            if device != "mps" or batch_size <= 1 or "memory" not in str(exc).lower():
                raise
            reductions.append({"from": batch_size, "to": batch_size // 2, "error": str(exc)})
            batch_size //= 2
            torch.mps.empty_cache()
    chunk_embeddings = np.concatenate(values)
    pooling_started = time.monotonic()
    document_embeddings = mean_pool_documents(chunk_embeddings, owners, len(selected))
    pooling_seconds = time.monotonic() - pooling_started
    del document_embeddings
    total_seconds = construction_seconds + encoding_seconds + pooling_seconds
    return {
        "sample_documents": len(selected),
        "sample_chunks": len(chunks),
        "tokenization_chunk_construction_seconds": construction_seconds,
        "encoding_seconds": encoding_seconds,
        "document_pooling_seconds": pooling_seconds,
        "total_benchmark_seconds": total_seconds,
        "chunks_per_second": len(chunks) / encoding_seconds,
        "documents_per_second": len(selected) / total_seconds,
        "batch_size_initial": initial_batch_size,
        "batch_size_final": batch_size,
        "batch_size_reductions": reductions,
        "mps_fallback_events": 0,
        "oom_events": len(reductions),
        "encoder_frozen": not any(parameter.requires_grad for parameter in model.parameters()),
        "gradients_enabled": False,
    }


def run_long_text_feasibility(
    *,
    development_dir: Path,
    protocol_report_dir: Path,
    reports_root: Path,
    report_dir: Path,
    protocol: FrozenProtocol,
    config_path: Path | None = None,
    stage: str = "estimate",
    resume: bool = True,
    cache_root: Path | None = None,
    checkpoint_root: Path | None = None,
) -> dict[str, Any]:
    if stage == "full":
        raise BenchmarkError("Phase B1.6 full run is not authorized by the feasibility task")
    config, config_hash = load_long_text_config(config_path)
    del resume, cache_root, checkpoint_root  # Reserved for the authorization-gated full stage.
    references = _verify_references(reports_root)
    frozen = json.loads((protocol_report_dir / "fingerprints.json").read_text())
    if frozen["protocol_sha256"] != protocol.fingerprint:
        raise BenchmarkError("protocol fingerprint drift")
    import torch

    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise BenchmarkError("Phase B1.6 requires available MPS")
    report_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = _load_frozen_model("mps")
    audit_rows, audit, candidates = audit_development_chunks(
        development_dir,
        tokenizer,
        frozen["development_membership_sha256"],
        seed=config["seed"],
    )
    _write_csv(report_dir / "chunking_audit.csv", audit_rows + [{"project": "ALL", **audit}])
    throughput = benchmark_chunks(
        model,
        tokenizer,
        candidates,
        config["target_benchmark_chunks"],
        config["batch_size"],
        "mps",
    )
    atomic_json(report_dir / "throughput_estimate.json", throughput)
    fit_predict_seconds = 929.918 + 6.199 + 197.319 + 0.363 + 595.533 + 1.957
    overhead_seconds = 2383.821 - fit_predict_seconds
    embedding_seconds = audit["total_chunks"] / throughput["chunks_per_second"]
    central = embedding_seconds + fit_predict_seconds + overhead_seconds
    conservative_embedding = embedding_seconds / 0.75
    conservative = conservative_embedding + fit_predict_seconds * 1.25 + overhead_seconds * 1.25
    raw_cache_bytes = audit["total_records"] * EMBEDDING_DIMENSION * 4
    existing_cache_bytes = 591_816_188
    projection = {
        "semantic_embedding_generation_seconds": embedding_seconds,
        "nine_fit_fusion_fit_prediction_seconds": fit_predict_seconds,
        "validation_reporting_overhead_seconds": overhead_seconds,
        "total_central_seconds": central,
        "conservative_embedding_seconds": conservative_embedding,
        "total_conservative_seconds": conservative,
        "conservative_assumptions": "75% measured chunk throughput; 25% fusion/overhead margin",
        "raw_float32_embedding_bytes": raw_cache_bytes,
        "projected_cache_bytes_central": existing_cache_bytes,
        "projected_cache_bytes_upper": raw_cache_bytes + 5_000_000,
        "cache_basis": (
            "same development rows, dimension, dtype, normalization, and shard format as B1 E2"
        ),
    }
    atomic_json(report_dir / "runtime_projection.json", projection)
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: version(name)
            for name in ("torch", "transformers", "sentence-transformers", "numpy", "pyarrow")
        },
        "mps_is_built": torch.backends.mps.is_built(),
        "mps_is_available": torch.backends.mps.is_available(),
        "device": "mps",
        "dtype": "float32",
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "mps_driver_allocated_bytes_after": torch.mps.driver_allocated_memory(),
        "model_id": MODEL_ID,
        "resolved_revision": MODEL_REVISION,
        "config_sha256": config_hash,
        "protocol_sha256": protocol.fingerprint,
        "development_membership_sha256": frozen["development_membership_sha256"],
        "reference_sha256": references,
        "full_run_started": False,
        "full_long_text_embeddings_materialized": False,
        "competitive_fits_started": 0,
        "locked_test_tokenized": False,
        "locked_test_embedded": False,
        "locked_test_model_performance_accessed": False,
    }
    atomic_json(report_dir / "environment.json", environment)
    return {
        "audit": audit,
        "throughput": throughput,
        "projection": projection,
        "environment": environment,
    }
