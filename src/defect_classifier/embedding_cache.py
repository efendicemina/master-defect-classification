"""Provenance-checked project-sharded cache for frozen development embeddings."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


class EmbeddingCacheError(RuntimeError):
    """Raised when a semantic embedding cache cannot be trusted."""


def stable_id(project: str, issue_id: str) -> str:
    return f"{project}:{issue_id}"


def provenance_fingerprint(provenance: dict[str, Any]) -> str:
    payload = json.dumps(provenance, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def validate_provenance(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    if provenance_fingerprint(actual) != provenance_fingerprint(expected):
        raise EmbeddingCacheError("embedding cache provenance mismatch")


def validate_membership(
    cache_ids: set[str], development_ids: set[str], locked_ids: set[str] | None = None
) -> None:
    if cache_ids != development_ids:
        raise EmbeddingCacheError("embedding cache is not the exact frozen DEVELOPMENT membership")
    if locked_ids is not None and cache_ids & locked_ids:
        raise EmbeddingCacheError("embedding cache intersects locked membership")


def validate_embeddings(values: np.ndarray, dimension: int, normalized: bool = True) -> None:
    if values.dtype != np.float32 or values.ndim != 2 or values.shape[1] != dimension:
        raise EmbeddingCacheError("embedding dtype, rank, or dimension mismatch")
    if not np.isfinite(values).all():
        raise EmbeddingCacheError("non-finite embedding detected")
    if normalized and not np.allclose(np.linalg.norm(values, axis=1), 1.0, atol=2e-4):
        raise EmbeddingCacheError("embedding normalization check failed")


def shard_path(cache_root: Path, encoder_id: str, project: str) -> Path:
    return cache_root / encoder_id / "shards" / f"{project}.parquet"


def write_shard(
    path: Path, project: str, issue_ids: list[str], embeddings: np.ndarray
) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    validate_embeddings(embeddings, embeddings.shape[1])
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = pa.array(embeddings.reshape(-1), type=pa.float32())
    table = pa.table(
        {
            "source_project": pa.array([project] * len(issue_ids)),
            "issue_id": pa.array(issue_ids),
            "embedding": pa.FixedSizeListArray.from_arrays(flat, embeddings.shape[1]),
        }
    )
    temporary = path.with_name(f".{path.name}.tmp")
    pq.write_table(table, temporary, compression="zstd", use_dictionary=["source_project"])
    os.replace(temporary, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "project": project,
        "row_count": len(issue_ids),
        "embedding_dimension": embeddings.shape[1],
        "relative_path": str(path),
        "sha256": digest,
        "size_bytes": path.stat().st_size,
    }


def read_shard(path: Path, dimension: int) -> tuple[list[str], np.ndarray]:
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["source_project", "issue_id", "embedding"])
    projects = table["source_project"].to_pylist()
    issues = table["issue_id"].to_pylist()
    embedding_column = table["embedding"].combine_chunks()
    embeddings = embedding_column.values.to_numpy(zero_copy_only=False).reshape(
        len(embedding_column), dimension
    )
    embeddings = embeddings.astype(np.float32, copy=False)
    validate_embeddings(embeddings, dimension)
    return [
        stable_id(project, issue) for project, issue in zip(projects, issues, strict=True)
    ], embeddings


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
