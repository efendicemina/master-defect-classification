"""Frozen transformer embedding materialization for protocol-v1 DEVELOPMENT data."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
import tomllib
from pathlib import Path
from typing import Any

import numpy as np

from defect_classifier.embedding_cache import (
    EmbeddingCacheError,
    atomic_json,
    provenance_fingerprint,
    read_shard,
    shard_path,
    stable_id,
    validate_embeddings,
    validate_membership,
    validate_provenance,
    write_shard,
)
from defect_classifier.preparation import _membership_fingerprint
from defect_classifier.protocol import FrozenProtocol


def default_semantic_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "semantic_embeddings_v1.toml"


def load_semantic_config(path: Path | None = None) -> tuple[dict[str, Any], str]:
    raw = (path or default_semantic_config_path()).read_bytes()
    config = tomllib.loads(raw.decode())
    validate_semantic_config(config)
    return config, hashlib.sha256(raw).hexdigest()


def validate_semantic_config(config: dict[str, Any]) -> None:
    expected_encoders = [
        ("E1", "sentence-transformers/all-MiniLM-L6-v2"),
        ("E2", "sentence-transformers/all-mpnet-base-v2"),
    ]
    if [(x["id"], x["model_id"]) for x in config["encoders"]] != expected_encoders:
        raise EmbeddingCacheError("semantic encoder matrix drift")
    if config["c"] != 1.0 or config["normalize_embeddings"] is not True:
        raise EmbeddingCacheError("frozen semantic method drift")
    if config["classifiers"] != ["LOGREG", "LINEARSVC"] or config["class_weights"] != [
        "NONE",
        "BALANCED",
    ]:
        raise EmbeddingCacheError("semantic classifier matrix drift")
    if config["tasks"] != ["S6", "S3", "S2"] or config["folds"] != [1, 2, 3]:
        raise EmbeddingCacheError("semantic task/fold matrix drift")


def search_space_size(config: dict[str, Any]) -> int:
    return math.prod(
        len(config[key]) for key in ("tasks", "encoders", "classifiers", "class_weights", "folds")
    )


def mps_preflight() -> dict[str, Any]:
    import torch

    return {
        "mps_is_built": torch.backends.mps.is_built(),
        "mps_is_available": torch.backends.mps.is_available(),
        "selected_device": "mps" if torch.backends.mps.is_available() else "cpu",
        "torch_version": torch.__version__,
    }


def _development_projects(development_dir: Path) -> dict[str, list[dict[str, str]]]:
    import pyarrow.parquet as pq

    projects = {}
    for path in sorted(development_dir.glob("*.parquet")):
        records = pq.read_table(
            path, columns=["source_project", "issue_id", "text_combined"]
        ).to_pylist()
        projects[path.stem] = records
    return projects


def _model_provenance(encoder: dict[str, str], device: str) -> tuple[Any, dict[str, Any]]:
    from huggingface_hub import model_info
    from sentence_transformers import SentenceTransformer

    revision = model_info(encoder["model_id"]).sha
    model = SentenceTransformer(
        encoder["model_id"], revision=revision, device=device, trust_remote_code=False
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    tokenizer = model.tokenizer
    provenance = {
        "encoder_id": encoder["id"],
        "model_id": encoder["model_id"],
        "resolved_revision": revision,
        "embedding_dimension": model.get_sentence_embedding_dimension(),
        "max_sequence_length": model.max_seq_length,
        "tokenizer_class": type(tokenizer).__name__,
        "model_class": type(model[0].auto_model).__name__,
        "pooling_mode": model[1].get_config_dict(),
        "normalize_embeddings": True,
        "dtype": "float32",
        "device": device,
        "trust_remote_code": False,
    }
    return model, provenance


def encoder_smoke(
    encoder: dict[str, str], texts: list[str], device: str, batch_size: int
) -> dict[str, Any]:
    import torch

    model, provenance = _model_provenance(encoder, device)
    with torch.inference_mode():
        values = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32, copy=False)
    validate_embeddings(values, provenance["embedding_dimension"])
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise EmbeddingCacheError("encoder parameter unexpectedly requires gradients")
    return {
        **provenance,
        "sample_count": len(texts),
        "shape": list(values.shape),
        "finite": bool(np.isfinite(values).all()),
        "normalized": True,
        "gradients_enabled": False,
    }


def _token_lengths(tokenizer: Any, texts: list[str], batch_size: int = 256) -> list[int]:
    lengths = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start : start + batch_size],
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_length=True,
        )
        lengths.extend(encoded["length"])
    return lengths


def materialize_encoder(
    *,
    encoder: dict[str, str],
    development_dir: Path,
    cache_root: Path,
    report_dir: Path,
    protocol: FrozenProtocol,
    frozen: dict[str, Any],
    config: dict[str, Any],
    config_fingerprint: str,
    device: str,
    resume: bool,
) -> dict[str, Any]:
    import torch

    started = time.monotonic()
    projects = _development_projects(development_dir)
    development_ids = {
        stable_id(row["source_project"], row["issue_id"])
        for records in projects.values()
        for row in records
    }
    if _membership_fingerprint(development_ids) != frozen["development_membership_sha256"]:
        raise EmbeddingCacheError("development membership fingerprint drift")
    model, model_meta = _model_provenance(encoder, device)
    provenance = {
        "protocol_sha256": protocol.fingerprint,
        "development_membership_sha256": frozen["development_membership_sha256"],
        "text_policy": {
            "fields": list(protocol.text_fields),
            "unicode_normalization": protocol.document["text"]["unicode_normalization"],
            "line_endings": protocol.document["text"]["line_endings"],
            "separator": protocol.text_separator,
        },
        "semantic_config_sha256": config_fingerprint,
        **model_meta,
    }
    metadata_path = cache_root / encoder["id"] / "metadata.json"
    if metadata_path.is_file():
        validate_provenance(json.loads(metadata_path.read_text()), provenance)
    batch_size = config["batch_size"]
    reductions: list[dict[str, Any]] = []
    manifest, truncation = [], []
    for project, records in projects.items():
        path = shard_path(cache_root, encoder["id"], project)
        texts = [row["text_combined"] for row in records]
        lengths = _token_lengths(model.tokenizer, texts)
        truncated = sum(length > model.max_seq_length for length in lengths)
        truncation.append(
            {
                "encoder_id": encoder["id"],
                "model_id": encoder["model_id"],
                "project": project,
                "development_records": len(records),
                "fits_without_truncation": len(records) - truncated,
                "requires_truncation": truncated,
                "truncation_percentage": truncated / len(records) * 100,
                "token_length_min": min(lengths),
                "token_length_mean": sum(lengths) / len(lengths),
                "token_length_max": max(lengths),
                "max_sequence_length": model.max_seq_length,
            }
        )
        if resume and path.is_file():
            ids, values = read_shard(path, model_meta["embedding_dimension"])
            expected = {stable_id(row["source_project"], row["issue_id"]) for row in records}
            validate_membership(set(ids), expected)
            manifest.append(_manifest_record(path, encoder, project, len(ids), values.shape[1]))
            continue
        while True:
            try:
                with torch.inference_mode():
                    values = model.encode(
                        texts,
                        batch_size=batch_size,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                        show_progress_bar=True,
                    ).astype(np.float32, copy=False)
                break
            except RuntimeError as exc:
                if device != "mps" or batch_size <= 1 or "memory" not in str(exc).lower():
                    raise
                new_size = max(1, batch_size // 2)
                reductions.append(
                    {"project": project, "from": batch_size, "to": new_size, "error": str(exc)}
                )
                batch_size = new_size
                torch.mps.empty_cache()
        validate_embeddings(values, model_meta["embedding_dimension"])
        record = write_shard(path, project, [row["issue_id"] for row in records], values)
        record.update({"encoder_id": encoder["id"], "model_id": encoder["model_id"]})
        manifest.append(record)
    cache_ids = set()
    for record in manifest:
        ids, _ = read_shard(Path(record["relative_path"]), model_meta["embedding_dimension"])
        cache_ids.update(ids)
    validate_membership(cache_ids, development_ids)
    atomic_json(metadata_path, provenance)
    result = {
        **model_meta,
        "development_embedding_count": len(cache_ids),
        "runtime_seconds": time.monotonic() - started,
        "batch_size_initial": config["batch_size"],
        "batch_size_final": batch_size,
        "batch_size_reductions": reductions,
        "cache_size_bytes": sum(row["size_bytes"] for row in manifest),
        "cache_membership_matches_development": True,
        "locked_membership_overlap": 0,
        "locked_overlap_basis": "exact cache membership equals frozen DEVELOPMENT partition",
        "provenance_fingerprint": provenance_fingerprint(provenance),
        "manifest": manifest,
        "truncation": truncation,
    }
    atomic_json(report_dir / ".work" / f"{encoder['id']}_materialization.json", result)
    return result


def _manifest_record(
    path: Path, encoder: dict[str, str], project: str, rows: int, dimension: int
) -> dict[str, Any]:
    return {
        "encoder_id": encoder["id"],
        "model_id": encoder["model_id"],
        "project": project,
        "row_count": rows,
        "embedding_dimension": dimension,
        "relative_path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def write_materialization_reports(report_dir: Path, results: list[dict[str, Any]]) -> None:
    atomic_json(
        report_dir / "embedding_models.json",
        [
            {key: value for key, value in row.items() if key not in ("manifest", "truncation")}
            for row in results
        ],
    )
    _write_csv(
        report_dir / "embedding_cache_manifest.csv", [x for row in results for x in row["manifest"]]
    )
    _write_csv(
        report_dir / "truncation_audit.csv", [x for row in results for x in row["truncation"]]
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
