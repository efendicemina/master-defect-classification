"""Orchestration for Phase B1 environment, embedding, and benchmark stages."""

from __future__ import annotations

import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path
from typing import Any

from defect_classifier.classical_benchmark import BenchmarkError
from defect_classifier.protocol import FrozenProtocol
from defect_classifier.semantic_benchmark import run_semantic_benchmark
from defect_classifier.semantic_embeddings import (
    encoder_smoke,
    load_semantic_config,
    materialize_encoder,
    mps_preflight,
    write_materialization_reports,
)


def _verify_reference_reports(a1_dir: Path, a2_dir: Path) -> dict[str, str]:
    paths = [
        a1_dir / "CLASSICAL_BENCHMARK_REPORT.md",
        a1_dir / "leaderboard.csv",
        a2_dir / "CLASSICAL_OPTIMIZATION_REPORT.md",
        a2_dir / "leaderboard.csv",
    ]
    if not all(path.is_file() for path in paths):
        raise BenchmarkError("missing immutable classical reference report")
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def run_semantic_pipeline(
    *,
    stage: str,
    development_dir: Path,
    manifest_dir: Path,
    protocol_report_dir: Path,
    a1_report_dir: Path,
    a2_report_dir: Path,
    cache_root: Path,
    report_dir: Path,
    protocol: FrozenProtocol,
    config_path: Path | None = None,
    task_filter: str | None = None,
    encoder_filter: str | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    config, config_fingerprint = load_semantic_config(config_path)
    references = _verify_reference_reports(a1_report_dir, a2_report_dir)
    frozen = json.loads((protocol_report_dir / "fingerprints.json").read_text())
    if frozen["protocol_sha256"] != protocol.fingerprint:
        raise BenchmarkError("protocol fingerprint drift")
    preflight = mps_preflight()
    environment = {
        **preflight,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            package: version(package)
            for package in ("torch", "transformers", "sentence-transformers", "huggingface-hub")
        },
        "semantic_config_sha256": config_fingerprint,
        "classical_reference_sha256": references,
        "locked_test_accessed": False,
        "locked_test_embedded": False,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    from defect_classifier.embedding_cache import atomic_json

    atomic_json(report_dir / "environment.json", environment)
    if stage == "preflight":
        return {"preflight": preflight}
    selected = [
        row for row in config["encoders"] if not encoder_filter or row["id"] == encoder_filter
    ]
    projects = sorted(development_dir.glob("*.parquet"))
    if not projects:
        raise BenchmarkError("missing frozen DEVELOPMENT artifacts")
    if stage in ("all", "smoke"):
        import pyarrow.parquet as pq

        sample = pq.read_table(projects[0], columns=["text_combined"])["text_combined"].to_pylist()
        sample = sample[: config["smoke"]["sample_size"]]
        smoke = [
            encoder_smoke(
                encoder, sample, preflight["selected_device"], config["smoke"]["batch_size"]
            )
            for encoder in selected
        ]
        atomic_json(report_dir / ".work" / "encoder_smoke.json", smoke)
        if stage == "smoke":
            return {"preflight": preflight, "smoke": smoke}
    materialized = []
    if stage in ("all", "materialize"):
        for encoder in selected:
            materialized.append(
                materialize_encoder(
                    encoder=encoder,
                    development_dir=development_dir,
                    cache_root=cache_root,
                    report_dir=report_dir,
                    protocol=protocol,
                    frozen=frozen,
                    config=config,
                    config_fingerprint=config_fingerprint,
                    device=preflight["selected_device"],
                    resume=resume,
                )
            )
        write_materialization_reports(report_dir, materialized)
        if stage == "materialize":
            return {"preflight": preflight, "materialized": materialized}
    benchmark = run_semantic_benchmark(
        development_dir,
        manifest_dir,
        protocol_report_dir,
        cache_root,
        report_dir,
        protocol,
        config_path=config_path,
        task_filter=task_filter,
        encoder_filter=encoder_filter,
        resume=resume,
    )
    return {"preflight": preflight, "materialized": materialized, "benchmark": benchmark}
