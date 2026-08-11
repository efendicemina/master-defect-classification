"""Command-line entry point."""

from __future__ import annotations

import argparse
import logging
import platform
import resource
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from defect_classifier.catalogue import CatalogueError, load_catalogue
from defect_classifier.classical_benchmark import BenchmarkError, run_classical_benchmark
from defect_classifier.classical_optimization import run_classical_optimization
from defect_classifier.config import ConfigurationError, resolve_data_root
from defect_classifier.cv_manifests import CvManifestError, materialize_frozen_cv_manifests
from defect_classifier.dataset_audits import AuditError, run_audit
from defect_classifier.lexical_semantic_fusion import run_fusion_benchmark
from defect_classifier.long_text_embeddings import (
    finalize_long_text_checkpoints,
    run_long_text_feasibility,
    run_long_text_full,
)
from defect_classifier.preparation import PreparationError, prepare_protocol_v1
from defect_classifier.protocol import ProtocolError, load_protocol
from defect_classifier.rta_fusion import finalize_rta_checkpoints, run_rta_feasibility, run_rta_full
from defect_classifier.semantic_pipeline import run_semantic_pipeline
from defect_classifier.transformer_finetuning import run_b2_pipeline
from defect_classifier.transformer_finetuning_lite import run_lite_pipeline
from defect_classifier.verification import verify_dataset


def _human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.2f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="defect-classifier")
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser(
        "verify-dataset", help="verify external Eclipse CSV presence without loading data"
    )
    verify.add_argument("--data-root", type=Path, help="override ECLIPSE_DATA_ROOT")
    verify.add_argument("--catalogue", type=Path, help="override configs/datasets.toml")
    audit = subparsers.add_parser(
        "audit-data", help="stream raw CSVs and write forensic dataset audit reports"
    )
    audit.add_argument("--data-root", type=Path, help="override ECLIPSE_DATA_ROOT")
    audit.add_argument("--catalogue", type=Path, help="override configs/datasets.toml")
    audit.add_argument(
        "--output-dir", type=Path, default=Path("reports/dataset_audit"), help="report directory"
    )
    audit.add_argument("--no-resume", action="store_true", help="ignore completed checkpoints")
    audit.add_argument("--progress-every", type=int, default=10_000, metavar="ROWS")
    prepare = subparsers.add_parser(
        "prepare-data", help="build the frozen protocol-v1 dataset and temporal split"
    )
    prepare.add_argument("--data-root", type=Path, help="override ECLIPSE_DATA_ROOT")
    prepare.add_argument("--catalogue", type=Path, help="override configs/datasets.toml")
    prepare.add_argument("--protocol", type=Path, help="override configs/protocol_v1.toml")
    prepare.add_argument("--processed-dir", type=Path, default=Path("data/processed/protocol_v1"))
    prepare.add_argument("--locked-dir", type=Path, default=Path("data/locked/protocol_v1"))
    prepare.add_argument("--report-dir", type=Path, default=Path("reports/protocol_v1"))
    prepare.add_argument("--progress-every", type=int, default=10_000, metavar="ROWS")
    materialize = subparsers.add_parser(
        "materialize-cv-manifests",
        help="persist already-frozen protocol-v1 development CV memberships",
    )
    materialize.add_argument("--protocol", type=Path, help="override configs/protocol_v1.toml")
    materialize.add_argument(
        "--project-dir", type=Path, default=Path("data/processed/protocol_v1/projects")
    )
    materialize.add_argument(
        "--manifest-dir", type=Path, default=Path("data/processed/protocol_v1/cv_manifests")
    )
    materialize.add_argument("--report-dir", type=Path, default=Path("reports/protocol_v1"))
    benchmark = subparsers.add_parser(
        "classical-benchmark", help="run the development-only Phase A1 classical benchmark"
    )
    benchmark.add_argument("--protocol", type=Path, help="override configs/protocol_v1.toml")
    benchmark.add_argument("--config", type=Path, help="override classical benchmark config")
    benchmark.add_argument(
        "--development-dir", type=Path, default=Path("data/processed/protocol_v1/development")
    )
    benchmark.add_argument(
        "--manifest-dir", type=Path, default=Path("data/processed/protocol_v1/cv_manifests")
    )
    benchmark.add_argument("--protocol-report-dir", type=Path, default=Path("reports/protocol_v1"))
    benchmark.add_argument(
        "--report-dir", type=Path, default=Path("reports/classical_benchmark_v1")
    )
    benchmark.add_argument("--stage", choices=("all", "smoke", "core"), default="all")
    benchmark.add_argument("--task", choices=("S6", "S3", "S2"))
    benchmark.add_argument("--no-resume", action="store_true")
    optimize = subparsers.add_parser(
        "run-classical-optimization", help="run development-only Phase A2 optimization"
    )
    optimize.add_argument("--protocol", type=Path, help="override configs/protocol_v1.toml")
    optimize.add_argument("--config", type=Path, help="override optimization config")
    optimize.add_argument(
        "--development-dir", type=Path, default=Path("data/processed/protocol_v1/development")
    )
    optimize.add_argument(
        "--manifest-dir", type=Path, default=Path("data/processed/protocol_v1/cv_manifests")
    )
    optimize.add_argument("--protocol-report-dir", type=Path, default=Path("reports/protocol_v1"))
    optimize.add_argument(
        "--a1-report-dir", type=Path, default=Path("reports/classical_benchmark_v1")
    )
    optimize.add_argument(
        "--report-dir", type=Path, default=Path("reports/classical_optimization_v1")
    )
    optimize.add_argument(
        "--stage", choices=("all", "smoke", "a2.1", "a2.2", "a2.3"), default="all"
    )
    optimize.add_argument("--task", choices=("S6", "S3", "S2"))
    optimize.add_argument("--no-resume", action="store_true")
    semantic = subparsers.add_parser(
        "run-semantic-embeddings", help="run development-only Phase B1 semantic embeddings"
    )
    semantic.add_argument("--protocol", type=Path, help="override protocol config")
    semantic.add_argument("--config", type=Path, help="override semantic config")
    semantic.add_argument(
        "--stage", choices=("all", "preflight", "smoke", "materialize", "benchmark"), default="all"
    )
    semantic.add_argument("--task", choices=("S6", "S3", "S2"))
    semantic.add_argument("--encoder", choices=("E1", "E2"))
    semantic.add_argument(
        "--development-dir", type=Path, default=Path("data/processed/protocol_v1/development")
    )
    semantic.add_argument(
        "--manifest-dir", type=Path, default=Path("data/processed/protocol_v1/cv_manifests")
    )
    semantic.add_argument("--protocol-report-dir", type=Path, default=Path("reports/protocol_v1"))
    semantic.add_argument(
        "--a1-report-dir", type=Path, default=Path("reports/classical_benchmark_v1")
    )
    semantic.add_argument(
        "--a2-report-dir", type=Path, default=Path("reports/classical_optimization_v1")
    )
    semantic.add_argument("--cache-root", type=Path, default=Path("data/processed/embeddings_v1"))
    semantic.add_argument("--report-dir", type=Path, default=Path("reports/semantic_embeddings_v1"))
    semantic.add_argument("--no-resume", action="store_true")
    fusion = subparsers.add_parser(
        "run-lexical-semantic-fusion",
        help="run development-only Phase B1.5 fixed lexical and MPNet fusion",
    )
    fusion.add_argument("--protocol", type=Path, help="override protocol config")
    fusion.add_argument("--config", type=Path, help="override fusion config")
    fusion.add_argument("--stage", choices=("all", "smoke", "benchmark"), default="all")
    fusion.add_argument("--task", choices=("S6", "S3", "S2"))
    fusion.add_argument(
        "--development-dir", type=Path, default=Path("data/processed/protocol_v1/development")
    )
    fusion.add_argument(
        "--manifest-dir", type=Path, default=Path("data/processed/protocol_v1/cv_manifests")
    )
    fusion.add_argument("--protocol-report-dir", type=Path, default=Path("reports/protocol_v1"))
    fusion.add_argument("--cache-root", type=Path, default=Path("data/processed/embeddings_v1"))
    fusion.add_argument("--reports-root", type=Path, default=Path("reports"))
    fusion.add_argument(
        "--report-dir", type=Path, default=Path("reports/lexical_semantic_fusion_v1")
    )
    fusion.add_argument("--no-resume", action="store_true")
    long_text = subparsers.add_parser(
        "run-long-text-fusion",
        help="run Phase B1.6 development-only long-text feasibility estimation",
    )
    long_text.add_argument("--protocol", type=Path, help="override protocol config")
    long_text.add_argument("--config", type=Path, help="override B1.6 config")
    long_text.add_argument("--stage", choices=("estimate", "full", "finalize"), default="estimate")
    long_text.add_argument(
        "--development-dir", type=Path, default=Path("data/processed/protocol_v1/development")
    )
    long_text.add_argument(
        "--manifest-dir", type=Path, default=Path("data/processed/protocol_v1/cv_manifests")
    )
    long_text.add_argument("--protocol-report-dir", type=Path, default=Path("reports/protocol_v1"))
    long_text.add_argument("--reports-root", type=Path, default=Path("reports"))
    long_text.add_argument("--report-dir", type=Path, default=Path("reports/long_text_fusion_v1"))
    long_text.add_argument(
        "--cache-root", type=Path, default=Path("data/processed/long_text_embeddings_v1")
    )
    long_text.add_argument(
        "--checkpoint-root", type=Path, default=Path("data/processed/long_text_fusion_v1")
    )
    long_text.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    rta = subparsers.add_parser(
        "run-rta-fusion", help="run Phase B3 development-only RTA feasibility estimation"
    )
    rta.add_argument("--protocol", type=Path, help="override protocol config")
    rta.add_argument("--config", type=Path, help="override B3 config")
    rta.add_argument(
        "--stage", choices=("feasibility", "full", "finalize"), default="feasibility"
    )
    rta.add_argument(
        "--development-dir", type=Path, default=Path("data/processed/protocol_v1/development")
    )
    rta.add_argument(
        "--manifest-dir", type=Path, default=Path("data/processed/protocol_v1/cv_manifests")
    )
    rta.add_argument("--protocol-report-dir", type=Path, default=Path("reports/protocol_v1"))
    rta.add_argument("--reports-root", type=Path, default=Path("reports"))
    rta.add_argument("--report-dir", type=Path, default=Path("reports/rta_fusion_v1"))
    rta.add_argument("--cache-root", type=Path, default=Path("data/processed/rta_embeddings_v1"))
    rta.add_argument("--checkpoint-root", type=Path, default=Path("data/processed/rta_fusion_v1"))
    rta.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    transformer = subparsers.add_parser(
        "run-transformer-finetuning",
        help="run development-only Phase B2 controlled DeBERTa fine-tuning",
    )
    transformer.add_argument("--protocol", type=Path, help="override protocol config")
    transformer.add_argument("--config", type=Path, help="override fine-tuning config")
    transformer.add_argument(
        "--stage",
        choices=("all", "preflight", "feasibility", "benchmark"),
        default="all",
    )
    transformer.add_argument(
        "--development-dir", type=Path, default=Path("data/processed/protocol_v1/development")
    )
    transformer.add_argument(
        "--manifest-dir", type=Path, default=Path("data/processed/protocol_v1/cv_manifests")
    )
    transformer.add_argument(
        "--protocol-report-dir", type=Path, default=Path("reports/protocol_v1")
    )
    transformer.add_argument("--reports-root", type=Path, default=Path("reports"))
    transformer.add_argument(
        "--report-dir", type=Path, default=Path("reports/transformer_finetuning_v1")
    )
    transformer.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("data/processed/transformer_checkpoints_v1"),
    )
    transformer.add_argument("--no-resume", action="store_true")
    lite = subparsers.add_parser(
        "run-transformer-finetuning-lite",
        help="run development-only Phase B2-LITE MiniLM fine-tuning",
    )
    lite.add_argument("--protocol", type=Path, help="override protocol config")
    lite.add_argument("--config", type=Path, help="override B2-LITE config")
    lite.add_argument(
        "--stage",
        choices=("all", "preflight", "feasibility", "benchmark", "finalize"),
        default="all",
    )
    lite.add_argument(
        "--development-dir", type=Path, default=Path("data/processed/protocol_v1/development")
    )
    lite.add_argument(
        "--manifest-dir", type=Path, default=Path("data/processed/protocol_v1/cv_manifests")
    )
    lite.add_argument("--protocol-report-dir", type=Path, default=Path("reports/protocol_v1"))
    lite.add_argument("--reports-root", type=Path, default=Path("reports"))
    lite.add_argument(
        "--report-dir", type=Path, default=Path("reports/transformer_finetuning_lite_v1")
    )
    lite.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("data/processed/transformer_lite_checkpoints_v1"),
    )
    lite.add_argument("--no-resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"Python: {platform.python_version()} ({sys.executable})")
    print(f"Platform: {platform.platform()}")
    if args.command == "run-rta-fusion":
        try:
            if args.stage == "finalize":
                result = finalize_rta_checkpoints(
                    protocol_report_dir=args.protocol_report_dir.resolve(),
                    reports_root=args.reports_root.resolve(),
                    report_dir=args.report_dir.resolve(),
                    cache_root=args.cache_root.resolve(),
                    checkpoint_root=args.checkpoint_root.resolve(),
                    protocol=load_protocol(args.protocol),
                    config_path=args.config,
                )
            elif args.stage == "full":
                result = run_rta_full(
                    development_dir=args.development_dir.resolve(),
                    manifest_dir=args.manifest_dir.resolve(),
                    protocol_report_dir=args.protocol_report_dir.resolve(),
                    reports_root=args.reports_root.resolve(),
                    report_dir=args.report_dir.resolve(),
                    cache_root=args.cache_root.resolve(),
                    checkpoint_root=args.checkpoint_root.resolve(),
                    protocol=load_protocol(args.protocol),
                    config_path=args.config,
                    resume=args.resume,
                )
            else:
                result = run_rta_feasibility(
                    development_dir=args.development_dir.resolve(),
                    protocol_report_dir=args.protocol_report_dir.resolve(),
                    reports_root=args.reports_root.resolve(),
                    report_dir=args.report_dir.resolve(),
                    protocol=load_protocol(args.protocol),
                    config_path=args.config,
                    stage=args.stage,
                )
        except (ProtocolError, BenchmarkError, OSError, KeyError, RuntimeError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        if args.stage in ("full", "finalize"):
            print(f"Successful competitive fits: {result['successful']}")
        else:
            print(f"Audited DEVELOPMENT rows: {result['audit']['total_records']}")
            print(f"Throughput sample documents: {result['throughput']['sample_documents']}")
        return 0
    if args.command == "run-long-text-fusion":
        try:
            if args.stage == "finalize":
                result = finalize_long_text_checkpoints(
                    protocol_report_dir=args.protocol_report_dir.resolve(),
                    reports_root=args.reports_root.resolve(),
                    report_dir=args.report_dir.resolve(),
                    cache_root=args.cache_root.resolve(),
                    checkpoint_root=args.checkpoint_root.resolve(),
                    protocol=load_protocol(args.protocol),
                    config_path=args.config,
                )
            elif args.stage == "full":
                result = run_long_text_full(
                    development_dir=args.development_dir.resolve(),
                    manifest_dir=args.manifest_dir.resolve(),
                    protocol_report_dir=args.protocol_report_dir.resolve(),
                    reports_root=args.reports_root.resolve(),
                    report_dir=args.report_dir.resolve(),
                    cache_root=args.cache_root.resolve(),
                    checkpoint_root=args.checkpoint_root.resolve(),
                    protocol=load_protocol(args.protocol),
                    config_path=args.config,
                    resume=args.resume,
                )
            else:
                result = run_long_text_feasibility(
                    development_dir=args.development_dir.resolve(),
                    protocol_report_dir=args.protocol_report_dir.resolve(),
                    reports_root=args.reports_root.resolve(),
                    report_dir=args.report_dir.resolve(),
                    protocol=load_protocol(args.protocol),
                    config_path=args.config,
                    stage=args.stage,
                )
        except (ProtocolError, BenchmarkError, OSError, KeyError, RuntimeError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        if args.stage in ("full", "finalize"):
            print(f"Successful competitive fits: {result['successful']}")
        else:
            print(f"Audited DEVELOPMENT rows: {result['audit']['total_records']}")
            print(f"Projected MPNet chunks: {result['audit']['total_chunks']}")
        return 0
    if args.command == "run-transformer-finetuning-lite":
        try:
            if args.stage == "finalize":
                from defect_classifier.transformer_finetuning_lite import (
                    finalize_lite_checkpoints,
                )

                result = finalize_lite_checkpoints(
                    checkpoint_root=args.checkpoint_root.resolve(),
                    protocol_report_dir=args.protocol_report_dir.resolve(),
                    reports_root=args.reports_root.resolve(),
                    report_dir=args.report_dir.resolve(),
                    protocol=load_protocol(args.protocol),
                    config_path=args.config,
                )
            else:
                result = run_lite_pipeline(
                    stage=args.stage,
                    development_dir=args.development_dir.resolve(),
                    manifest_dir=args.manifest_dir.resolve(),
                    protocol_report_dir=args.protocol_report_dir.resolve(),
                    reports_root=args.reports_root.resolve(),
                    report_dir=args.report_dir.resolve(),
                    checkpoint_root=args.checkpoint_root.resolve(),
                    protocol=load_protocol(args.protocol),
                    config_path=args.config,
                    resume=not args.no_resume,
                )
        except (ProtocolError, BenchmarkError, OSError, KeyError, RuntimeError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        print(f"Successful competitive fits: {result.get('successful', 0)}")
        print(f"Failed competitive fits: {result.get('failed', 0)}")
        if result.get("stopped_before_competitive"):
            print("Competitive execution stopped by the 24-hour feasibility rule")
        return 0
    if args.command == "run-transformer-finetuning":
        try:
            result = run_b2_pipeline(
                stage=args.stage,
                development_dir=args.development_dir.resolve(),
                manifest_dir=args.manifest_dir.resolve(),
                protocol_report_dir=args.protocol_report_dir.resolve(),
                reports_root=args.reports_root.resolve(),
                report_dir=args.report_dir.resolve(),
                checkpoint_root=args.checkpoint_root.resolve(),
                protocol=load_protocol(args.protocol),
                config_path=args.config,
                resume=not args.no_resume,
            )
        except (ProtocolError, BenchmarkError, OSError, KeyError, RuntimeError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        print(f"Successful competitive fits: {result.get('successful', 0)}")
        print(f"Failed competitive fits: {result.get('failed', 0)}")
        if result.get("stopped_before_competitive"):
            print("Competitive execution stopped by the 24-hour feasibility rule")
        return 0
    if args.command == "run-lexical-semantic-fusion":
        try:
            result = run_fusion_benchmark(
                args.development_dir.resolve(),
                args.manifest_dir.resolve(),
                args.protocol_report_dir.resolve(),
                args.cache_root.resolve(),
                args.reports_root.resolve(),
                args.report_dir.resolve(),
                load_protocol(args.protocol),
                config_path=args.config,
                stage=args.stage,
                task_filter=args.task,
                resume=not args.no_resume,
            )
        except (ProtocolError, BenchmarkError, OSError, KeyError, RuntimeError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        print(f"Successful competitive fits: {result.get('successful', 0)}")
        print(f"Failed competitive fits: {result.get('failed', 0)}")
        print(f"Runtime: {result['runtime_seconds']:.3f} seconds")
        return 0
    if args.command == "run-semantic-embeddings":
        try:
            result = run_semantic_pipeline(
                stage=args.stage,
                development_dir=args.development_dir.resolve(),
                manifest_dir=args.manifest_dir.resolve(),
                protocol_report_dir=args.protocol_report_dir.resolve(),
                a1_report_dir=args.a1_report_dir.resolve(),
                a2_report_dir=args.a2_report_dir.resolve(),
                cache_root=args.cache_root.resolve(),
                report_dir=args.report_dir.resolve(),
                protocol=load_protocol(args.protocol),
                config_path=args.config,
                task_filter=args.task,
                encoder_filter=args.encoder,
                resume=not args.no_resume,
            )
        except (ProtocolError, BenchmarkError, OSError, KeyError, RuntimeError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        benchmark = result.get("benchmark", {})
        print(f"MPS built: {result['preflight']['mps_is_built']}")
        print(f"MPS available: {result['preflight']['mps_is_available']}")
        print(f"Successful competitive fits: {benchmark.get('successful', 0)}")
        print(f"Failed competitive fits: {benchmark.get('failed', 0)}")
        return 0
    if args.command == "run-classical-optimization":
        try:
            protocol = load_protocol(args.protocol)
            result = run_classical_optimization(
                args.development_dir.resolve(),
                args.manifest_dir.resolve(),
                args.protocol_report_dir.resolve(),
                args.a1_report_dir.resolve(),
                args.report_dir.resolve(),
                protocol,
                config_path=args.config,
                stage=args.stage,
                task_filter=args.task,
                resume=not args.no_resume,
            )
        except (ProtocolError, BenchmarkError, OSError, KeyError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        print(f"Successful new competitive fits: {result.get('successful', 0)}")
        print(f"Failed new competitive fits: {result.get('failed', 0)}")
        print(f"Runtime: {result['runtime_seconds']:.3f} seconds")
        return 0
    if args.command == "classical-benchmark":
        try:
            protocol = load_protocol(args.protocol)
            result = run_classical_benchmark(
                args.development_dir.resolve(),
                args.manifest_dir.resolve(),
                args.protocol_report_dir.resolve(),
                args.report_dir.resolve(),
                protocol,
                benchmark_config_path=args.config,
                stage=args.stage,
                task_filter=args.task,
                resume=not args.no_resume,
            )
        except (ProtocolError, BenchmarkError, OSError, KeyError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        print(f"Stage 0: {result['stage0'].get('status', 'SKIPPED')}")
        print(f"Successful competitive fits: {result.get('successful', 0)}")
        print(f"Failed competitive fits: {result.get('failed', 0)}")
        print(f"Runtime: {result['total_runtime_seconds']:.3f} seconds")
        return 0
    if args.command == "materialize-cv-manifests":
        try:
            protocol = load_protocol(args.protocol)
            result = materialize_frozen_cv_manifests(
                args.project_dir.resolve(),
                args.manifest_dir.resolve(),
                args.report_dir.resolve(),
                protocol,
            )
        except (ProtocolError, CvManifestError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        peak_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
        if sys.platform != "darwin":
            peak_mib *= 1024
        print(f"Manifest files: {result['manifest_file_count']}")
        print(f"Manifest records: {result['manifest_record_count']}")
        print(f"Fingerprints matched: {result['fingerprints_matched']}/12")
        print(f"Runtime: {result['runtime_seconds']:.3f} seconds")
        print(f"Peak process RSS: {peak_mib:.2f} MiB")
        return 0
    try:
        root = resolve_data_root(args.data_root)
        catalogue = load_catalogue(args.catalogue)
    except (ConfigurationError, CatalogueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"ECLIPSE_DATA_ROOT: {root}", flush=True)
    print(f"Catalogue version: {catalogue.version}", flush=True)
    if args.command == "prepare-data":
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        if args.progress_every < 0:
            print("ERROR: --progress-every must be non-negative", file=sys.stderr)
            return 2
        try:
            protocol = load_protocol(args.protocol)
            result = prepare_protocol_v1(
                root,
                catalogue,
                protocol,
                args.processed_dir.resolve(),
                args.locked_dir.resolve(),
                args.report_dir.resolve(),
                progress_every=args.progress_every,
            )
        except (ProtocolError, PreparationError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        peak_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
        if sys.platform != "darwin":
            peak_mib *= 1024
        print(f"Eligible rows: {result['eligible_rows']}")
        print(f"Development rows: {result['development_rows']}")
        print(f"Final locked-test rows: {result['final_locked_test_rows']}")
        print(f"Runtime: {result['elapsed_seconds']:.3f} seconds")
        print(f"Peak process RSS: {peak_mib:.2f} MiB")
        print(f"Protocol SHA-256: {protocol.fingerprint}")
        return 0
    if args.command == "audit-data":
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        if args.progress_every < 0:
            print("ERROR: --progress-every must be non-negative", file=sys.stderr)
            return 2
        started = time.monotonic()
        try:
            results = run_audit(
                root,
                catalogue,
                args.output_dir.resolve(),
                resume=not args.no_resume,
                progress_every=args.progress_every,
            )
        except AuditError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        elapsed = time.monotonic() - started
        peak_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
        # macOS reports bytes; Linux reports KiB.
        if sys.platform != "darwin":
            peak_mib *= 1024
        print(f"Audited {len(results)} projects in {elapsed:.3f} seconds.")
        print(f"Peak process RSS: {peak_mib:.2f} MiB")
        print(f"Reports: {args.output_dir.resolve()}")
        return 0

    if args.command != "verify-dataset":
        return 2
    checks = verify_dataset(root, catalogue)
    for check in checks:
        if check.exists:
            print(
                f"FOUND {check.project:<8} {check.relative_path} "
                f"({check.size_bytes} bytes, {_human_size(check.size_bytes or 0)})"
            )
        else:
            print(f"MISSING {check.project:<8} {check.relative_path}")

    missing = [check for check in checks if not check.exists]
    if missing:
        print(f"ERROR: {len(missing)} of {len(checks)} required files are missing", file=sys.stderr)
        return 1
    print(f"Verified {len(checks)} of {len(checks)} required files (metadata only).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
