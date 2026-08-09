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
from defect_classifier.config import ConfigurationError, resolve_data_root
from defect_classifier.cv_manifests import CvManifestError, materialize_frozen_cv_manifests
from defect_classifier.dataset_audits import AuditError, run_audit
from defect_classifier.preparation import PreparationError, prepare_protocol_v1
from defect_classifier.protocol import ProtocolError, load_protocol
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"Python: {platform.python_version()} ({sys.executable})")
    print(f"Platform: {platform.platform()}")
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
