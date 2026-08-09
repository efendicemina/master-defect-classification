"""Command-line entry point."""

from __future__ import annotations

import argparse
import platform
import sys
from collections.abc import Sequence
from pathlib import Path

from defect_classifier.catalogue import CatalogueError, load_catalogue
from defect_classifier.config import ConfigurationError, resolve_data_root
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "verify-dataset":
        return 2

    print(f"Python: {platform.python_version()} ({sys.executable})")
    print(f"Platform: {platform.platform()}")
    try:
        root = resolve_data_root(args.data_root)
        catalogue = load_catalogue(args.catalogue)
    except (ConfigurationError, CatalogueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"ECLIPSE_DATA_ROOT: {root}")
    print(f"Catalogue version: {catalogue.version}")
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
