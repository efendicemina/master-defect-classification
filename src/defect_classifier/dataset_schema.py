"""Bounded CSV schema and dialect discovery."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

SAMPLE_BYTES = 256 * 1024


class DatasetSchemaError(ValueError):
    """Raised when a source CSV does not satisfy the audit ingestion contract."""


@dataclass(frozen=True)
class CsvSchema:
    columns: tuple[str, ...]
    delimiter: str
    encoding: str
    has_utf8_bom: bool
    duplicate_columns: tuple[str, ...]


def discover_csv_schema(path: Path) -> CsvSchema:
    """Inspect a bounded prefix and the header without consuming dataset records."""
    with path.open("rb") as handle:
        sample = handle.read(SAMPLE_BYTES)
    has_bom = sample.startswith(b"\xef\xbb\xbf")
    try:
        decoded = sample.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DatasetSchemaError(f"{path} is not valid UTF-8 near byte {exc.start}") from exc
    try:
        dialect = csv.Sniffer().sniff(decoded, delimiters=",;\t|")
    except csv.Error as exc:
        header_line = decoded.splitlines()[0] if decoded.splitlines() else ""
        candidate = max(",;\t|", key=header_line.count)
        if header_line.count(candidate) == 0:
            raise DatasetSchemaError(f"cannot infer delimiter for {path}: {exc}") from exc
        delimiter = candidate
    else:
        delimiter = dialect.delimiter

    try:
        with path.open("r", encoding="utf-8-sig", errors="strict", newline="") as handle:
            columns = tuple(next(csv.reader(handle, delimiter=delimiter, strict=True)))
    except (OSError, UnicodeError, csv.Error, StopIteration) as exc:
        raise DatasetSchemaError(f"cannot read CSV header for {path}: {exc}") from exc
    if not columns:
        raise DatasetSchemaError(f"CSV has an empty header: {path}")
    counts = Counter(columns)
    duplicates = tuple(name for name, count in counts.items() if count > 1)
    return CsvSchema(
        columns=columns,
        delimiter=delimiter,
        encoding="utf-8-sig",
        has_utf8_bom=has_bom,
        duplicate_columns=duplicates,
    )
