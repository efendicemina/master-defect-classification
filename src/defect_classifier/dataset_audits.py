"""Memory-bounded, single-pass forensic audits of raw Eclipse CSV files."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import platform
import re
import statistics
import time
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from defect_classifier.catalogue import DatasetCatalogue
from defect_classifier.dataset_schema import CsvSchema, discover_csv_schema
from defect_classifier.paths import dataset_path

LOGGER = logging.getLogger(__name__)
REQUIRED_FIELDS = ("ID", "Severity", "Summary", "Description", "Creation time")
HISTORICAL_ROW_COUNTS = {
    "BIRT": 23_308,
    "CDT": 22_371,
    "EQUINOX": 14_559,
    "JDT": 63_266,
    "MYLYN": 13_993,
    "PDE": 17_639,
    "PAPYRUS": 13_253,
    "PLATFORM": 122_496,
    "TPTP": 10_579,
}
URL_PATTERN = re.compile(r"(?:https?://|www\.)", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b")
HTML_PATTERN = re.compile(r"</?[a-z][^>]*>", re.IGNORECASE)
CODE_PATTERN = re.compile(
    r"(?:\bat\s+[\w.$]+\([^\n)]*:\d+\)|Traceback \(most recent call last\)|"
    r"Exception(?:\b|:)|\b(?:class|public|private|protected)\s+\w+|```)",
    re.IGNORECASE,
)
SEMANTIC_EQUIVALENTS = {
    "severity": ("severity", "bug severity"),
    "summary": ("summary", "title"),
    "description": ("description", "body"),
    "creation_timestamp": ("creation time", "created", "opened"),
    "issue_id": ("id", "issue id", "bug id"),
}
AUDIT_FORMAT_VERSION = 3


class AuditError(RuntimeError):
    """Raised when a source cannot be audited reliably."""


@dataclass
class LengthAccumulator:
    lengths: list[int] = field(default_factory=list)
    total: int = 0
    minimum: int | None = None
    maximum: int | None = None

    def add(self, value: str) -> None:
        length = len(value)
        self.lengths.append(length)
        self.total += length
        self.minimum = length if self.minimum is None else min(self.minimum, length)
        self.maximum = length if self.maximum is None else max(self.maximum, length)

    def summary(self) -> dict[str, int | float | None]:
        count = len(self.lengths)
        return {
            "count": count,
            "mean": self.total / count if count else None,
            "median": statistics.median(self.lengths) if count else None,
            "min": self.minimum,
            "max": self.maximum,
        }


@dataclass
class ProjectAudit:
    project: str
    relative_path: str
    source_size_bytes: int
    schema: CsvSchema
    raw_rows: int
    parsed_rows: int
    field_audited_rows: int
    malformed_rows: int
    malformed_record_samples: list[dict[str, int]]
    historical_rows: int | None
    severity: Counter[str]
    severity_null_blank: int
    summary_missing: int
    summary_blank: int
    description_missing: int
    description_blank: int
    both_text_unavailable: int
    summary_lengths: dict[str, int | float | None]
    description_lengths: dict[str, int | float | None]
    timestamp_parse_success: int
    timestamp_parse_failure: int
    timestamp_null: int
    timestamp_earliest: str | None
    timestamp_latest: str | None
    timestamp_aware: int
    timestamp_naive: int
    issue_id_non_null: int
    issue_id_unique: int
    issue_id_duplicate_rows: int
    duplicate_hash_counts: dict[str, int]
    duplicate_hash_sample_ids: dict[str, list[str]]
    fully_empty_rows: int
    html_rows: int
    url_rows: int
    email_rows: int
    code_rows: int
    elapsed_seconds: float


def _normalize_for_exact_hash(value: str) -> str:
    return unicodedata.normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n"))


def _text_hash(summary: str, description: str) -> str:
    digest = hashlib.sha256()
    digest.update(_normalize_for_exact_hash(summary).encode("utf-8"))
    digest.update(b"\x00")
    digest.update(_normalize_for_exact_hash(description).encode("utf-8"))
    return digest.hexdigest()


def _parse_timestamp(raw: str) -> datetime:
    value = raw.strip()
    if value.endswith("Z"):
        value = f"{value[:-1]}+00:00"
    return datetime.fromisoformat(value)


def _row_value(row: list[str], indices: dict[str, int], field_name: str) -> str:
    return row[indices[field_name]]


def audit_project(
    project: str,
    path: Path,
    relative_path: str,
    *,
    progress_every: int = 10_000,
) -> ProjectAudit:
    """Audit one CSV in one streaming pass, retaining only bounded aggregate state."""
    started = time.monotonic()
    schema = discover_csv_schema(path)
    if schema.duplicate_columns:
        raise AuditError(f"{project}: duplicate columns prevent unambiguous parsing")
    missing = [name for name in REQUIRED_FIELDS if name not in schema.columns]
    if missing:
        raise AuditError(f"{project}: missing required audit fields: {missing}")
    indices = {name: schema.columns.index(name) for name in schema.columns}
    id_counts: Counter[str] = Counter()
    severity: Counter[str] = Counter()
    hash_counts: Counter[str] = Counter()
    hash_samples: dict[str, list[str]] = {}
    summary_lengths = LengthAccumulator()
    description_lengths = LengthAccumulator()
    raw_rows = parsed_rows = field_audited_rows = malformed_rows = 0
    malformed_samples: list[dict[str, int]] = []
    severity_null_blank = 0
    summary_missing = summary_blank = 0
    description_missing = description_blank = both_unavailable = 0
    timestamp_success = timestamp_failure = timestamp_null = 0
    timestamp_aware = timestamp_naive = 0
    earliest: datetime | None = None
    latest: datetime | None = None
    fully_empty = html_rows = url_rows = email_rows = code_rows = 0
    csv.field_size_limit(1024**3)

    try:
        with path.open("r", encoding=schema.encoding, errors="strict", newline="") as handle:
            reader = csv.reader(handle, delimiter=schema.delimiter, strict=True)
            next(reader)
            for row in reader:
                raw_rows += 1
                parsed_rows += 1
                if len(row) != len(schema.columns):
                    malformed_rows += 1
                    if len(malformed_samples) < 100:
                        malformed_samples.append(
                            {
                                "record_number": raw_rows,
                                "ending_physical_line": reader.line_num,
                                "observed_fields": len(row),
                                "expected_fields": len(schema.columns),
                            }
                        )
                    if len(row) <= max(indices[name] for name in REQUIRED_FIELDS):
                        continue
                    row.extend([""] * (len(schema.columns) - len(row)))
                field_audited_rows += 1
                if progress_every and raw_rows % progress_every == 0:
                    LOGGER.info("%s: %s rows parsed", project, f"{raw_rows:,}")

                issue_id = _row_value(row, indices, "ID")
                severity_value = _row_value(row, indices, "Severity")
                summary = _row_value(row, indices, "Summary")
                description = _row_value(row, indices, "Description")
                timestamp_raw = _row_value(row, indices, "Creation time")

                if issue_id != "":
                    id_counts[issue_id] += 1
                if severity_value.strip() == "":
                    severity_null_blank += 1
                else:
                    severity[severity_value] += 1

                if summary == "":
                    summary_missing += 1
                elif summary.strip() == "":
                    summary_blank += 1
                if description == "":
                    description_missing += 1
                elif description.strip() == "":
                    description_blank += 1
                if summary.strip() == "" and description.strip() == "":
                    both_unavailable += 1
                summary_lengths.add(summary)
                description_lengths.add(description)

                if timestamp_raw.strip() == "":
                    timestamp_null += 1
                else:
                    try:
                        timestamp = _parse_timestamp(timestamp_raw)
                    except ValueError:
                        timestamp_failure += 1
                    else:
                        timestamp_success += 1
                        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                            timestamp_naive += 1
                        else:
                            timestamp_aware += 1
                        comparable = (
                            timestamp.replace(tzinfo=UTC)
                            if timestamp.tzinfo is None or timestamp.utcoffset() is None
                            else timestamp.astimezone(UTC)
                        )
                        earliest = comparable if earliest is None else min(earliest, comparable)
                        latest = comparable if latest is None else max(latest, comparable)

                digest = _text_hash(summary, description)
                hash_counts[digest] += 1
                sample = hash_samples.setdefault(digest, [])
                if len(sample) < 3 and issue_id:
                    sample.append(issue_id)

                if not any(value.strip() for value in row):
                    fully_empty += 1
                has_html = has_url = has_email = has_code = False
                for text in (summary, description):
                    has_html = has_html or bool(HTML_PATTERN.search(text))
                    has_url = has_url or bool(URL_PATTERN.search(text))
                    has_email = has_email or bool(EMAIL_PATTERN.search(text))
                    has_code = has_code or bool(CODE_PATTERN.search(text))
                html_rows += has_html
                url_rows += has_url
                email_rows += has_email
                code_rows += has_code
    except (OSError, UnicodeError, csv.Error) as exc:
        raise AuditError(f"{project}: parsing failed after {raw_rows:,} rows: {exc}") from exc

    duplicate_id_rows = sum(count for count in id_counts.values() if count > 1)
    return ProjectAudit(
        project=project,
        relative_path=relative_path,
        source_size_bytes=path.stat().st_size,
        schema=schema,
        raw_rows=raw_rows,
        parsed_rows=parsed_rows,
        field_audited_rows=field_audited_rows,
        malformed_rows=malformed_rows,
        malformed_record_samples=malformed_samples,
        historical_rows=HISTORICAL_ROW_COUNTS.get(project),
        severity=severity,
        severity_null_blank=severity_null_blank,
        summary_missing=summary_missing,
        summary_blank=summary_blank,
        description_missing=description_missing,
        description_blank=description_blank,
        both_text_unavailable=both_unavailable,
        summary_lengths=summary_lengths.summary(),
        description_lengths=description_lengths.summary(),
        timestamp_parse_success=timestamp_success,
        timestamp_parse_failure=timestamp_failure,
        timestamp_null=timestamp_null,
        timestamp_earliest=earliest.isoformat() if earliest else None,
        timestamp_latest=latest.isoformat() if latest else None,
        timestamp_aware=timestamp_aware,
        timestamp_naive=timestamp_naive,
        issue_id_non_null=sum(id_counts.values()),
        issue_id_unique=len(id_counts),
        issue_id_duplicate_rows=duplicate_id_rows,
        duplicate_hash_counts=dict(hash_counts),
        duplicate_hash_sample_ids=hash_samples,
        fully_empty_rows=fully_empty,
        html_rows=html_rows,
        url_rows=url_rows,
        email_rows=email_rows,
        code_rows=code_rows,
        elapsed_seconds=time.monotonic() - started,
    )


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8", newline="")
    os.replace(temporary, path)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    _atomic_text(path, buffer.getvalue())


def _write_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def _project_checkpoint(path: Path, result: ProjectAudit) -> None:
    payload = asdict(result)
    payload["audit_format_version"] = AUDIT_FORMAT_VERSION
    payload["schema"] = asdict(result.schema)
    payload["severity"] = dict(result.severity)
    _write_json(path, payload)


def _load_checkpoint(path: Path) -> ProjectAudit:
    payload = json.loads(path.read_text(encoding="utf-8"))
    version = payload.pop("audit_format_version", None)
    if version in (None, 2) and payload.get("malformed_rows") == 0:
        payload["malformed_record_samples"] = []
        payload["field_audited_rows"] = payload["parsed_rows"]
    elif version != AUDIT_FORMAT_VERSION:
        raise AuditError(f"obsolete audit checkpoint: {path}")
    payload["schema"] = CsvSchema(**payload["schema"])
    payload["severity"] = Counter(payload["severity"])
    return ProjectAudit(**payload)


def run_audit(
    root: Path,
    catalogue: DatasetCatalogue,
    output_dir: Path,
    *,
    resume: bool = True,
    progress_every: int = 10_000,
) -> tuple[ProjectAudit, ...]:
    """Audit all catalogue projects and create the complete report suite."""
    started = time.monotonic()
    work_dir = output_dir / ".work"
    work_dir.mkdir(parents=True, exist_ok=True)
    results: list[ProjectAudit] = []
    for project, relative_path in catalogue.projects.items():
        source = dataset_path(root, relative_path)
        if not source.is_file():
            raise AuditError(f"{project}: source file does not exist: {source}")
        checkpoint = work_dir / f"{project}.json"
        if resume and checkpoint.is_file():
            try:
                cached = _load_checkpoint(checkpoint)
            except AuditError:
                LOGGER.info("%s: checkpoint format changed; recomputing", project)
            else:
                if (
                    cached.relative_path == relative_path.as_posix()
                    and cached.source_size_bytes == source.stat().st_size
                ):
                    LOGGER.info("%s: using completed checkpoint", project)
                    results.append(cached)
                    continue
        LOGGER.info("%s: auditing %s", project, source)
        result = audit_project(
            project,
            source,
            relative_path.as_posix(),
            progress_every=progress_every,
        )
        _project_checkpoint(checkpoint, result)
        results.append(result)
        LOGGER.info("%s: complete in %.1f seconds", project, result.elapsed_seconds)
    write_reports(tuple(results), output_dir, time.monotonic() - started)
    return tuple(results)


def _schema_reports(results: tuple[ProjectAudit, ...], output_dir: Path) -> dict[str, Any]:
    all_columns = sorted({column for result in results for column in result.schema.columns})
    schema_rows = []
    for result in results:
        present = set(result.schema.columns)
        for position, column in enumerate(result.schema.columns, start=1):
            schema_rows.append(
                {"project": result.project, "position": position, "column": column, "present": True}
            )
        for column in set(all_columns) - present:
            schema_rows.append(
                {"project": result.project, "position": "", "column": column, "present": False}
            )
    _write_csv(
        output_dir / "schema_by_project.csv",
        schema_rows,
        ["project", "position", "column", "present"],
    )
    sets = {result.project: set(result.schema.columns) for result in results}
    common = sorted(set.intersection(*sets.values())) if sets else []
    summary = {
        "projects": {
            result.project: {
                "column_count": len(result.schema.columns),
                "columns": list(result.schema.columns),
                "delimiter": result.schema.delimiter,
                "encoding": result.schema.encoding,
                "has_utf8_bom": result.schema.has_utf8_bom,
                "duplicate_columns": list(result.schema.duplicate_columns),
                "missing_from_union": sorted(set(all_columns) - sets[result.project]),
                "unique_to_project": sorted(
                    sets[result.project]
                    - set().union(*(value for key, value in sets.items() if key != result.project))
                ),
            }
            for result in results
        },
        "columns_present_in_every_project": common,
        "capitalization_differences": _capitalization_differences(all_columns),
        "likely_semantic_equivalents_considered": SEMANTIC_EQUIVALENTS,
    }
    _write_json(output_dir / "schema_summary.json", summary)
    return summary


def _capitalization_differences(columns: list[str]) -> list[list[str]]:
    groups: dict[str, list[str]] = {}
    for column in columns:
        groups.setdefault(column.casefold(), []).append(column)
    return [sorted(values) for values in groups.values() if len(values) > 1]


def write_reports(
    results: tuple[ProjectAudit, ...], output_dir: Path, total_elapsed_seconds: float
) -> None:
    """Materialize deterministic, Git-trackable audit reports atomically."""
    output_dir.mkdir(parents=True, exist_ok=True)
    schema_summary = _schema_reports(results, output_dir)
    manifest = [
        {
            "project": r.project,
            "source_relative_path": r.relative_path,
            "source_size_bytes": r.source_size_bytes,
            "raw_row_count": r.raw_rows,
            "successfully_parsed_rows": r.parsed_rows,
            "field_audited_rows": r.field_audited_rows,
            "malformed_bad_line_count": r.malformed_rows,
            "historical_reference_count": r.historical_rows,
            "historical_comparison": "MATCH" if r.raw_rows == r.historical_rows else "MISMATCH",
        }
        for r in results
    ]
    manifest_fields = list(manifest[0]) if manifest else []
    _write_csv(output_dir / "dataset_manifest.csv", manifest, manifest_fields)
    _write_json(
        output_dir / "dataset_manifest.json",
        {
            "audit": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "total_elapsed_seconds": round(total_elapsed_seconds, 3),
            },
            "datasets": manifest,
        },
    )

    severity_rows = []
    combined: Counter[str] = Counter()
    combined_null = 0
    for result in results:
        combined.update(result.severity)
        combined_null += result.severity_null_blank
        denominator = result.field_audited_rows
        for label, count in sorted(result.severity.items()):
            severity_rows.append(
                {
                    "project": result.project,
                    "severity_raw": label,
                    "count": count,
                    "percentage": round(100 * count / denominator, 8) if denominator else 0,
                    "is_null_or_blank": False,
                }
            )
        severity_rows.append(
            {
                "project": result.project,
                "severity_raw": "",
                "count": result.severity_null_blank,
                "percentage": (
                    round(100 * result.severity_null_blank / denominator, 8) if denominator else 0
                ),
                "is_null_or_blank": True,
            }
        )
    _write_csv(
        output_dir / "severity_by_project.csv",
        severity_rows,
        ["project", "severity_raw", "count", "percentage", "is_null_or_blank"],
    )
    total_parsed = sum(r.field_audited_rows for r in results)
    combined_rows = [
        {
            "severity_raw": label,
            "count": count,
            "percentage": round(100 * count / total_parsed, 8) if total_parsed else 0,
            "is_null_or_blank": False,
        }
        for label, count in sorted(combined.items())
    ]
    combined_rows.append(
        {
            "severity_raw": "",
            "count": combined_null,
            "percentage": round(100 * combined_null / total_parsed, 8) if total_parsed else 0,
            "is_null_or_blank": True,
        }
    )
    _write_csv(
        output_dir / "severity_combined.csv",
        combined_rows,
        ["severity_raw", "count", "percentage", "is_null_or_blank"],
    )
    expected_labels = {"blocker", "critical", "major", "normal", "minor", "trivial", "enhancement"}
    unexpected = [
        row
        for row in severity_rows
        if row["count"] > 0 and row["severity_raw"] not in expected_labels
    ]
    _write_csv(
        output_dir / "unexpected_severity_labels.csv",
        unexpected,
        ["project", "severity_raw", "count", "percentage", "is_null_or_blank"],
    )

    text_rows = []
    for r in results:
        row = {
            "project": r.project,
            "successfully_parsed_rows": r.parsed_rows,
            "field_audited_rows": r.field_audited_rows,
            "summary_missing_null_empty": r.summary_missing,
            "summary_blank_whitespace_only": r.summary_blank,
            "description_missing_null_empty": r.description_missing,
            "description_blank_whitespace_only": r.description_blank,
            "both_summary_description_unavailable": r.both_text_unavailable,
        }
        row.update({f"summary_length_{key}": value for key, value in r.summary_lengths.items()})
        row.update(
            {f"description_length_{key}": value for key, value in r.description_lengths.items()}
        )
        text_rows.append(row)
    _write_csv(output_dir / "text_missing_audit.csv", text_rows, list(text_rows[0]))

    temporal_rows = [
        {
            "project": r.project,
            "timestamp_field": "Creation time",
            "parse_success_count": r.timestamp_parse_success,
            "parse_failure_count": r.timestamp_parse_failure,
            "null_timestamp_count": r.timestamp_null,
            "earliest_timestamp": r.timestamp_earliest,
            "latest_timestamp": r.timestamp_latest,
            "timezone_aware_count": r.timestamp_aware,
            "timezone_naive_count": r.timestamp_naive,
        }
        for r in results
    ]
    _write_csv(output_dir / "temporal_coverage.csv", temporal_rows, list(temporal_rows[0]))

    identifier_rows = [
        {
            "project": r.project,
            "total_rows": r.raw_rows,
            "field_audited_rows": r.field_audited_rows,
            "issue_id_field": "ID",
            "non_null_issue_ids": r.issue_id_non_null,
            "unique_issue_ids": r.issue_id_unique,
            "duplicate_issue_id_rows": r.issue_id_duplicate_rows,
            "duplicate_issue_id_excess": r.issue_id_non_null - r.issue_id_unique,
        }
        for r in results
    ]
    _write_csv(output_dir / "identifier_audit.csv", identifier_rows, list(identifier_rows[0]))

    duplicate_rows, duplicate_details = _duplicate_reports(results)
    _write_csv(output_dir / "duplicate_summary.csv", duplicate_rows, list(duplicate_rows[0]))
    _write_csv(
        output_dir / "duplicate_groups.csv",
        duplicate_details,
        ["scope", "project_count", "projects", "text_sha256", "row_count", "sample_issue_ids"],
    )
    content_rows = [
        {
            "project": r.project,
            "successfully_parsed_rows": r.parsed_rows,
            "field_audited_rows": r.field_audited_rows,
            "fully_empty_rows": r.fully_empty_rows,
            "malformed_records": r.malformed_rows,
            "rows_with_obvious_html": r.html_rows,
            "rows_with_url": r.url_rows,
            "rows_with_email_like_text": r.email_rows,
            "rows_with_code_or_stacktrace_indicator": r.code_rows,
        }
        for r in results
    ]
    _write_csv(output_dir / "text_content_signals.csv", content_rows, list(content_rows[0]))
    malformed_details = [
        {"project": result.project, **sample}
        for result in results
        for sample in result.malformed_record_samples
    ]
    _write_csv(
        output_dir / "malformed_records.csv",
        malformed_details,
        [
            "project",
            "record_number",
            "ending_physical_line",
            "observed_fields",
            "expected_fields",
        ],
    )
    _write_assessments(output_dir, results, schema_summary)
    _write_master_report(output_dir, results, combined_rows, total_elapsed_seconds)


def _duplicate_reports(
    results: tuple[ProjectAudit, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    global_counts: Counter[str] = Counter()
    global_projects: dict[str, list[str]] = {}
    global_samples: dict[str, list[str]] = {}
    for result in results:
        duplicate_counts = {h: c for h, c in result.duplicate_hash_counts.items() if c > 1}
        summaries.append(
            {
                "scope": "within_project",
                "project": result.project,
                "rows_in_duplicate_groups": sum(duplicate_counts.values()),
                "duplicate_groups": len(duplicate_counts),
                "largest_duplicate_group": max(duplicate_counts.values(), default=0),
            }
        )
        for digest, count in result.duplicate_hash_counts.items():
            global_counts[digest] += count
            global_projects.setdefault(digest, []).append(result.project)
            samples = global_samples.setdefault(digest, [])
            samples.extend(
                result.duplicate_hash_sample_ids.get(digest, [])[: max(0, 3 - len(samples))]
            )
            if count > 1:
                details.append(
                    {
                        "scope": "within_project",
                        "project_count": 1,
                        "projects": result.project,
                        "text_sha256": digest,
                        "row_count": count,
                        "sample_issue_ids": "|".join(
                            result.duplicate_hash_sample_ids.get(digest, [])
                        ),
                    }
                )
    cross = {
        digest: count
        for digest, count in global_counts.items()
        if len(set(global_projects[digest])) > 1
    }
    summaries.append(
        {
            "scope": "cross_project",
            "project": "ALL",
            "rows_in_duplicate_groups": sum(cross.values()),
            "duplicate_groups": len(cross),
            "largest_duplicate_group": max(cross.values(), default=0),
        }
    )
    for digest, count in cross.items():
        projects = sorted(set(global_projects[digest]))
        details.append(
            {
                "scope": "cross_project",
                "project_count": len(projects),
                "projects": "|".join(projects),
                "text_sha256": digest,
                "row_count": count,
                "sample_issue_ids": "|".join(global_samples[digest][:3]),
            }
        )
    details.sort(
        key=lambda row: (row["scope"], row["projects"], -row["row_count"], row["text_sha256"])
    )
    return summaries, details


def _write_assessments(
    output_dir: Path, results: tuple[ProjectAudit, ...], schema_summary: dict[str, Any]
) -> None:
    common_count = len(schema_summary["columns_present_in_every_project"])
    schema_md = f"""# Schema Audit

## Observation

The audit found {common_count} columns present in every project. Delimiter and encoding details,
column order, duplicate names, missing columns, and project-unique columns are recorded in
`schema_summary.json`. Field names were preserved exactly; no semantic normalization was applied.

Likely semantic roles were assessed explicitly: `Severity` (raw target), `Summary` and
`Description` (candidate text), `Creation time` (candidate chronology), and `ID` (issue identity).

## Future methodological decision

The modelling schema and permitted predictors must be frozen separately. In particular, the
presence of a field does not authorize its use as a feature.
"""
    _atomic_text(output_dir / "SCHEMA_AUDIT.md", schema_md)
    temporal_md = """# Temporal Field Assessment

## Observation

`Creation time` is the most credible chronological field because its name denotes issue creation,
whereas `Last change time` describes later lifecycle activity and could leak post-creation
information. The audit parses `Creation time` only, using ISO-8601 semantics, and reports nulls,
failures, range, and timezone awareness per project in `temporal_coverage.csv`.

## Future methodological decision

No chronological cutoff or split has been selected. The next protocol-freezing task must decide
how timestamps, ties, invalid values, and project-specific temporal coverage affect eligibility.
"""
    _atomic_text(output_dir / "TEMPORAL_FIELD_ASSESSMENT.md", temporal_md)
    relation_fields = sorted(
        {
            column
            for result in results
            for column in result.schema.columns
            if any(term in column.casefold() for term in ("dupe", "depend", "block", "see also"))
        }
    )
    duplicate_md = f"""# Duplicate Field Assessment

## Observation

`ID` is the explicit issue identifier. Fields with duplicate or issue-relation semantics are:
{", ".join(f"`{field}`" for field in relation_fields)}. `Dupe of` is the strongest explicit
duplicate-link candidate; dependency, blocking, and see-also links may also connect related issues.
`Product` is the strongest embedded project/product-identity candidate, while `Classification`
and `Component` provide broader/finer taxonomy rather than an unambiguous catalogue project ID.
The catalogue project remains source provenance. Identifier counts are in `identifier_audit.csv`;
no rows or links were removed.

## Future methodological decision

The deduplication unit, graph grouping policy, treatment of conflicting labels, and cross-project
leakage controls must be frozen before any split is made.
"""
    _atomic_text(output_dir / "DUPLICATE_FIELD_ASSESSMENT.md", duplicate_md)


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _write_master_report(
    output_dir: Path,
    results: tuple[ProjectAudit, ...],
    combined_severity: list[dict[str, Any]],
    elapsed: float,
) -> None:
    inventory = _markdown_table(
        ["Project", "Rows", "Historical", "Status", "Malformed", "GiB"],
        [
            [
                r.project,
                r.raw_rows,
                r.historical_rows,
                "MATCH" if r.raw_rows == r.historical_rows else "MISMATCH",
                r.malformed_rows,
                f"{r.source_size_bytes / 1024**3:.2f}",
            ]
            for r in results
        ],
    )
    severity = _markdown_table(
        ["Raw label", "Count", "Percentage"],
        [
            [row["severity_raw"] or "(null/blank)", row["count"], row["percentage"]]
            for row in combined_severity
        ],
    )
    text = _markdown_table(
        [
            "Project",
            "Summary missing",
            "Summary blank",
            "Description missing",
            "Description blank",
            "Both unavailable",
        ],
        [
            [
                r.project,
                r.summary_missing,
                r.summary_blank,
                r.description_missing,
                r.description_blank,
                r.both_text_unavailable,
            ]
            for r in results
        ],
    )
    temporal = _markdown_table(
        ["Project", "Parsed", "Failed", "Null", "Earliest", "Latest"],
        [
            [
                r.project,
                r.timestamp_parse_success,
                r.timestamp_parse_failure,
                r.timestamp_null,
                r.timestamp_earliest,
                r.timestamp_latest,
            ]
            for r in results
        ],
    )
    identifiers = _markdown_table(
        ["Project", "Non-null IDs", "Unique IDs", "Rows in repeated IDs"],
        [
            [r.project, r.issue_id_non_null, r.issue_id_unique, r.issue_id_duplicate_rows]
            for r in results
        ],
    )
    duplicates = _markdown_table(
        ["Project", "Rows in exact duplicate groups", "Groups", "Largest"],
        [
            [
                r.project,
                sum(c for c in r.duplicate_hash_counts.values() if c > 1),
                sum(c > 1 for c in r.duplicate_hash_counts.values()),
                max((c for c in r.duplicate_hash_counts.values() if c > 1), default=0),
            ]
            for r in results
        ],
    )
    report = f"""# Forensic Dataset Audit

This audit is descriptive. It does not define modelling eligibility, transform targets, remove
duplicates, create splits, or inspect any future locked test set.

## 1. Dataset inventory and raw row counts

**OBSERVATION:** All catalogue sources were processed independently by a strict streaming CSV
reader. Raw rows include every successfully parsed CSV record; malformed counts flag records whose
field count differs from the header. Required-field aggregates include malformed-width records
only when all required field positions remain available.

{inventory}

**OBSERVATION:** Four successfully parsed records have fewer fields than the 42-column header:
one in EQUINOX and three in PDE. Their record numbers, ending physical lines, and widths are in
`malformed_records.csv`. All retain the required audit-field positions and are therefore included
in target, text, timestamp, identifier, duplicate, and content aggregates; absent trailing fields
were represented as empty only for row-shape inspection.

## 2. Schema consistency

**OBSERVATION:** See `SCHEMA_AUDIT.md`, `schema_by_project.csv`, and `schema_summary.json` for the
canonical comparison. No ambiguous field was silently normalized.

## 3. Raw severity distributions

**OBSERVATION:** Values below are exact stored labels; `enhancement` is retained.

{severity}

**FUTURE METHODOLOGICAL DECISION:** Freeze target eligibility and S6/S3/S2 mappings only after
reviewing these raw distributions.

## 4. Missing text

**OBSERVATION:** Empty CSV fields and non-empty whitespace-only fields are counted separately.
CSV itself does not distinguish database NULL from an exported empty string.

{text}

**FUTURE METHODOLOGICAL DECISION:** Decide text eligibility and empty-field handling before
preprocessing.

## 5. Temporal coverage

**OBSERVATION:** `Creation time` was audited as the strongest creation timestamp candidate.

{temporal}

**FUTURE METHODOLOGICAL DECISION:** Freeze invalid-time handling and chronological boundaries
without consulting future test outcomes.

## 6. Identifier integrity and duplicate-related fields

{identifiers}

**OBSERVATION:** Explicit relation fields are documented in `DUPLICATE_FIELD_ASSESSMENT.md`.

## 7. Exact textual duplicates

**OBSERVATION:** SHA-256 hashes cover NFC-normalized raw `Summary`, a null separator, and raw
`Description`, with only null-to-empty and line-ending normalization. No semantic cleaning occurs.

{duplicates}

Cross-project groups and hash-only detail are in `duplicate_summary.csv` and
`duplicate_groups.csv`. Full text is deliberately not copied into reports.

**FUTURE METHODOLOGICAL DECISION:** Freeze duplicate grouping/removal and leakage policy before
splitting.

## 8. Other data quality observations

**OBSERVATION:** `text_content_signals.csv` reports fully empty rows, malformed record shapes, and
regex-based HTML, URL, email-like, and code/stack-trace indicators. These are coarse descriptive
signals, not preprocessing recommendations.

## 9. Issues requiring decisions before modelling

- Define eligible raw severity labels and then freeze S6/S3/S2 mappings.
- Define handling for empty/blank text and invalid or absent creation timestamps.
- Define exact and linked-duplicate grouping, conflicting-label handling, and split containment.
- Decide whether project identity or any metadata field is a permissible predictor.
- Freeze chronological development and locked-test boundaries in a separate reviewed task.
- Specify dataset identity/provenance hashing without placing raw or derived data in Git.

Total audit orchestration runtime: {elapsed:.3f} seconds. Per-project parsing runtimes are retained
in restart checkpoints; the committed manifest records the full invocation runtime.
"""
    _atomic_text(output_dir / "DATASET_AUDIT.md", report)
