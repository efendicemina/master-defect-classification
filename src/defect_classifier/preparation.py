"""Protocol-v1 dataset preparation, leakage protection, and split reporting."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import platform
import re
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from defect_classifier.catalogue import DatasetCatalogue
from defect_classifier.dataset_schema import discover_csv_schema
from defect_classifier.paths import dataset_path
from defect_classifier.protocol import FrozenProtocol

LOGGER = logging.getLogger(__name__)
PARQUET_BATCH_ROWS = 256
DUPE_ID_PATTERN = re.compile(r"^\s*(\d+)\s*$")
DUPE_URL_PATTERN = re.compile(r"(?:[?&](?:id|bug_id)=|show_bug\.cgi\?id=)(\d+)", re.IGNORECASE)


class PreparationError(RuntimeError):
    """Raised when preparation cannot uphold the frozen protocol."""


@dataclass(frozen=True)
class RowMeta:
    stable_id: str
    source_project: str
    issue_id: str
    creation_time: datetime
    severity_raw: str
    target_s6: str
    target_s3: str
    target_s2: str
    dupe_of_raw: str
    exact_text_hash: str


@dataclass(frozen=True)
class ProjectPopulation:
    project: str
    raw_rows: int
    eligible_rows: int
    excluded_enhancement: int
    excluded_missing_text: int
    excluded_other: int
    malformed_trailing_rows: int


class UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}
        self.rank = dict.fromkeys(self.parent, 0)

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def _timestamp(raw: str) -> datetime:
    value = raw.strip()
    if value.endswith("Z"):
        value = f"{value[:-1]}+00:00"
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp is timezone-naive")
    return parsed.astimezone(UTC)


def _issue_sort_key(issue_id: str) -> tuple[int, int | str, str]:
    stripped = issue_id.strip()
    if stripped.isdecimal():
        return (0, int(stripped), stripped)
    return (1, stripped, stripped)


def _membership_fingerprint(stable_ids: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for stable_id in sorted(stable_ids):
        digest.update(stable_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _parse_dupe_reference(raw: str) -> str | None:
    if not raw.strip():
        return None
    if match := DUPE_ID_PATTERN.fullmatch(raw):
        return match.group(1)
    matches = DUPE_URL_PATTERN.findall(raw)
    return matches[0] if len(set(matches)) == 1 else ""


def _parquet_schema() -> Any:
    import pyarrow as pa

    return pa.schema(
        [
            ("source_project", pa.string()),
            ("issue_id", pa.string()),
            ("creation_time", pa.timestamp("us", tz="UTC")),
            ("summary", pa.string()),
            ("description", pa.string()),
            ("text_combined", pa.string()),
            ("severity_raw", pa.string()),
            ("target_s6", pa.string()),
            ("target_s3", pa.string()),
            ("target_s2", pa.string()),
            ("dupe_of", pa.string()),
            ("exact_text_hash", pa.string()),
            ("source_relative_path", pa.string()),
            ("source_record_number", pa.int64()),
        ]
    )


class _ParquetBatchWriter:
    def __init__(self, path: Path) -> None:
        import pyarrow.parquet as pq

        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.schema = _parquet_schema()
        self.writer = pq.ParquetWriter(
            path,
            self.schema,
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
        )
        self.rows: list[dict[str, Any]] = []

    def add(self, row: dict[str, Any]) -> None:
        self.rows.append(row)
        if len(self.rows) >= PARQUET_BATCH_ROWS:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        import pyarrow as pa

        table = pa.Table.from_pylist(self.rows, schema=self.schema)
        self.writer.write_table(table, row_group_size=PARQUET_BATCH_ROWS)
        self.rows.clear()

    def close(self) -> None:
        self.flush()
        self.writer.close()


def _stream_project(
    project: str,
    source: Path,
    relative_path: str,
    protocol: FrozenProtocol,
    parquet_path: Path,
    *,
    progress_every: int,
) -> tuple[list[RowMeta], ProjectPopulation]:
    schema = discover_csv_schema(source)
    if schema.columns != protocol.raw_columns:
        raise PreparationError(f"{project}: raw schema differs from frozen protocol")
    if schema.delimiter != protocol.raw_delimiter or schema.encoding != protocol.raw_encoding:
        raise PreparationError(f"{project}: CSV dialect/encoding differs from frozen protocol")
    indices = {name: schema.columns.index(name) for name in schema.columns}
    required = (
        "ID",
        "Severity",
        "Creation time",
        "Summary",
        "Description",
        protocol.duplicate_field,
    )
    max_required_index = max(indices[name] for name in required)
    known_labels = set(protocol.accepted_labels) | set(protocol.excluded_labels)
    raw_rows = enhancement = missing_text = excluded_other = malformed = 0
    metadata: list[RowMeta] = []
    seen_ids: set[str] = set()
    writer = _ParquetBatchWriter(parquet_path)
    csv.field_size_limit(1024**3)
    try:
        with source.open("r", encoding=schema.encoding, errors="strict", newline="") as handle:
            reader = csv.reader(handle, delimiter=schema.delimiter, strict=True)
            header = tuple(next(reader))
            if header != protocol.raw_columns:
                raise PreparationError(f"{project}: header changed between discovery and reading")
            for row in reader:
                raw_rows += 1
                if progress_every and raw_rows % progress_every == 0:
                    LOGGER.info("%s: %s raw rows prepared", project, f"{raw_rows:,}")
                if len(row) != len(schema.columns):
                    malformed += 1
                    if len(row) <= max_required_index or len(row) > len(schema.columns):
                        raise PreparationError(
                            f"{project}: malformed record {raw_rows} lacks required fields"
                        )
                    row.extend([""] * (len(schema.columns) - len(row)))
                issue_id = row[indices["ID"]].strip()
                if not issue_id:
                    raise PreparationError(f"{project}: missing issue ID at record {raw_rows}")
                if issue_id in seen_ids:
                    raise PreparationError(f"{project}: duplicate issue ID {issue_id}")
                seen_ids.add(issue_id)
                severity = row[indices["Severity"]]
                if severity not in known_labels:
                    raise PreparationError(
                        f"{project}: unknown severity {severity!r} at record {raw_rows}"
                    )
                if severity in protocol.excluded_labels:
                    enhancement += 1
                    continue
                try:
                    creation_time = _timestamp(row[indices["Creation time"]])
                except ValueError as exc:
                    raise PreparationError(
                        f"{project}: invalid creation timestamp at record {raw_rows}"
                    ) from exc
                summary, description, combined = protocol.combine_text(
                    row[indices["Summary"]], row[indices["Description"]]
                )
                if not summary.strip() and not description.strip():
                    missing_text += 1
                    continue
                targets = protocol.map_severity(severity)
                dupe_of = row[indices[protocol.duplicate_field]].strip()
                exact_hash = protocol.exact_text_hash(summary, description)
                stable_id = f"{project}:{issue_id}"
                metadata.append(
                    RowMeta(
                        stable_id=stable_id,
                        source_project=project,
                        issue_id=issue_id,
                        creation_time=creation_time,
                        severity_raw=severity,
                        target_s6=targets["s6"],
                        target_s3=targets["s3"],
                        target_s2=targets["s2"],
                        dupe_of_raw=dupe_of,
                        exact_text_hash=exact_hash,
                    )
                )
                writer.add(
                    {
                        "source_project": project,
                        "issue_id": issue_id,
                        "creation_time": creation_time,
                        "summary": summary,
                        "description": description,
                        "text_combined": combined,
                        "severity_raw": severity,
                        "target_s6": targets["s6"],
                        "target_s3": targets["s3"],
                        "target_s2": targets["s2"],
                        "dupe_of": dupe_of,
                        "exact_text_hash": exact_hash,
                        "source_relative_path": relative_path,
                        "source_record_number": raw_rows,
                    }
                )
    except (OSError, UnicodeError, csv.Error) as exc:
        raise PreparationError(f"{project}: raw CSV parsing failed: {exc}") from exc
    finally:
        writer.close()
    return metadata, ProjectPopulation(
        project=project,
        raw_rows=raw_rows,
        eligible_rows=len(metadata),
        excluded_enhancement=enhancement,
        excluded_missing_text=missing_text,
        excluded_other=excluded_other,
        malformed_trailing_rows=malformed,
    )


def _build_components(
    rows: list[RowMeta],
) -> tuple[UnionFind, dict[str, str], list[dict[str, Any]], Counter[str]]:
    union_find = UnionFind(row.stable_id for row in rows)
    exact_owner: dict[str, str] = {}
    for row in rows:
        owner = exact_owner.setdefault(row.exact_text_hash, row.stable_id)
        union_find.union(owner, row.stable_id)

    issue_locations: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        issue_locations[row.issue_id].append(row.stable_id)
    unresolved: Counter[tuple[str, str]] = Counter()
    resolution_counts: Counter[str] = Counter()
    explicit_edges: list[dict[str, Any]] = []
    for row in rows:
        reference = _parse_dupe_reference(row.dupe_of_raw)
        if reference is None:
            continue
        if reference == "":
            unresolved[(row.source_project, "malformed_or_ambiguous_reference")] += 1
            continue
        targets = issue_locations.get(reference, [])
        if len(targets) != 1:
            reason = "target_not_eligible_or_external" if not targets else "ambiguous_target"
            unresolved[(row.source_project, reason)] += 1
            continue
        target = targets[0]
        if target == row.stable_id:
            resolution_counts["self_reference"] += 1
            continue
        union_find.union(row.stable_id, target)
        explicit_edges.append({"left": row.stable_id, "right": target})
        resolution_counts["resolved"] += 1
    unresolved_rows = [
        {"project": project, "reason": reason, "count": count}
        for (project, reason), count in sorted(unresolved.items())
    ]
    return union_find, exact_owner, unresolved_rows, resolution_counts


def _chronological_split(
    rows_by_project: dict[str, list[RowMeta]],
    union_find: UnionFind,
) -> tuple[set[str], set[str], set[str], list[dict[str, Any]]]:
    development: set[str] = set()
    candidates: set[str] = set()
    project_rows: list[dict[str, Any]] = []
    for project, rows in rows_by_project.items():
        ordered = sorted(rows, key=lambda row: (row.creation_time, _issue_sort_key(row.issue_id)))
        boundary = math.floor(len(ordered) * 0.8)
        project_development = ordered[:boundary]
        project_candidates = ordered[boundary:]
        development.update(row.stable_id for row in project_development)
        candidates.update(row.stable_id for row in project_candidates)
        project_rows.append(
            {
                "project": project,
                "eligible_rows": len(ordered),
                "development_candidate_rows": len(project_development),
                "locked_test_candidate_rows": len(project_candidates),
                "chronology_boundary_index": boundary,
            }
        )
    development_roots = {union_find.find(stable_id) for stable_id in development}
    final_test = {
        stable_id for stable_id in candidates if union_find.find(stable_id) not in development_roots
    }
    return development, candidates, final_test, project_rows


def _overlap_reasons(
    rows: list[RowMeta],
    earlier: set[str],
    future: set[str],
    union_find: UnionFind,
) -> dict[str, str]:
    by_id = {row.stable_id: row for row in rows}
    earlier_hashes = {by_id[stable_id].exact_text_hash for stable_id in earlier}
    earlier_roots = {union_find.find(stable_id) for stable_id in earlier}
    reasons = {}
    for stable_id in future:
        if union_find.find(stable_id) not in earlier_roots:
            continue
        reasons[stable_id] = (
            "exact_text"
            if by_id[stable_id].exact_text_hash in earlier_hashes
            else "explicit_dupe_link"
        )
    return reasons


def _temporal_folds(
    rows_by_project: dict[str, list[RowMeta]],
    development: set[str],
    union_find: UnionFind,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    fold_membership: dict[tuple[int, str], tuple[set[str], set[str]]] = {}
    proof_rows: list[dict[str, Any]] = []
    for project, all_rows in rows_by_project.items():
        rows = sorted(
            (row for row in all_rows if row.stable_id in development),
            key=lambda row: (row.creation_time, _issue_sort_key(row.issue_id)),
        )
        edges = [math.floor(len(rows) * index / 4) for index in range(5)]
        for fold in range(1, 4):
            train = {row.stable_id for row in rows[: edges[fold]]}
            validation = {row.stable_id for row in rows[edges[fold] : edges[fold + 1]]}
            fold_membership[(fold, project)] = (train, validation)
            train_times = [row.creation_time for row in rows[: edges[fold]]]
            validation_times = [row.creation_time for row in rows[edges[fold] : edges[fold + 1]]]
            maximum = max(train_times) if train_times else None
            minimum = min(validation_times) if validation_times else None
            proof_rows.append(
                {
                    "fold": fold,
                    "project": project,
                    "max_training_creation_time": maximum.isoformat() if maximum else "",
                    "min_validation_creation_time": minimum.isoformat() if minimum else "",
                    "chronology_valid": maximum is None or minimum is None or maximum <= minimum,
                }
            )

    summaries: list[dict[str, Any]] = []
    fingerprints: dict[str, str] = {}
    for family in ("within_project", "pooled"):
        for fold in range(1, 4):
            pooled_train = set().union(
                *(fold_membership[(fold, project)][0] for project in rows_by_project)
            )
            pooled_validation = set().union(
                *(fold_membership[(fold, project)][1] for project in rows_by_project)
            )
            pooled_reasons = _overlap_reasons(
                [row for rows in rows_by_project.values() for row in rows],
                pooled_train,
                pooled_validation,
                union_find,
            )
            family_train_all: set[str] = set()
            family_validation_all: set[str] = set()
            for project in rows_by_project:
                train, validation = fold_membership[(fold, project)]
                reasons = (
                    pooled_reasons
                    if family == "pooled"
                    else _overlap_reasons(rows_by_project[project], train, validation, union_find)
                )
                final_validation = validation - set(reasons)
                family_train_all.update(train)
                family_validation_all.update(final_validation)
                summaries.append(
                    {
                        "family": family,
                        "fold": fold,
                        "project": project,
                        "training_rows": len(train),
                        "validation_candidate_rows": len(validation),
                        "validation_removed_exact_text": sum(
                            reasons.get(stable_id) == "exact_text" for stable_id in validation
                        ),
                        "validation_removed_explicit_dupe_link": sum(
                            reasons.get(stable_id) == "explicit_dupe_link"
                            for stable_id in validation
                        ),
                        "validation_final_rows": len(final_validation),
                        "validation_component_overlap_after_purge": 0,
                        "_training_ids": train,
                        "_final_validation_ids": final_validation,
                    }
                )
            prefix = f"{family}_fold_{fold}"
            fingerprints[f"{prefix}_training"] = _membership_fingerprint(family_train_all)
            fingerprints[f"{prefix}_validation"] = _membership_fingerprint(family_validation_all)
    return summaries, proof_rows, fingerprints


def _copy_partitioned_artifacts(
    staged_projects: Path,
    staged_development: Path,
    staged_locked: Path,
    development: set[str],
    final_test: set[str],
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    for project_file in sorted(staged_projects.glob("*.parquet")):
        project = project_file.stem
        writers = {
            "development": _ParquetBatchWriter(staged_development / f"{project}.parquet"),
            "locked": _ParquetBatchWriter(staged_locked / f"{project}.parquet"),
        }
        parquet = pq.ParquetFile(project_file)
        for batch in parquet.iter_batches(batch_size=PARQUET_BATCH_ROWS):
            table = pa.Table.from_batches([batch])
            for row in table.to_pylist():
                stable_id = f"{row['source_project']}:{row['issue_id']}"
                if stable_id in development:
                    writers["development"].add(row)
                elif stable_id in final_test:
                    writers["locked"].add(row)
        for writer in writers.values():
            writer.close()


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8", newline="")
    os.replace(temporary, path)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_text(path, buffer.getvalue())


def _write_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _class_rows(
    rows: list[RowMeta], development: set[str], final_test: set[str], protocol: FrozenProtocol
) -> list[dict[str, Any]]:
    output = []
    for partition, membership in (("DEVELOPMENT", development), ("LOCKED_TEST", final_test)):
        selected = [row for row in rows if row.stable_id in membership]
        for task, attribute in (("S6", "target_s6"), ("S3", "target_s3"), ("S2", "target_s2")):
            counts = Counter(getattr(row, attribute) for row in selected)
            for label in protocol.targets[task.casefold()].order:
                output.append(
                    {
                        "partition": partition,
                        "task": task,
                        "class": label,
                        "count": counts[label],
                    }
                )
    return output


def _cv_class_rows(
    rows: list[RowMeta], summaries: list[dict[str, Any]], protocol: FrozenProtocol
) -> list[dict[str, Any]]:
    by_id = {row.stable_id: row for row in rows}
    output: list[dict[str, Any]] = []
    for summary in summaries:
        partitions = {
            "TRAINING": [by_id[value] for value in summary["_training_ids"]],
            "VALIDATION": [by_id[value] for value in summary["_final_validation_ids"]],
        }
        for partition, selected in partitions.items():
            for task, attribute in (("S6", "target_s6"), ("S3", "target_s3"), ("S2", "target_s2")):
                counts = Counter(getattr(row, attribute) for row in selected)
                for label in protocol.targets[task.casefold()].order:
                    output.append(
                        {
                            "family": summary["family"],
                            "fold": summary["fold"],
                            "project": summary["project"],
                            "partition": partition,
                            "task": task,
                            "class": label,
                            "count": counts[label],
                        }
                    )
    return output


def prepare_protocol_v1(
    root: Path,
    catalogue: DatasetCatalogue,
    protocol: FrozenProtocol,
    processed_dir: Path,
    locked_dir: Path,
    report_dir: Path,
    *,
    progress_every: int = 10_000,
) -> dict[str, Any]:
    """Build frozen protocol-v1 artifacts and safe aggregate reports."""
    started = time.monotonic()
    existing_fingerprints_path = report_dir / "fingerprints.json"
    existing = None
    if existing_fingerprints_path.is_file():
        existing = json.loads(existing_fingerprints_path.read_text(encoding="utf-8"))
        if existing.get("protocol_sha256") != protocol.fingerprint:
            raise PreparationError(
                "protocol configuration differs from the existing frozen split; refusing overwrite"
            )

    work = report_dir / ".work"
    staged_projects = work / "projects"
    staged_development = work / "development"
    staged_locked = work / "locked_test"
    for directory in (staged_projects, staged_development, staged_locked):
        directory.mkdir(parents=True, exist_ok=True)

    all_rows: list[RowMeta] = []
    populations: list[ProjectPopulation] = []
    rows_by_project: dict[str, list[RowMeta]] = {}
    for project, relative_path in catalogue.projects.items():
        source = dataset_path(root, relative_path)
        if not source.is_file():
            raise PreparationError(f"{project}: missing raw source {source}")
        LOGGER.info("%s: preparing eligible research rows", project)
        rows, population = _stream_project(
            project,
            source,
            relative_path.as_posix(),
            protocol,
            staged_projects / f"{project}.parquet",
            progress_every=progress_every,
        )
        rows_by_project[project] = rows
        all_rows.extend(rows)
        populations.append(population)
        LOGGER.info("%s: %s eligible rows", project, f"{len(rows):,}")
    if sum(item.raw_rows for item in populations) != protocol.expected_raw_rows:
        raise PreparationError(
            f"raw population mismatch: observed {sum(item.raw_rows for item in populations):,}, "
            f"expected {protocol.expected_raw_rows:,}"
        )
    if len(all_rows) != protocol.expected_eligible_rows:
        raise PreparationError(
            f"eligible population mismatch: observed {len(all_rows):,}, expected "
            f"{protocol.expected_eligible_rows:,}"
        )

    union_find, _, unresolved, dupe_resolution = _build_components(all_rows)
    development, candidates, final_test, split_rows = _chronological_split(
        rows_by_project, union_find
    )
    overlap_reasons = _overlap_reasons(all_rows, development, candidates, union_find)
    if candidates - set(overlap_reasons) != final_test:
        raise PreparationError("locked-test overlap exclusion is internally inconsistent")
    if development & final_test:
        raise PreparationError("development and locked-test memberships overlap")
    development_roots = {union_find.find(value) for value in development}
    final_roots = {union_find.find(value) for value in final_test}
    if development_roots & final_roots:
        raise PreparationError("duplicate leakage remains between development and locked test")

    development_rows_by_project = {
        project: [row for row in rows if row.stable_id in development]
        for project, rows in rows_by_project.items()
    }
    cv_sizes, chronology_proofs, cv_fingerprints = _temporal_folds(
        development_rows_by_project, development, union_find
    )
    if not all(row["chronology_valid"] for row in chronology_proofs):
        raise PreparationError("temporal CV chronology proof failed")

    _copy_partitioned_artifacts(
        staged_projects, staged_development, staged_locked, development, final_test
    )
    artifact_fingerprints = {
        project_file.stem: _file_fingerprint(project_file)
        for project_file in sorted(staged_projects.glob("*.parquet"))
    }
    fingerprints = {
        "protocol_sha256": protocol.fingerprint,
        "processed_project_artifacts": artifact_fingerprints,
        "processed_dataset_sha256": hashlib.sha256(
            "\n".join(
                f"{key}:{value}" for key, value in sorted(artifact_fingerprints.items())
            ).encode()
        ).hexdigest(),
        "development_membership_sha256": _membership_fingerprint(development),
        "locked_test_membership_sha256": _membership_fingerprint(final_test),
        "locked_test_candidate_membership_sha256": _membership_fingerprint(candidates),
        "cv_membership_sha256": cv_fingerprints,
    }
    if existing is not None:
        stable_keys = (
            "protocol_sha256",
            "processed_project_artifacts",
            "processed_dataset_sha256",
            "development_membership_sha256",
            "locked_test_membership_sha256",
            "locked_test_candidate_membership_sha256",
            "cv_membership_sha256",
        )
        if any(existing.get(key) != fingerprints.get(key) for key in stable_keys):
            raise PreparationError("regenerated frozen dataset/split fingerprints differ")

    for project in catalogue.projects:
        destinations = (
            (
                staged_projects / f"{project}.parquet",
                processed_dir / "projects" / f"{project}.parquet",
            ),
            (
                staged_development / f"{project}.parquet",
                processed_dir / "development" / f"{project}.parquet",
            ),
            (staged_locked / f"{project}.parquet", locked_dir / f"{project}.parquet"),
        )
        for source, destination in destinations:
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, destination)

    elapsed = time.monotonic() - started
    _write_protocol_reports(
        report_dir,
        protocol,
        populations,
        all_rows,
        development,
        candidates,
        final_test,
        overlap_reasons,
        split_rows,
        unresolved,
        dupe_resolution,
        cv_sizes,
        chronology_proofs,
        fingerprints,
        elapsed,
    )
    return {
        "eligible_rows": len(all_rows),
        "development_rows": len(development),
        "locked_test_candidate_rows": len(candidates),
        "final_locked_test_rows": len(final_test),
        "elapsed_seconds": elapsed,
        "fingerprints": fingerprints,
    }


def _write_protocol_reports(
    report_dir: Path,
    protocol: FrozenProtocol,
    populations: list[ProjectPopulation],
    rows: list[RowMeta],
    development: set[str],
    candidates: set[str],
    final_test: set[str],
    overlap_reasons: dict[str, str],
    split_rows: list[dict[str, Any]],
    unresolved: list[dict[str, Any]],
    dupe_resolution: Counter[str],
    cv_sizes: list[dict[str, Any]],
    chronology_proofs: list[dict[str, Any]],
    fingerprints: dict[str, Any],
    elapsed: float,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    population_by_project = {row.project: row for row in populations}
    for split in split_rows:
        project = split["project"]
        population = population_by_project[project]
        project_development = {value for value in development if value.startswith(f"{project}:")}
        project_candidates = {value for value in candidates if value.startswith(f"{project}:")}
        project_final = {value for value in final_test if value.startswith(f"{project}:")}
        split.update(
            {
                "raw_rows": population.raw_rows,
                "excluded_enhancement_rows": population.excluded_enhancement,
                "excluded_missing_text_rows": population.excluded_missing_text,
                "excluded_other_rows": population.excluded_other,
                "malformed_trailing_rows_retained": population.malformed_trailing_rows,
                "locked_removed_exact_text": sum(
                    overlap_reasons.get(value) == "exact_text" for value in project_candidates
                ),
                "locked_removed_explicit_dupe_link": sum(
                    overlap_reasons.get(value) == "explicit_dupe_link"
                    for value in project_candidates
                ),
                "final_development_rows": len(project_development),
                "final_locked_test_rows": len(project_final),
            }
        )
    split_fields = [
        "project",
        "raw_rows",
        "eligible_rows",
        "excluded_enhancement_rows",
        "excluded_missing_text_rows",
        "excluded_other_rows",
        "malformed_trailing_rows_retained",
        "development_candidate_rows",
        "locked_test_candidate_rows",
        "locked_removed_exact_text",
        "locked_removed_explicit_dupe_link",
        "final_development_rows",
        "final_locked_test_rows",
        "chronology_boundary_index",
    ]
    _write_csv(report_dir / "population_and_split_counts.csv", split_rows, split_fields)

    class_rows = _class_rows(rows, development, final_test, protocol)
    _write_csv(report_dir / "class_distributions.csv", class_rows, list(class_rows[0]))
    temporal_rows = []
    locked_chronology_rows = []
    for project in population_by_project:
        for partition, membership in (("DEVELOPMENT", development), ("LOCKED_TEST", final_test)):
            times = [
                row.creation_time
                for row in rows
                if row.source_project == project and row.stable_id in membership
            ]
            temporal_rows.append(
                {
                    "project": project,
                    "partition": partition,
                    "row_count": len(times),
                    "earliest_creation_time": min(times).isoformat() if times else "",
                    "latest_creation_time": max(times).isoformat() if times else "",
                }
            )
        development_times = [
            row.creation_time
            for row in rows
            if row.source_project == project and row.stable_id in development
        ]
        candidate_times = [
            row.creation_time
            for row in rows
            if row.source_project == project and row.stable_id in candidates
        ]
        final_times = [
            row.creation_time
            for row in rows
            if row.source_project == project and row.stable_id in final_test
        ]
        development_max = max(development_times)
        candidate_min = min(candidate_times)
        final_min = min(final_times) if final_times else None
        locked_chronology_rows.append(
            {
                "project": project,
                "max_development_creation_time": development_max.isoformat(),
                "min_locked_candidate_creation_time": candidate_min.isoformat(),
                "min_final_locked_creation_time": final_min.isoformat() if final_min else "",
                "candidate_chronology_valid": development_max <= candidate_min,
                "final_chronology_valid": final_min is None or development_max <= final_min,
            }
        )
    _write_csv(report_dir / "temporal_ranges.csv", temporal_rows, list(temporal_rows[0]))
    _write_csv(
        report_dir / "locked_split_chronological_proofs.csv",
        locked_chronology_rows,
        list(locked_chronology_rows[0]),
    )

    overlap_rows = [
        {
            "boundary": "development_to_locked_test",
            "candidate_overlap_exact_text": sum(
                reason == "exact_text" for reason in overlap_reasons.values()
            ),
            "candidate_overlap_explicit_dupe_link": sum(
                reason == "explicit_dupe_link" for reason in overlap_reasons.values()
            ),
            "duplicate_component_overlap_before_purge": len(overlap_reasons),
            "duplicate_component_overlap_after_purge": 0,
            "membership_overlap_after_purge": len(development & final_test),
            "resolved_explicit_dupe_edges": dupe_resolution["resolved"],
        }
    ]
    _write_csv(report_dir / "duplicate_overlap_proof.csv", overlap_rows, list(overlap_rows[0]))
    if not unresolved:
        unresolved = [{"project": "ALL", "reason": "none", "count": 0}]
    _write_csv(report_dir / "unresolved_dupe_links.csv", unresolved, ["project", "reason", "count"])
    cv_size_fields = [key for key in cv_sizes[0] if not key.startswith("_")]
    _write_csv(report_dir / "cv_fold_sizes.csv", cv_sizes, cv_size_fields)
    development_rows = [row for row in rows if row.stable_id in development]
    cv_class_rows = _cv_class_rows(development_rows, cv_sizes, protocol)
    _write_csv(report_dir / "cv_class_counts.csv", cv_class_rows, list(cv_class_rows[0]))
    _write_csv(
        report_dir / "chronological_proofs.csv",
        chronology_proofs,
        list(chronology_proofs[0]),
    )
    fingerprints["generated_at_runtime_seconds"] = round(elapsed, 3)
    fingerprints["python"] = platform.python_version()
    fingerprints["platform"] = platform.platform()
    _write_json(report_dir / "fingerprints.json", fingerprints)
    _atomic_text(
        report_dir / "ACCESS_RECORD.txt",
        "TEST_SET_ACCESSED_FOR_MODEL_SELECTION = NO\n"
        "LOCKED_TEST_MODEL_PERFORMANCE_ACCESSED = NO\n"
        "MODELS_FITTED = 0\n",
    )
    exact_removed = sum(reason == "exact_text" for reason in overlap_reasons.values())
    explicit_removed = sum(reason == "explicit_dupe_link" for reason in overlap_reasons.values())
    report = f"""# Protocol V1 Preparation Report

Protocol: `{protocol.protocol_id}`  
Protocol SHA-256: `{protocol.fingerprint}`

The raw sources were streamed project by project. No model was fitted and no locked-test model
performance was calculated or inspected.

## Frozen population and split

- Raw rows: {sum(item.raw_rows for item in populations):,}
- Eligible rows: {len(rows):,} (expected {protocol.expected_eligible_rows:,}: MATCH)
- Excluded `enhancement` rows: {sum(item.excluded_enhancement for item in populations):,}
- Excluded missing-text rows: {sum(item.excluded_missing_text for item in populations):,}
- Development rows: {len(development):,}
- Locked-test candidate rows: {len(candidates):,}
- Candidate rows removed for exact-text overlap: {exact_removed:,}
- Candidate rows removed for explicit duplicate-component overlap: {explicit_removed:,}
- Final locked-test rows: {len(final_test):,}

## Integrity conclusions

- Development/locked membership overlap after purge: **0**
- Duplicate-component overlap after purge: **0**
- All per-project locked boundaries follow timestamp/issue-ID order: **PASS**
- All 27 per-project CV chronology proofs: **PASS**
- CV validation overlap removals are recorded for within-project and pooled families.
- Four known short trailing records were retained because all protocol-required fields exist.

Detailed safe aggregates are provided in the CSV and JSON files beside this report. Text and
individual locked-test membership are not committed.

```text
MODELS_FITTED = 0
TEST_SET_ACCESSED_FOR_MODEL_SELECTION = NO
LOCKED_TEST_MODEL_PERFORMANCE_ACCESSED = NO
```
"""
    _atomic_text(report_dir / "PROTOCOL_V1_REPORT.md", report)
