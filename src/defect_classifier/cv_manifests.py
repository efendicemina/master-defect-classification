"""Persistence and fail-closed loading of frozen protocol-v1 CV memberships."""

from __future__ import annotations

import csv
import json
import os
import platform
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from defect_classifier.preparation import (
    RowMeta,
    UnionFind,
    _build_components,
    _chronological_split,
    _file_fingerprint,
    _membership_fingerprint,
    _temporal_folds,
)
from defect_classifier.protocol import FrozenProtocol

FAMILIES = ("within_project", "pooled")
ROLES = ("TRAIN", "VALIDATION")


class CvManifestError(RuntimeError):
    """Raised when frozen CV membership cannot be reproduced or trusted."""


@dataclass(frozen=True)
class CvManifestRow:
    family: str
    fold: int
    role: str
    source_project: str
    issue_id: str
    creation_time: Any

    @property
    def stable_id(self) -> str:
        return f"{self.source_project}:{self.issue_id}"


def _fingerprint_key(family: str, fold: int, role: str) -> str:
    suffix = "training" if role == "TRAIN" else "validation"
    return f"{family}_fold_{fold}_{suffix}"


def _manifest_path(root: Path, family: str, fold: int, role: str) -> Path:
    return root / family / f"fold_{fold}" / f"{role.casefold()}.parquet"


def _read_frozen_fingerprints(path: Path, protocol: FrozenProtocol) -> dict[str, Any]:
    try:
        fingerprints = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CvManifestError(f"cannot read frozen fingerprints {path}: {exc}") from exc
    if fingerprints.get("protocol_sha256") != protocol.fingerprint:
        raise CvManifestError("protocol drift detected against frozen fingerprints")
    expected_cv = fingerprints.get("cv_membership_sha256")
    expected_keys = {
        _fingerprint_key(family, fold, role)
        for family in FAMILIES
        for fold in range(1, protocol.cv_fold_count + 1)
        for role in ROLES
    }
    if not isinstance(expected_cv, dict) or set(expected_cv) != expected_keys:
        raise CvManifestError("frozen fingerprints do not contain the complete 12 CV memberships")
    return fingerprints


def _load_authoritative_rows(
    project_dir: Path, frozen_fingerprints: dict[str, Any]
) -> tuple[list[RowMeta], dict[str, list[RowMeta]]]:
    import pyarrow.parquet as pq

    columns = [
        "source_project",
        "issue_id",
        "creation_time",
        "severity_raw",
        "target_s6",
        "target_s3",
        "target_s2",
        "dupe_of",
        "exact_text_hash",
    ]
    expected_artifacts = frozen_fingerprints.get("processed_project_artifacts", {})
    rows: list[RowMeta] = []
    by_project: dict[str, list[RowMeta]] = {}
    for project, expected_hash in sorted(expected_artifacts.items()):
        path = project_dir / f"{project}.parquet"
        if not path.is_file():
            raise CvManifestError(f"missing authoritative eligible artifact: {path}")
        actual_hash = _file_fingerprint(path)
        if actual_hash != expected_hash:
            raise CvManifestError(f"processed project artifact fingerprint mismatch: {project}")
        project_rows = []
        for record in pq.read_table(path, columns=columns).to_pylist():
            row = RowMeta(
                stable_id=f"{record['source_project']}:{record['issue_id']}",
                source_project=record["source_project"],
                issue_id=record["issue_id"],
                creation_time=record["creation_time"],
                severity_raw=record["severity_raw"],
                target_s6=record["target_s6"],
                target_s3=record["target_s3"],
                target_s2=record["target_s2"],
                dupe_of_raw=record["dupe_of"],
                exact_text_hash=record["exact_text_hash"],
            )
            project_rows.append(row)
            rows.append(row)
        by_project[project] = project_rows
    return rows, by_project


def memberships_from_fold_summaries(
    summaries: list[dict[str, Any]],
) -> dict[tuple[str, int, str], set[str]]:
    """Expose exact memberships emitted by the one authoritative fold generator."""
    memberships: dict[tuple[str, int, str], set[str]] = defaultdict(set)
    for summary in summaries:
        family = summary["family"]
        fold = summary["fold"]
        memberships[(family, fold, "TRAIN")].update(summary["_training_ids"])
        memberships[(family, fold, "VALIDATION")].update(summary["_final_validation_ids"])
    return dict(memberships)


def _validate_fingerprints(
    memberships: dict[tuple[str, int, str], set[str]], expected: dict[str, str]
) -> list[dict[str, Any]]:
    rows = []
    for (family, fold, role), stable_ids in sorted(memberships.items()):
        artifact = _fingerprint_key(family, fold, role)
        actual = _membership_fingerprint(stable_ids)
        expected_value = expected[artifact]
        rows.append(
            {
                "artifact": artifact,
                "family": family,
                "fold": fold,
                "role": role,
                "row_count": len(stable_ids),
                "expected_fingerprint": expected_value,
                "actual_fingerprint": actual,
                "status": "MATCH" if actual == expected_value else "MISMATCH",
            }
        )
    if len(rows) != 12 or any(row["status"] != "MATCH" for row in rows):
        raise CvManifestError("regenerated CV membership differs from frozen fingerprints")
    return rows


def _validate_counts(summaries: list[dict[str, Any]], frozen_counts_path: Path) -> None:
    try:
        with frozen_counts_path.open(encoding="utf-8", newline="") as handle:
            expected = list(csv.DictReader(handle))
    except OSError as exc:
        raise CvManifestError(f"cannot read frozen CV counts: {exc}") from exc
    actual_by_key = {(row["family"], str(row["fold"]), row["project"]): row for row in summaries}
    fields = (
        "training_rows",
        "validation_candidate_rows",
        "validation_removed_exact_text",
        "validation_removed_explicit_dupe_link",
        "validation_final_rows",
        "validation_component_overlap_after_purge",
    )
    if len(expected) != len(summaries):
        raise CvManifestError("frozen CV count row count differs from regenerated folds")
    for frozen in expected:
        key = (frozen["family"], frozen["fold"], frozen["project"])
        actual = actual_by_key.get(key)
        if actual is None or any(int(frozen[field]) != int(actual[field]) for field in fields):
            raise CvManifestError(f"CV count mismatch for family/fold/project {key}")


def _validate_integrity(
    memberships: dict[tuple[str, int, str], set[str]],
    rows: list[RowMeta],
    development: set[str],
    final_locked: set[str],
    union_find: UnionFind,
) -> list[dict[str, Any]]:
    by_id = {row.stable_id: row for row in rows}
    checks = []
    for family in FAMILIES:
        for fold in range(1, 4):
            training = memberships[(family, fold, "TRAIN")]
            validation = memberships[(family, fold, "VALIDATION")]
            if not training | validation <= development:
                raise CvManifestError(f"{family} fold {fold} contains a non-development row")
            if training & validation:
                raise CvManifestError(f"{family} fold {fold} has TRAIN/VALIDATION overlap")
            if (training | validation) & final_locked:
                raise CvManifestError(f"{family} fold {fold} contains final locked-test membership")
            component_scopes = (
                [None]
                if family == "pooled"
                else sorted({by_id[value].source_project for value in training | validation})
            )
            for component_project in component_scopes:
                scoped_training = {
                    value
                    for value in training
                    if component_project is None or by_id[value].source_project == component_project
                }
                scoped_validation = {
                    value
                    for value in validation
                    if component_project is None or by_id[value].source_project == component_project
                }
                training_roots = {union_find.find(value) for value in scoped_training}
                validation_roots = {union_find.find(value) for value in scoped_validation}
                if training_roots & validation_roots:
                    raise CvManifestError(
                        f"{family} fold {fold} retains duplicate-component leakage"
                    )
            for project in sorted({by_id[value].source_project for value in validation}):
                train_times = [
                    by_id[value].creation_time
                    for value in training
                    if by_id[value].source_project == project
                ]
                validation_times = [
                    by_id[value].creation_time
                    for value in validation
                    if by_id[value].source_project == project
                ]
                if train_times and validation_times and max(train_times) > min(validation_times):
                    raise CvManifestError(f"{family} fold {fold} chronology violated for {project}")
            checks.append(
                {
                    "family": family,
                    "fold": fold,
                    "train_validation_overlap": 0,
                    "cv_final_locked_overlap": 0,
                    "duplicate_component_overlap": 0,
                    "chronology": "PASS",
                }
            )
    return checks


def persist_cv_memberships(
    manifest_root: Path,
    memberships: dict[tuple[str, int, str], set[str]],
    rows: list[RowMeta],
) -> int:
    """Persist metadata-only row membership into deterministic Parquet files."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    by_id = {row.stable_id: row for row in rows}
    total = 0
    for (family, fold, role), stable_ids in sorted(memberships.items()):
        records = [
            {
                "family": family,
                "fold": fold,
                "role": role,
                "source_project": by_id[value].source_project,
                "issue_id": by_id[value].issue_id,
                "creation_time": by_id[value].creation_time,
            }
            for value in sorted(stable_ids)
        ]
        path = _manifest_path(manifest_root, family, fold, role)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        table = pa.Table.from_pylist(records)
        pq.write_table(table, temporary, compression="zstd", use_dictionary=True)
        os.replace(temporary, path)
        total += len(records)
    return total


def load_cv_membership(
    manifest_root: Path,
    frozen_fingerprints_path: Path,
    protocol: FrozenProtocol,
    *,
    family: str,
    fold: int,
    role: str,
    project: str | None = None,
) -> tuple[CvManifestRow, ...]:
    """Load and authenticate persisted development-only membership; never infer folds."""
    import pyarrow.parquet as pq

    normalized_role = role.upper()
    if family not in FAMILIES or fold not in range(1, protocol.cv_fold_count + 1):
        raise CvManifestError("unknown frozen CV family or fold")
    if normalized_role not in ROLES:
        raise CvManifestError("CV role must be TRAIN or VALIDATION")
    fingerprints = _read_frozen_fingerprints(frozen_fingerprints_path, protocol)
    path = _manifest_path(manifest_root, family, fold, normalized_role)
    if not path.is_file():
        raise CvManifestError(f"missing frozen CV membership manifest: {path}")
    records = tuple(
        CvManifestRow(**record)
        for record in pq.read_table(
            path,
            columns=["family", "fold", "role", "source_project", "issue_id", "creation_time"],
        ).to_pylist()
    )
    if any(
        row.family != family or row.fold != fold or row.role != normalized_role for row in records
    ):
        raise CvManifestError("manifest content does not match requested frozen fold")
    actual = _membership_fingerprint(row.stable_id for row in records)
    expected = fingerprints["cv_membership_sha256"][_fingerprint_key(family, fold, normalized_role)]
    if actual != expected:
        raise CvManifestError("persisted CV membership fingerprint mismatch")
    return tuple(row for row in records if project is None or row.source_project == project)


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8", newline="")
    os.replace(temporary, path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_text(path, buffer.getvalue())


def materialize_frozen_cv_manifests(
    project_dir: Path,
    manifest_root: Path,
    report_dir: Path,
    protocol: FrozenProtocol,
) -> dict[str, Any]:
    """Regenerate exact frozen memberships, validate, then persist them."""
    started = time.monotonic()
    fingerprint_path = report_dir / "fingerprints.json"
    frozen = _read_frozen_fingerprints(fingerprint_path, protocol)
    rows, by_project = _load_authoritative_rows(project_dir, frozen)
    union_find, _, _, _ = _build_components(rows)
    development, _, final_locked, _ = _chronological_split(by_project, union_find)
    if _membership_fingerprint(development) != frozen["development_membership_sha256"]:
        raise CvManifestError("regenerated development membership fingerprint mismatch")
    if _membership_fingerprint(final_locked) != frozen["locked_test_membership_sha256"]:
        raise CvManifestError("regenerated locked-test membership fingerprint mismatch")
    summaries, chronology, actual_cv = _temporal_folds(by_project, development, union_find)
    if not all(row["chronology_valid"] for row in chronology):
        raise CvManifestError("authoritative temporal fold generation failed chronology")
    memberships = memberships_from_fold_summaries(summaries)
    fingerprint_rows = _validate_fingerprints(memberships, frozen["cv_membership_sha256"])
    if actual_cv != frozen["cv_membership_sha256"]:
        raise CvManifestError("authoritative fold fingerprints differ from frozen fingerprints")
    _validate_counts(summaries, report_dir / "cv_fold_sizes.csv")
    integrity = _validate_integrity(memberships, rows, development, final_locked, union_find)

    staging = report_dir / ".work" / "cv_manifests"
    total_records = persist_cv_memberships(staging, memberships, rows)
    for family in FAMILIES:
        for fold in range(1, protocol.cv_fold_count + 1):
            for role in ROLES:
                staged = _manifest_path(staging, family, fold, role)
                destination = _manifest_path(manifest_root, family, fold, role)
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staged, destination)

    for family in FAMILIES:
        for fold in range(1, protocol.cv_fold_count + 1):
            for role in ROLES:
                load_cv_membership(
                    manifest_root,
                    fingerprint_path,
                    protocol,
                    family=family,
                    fold=fold,
                    role=role,
                )
    elapsed = time.monotonic() - started
    _write_csv(report_dir / "cv_manifest_validation.csv", fingerprint_rows)
    result = {
        "protocol_sha256": protocol.fingerprint,
        "manifest_record_count": total_records,
        "manifest_file_count": 12,
        "fingerprints_matched": 12,
        "count_rows_validated": len(summaries),
        "chronology_checks": len(chronology),
        "integrity_checks": integrity,
        "runtime_seconds": round(elapsed, 3),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    _atomic_text(
        report_dir / "cv_manifest_materialization.json",
        json.dumps(result, indent=2, sort_keys=True) + "\n",
    )
    _atomic_text(
        report_dir / "CV_MANIFEST_MATERIALIZATION.md",
        f"""# Frozen CV Manifest Materialization

This corrective task persisted row-level membership omitted by the original protocol-v1 build.
It reused the authoritative component, chronological split, and temporal-fold implementation;
protocol definitions and existing fingerprints were not changed.

- Manifest files: 12
- Manifest records: {total_records:,}
- Frozen CV fingerprints matched: 12/12
- Frozen family/fold/project count rows matched: {len(summaries)}/54
- Chronological project/fold checks: {len(chronology)}/27 PASS
- TRAIN/VALIDATION overlap: 0 for all six family/fold combinations
- CV/final-locked overlap: 0 for all six family/fold combinations
- Duplicate-component overlap: 0 for all six family/fold combinations
- Runtime: {elapsed:.3f} seconds

The manifests contain only family, fold, role, source project, issue ID, and creation time. They
contain no bug-report text, targets, predictions, or model metrics.

```text
PROTOCOL_DEFINITIONS_CHANGED = NO
FROZEN_FINGERPRINTS_CHANGED = NO
MODELS_FITTED = NO
```
""",
    )
    return result
