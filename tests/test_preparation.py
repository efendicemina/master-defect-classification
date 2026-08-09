import csv
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from defect_classifier.preparation import (
    PreparationError,
    RowMeta,
    _build_components,
    _chronological_split,
    _membership_fingerprint,
    _overlap_reasons,
    _stream_project,
    _temporal_folds,
    prepare_protocol_v1,
)
from defect_classifier.protocol import load_protocol


def _raw_row(protocol, **overrides: str) -> list[str]:
    values = dict.fromkeys(protocol.raw_columns, "")
    values.update(
        {
            "ID": "1",
            "Severity": "normal",
            "Creation time": "2020-01-01T00:00:00Z",
            "Summary": "summary",
            "Description": "description",
        }
    )
    values.update(overrides)
    return [values[column] for column in protocol.raw_columns]


def _write_raw(path: Path, protocol, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(protocol.raw_columns)
        writer.writerows(rows)


def _meta(
    project: str,
    issue_id: str,
    day: int,
    *,
    text_hash: str | None = None,
    dupe_of: str = "",
) -> RowMeta:
    return RowMeta(
        stable_id=f"{project}:{issue_id}",
        source_project=project,
        issue_id=issue_id,
        creation_time=datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=day),
        severity_raw="normal",
        target_s6="normal",
        target_s3="MEDIUM",
        target_s2="LOWER_IMPACT",
        dupe_of_raw=dupe_of,
        exact_text_hash=text_hash or f"hash-{project}-{issue_id}",
    )


def test_eligibility_enhancement_and_missing_text_rules(tmp_path: Path) -> None:
    protocol = load_protocol()
    source = tmp_path / "issues.csv"
    rows = [
        _raw_row(protocol, ID="1", Severity="enhancement"),
        _raw_row(protocol, ID="2", Summary="", Description="description"),
        _raw_row(protocol, ID="3", Summary="summary", Description=""),
        _raw_row(protocol, ID="4", Summary="", Description="   "),
    ]
    _write_raw(source, protocol, rows)
    metadata, population = _stream_project(
        "TEST", source, "issues.csv", protocol, tmp_path / "output.parquet", progress_every=0
    )
    assert [row.issue_id for row in metadata] == ["2", "3"]
    assert population.excluded_enhancement == 1
    assert population.excluded_missing_text == 1


def test_unknown_severity_fails_closed(tmp_path: Path) -> None:
    protocol = load_protocol()
    source = tmp_path / "issues.csv"
    _write_raw(source, protocol, [_raw_row(protocol, Severity="unknown")])
    with pytest.raises(PreparationError, match="unknown severity"):
        _stream_project(
            "TEST", source, "issues.csv", protocol, tmp_path / "output.parquet", progress_every=0
        )


def test_trailing_unused_fields_do_not_invalidate_row(tmp_path: Path) -> None:
    protocol = load_protocol()
    source = tmp_path / "issues.csv"
    row = _raw_row(protocol)
    _write_raw(source, protocol, [row[:-3]])
    metadata, population = _stream_project(
        "TEST", source, "issues.csv", protocol, tmp_path / "output.parquet", progress_every=0
    )
    assert len(metadata) == 1
    assert population.malformed_trailing_rows == 1


def test_chronological_split_is_deterministic_and_uses_issue_id_ties() -> None:
    rows = [_meta("P", str(issue_id), 0) for issue_id in (10, 2, 1, 3, 4)]
    union_find, _, _, _ = _build_components(rows)
    development, candidates, final, split = _chronological_split({"P": rows}, union_find)
    assert development == {"P:1", "P:2", "P:3", "P:4"}
    assert candidates == final == {"P:10"}
    assert split[0]["chronology_boundary_index"] == 4


def test_exact_and_explicit_components_remove_future_not_earlier() -> None:
    rows = [
        _meta("A", "1", 0, text_hash="same"),
        _meta("A", "2", 1, dupe_of="1"),
        _meta("B", "3", 2, text_hash="same"),
        _meta("A", "4", 3),
        _meta("A", "5", 4),
        _meta("A", "6", 5, dupe_of="1"),
    ]
    union_find, _, unresolved, counts = _build_components(rows)
    assert union_find.find("A:1") == union_find.find("A:2")
    assert union_find.find("A:1") == union_find.find("B:3")
    assert not unresolved
    assert counts["resolved"] == 2
    development = {"A:1", "A:2", "A:4"}
    candidates = {"B:3", "A:5", "A:6"}
    reasons = _overlap_reasons(rows, development, candidates, union_find)
    assert reasons == {"B:3": "exact_text", "A:6": "explicit_dupe_link"}
    assert "B:3" not in candidates - set(reasons)
    assert "A:1" in development


def test_expanding_folds_have_no_reversal_and_protect_duplicates() -> None:
    rows = [_meta("P", str(index), index) for index in range(8)]
    rows[1] = replace(rows[1], exact_text_hash="duplicate")
    rows[2] = replace(rows[2], exact_text_hash="duplicate")
    union_find, _, _, _ = _build_components(rows)
    development = {row.stable_id for row in rows}
    sizes, proofs, fingerprints = _temporal_folds({"P": rows}, development, union_find)
    within_fold_one = next(
        row for row in sizes if row["family"] == "within_project" and row["fold"] == 1
    )
    assert within_fold_one["training_rows"] == 2
    assert within_fold_one["validation_candidate_rows"] == 2
    assert within_fold_one["validation_removed_exact_text"] == 1
    assert within_fold_one["validation_final_rows"] == 1
    assert all(row["chronology_valid"] for row in proofs)
    assert len(fingerprints) == 12


def test_membership_fingerprints_are_order_independent_and_sensitive() -> None:
    assert _membership_fingerprint(["B:2", "A:1"]) == _membership_fingerprint(["A:1", "B:2"])
    assert _membership_fingerprint(["A:1"]) != _membership_fingerprint(["A:2"])


def test_existing_frozen_split_rejects_protocol_drift(tmp_path: Path) -> None:
    protocol = load_protocol()
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    (report_dir / "fingerprints.json").write_text(
        '{"protocol_sha256": "different"}\n', encoding="utf-8"
    )
    with pytest.raises(PreparationError, match="protocol configuration differs"):
        prepare_protocol_v1(
            tmp_path,
            object(),  # type: ignore[arg-type]
            protocol,
            tmp_path / "processed",
            tmp_path / "locked",
            report_dir,
            progress_every=0,
        )
