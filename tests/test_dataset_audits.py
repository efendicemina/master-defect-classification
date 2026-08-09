import csv
from collections import Counter
from pathlib import Path, PurePosixPath

import pytest

from defect_classifier.catalogue import EXPECTED_PROJECTS, DatasetCatalogue
from defect_classifier.dataset_audits import AuditError, audit_project, run_audit, write_reports

FIELDS = [
    "ID",
    "Severity",
    "Summary",
    "Description",
    "Creation time",
    "Dupe of",
    "Depends on",
]


def _write_csv(path: Path, rows: list[list[str]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields or FIELDS)
        writer.writerows(rows)


def test_project_audit_aggregates_quality_signals(tmp_path: Path) -> None:
    path = tmp_path / "issues.csv"
    _write_csv(
        path,
        [
            ["1", "normal", "Same", "Body\r\nline", "2020-01-01T00:00:00Z", "", ""],
            ["1", "critical", "Same", "Body\nline", "invalid", "", ""],
            ["3", "enhancement", "   ", "", "", "1", ""],
            [
                "4",
                "odd-label",
                "<b>Title</b>",
                "https://example.test a@b.test",
                "2021-01-01T00:00:00",
                "",
                "",
            ],
        ],
    )

    result = audit_project("SYNTHETIC", path, "issues.csv", progress_every=0)

    assert result.raw_rows == result.parsed_rows == result.field_audited_rows == 4
    assert result.severity == Counter(
        {"normal": 1, "critical": 1, "enhancement": 1, "odd-label": 1}
    )
    assert result.summary_missing == 0
    assert result.summary_blank == 1
    assert result.description_missing == 1
    assert result.description_blank == 0
    assert result.both_text_unavailable == 1
    assert result.timestamp_parse_success == 2
    assert result.timestamp_parse_failure == 1
    assert result.timestamp_null == 1
    assert result.timestamp_aware == 1
    assert result.timestamp_naive == 1
    assert result.issue_id_non_null == 4
    assert result.issue_id_unique == 3
    assert result.issue_id_duplicate_rows == 2
    duplicates = [count for count in result.duplicate_hash_counts.values() if count > 1]
    assert duplicates == [2]
    assert result.html_rows == 1
    assert result.url_rows == 1
    assert result.email_rows == 1


def test_project_audit_counts_wrong_width_as_malformed(tmp_path: Path) -> None:
    path = tmp_path / "issues.csv"
    path.write_text(
        ",".join(FIELDS) + "\n1,normal,summary\n2,normal,s,d,2020-01-01T00:00:00Z,,\n",
        encoding="utf-8",
    )
    result = audit_project("SYNTHETIC", path, "issues.csv", progress_every=0)
    assert result.raw_rows == 2
    assert result.parsed_rows == 2
    assert result.field_audited_rows == 1
    assert result.malformed_rows == 1
    assert result.malformed_record_samples == [
        {
            "record_number": 1,
            "ending_physical_line": 2,
            "observed_fields": 3,
            "expected_fields": 7,
        }
    ]


def test_project_audit_rejects_missing_required_field(tmp_path: Path) -> None:
    path = tmp_path / "issues.csv"
    _write_csv(
        path,
        [["1", "normal", "summary", "2020-01-01T00:00:00Z", "", ""]],
        fields=[field for field in FIELDS if field != "Description"],
    )
    with pytest.raises(AuditError, match="Description"):
        audit_project("SYNTHETIC", path, "issues.csv", progress_every=0)


def test_catalogue_integration_and_schema_differences(tmp_path: Path) -> None:
    projects = {}
    for index, project in enumerate(EXPECTED_PROJECTS):
        relative = PurePosixPath(project) / "issues.csv"
        projects[project] = relative
        fields = FIELDS + (["Project-specific"] if index == 0 else [])
        row = [str(index), "normal", "summary", "description", "2020-01-01T00:00:00Z", "", ""]
        if index == 0:
            row.append("value")
        _write_csv(tmp_path.joinpath(*relative.parts), [row], fields)
    output = tmp_path / "reports"
    results = run_audit(
        tmp_path,
        DatasetCatalogue(projects=projects, version=1),
        output,
        progress_every=0,
    )

    assert len(results) == 9
    summary = (output / "schema_summary.json").read_text(encoding="utf-8")
    assert '"unique_to_project": [\n        "Project-specific"' in summary
    assert (output / "DATASET_AUDIT.md").is_file()


def test_report_output_is_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "issues.csv"
    _write_csv(source, [["1", "normal", "s", "d", "2020-01-01T00:00:00Z", "", ""]])
    result = audit_project("SYNTHETIC", source, "issues.csv", progress_every=0)
    first = tmp_path / "first"
    second = tmp_path / "second"

    write_reports((result,), first, 1.25)
    write_reports((result,), second, 1.25)

    first_files = sorted(path.relative_to(first) for path in first.iterdir())
    second_files = sorted(path.relative_to(second) for path in second.iterdir())
    assert first_files == second_files
    for relative in first_files:
        assert (first / relative).read_bytes() == (second / relative).read_bytes()
