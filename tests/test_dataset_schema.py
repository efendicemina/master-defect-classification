from pathlib import Path

import pytest

from defect_classifier.dataset_schema import DatasetSchemaError, discover_csv_schema


def test_schema_discovers_delimiter_columns_and_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "issues.csv"
    path.write_text("ID;Summary;Summary\n1;first;second\n", encoding="utf-8")

    schema = discover_csv_schema(path)

    assert schema.delimiter == ";"
    assert schema.columns == ("ID", "Summary", "Summary")
    assert schema.duplicate_columns == ("Summary",)
    assert schema.encoding == "utf-8-sig"


def test_schema_rejects_non_utf8(tmp_path: Path) -> None:
    path = tmp_path / "issues.csv"
    path.write_bytes(b"ID,Summary\n1,\xff\n")
    with pytest.raises(DatasetSchemaError, match="not valid UTF-8"):
        discover_csv_schema(path)
