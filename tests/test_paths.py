from pathlib import Path, PurePosixPath

import pytest

from defect_classifier.paths import dataset_path


def test_dataset_path_resolves_beneath_root(tmp_path: Path) -> None:
    expected = tmp_path / "project" / "issues.csv"
    assert dataset_path(tmp_path, PurePosixPath("project/issues.csv")) == expected.resolve()


def test_dataset_path_rejects_escape(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="escapes"):
        dataset_path(tmp_path, PurePosixPath("../outside.csv"))
