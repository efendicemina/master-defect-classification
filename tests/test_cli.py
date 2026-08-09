from pathlib import Path

from defect_classifier.catalogue import EXPECTED_PROJECTS
from defect_classifier.cli import main


def _catalogue(path: Path) -> Path:
    projects = "\n".join(f'{name} = "{name}/issues.csv"' for name in EXPECTED_PROJECTS)
    path.write_text(f"version = 1\n[projects]\n{projects}\n", encoding="utf-8")
    return path


def test_cli_returns_nonzero_for_missing_dataset(tmp_path: Path) -> None:
    catalogue = _catalogue(tmp_path / "datasets.toml")
    assert (
        main(["verify-dataset", "--data-root", str(tmp_path), "--catalogue", str(catalogue)]) == 1
    )


def test_cli_succeeds_for_complete_fake_dataset(tmp_path: Path) -> None:
    catalogue = _catalogue(tmp_path / "datasets.toml")
    for project in EXPECTED_PROJECTS:
        csv = tmp_path / project / "issues.csv"
        csv.parent.mkdir()
        csv.touch()
    assert (
        main(["verify-dataset", "--data-root", str(tmp_path), "--catalogue", str(catalogue)]) == 0
    )
