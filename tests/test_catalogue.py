from pathlib import Path

from defect_classifier.catalogue import EXPECTED_PROJECTS, load_catalogue


def test_committed_catalogue_is_complete() -> None:
    catalogue = load_catalogue()
    assert tuple(catalogue.projects) == EXPECTED_PROJECTS
    assert len(set(catalogue.projects.values())) == 9
    assert all(not path.is_absolute() for path in catalogue.projects.values())


def test_catalogue_can_be_loaded_from_explicit_path(tmp_path: Path) -> None:
    projects = "\n".join(f'{name} = "folder/{name}.csv"' for name in EXPECTED_PROJECTS)
    path = tmp_path / "catalogue.toml"
    path.write_text(f"version = 2\n[projects]\n{projects}\n", encoding="utf-8")
    assert load_catalogue(path).version == 2
