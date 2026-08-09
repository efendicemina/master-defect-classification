"""Canonical dataset catalogue loading and validation."""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

EXPECTED_PROJECTS = (
    "BIRT",
    "MYLYN",
    "CDT",
    "EQUINOX",
    "JDT",
    "PDE",
    "PAPYRUS",
    "PLATFORM",
    "TPTP",
)


class CatalogueError(ValueError):
    """Raised when the catalogue violates its data contract."""


@dataclass(frozen=True)
class DatasetCatalogue:
    """Validated project-to-relative-CSV mapping."""

    projects: Mapping[str, PurePosixPath]
    version: int


def default_catalogue_path() -> Path:
    """Return the repository catalogue path for source or editable installations."""
    return Path(__file__).resolve().parents[2] / "configs" / "datasets.toml"


def load_catalogue(path: str | Path | None = None) -> DatasetCatalogue:
    catalogue_path = Path(path) if path is not None else default_catalogue_path()
    try:
        with catalogue_path.open("rb") as handle:
            document = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise CatalogueError(f"cannot read dataset catalogue {catalogue_path}: {exc}") from exc

    raw_projects = document.get("projects")
    if not isinstance(raw_projects, dict):
        raise CatalogueError("catalogue must contain a [projects] table")

    actual = set(raw_projects)
    expected = set(EXPECTED_PROJECTS)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise CatalogueError(f"catalogue project mismatch; missing={missing}, extra={extra}")

    projects: dict[str, PurePosixPath] = {}
    for project in EXPECTED_PROJECTS:
        raw_path = raw_projects[project]
        if not isinstance(raw_path, str):
            raise CatalogueError(f"path for {project} must be a string")
        relative_path = PurePosixPath(raw_path)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise CatalogueError(f"path for {project} must be a safe relative path")
        if relative_path.suffix.lower() != ".csv":
            raise CatalogueError(f"path for {project} must identify a CSV file")
        projects[project] = relative_path

    version = document.get("version")
    if not isinstance(version, int) or version < 1:
        raise CatalogueError("catalogue version must be a positive integer")
    return DatasetCatalogue(projects=projects, version=version)
