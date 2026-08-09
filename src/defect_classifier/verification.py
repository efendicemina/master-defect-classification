"""Metadata-only external dataset verification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from defect_classifier.catalogue import DatasetCatalogue
from defect_classifier.paths import dataset_path


@dataclass(frozen=True)
class FileCheck:
    project: str
    relative_path: str
    path: Path
    exists: bool
    size_bytes: int | None


def verify_dataset(root: Path, catalogue: DatasetCatalogue) -> tuple[FileCheck, ...]:
    """Inspect presence and size only; never open or parse a CSV."""
    checks = []
    for project, relative_path in catalogue.projects.items():
        path = dataset_path(root, relative_path)
        is_file = path.is_file()
        checks.append(
            FileCheck(
                project=project,
                relative_path=relative_path.as_posix(),
                path=path,
                exists=is_file,
                size_bytes=path.stat().st_size if is_file else None,
            )
        )
    return tuple(checks)
