"""Safe resolution of external dataset paths."""

from __future__ import annotations

from pathlib import Path, PurePosixPath


def dataset_path(root: Path, relative_path: PurePosixPath) -> Path:
    """Resolve a validated catalogue path beneath the external data root."""
    root = root.resolve()
    candidate = root.joinpath(*relative_path.parts).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"dataset path escapes configured root: {relative_path}")
    return candidate
