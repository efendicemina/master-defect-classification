"""Environment-backed project configuration."""

from __future__ import annotations

import os
from pathlib import Path

DATA_ROOT_ENV = "ECLIPSE_DATA_ROOT"


class ConfigurationError(ValueError):
    """Raised when required local configuration is absent or invalid."""


def resolve_data_root(value: str | Path | None = None) -> Path:
    """Resolve the external dataset root without requiring it to exist yet."""
    raw_value = value if value is not None else os.environ.get(DATA_ROOT_ENV)
    if raw_value is None or not str(raw_value).strip():
        raise ConfigurationError(
            f"{DATA_ROOT_ENV} is not set; export it as the external Eclipse dataset directory"
        )
    return Path(raw_value).expanduser().resolve()
