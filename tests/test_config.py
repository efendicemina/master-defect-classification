from pathlib import Path

import pytest

from defect_classifier.config import DATA_ROOT_ENV, ConfigurationError, resolve_data_root


def test_resolve_data_root_from_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path))
    assert resolve_data_root() == tmp_path.resolve()


def test_missing_environment_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(DATA_ROOT_ENV, raising=False)
    with pytest.raises(ConfigurationError, match=DATA_ROOT_ENV):
        resolve_data_root()
