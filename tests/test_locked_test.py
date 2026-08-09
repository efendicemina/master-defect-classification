import pytest

from defect_classifier.locked_test import LockedTestAccessError, require_locked_test_unlock
from defect_classifier.protocol import load_protocol


def test_locked_test_guard_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    protocol = load_protocol()
    monkeypatch.delenv(protocol.unlock_environment_variable, raising=False)
    with pytest.raises(LockedTestAccessError):
        require_locked_test_unlock(protocol)


def test_locked_test_guard_requires_exact_explicit_value(monkeypatch: pytest.MonkeyPatch) -> None:
    protocol = load_protocol()
    monkeypatch.setenv(protocol.unlock_environment_variable, "wrong")
    with pytest.raises(LockedTestAccessError):
        require_locked_test_unlock(protocol)
    monkeypatch.setenv(protocol.unlock_environment_variable, protocol.unlock_value)
    require_locked_test_unlock(protocol)
