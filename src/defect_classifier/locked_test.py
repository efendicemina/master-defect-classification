"""Fail-closed guard for future locked-test reads."""

from __future__ import annotations

import os

from defect_classifier.protocol import FrozenProtocol


class LockedTestAccessError(PermissionError):
    """Raised when development code attempts to access locked-test artifacts."""


def require_locked_test_unlock(protocol: FrozenProtocol) -> None:
    """Require the explicit final-evaluation unlock configured by the frozen protocol."""
    actual = os.environ.get(protocol.unlock_environment_variable)
    if actual != protocol.unlock_value:
        raise LockedTestAccessError(
            "locked-test access denied; use only the separate final-evaluation workflow"
        )
