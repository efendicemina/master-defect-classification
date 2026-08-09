import random

import pytest

from defect_classifier.reproducibility import seed_everything


def test_seed_everything_is_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    independent_a = seed_everything(2026)
    global_a = [random.random() for _ in range(3)]
    independent_values_a = [independent_a.random() for _ in range(3)]

    independent_b = seed_everything(2026)
    assert [random.random() for _ in range(3)] == global_a
    assert [independent_b.random() for _ in range(3)] == independent_values_a
    assert __import__("os").environ["PYTHONHASHSEED"] == "2026"


def test_seed_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        seed_everything(-1)
