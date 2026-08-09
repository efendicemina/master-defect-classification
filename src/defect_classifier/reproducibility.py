"""Small reproducibility utilities shared by future experiments."""

from __future__ import annotations

import os
import random


def seed_everything(seed: int) -> random.Random:
    """Seed Python randomness and return an independent identically seeded generator.

    Setting ``PYTHONHASHSEED`` records the requested value for child processes. Python's hash
    seed for the current interpreter must still be supplied before interpreter startup.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    return random.Random(seed)
