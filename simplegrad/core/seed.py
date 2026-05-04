import numpy as np
import random

_SEED = None


def seed(value: int) -> None:
    """Set the global random seed for reproducibility."""
    random.seed(value)
    np.random.seed(value)
    global _SEED
    _SEED = value


def get_seed() -> int | None:
    return _SEED
