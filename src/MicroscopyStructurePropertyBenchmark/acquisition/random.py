from __future__ import annotations

import numpy as np


def random_scores(n_candidates: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random(n_candidates)
