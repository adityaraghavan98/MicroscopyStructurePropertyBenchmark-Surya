from __future__ import annotations

import numpy as np


def observed_value_coverage(acquired_indices: list[int], target: np.ndarray, n_bins: int = 100) -> float:
    """Fraction of target-value bins touched by acquired points."""

    if not acquired_indices:
        return 0.0
    target = np.asarray(target)
    bins = np.linspace(float(target.min()), float(target.max()), n_bins + 1)
    labels = np.digitize(target, bins[1:-1], right=False)
    return float(len(set(labels[acquired_indices])) / max(1, len(set(labels))))
