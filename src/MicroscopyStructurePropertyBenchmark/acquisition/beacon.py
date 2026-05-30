from __future__ import annotations

import numpy as np


def beacon_scores(
    candidate_features: np.ndarray,
    acquired_features: np.ndarray,
    acquired_values: np.ndarray,
    mean: np.ndarray,
    elite_fraction: float = 0.1,
) -> np.ndarray:
    """A lightweight BEACON-style score over representation space.

    Candidates are rewarded for predicted value and distance from the current
    elite set. This keeps the benchmark interface simple while the DKL-BEACON
    notebook logic is distilled into a production implementation.
    """

    if acquired_features.size == 0:
        return mean

    candidate_features = _as_feature_matrix(candidate_features)
    acquired_features = _as_feature_matrix(acquired_features)
    n_elite = max(1, int(np.ceil(len(acquired_values) * elite_fraction)))
    elite_idx = np.argsort(acquired_values)[-n_elite:]
    elite_features = acquired_features[elite_idx]

    distances = np.linalg.norm(candidate_features[:, None, :] - elite_features[None, :, :], axis=2)
    novelty = distances.min(axis=1)
    novelty = _normalize(novelty)
    value = _normalize(mean)
    return 0.5 * value + 0.5 * novelty


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    span = values.max() - values.min()
    if np.isclose(span, 0.0):
        return np.zeros_like(values)
    return (values - values.min()) / span


def _as_feature_matrix(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features)
    return features.reshape(features.shape[0], -1)
