from __future__ import annotations

import numpy as np


ENERGY_RANGES: dict[str, tuple[float, float]] = {
    "dipole": (0.35, 0.55),
    "edge": (0.60, 0.75),
    "bulk": (0.80, 1.00),
}


def spectrum_sum_scalarizer(
    spectrum_image: np.ndarray,
    energy_axis: np.ndarray,
    reward: str = "dipole",
    normalize: bool = True,
) -> np.ndarray:
    """Return a scalar reward map by summing spectra over a named energy window."""

    if reward == "zero":
        scalarizer = np.zeros(spectrum_image.shape[:2], dtype=np.float32)
    else:
        if reward not in ENERGY_RANGES:
            valid = ", ".join(sorted([*ENERGY_RANGES, "zero"]))
            raise ValueError(f"Unknown reward '{reward}'. Expected one of: {valid}.")
        e_min, e_max = ENERGY_RANGES[reward]
        start = int(np.abs(energy_axis - e_min).argmin())
        end = int(np.abs(energy_axis - e_max).argmin())
        if end < start:
            start, end = end, start
        scalarizer = spectrum_image[:, :, start : end + 1].sum(axis=-1).astype(np.float32)

    return normalize_values(scalarizer) if normalize else scalarizer


def normalize_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if np.isclose(max_value, min_value):
        return np.zeros_like(values, dtype=np.float32)
    return ((values - min_value) / (max_value - min_value)).astype(np.float32)
