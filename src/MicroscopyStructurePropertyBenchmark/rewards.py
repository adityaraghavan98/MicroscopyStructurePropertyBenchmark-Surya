from __future__ import annotations

import numpy as np


ENERGY_RANGES: dict[str, tuple[float, float]] = {
    "dipole": (0.35, 0.55),
    "edge": (0.60, 0.75),
    "bulk": (0.80, 1.00),
}

STRUCTURAL_REWARDS = {"defect", "gradient"}
WINDOWED_REWARDS = {"composition", "peak", "peak_intensity"}


def spectrum_sum_scalarizer(
    spectrum_image: np.ndarray,
    energy_axis: np.ndarray,
    reward: str = "dipole",
    normalize: bool = True,
    energy_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """Return a scalar reward map by summing spectra over a named energy window."""

    if reward == "zero":
        scalarizer = np.zeros(spectrum_image.shape[:2], dtype=np.float32)
    elif reward == "defect":
        scalarizer = defect_scalarizer(spectrum_image)
    elif reward == "gradient":
        scalarizer = gradient_scalarizer(spectrum_image)
    elif reward == "composition":
        scalarizer = composition_scalarizer(spectrum_image, energy_axis, energy_range)
    elif reward in {"peak", "peak_intensity"}:
        scalarizer = peak_intensity_scalarizer(spectrum_image, energy_axis, energy_range)
    else:
        if reward not in ENERGY_RANGES:
            valid = ", ".join(sorted([*ENERGY_RANGES, *STRUCTURAL_REWARDS, *WINDOWED_REWARDS, "zero"]))
            raise ValueError(f"Unknown reward '{reward}'. Expected one of: {valid}.")
        start, end = _energy_window_indices(energy_axis, ENERGY_RANGES[reward])
        scalarizer = spectrum_image[:, :, start : end + 1].sum(axis=-1).astype(np.float32)

    return normalize_values(scalarizer) if normalize else scalarizer


def composition_scalarizer(
    spectrum_image: np.ndarray,
    energy_axis: np.ndarray,
    energy_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """Return high values where total signal is strong in a chosen elemental window."""

    start, end = _energy_window_indices(energy_axis, energy_range)
    return spectrum_image[:, :, start : end + 1].sum(axis=-1).astype(np.float32)


def peak_intensity_scalarizer(
    spectrum_image: np.ndarray,
    energy_axis: np.ndarray,
    energy_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """Return high values where the strongest spectral peak is large."""

    start, end = _energy_window_indices(energy_axis, energy_range)
    return spectrum_image[:, :, start : end + 1].max(axis=-1).astype(np.float32)


def defect_scalarizer(spectrum_image: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Return high values for spectra that are robustly far from the median spectrum."""

    spectra = np.asarray(spectrum_image, dtype=np.float32)
    flat = spectra.reshape(-1, spectra.shape[-1])
    median_spectrum = np.median(flat, axis=0)
    mad_spectrum = np.median(np.abs(flat - median_spectrum), axis=0)
    robust_z = (flat - median_spectrum) / (1.4826 * mad_spectrum + eps)
    scores = np.sqrt(np.mean(robust_z**2, axis=1))
    return scores.reshape(spectra.shape[:2]).astype(np.float32)


def gradient_scalarizer(spectrum_image: np.ndarray) -> np.ndarray:
    """Return high values where the integrated spectral signal changes quickly."""

    intensity = np.asarray(spectrum_image, dtype=np.float32).sum(axis=-1)
    grad_y, grad_x = np.gradient(intensity)
    return np.sqrt(grad_x**2 + grad_y**2).astype(np.float32)


def _energy_window_indices(
    energy_axis: np.ndarray,
    energy_range: tuple[float, float] | None = None,
) -> tuple[int, int]:
    if energy_range is None:
        return 0, len(energy_axis) - 1
    e_min, e_max = energy_range
    start = int(np.abs(energy_axis - e_min).argmin())
    end = int(np.abs(energy_axis - e_max).argmin())
    if end < start:
        start, end = end, start
    return start, end


def normalize_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if np.isclose(max_value, min_value):
        return np.zeros_like(values, dtype=np.float32)
    return ((values - min_value) / (max_value - min_value)).astype(np.float32)
