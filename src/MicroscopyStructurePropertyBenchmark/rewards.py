from __future__ import annotations

import numpy as np


ENERGY_RANGES: dict[str, tuple[float, float]] = {
    "dipole": (0.35, 0.55),
    "edge": (0.60, 0.75),
    "bulk": (0.80, 1.00),
}

STRUCTURAL_REWARDS = {"defect", "gradient"}
WINDOWED_REWARDS = {
    "composition",
    "peak",
    "peak_intensity",
    "fitted_peak_area",
    "fitted_peak_center",
    "fitted_peak_width",
}


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
    elif reward in {"fitted_peak_area", "fitted_peak_center", "fitted_peak_width"}:
        scalarizer = fitted_peak_scalarizer(
            spectrum_image,
            energy_axis,
            energy_range,
            parameter=reward.removeprefix("fitted_peak_"),
        )
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


def fitted_peak_scalarizer(
    spectrum_image: np.ndarray,
    energy_axis: np.ndarray,
    energy_range: tuple[float, float] | None = None,
    parameter: str = "area",
) -> np.ndarray:
    """Fit a Gaussian peak plus linear background and return a fitted peak parameter."""

    start, end = _energy_window_indices(energy_axis, energy_range)
    energy = np.asarray(energy_axis[start : end + 1], dtype=np.float32)
    spectra = np.asarray(spectrum_image[:, :, start : end + 1], dtype=np.float32)
    flat = spectra.reshape(-1, spectra.shape[-1])
    fitted = np.array([_fit_gaussian_peak(energy, spectrum, parameter) for spectrum in flat], dtype=np.float32)
    return fitted.reshape(spectra.shape[:2])


def _fit_gaussian_peak(energy: np.ndarray, spectrum: np.ndarray, parameter: str) -> float:
    if energy.size < 4 or np.isclose(float(np.ptp(energy)), 0.0):
        return 0.0
