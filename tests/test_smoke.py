import numpy as np
import csv

from MicroscopyStructurePropertyBenchmark.datasets import make_synthetic_dataset
from MicroscopyStructurePropertyBenchmark.metrics import observed_value_coverage
from MicroscopyStructurePropertyBenchmark.rewards import spectrum_sum_scalarizer
from MicroscopyStructurePropertyBenchmark.runners.active_learning import run_benchmark
from MicroscopyStructurePropertyBenchmark.runners.sweep import run_sweep


def test_synthetic_dataset_shapes():
    dataset = make_synthetic_dataset(grid_shape=(8, 9), patch_size=4, seed=1)
    assert dataset.image.shape == (8, 9)
    assert dataset.coords.shape == (72, 2)
    assert dataset.patches.shape == (72, 4, 4)
    assert dataset.target.shape == (72,)


def test_observed_value_coverage_is_bounded():
    target = np.linspace(0, 1, 10)
    coverage = observed_value_coverage([0, 5, 9], target, n_bins=5)
    assert 0.0 <= coverage <= 1.0


def test_defect_reward_finds_abnormal_spectrum():
    spectrum_image = np.ones((5, 5, 6), dtype=np.float32)
    spectrum_image[2, 3] = np.array([1, 1, 8, 8, 1, 1], dtype=np.float32)
    energy_axis = np.linspace(0.0, 1.0, 6, dtype=np.float32)

    reward = spectrum_sum_scalarizer(spectrum_image, energy_axis, reward="defect")

    assert np.unravel_index(int(np.argmax(reward)), reward.shape) == (2, 3)
    assert reward[2, 3] == 1.0


def test_gradient_reward_scores_fast_signal_changes():
    spectrum_image = np.zeros((5, 5, 4), dtype=np.float32)
    spectrum_image[:, 3:, :] = 5.0
    energy_axis = np.linspace(0.0, 1.0, 4, dtype=np.float32)

    reward = spectrum_sum_scalarizer(spectrum_image, energy_axis, reward="gradient")

    assert reward[:, 2].mean() > reward[:, 0].mean()
    assert reward[:, 2].mean() > reward[:, 4].mean()


def test_composition_reward_uses_requested_energy_range():
    spectrum_image = np.zeros((4, 4, 5), dtype=np.float32)
    spectrum_image[1, 2, 2] = 8.0
    spectrum_image[3, 3, 4] = 20.0
    energy_axis = np.linspace(0.0, 1.0, 5, dtype=np.float32)

    reward = spectrum_sum_scalarizer(
        spectrum_image,
        energy_axis,
        reward="composition",
        energy_range=(0.45, 0.55),
    )

    assert np.unravel_index(int(np.argmax(reward)), reward.shape) == (1, 2)
    assert reward[3, 3] == 0.0


def test_peak_intensity_reward_uses_requested_energy_range():
    spectrum_image = np.zeros((4, 4, 5), dtype=np.float32)
    spectrum_image[0, 1, 1] = 4.0
    spectrum_image[2, 2, 2] = 9.0
    spectrum_image[3, 3, 4] = 30.0
    energy_axis = np.linspace(0.0, 1.0, 5, dtype=np.float32)

    reward = spectrum_sum_scalarizer(
        spectrum_image,
        energy_axis,
        reward="peak_intensity",
        energy_range=(0.20, 0.55),
    )

    assert np.unravel_index(int(np.argmax(reward)), reward.shape) == (2, 2)
    assert reward[3, 3] == 0.0


def test_fitted_peak_area_reward_scores_fitted_gaussian_area():
    energy_axis = np.linspace(0.0, 1.0, 41, dtype=np.float32)
    spectrum_image = np.zeros((3, 3, energy_axis.size), dtype=np.float32)
    baseline = 0.2 + 0.1 * energy_axis
    gaussian = np.exp(-0.5 * ((energy_axis - 0.52) / 0.06) ** 2)
    spectrum_image[:] = baseline
    spectrum_image[1, 1] += 6.0 * gaussian
    spectrum_image[0, 2] += 3.0 * gaussian

    reward = spectrum_sum_scalarizer(
        spectrum_image,
        energy_axis,
        reward="fitted_peak_area",
        energy_range=(0.30, 0.75),
    )

    assert np.unravel_index(int(np.argmax(reward)), reward.shape) == (1, 1)
    assert reward[1, 1] == 1.0
