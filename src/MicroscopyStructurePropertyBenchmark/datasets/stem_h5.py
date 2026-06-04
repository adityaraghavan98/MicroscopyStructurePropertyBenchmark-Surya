from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from MicroscopyStructurePropertyBenchmark.datasets.synthetic import MicroscopyDataset
from MicroscopyStructurePropertyBenchmark.rewards import spectrum_sum_scalarizer


def load_stem_h5_dataset(
    path: str | Path,
    patch_size: int = 8,
    reward: str = "dipole",
    normalize_reward: bool = True,
    reward_energy_range: tuple[float, float] | None = None,
    image_path: str = "Measurement_000/Channel_000/generic/generic",
    spectrum_path: str = "Measurement_000/Channel_001/generic/generic",
    energy_path: str = "Measurement_000/Channel_001/generic/energy_scale",
) -> MicroscopyDataset:
    """Load the DTMicroscope STEM H5 test data as a benchmark dataset."""

    with h5py.File(path, "r") as h5:
        image = np.asarray(h5[image_path], dtype=np.float32)
        spectrum_image = np.asarray(h5[spectrum_path], dtype=np.float32)
        energy_axis = np.asarray(h5[energy_path], dtype=np.float32)

    target_img = spectrum_sum_scalarizer(
        spectrum_image=spectrum_image,
        energy_axis=energy_axis,
        reward=reward,
        normalize=normalize_reward,
        energy_range=reward_energy_range,
    )
    coords = _coord_grid(image.shape)
    patches = _patches_from_image(image, coords, patch_size)
    target = target_img[coords[:, 0], coords[:, 1]].astype(np.float32)
    image = _normalize_image(image)
    return MicroscopyDataset(image=image, patches=patches, coords=coords, target=target)


def _coord_grid(shape: tuple[int, int]) -> np.ndarray:
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    return np.column_stack([yy.ravel(), xx.ravel()]).astype(np.int64)


def _patches_from_image(image: np.ndarray, coords: np.ndarray, patch_size: int) -> np.ndarray:
    pad_before = patch_size // 2
    pad_after = patch_size - pad_before - 1
    padded = np.pad(image, ((pad_before, pad_after), (pad_before, pad_after)), mode="reflect")
    patches = []
    for row, col in coords:
        row_start = int(row)
        col_start = int(col)
        patch = padded[row_start : row_start + patch_size, col_start : col_start + patch_size]
        patches.append(patch.astype(np.float32))
    return np.stack(patches, axis=0)


def _normalize_image(image: np.ndarray) -> np.ndarray:
    min_value = float(np.min(image))
    max_value = float(np.max(image))
    if np.isclose(max_value, min_value):
        return np.zeros_like(image, dtype=np.float32)
    return ((image - min_value) / (max_value - min_value)).astype(np.float32)
