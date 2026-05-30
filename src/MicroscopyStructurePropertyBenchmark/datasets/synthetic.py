from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MicroscopyDataset:
    """Discrete microscopy benchmark data."""

    image: np.ndarray
    patches: np.ndarray
    coords: np.ndarray
    target: np.ndarray


def make_synthetic_dataset(
    grid_shape: tuple[int, int] = (24, 24),
    patch_size: int = 8,
    noise: float = 0.03,
    seed: int = 0,
) -> MicroscopyDataset:
    """Create a small microscopy-like patch dataset with a hidden scalarizer."""

    rng = np.random.default_rng(seed)
    rows, cols = grid_shape
    yy, xx = np.mgrid[0:rows, 0:cols]
    x = xx / max(cols - 1, 1)
    y = yy / max(rows - 1, 1)

    blob_a = np.exp(-((x - 0.72) ** 2 + (y - 0.28) ** 2) / 0.018)
    blob_b = 0.75 * np.exp(-((x - 0.32) ** 2 + (y - 0.70) ** 2) / 0.035)
    ripple = 0.18 * np.sin(4 * np.pi * x) * np.cos(3 * np.pi * y)
    image = blob_a + blob_b + ripple + noise * rng.normal(size=grid_shape)
    image = _normalize(image)

    coords = np.column_stack([yy.ravel(), xx.ravel()]).astype(np.int64)
    target = _normalize((0.65 * blob_a + 0.35 * blob_b + 0.10 * ripple).ravel())
    patches = _patches_from_image(image, coords, patch_size)
    return MicroscopyDataset(image=image, patches=patches, coords=coords, target=target)


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


def _normalize(values: np.ndarray) -> np.ndarray:
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if np.isclose(max_value, min_value):
        return np.zeros_like(values, dtype=np.float32)
    return ((values - min_value) / (max_value - min_value)).astype(np.float32)
