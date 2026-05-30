from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


class PCARepresentation:
    """Flatten image patches and project them with sklearn PCA."""

    def __init__(self, n_components: int = 12, seed: int = 0):
        self.n_components = n_components
        self.seed = seed
        self._pca = PCA(n_components=n_components, random_state=seed)

    def fit_transform(self, patches: np.ndarray) -> np.ndarray:
        flat = patches.reshape(patches.shape[0], -1)
        return self._pca.fit_transform(flat).astype(np.float32)

    def transform(self, patches: np.ndarray) -> np.ndarray:
        flat = patches.reshape(patches.shape[0], -1)
        return self._pca.transform(flat).astype(np.float32)

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        return self._pca.explained_variance_ratio_
