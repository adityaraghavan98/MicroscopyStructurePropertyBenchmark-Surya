from __future__ import annotations

import numpy as np
import torch


def upper_confidence_bound_scores(model, candidate_features: np.ndarray, beta: float = 100000.0) -> np.ndarray:
    """BoTorch UpperConfidenceBound scores for a discrete candidate set."""

    from botorch.acquisition import UpperConfidenceBound

    candidates = torch.as_tensor(candidate_features, dtype=torch.float32)
    candidates = candidates.reshape(candidates.shape[0], -1).unsqueeze(1)
    acquisition = UpperConfidenceBound(model=model, beta=float(beta))
    with torch.no_grad():
        scores = acquisition(candidates)
    return scores.detach().cpu().numpy()
