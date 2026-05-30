from __future__ import annotations

import numpy as np
import torch


def expected_improvement_scores(model, candidate_features: np.ndarray, best_observed: float) -> np.ndarray:
    """BoTorch LogExpectedImprovement scores for a discrete candidate set."""

    from botorch.acquisition import LogExpectedImprovement

    candidates = torch.as_tensor(candidate_features, dtype=torch.float32)
    candidates = candidates.reshape(candidates.shape[0], -1).unsqueeze(1)
    best_f = torch.tensor(best_observed, dtype=torch.float32)
    acquisition = LogExpectedImprovement(model=model, best_f=best_f)
    with torch.no_grad():
        scores = acquisition(candidates)
    return scores.detach().cpu().numpy()
