from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gpytorch
import numpy as np
import torch


class _ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor, likelihood: gpytorch.likelihoods.GaussianLikelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


@dataclass
class Prediction:
    mean: np.ndarray
    variance: np.ndarray


class GPyTorchRegressor:
    """Small exact GP wrapper for PCA features."""

    def __init__(self, training_steps: int = 50, learning_rate: float = 0.08, seed: int = 0, load_model_path: str | None = None):
        self.training_steps = training_steps
        self.learning_rate = learning_rate
        self.seed = seed
        self.load_model_path = load_model_path
        self.likelihood: gpytorch.likelihoods.GaussianLikelihood | None = None
        self.model: _ExactGPModel | None = None
        self.training_loss_trace: list[float] = []

    def fit(self, features: np.ndarray, target: np.ndarray) -> "GPyTorchRegressor":
        torch.manual_seed(self.seed)
        train_x = torch.as_tensor(features, dtype=torch.float32)
        train_y = torch.as_tensor(target, dtype=torch.float32)

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = _ExactGPModel(train_x, train_y, self.likelihood)
        self._load_initial_state()
        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        self.training_loss_trace = []

        for _ in range(self.training_steps):
            optimizer.zero_grad()
            output = self.model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
            self.training_loss_trace.append(float(loss.detach().cpu()))
        return self

    def predict(self, features: np.ndarray) -> Prediction:
        if self.model is None or self.likelihood is None:
            raise RuntimeError("Model must be fit before predict().")

        self.model.eval()
        self.likelihood.eval()
        x = torch.as_tensor(features, dtype=torch.float32)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = self.likelihood(self.model(x))
        variance = posterior.variance.clamp_min(1e-9)
        return Prediction(
            mean=posterior.mean.cpu().numpy(),
            variance=variance.cpu().numpy(),
        )

    def posterior(self, X: torch.Tensor, output_indices=None, observation_noise: bool = False, **kwargs):
        """Expose a BoTorch-compatible posterior for acquisition functions."""

        from botorch.posteriors.gpytorch import GPyTorchPosterior

        if self.model is None or self.likelihood is None:
            raise RuntimeError("Model must be fit before posterior().")

        self.model.eval()
        self.likelihood.eval()
        X = torch.as_tensor(X, dtype=torch.float32)
        if X.ndim >= 3 and X.shape[-2] == 1:
            X = X.squeeze(-2)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            dist = self.likelihood(self.model(X)) if observation_noise else self.model(X)
        return GPyTorchPosterior(dist)

    @property
    def num_outputs(self) -> int:
        return 1

    def diagnostics(self) -> dict[str, float | int | list[float]]:
        data: dict[str, float | int | list[float]] = {
            "training_steps": self.training_steps,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "loss_initial": self.training_loss_trace[0] if self.training_loss_trace else float("nan"),
            "loss_final": self.training_loss_trace[-1] if self.training_loss_trace else float("nan"),
            "loss_trace": self.training_loss_trace,
        }
        if self.likelihood is not None:
            data["likelihood_noise"] = float(self.likelihood.noise.detach().cpu().item())
        if self.model is not None:
            data["mean_constant"] = float(self.model.mean_module.constant.detach().cpu().item())
            data["outputscale"] = float(self.model.covar_module.outputscale.detach().cpu().item())
            data["lengthscale_mean"] = float(self.model.covar_module.base_kernel.lengthscale.detach().cpu().mean().item())
        return data

    def checkpoint_payload(self, step: int, metadata: dict | None = None) -> dict:
        if self.model is None or self.likelihood is None:
            raise RuntimeError("Model must be fit before checkpoint_payload().")
        return {
            "model_type": "gpytorch_gp",
            "step": step,
            "metadata": metadata or {},
            "config": {
                "training_steps": self.training_steps,
                "learning_rate": self.learning_rate,
                "seed": self.seed,
            },
            "model_state_dict": self.model.state_dict(),
            "likelihood_state_dict": self.likelihood.state_dict(),
            "training_loss_trace": self.training_loss_trace,
        }

    def _load_initial_state(self) -> None:
        if not self.load_model_path:
            return
        if self.model is None or self.likelihood is None:
            raise RuntimeError("Model must exist before loading checkpoint.")
        checkpoint = torch.load(Path(self.load_model_path), map_location="cpu", weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        self.likelihood.load_state_dict(checkpoint["likelihood_state_dict"], strict=False)
