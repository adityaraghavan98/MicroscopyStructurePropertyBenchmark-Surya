from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import gpytorch
import numpy as np
import torch
import torch.nn as nn
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy


class ConvNetFeatureExtractor(nn.Module):
    """Small CNN feature extractor for patch-based DKL."""

    def __init__(self, input_channels: int = 1, output_dim: int = 32):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.projection = nn.Linear(32 * 2 * 2, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        x = self.conv_layers(x)
        return self.projection(torch.flatten(x, start_dim=1))


class _ApproximateDKLGP(ApproximateGP):
    def __init__(self, inducing_points: torch.Tensor, feature_extractor: ConvNetFeatureExtractor):
        inducing_features = feature_extractor(inducing_points).detach()
        variational_distribution = CholeskyVariationalDistribution(inducing_features.size(0))
        variational_strategy = VariationalStrategy(
            self,
            inducing_features,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)
        self.feature_extractor = feature_extractor
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def __call__(self, x: torch.Tensor, use_feature_extractor: bool = True, *args, **kwargs):
        if use_feature_extractor:
            x = self.feature_extractor(x)
        return super().__call__(x, *args, **kwargs)


@dataclass
class Prediction:
    mean: np.ndarray
    variance: np.ndarray


class DKLRegressor:
    """Stochastic variational deep-kernel GP for image patches."""

    def __init__(
        self,
        training_steps: int = 50,
        learning_rate: float = 0.01,
        feature_dim: int = 32,
        inducing_points: int = 16,
        seed: int = 0,
        load_model_path: str | None = None,
    ):
        self.training_steps = training_steps
        self.learning_rate = learning_rate
        self.feature_dim = feature_dim
        self.inducing_points = inducing_points
        self.seed = seed
        self.load_model_path = load_model_path
        self.likelihood: gpytorch.likelihoods.GaussianLikelihood | None = None
        self.model: _ApproximateDKLGP | None = None
        self.training_loss_trace: list[float] = []

    def fit(self, patches: np.ndarray, target: np.ndarray) -> "DKLRegressor":
        torch.manual_seed(self.seed)
        train_x = _patch_tensor(patches)
        train_y = torch.as_tensor(target, dtype=torch.float32)

        feature_extractor = ConvNetFeatureExtractor(output_dim=self.feature_dim)
        n_inducing = min(self.inducing_points, train_x.shape[0])
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = _ApproximateDKLGP(train_x[:n_inducing], feature_extractor)
        self._load_initial_state()
        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.likelihood.parameters()),
            lr=self.learning_rate,
        )
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=train_y.numel())
        self.training_loss_trace = []

        for _ in range(self.training_steps):
            optimizer.zero_grad()
            output = self.model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
            self.training_loss_trace.append(float(loss.detach().cpu()))
        return self

    def predict(self, patches: np.ndarray) -> Prediction:
        if self.model is None or self.likelihood is None:
            raise RuntimeError("Model must be fit before predict().")

        self.model.eval()
        self.likelihood.eval()
        x = _patch_tensor(patches)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = self.likelihood(self.model(x))
        variance = posterior.variance.clamp_min(1e-9)
        return Prediction(mean=posterior.mean.cpu().numpy(), variance=variance.cpu().numpy())

    def posterior(self, X: torch.Tensor, output_indices=None, observation_noise: bool = False, **kwargs):
        """Expose a BoTorch-compatible posterior for LogExpectedImprovement."""

        from botorch.posteriors.gpytorch import GPyTorchPosterior

        if self.model is None or self.likelihood is None:
            raise RuntimeError("Model must be fit before posterior().")

        self.model.eval()
        self.likelihood.eval()
        x = torch.as_tensor(X, dtype=torch.float32)
        x = _botorch_candidate_tensor_to_patches(x)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            dist = self.likelihood(self.model(x)) if observation_noise else self.model(x)
        return GPyTorchPosterior(dist)

    @property
    def num_outputs(self) -> int:
        return 1

    def diagnostics(self) -> dict[str, float | int | list[float]]:
        data: dict[str, float | int | list[float]] = {
            "training_steps": self.training_steps,
            "learning_rate": self.learning_rate,
            "feature_dim": self.feature_dim,
            "inducing_points": self.inducing_points,
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
            "model_type": "dkl",
            "step": step,
            "metadata": metadata or {},
            "config": {
                "training_steps": self.training_steps,
                "learning_rate": self.learning_rate,
                "feature_dim": self.feature_dim,
                "inducing_points": self.inducing_points,
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
        try:
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            self.likelihood.load_state_dict(checkpoint["likelihood_state_dict"], strict=False)
        except RuntimeError as exc:
            raise RuntimeError(
                "Could not load DKL checkpoint. Check that feature_dim and inducing_points match the saved model."
            ) from exc


def _patch_tensor(patches: np.ndarray) -> torch.Tensor:
    x = torch.as_tensor(patches, dtype=torch.float32)
    if x.ndim == 3:
        x = x.unsqueeze(1)
    return x


def _botorch_candidate_tensor_to_patches(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 5 and x.shape[-4] == 1:
        return x.squeeze(-4)
    if x.ndim == 3 and x.shape[-2] == 1:
        x = x.squeeze(-2)
    if x.ndim == 2:
        patch_width = int(math.sqrt(x.shape[-1]))
        if patch_width * patch_width != x.shape[-1]:
            raise ValueError(f"Cannot reshape flattened candidate dimension {x.shape[-1]} into a square patch.")
        return x.reshape(x.shape[0], 1, patch_width, patch_width)
    return x
