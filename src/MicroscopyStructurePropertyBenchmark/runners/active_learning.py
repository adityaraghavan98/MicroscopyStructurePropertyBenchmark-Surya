from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from MicroscopyStructurePropertyBenchmark.acquisition import beacon_scores, expected_improvement_scores, random_scores, upper_confidence_bound_scores
from MicroscopyStructurePropertyBenchmark.datasets import load_stem_h5_dataset, make_synthetic_dataset
from MicroscopyStructurePropertyBenchmark.io import OutputWriter
from MicroscopyStructurePropertyBenchmark.metrics import mae, mse, nlpd, observed_value_coverage
from MicroscopyStructurePropertyBenchmark.models import DKLRegressor, GPyTorchRegressor
from MicroscopyStructurePropertyBenchmark.representations import PCARepresentation


@dataclass
class BenchmarkResult:
    acquired_order: list[int]
    seed_indices: list[int]
    output_dir: str | None = None
    mse_trace: list[float] = field(default_factory=list)
    mae_trace: list[float] = field(default_factory=list)
    nlpd_trace: list[float] = field(default_factory=list)
    coverage_trace: list[float] = field(default_factory=list)
    mean_prediction_trace: list[float] = field(default_factory=list)
    mean_variance_trace: list[float] = field(default_factory=list)
    loss_initial_trace: list[float] = field(default_factory=list)
    loss_final_trace: list[float] = field(default_factory=list)

    def summary(self) -> dict[str, float | int | list[int]]:
        return {
            "n_seed": len(self.seed_indices),
            "n_acquired": len(self.acquired_order),
            "final_mse": self.mse_trace[-1],
            "final_mae": self.mae_trace[-1],
            "final_nlpd": self.nlpd_trace[-1],
            "final_observed_value_coverage": self.coverage_trace[-1],
            "first_acquired": self.acquired_order[:5],
            "output_dir": self.output_dir,
        }


def run_benchmark(config: dict) -> BenchmarkResult:
    seed = int(config.get("seed", 0))
    dataset_cfg = config.get("dataset", {})
    dataset = _make_dataset(dataset_cfg, seed=seed)

    model_cfg = config.get("model", {})
    checkpoint_cfg = config.get("checkpoint", {})
    features = _make_features(config.get("representation", {}), model_cfg, dataset.patches, seed=seed)

    benchmark_cfg = config.get("benchmark", {})
    initial_points = int(benchmark_cfg.get("initial_points", 12))
    steps = int(benchmark_cfg.get("steps", 20))

    rng = np.random.default_rng(seed)
    all_indices = np.arange(len(dataset.target))
    seed_indices = rng.choice(all_indices, size=initial_points, replace=False).astype(int).tolist()
    acquired = list(seed_indices)
    acquired_order: list[int] = []
    unacquired = [int(i) for i in all_indices if int(i) not in acquired]

    result = BenchmarkResult(acquired_order=acquired_order, seed_indices=seed_indices)
    output_cfg = dict(config.get("output", {}))
    if bool(checkpoint_cfg.get("save_model", False)):
        output_cfg["enabled"] = True
        output_cfg.setdefault("save_step_plots", False)
        output_cfg.setdefault("save_step_pickles", False)
        output_cfg.setdefault("save_trajectory_plot", False)
    output = OutputWriter(
        output_cfg,
        seed=seed,
        method_name=f"{model_cfg.get('name', 'gpytorch_gp')}_{config.get('acquisition', {}).get('name', 'expected_improvement')}",
    )
    result.output_dir = str(output.run_dir) if output.run_dir is not None else None
    output.log_run_start(
        config=config,
        dataset_summary={
            "name": dataset_cfg.get("name", "synthetic"),
            "image_shape": tuple(int(v) for v in dataset.image.shape),
            "patches_shape": tuple(int(v) for v in dataset.patches.shape),
            "coords_shape": tuple(int(v) for v in dataset.coords.shape),
            "target_min": float(np.min(dataset.target)),
            "target_max": float(np.max(dataset.target)),
            "target_mean": float(np.mean(dataset.target)),
            "target_std": float(np.std(dataset.target)),
        },
        feature_summary={
            "shape": tuple(int(v) for v in features.shape),
            "min": float(np.min(features)),
            "max": float(np.max(features)),
            "mean": float(np.mean(features)),
            "std": float(np.std(features)),
        },
    )
    for step in range(steps):
        train_idx = np.asarray(acquired, dtype=int)
        candidate_idx = np.asarray(unacquired, dtype=int)

        model = _make_model(model_cfg, seed=seed + step, load_model_path=checkpoint_cfg.get("load_model_path"))
        model.fit(features[train_idx], dataset.target[train_idx])
        full_pred = model.predict(features)
        candidate_pred = model.predict(features[candidate_idx])

        scores = _score_candidates(
            config.get("acquisition", {}),
            model=model,
            candidate_idx=candidate_idx,
            features=features,
            acquired=acquired,
            target=dataset.target,
            mean=candidate_pred.mean,
            variance=candidate_pred.variance,
            seed=seed + step,
        )
        selected = int(candidate_idx[int(np.argmax(scores))])

        acquired.append(selected)
        acquired_order.append(selected)
        unacquired.remove(selected)

        mse_value = mse(dataset.target, full_pred.mean)
        mae_value = mae(dataset.target, full_pred.mean)
        nlpd_value = nlpd(dataset.target, full_pred.mean, full_pred.variance)
        coverage_value = observed_value_coverage(acquired, dataset.target)

        result.mse_trace.append(mse_value)
        result.mae_trace.append(mae_value)
        result.nlpd_trace.append(nlpd_value)
        result.coverage_trace.append(coverage_value)
        result.mean_prediction_trace.append(float(np.mean(full_pred.mean)))
        result.mean_variance_trace.append(float(np.mean(full_pred.variance)))
        model_diagnostics = model.diagnostics()
        result.loss_initial_trace.append(float(model_diagnostics["loss_initial"]))
        result.loss_final_trace.append(float(model_diagnostics["loss_final"]))

        output.save_step(
            step=step,
            image=dataset.image,
            coords=dataset.coords,
            target=dataset.target,
            mean=full_pred.mean,
            variance=full_pred.variance,
            selected_index=selected,
            seed_indices=seed_indices,
            acquired_order=acquired_order,
            mse_value=mse_value,
            mae_value=mae_value,
            nlpd_value=nlpd_value,
            coverage_value=coverage_value,
            mean_prediction=result.mean_prediction_trace[-1],
            mean_variance=result.mean_variance_trace[-1],
            model_diagnostics=model_diagnostics,
        )
        if bool(checkpoint_cfg.get("save_model", False)):
            output.save_model_checkpoint(
                step=step,
                payload=model.checkpoint_payload(
                    step=step,
                    metadata={
                        "selected_index": selected,
                        "seed_indices": seed_indices,
                        "acquired_order": acquired_order,
                        "unacquired_indices": unacquired,
                        "mse": mse_value,
                        "mae": mae_value,
                        "nlpd": nlpd_value,
                        "coverage": coverage_value,
                    },
                ),
            )

    output.save_final(
        image=dataset.image,
        features=features,
        coords=dataset.coords,
        seed_indices=seed_indices,
        acquired_order=acquired_order,
        unacquired_indices=unacquired,
        mean_trace=result.mean_prediction_trace,
        variance_trace=result.mean_variance_trace,
        mse_trace=result.mse_trace,
        mae_trace=result.mae_trace,
        nlpd_trace=result.nlpd_trace,
        coverage_trace=result.coverage_trace,
    )
    return result


def _make_features(rep_config: dict, model_config: dict, patches: np.ndarray, seed: int) -> np.ndarray:
    model_name = model_config.get("name", "gpytorch_gp")
    rep_name = rep_config.get("name", "pca" if model_name == "gpytorch_gp" else "patches")
    if rep_name == "pca":
        representation = PCARepresentation(n_components=int(rep_config.get("n_components", 12)), seed=seed)
        return representation.fit_transform(patches)
    if rep_name in {"patches", "raw_patches"}:
        return patches[:, None, :, :].astype(np.float32)
    raise ValueError(f"Unknown representation: {rep_name}")


def _make_model(config: dict, seed: int, load_model_path: str | None = None) -> GPyTorchRegressor | DKLRegressor:
    name = config.get("name", "gpytorch_gp")
    if name == "gpytorch_gp":
        return GPyTorchRegressor(
            training_steps=int(config.get("training_steps", 50)),
            learning_rate=float(config.get("learning_rate", 0.08)),
            seed=seed,
            load_model_path=load_model_path,
        )
    if name == "dkl":
        return DKLRegressor(
            training_steps=int(config.get("training_steps", 50)),
            learning_rate=float(config.get("learning_rate", 0.01)),
            feature_dim=int(config.get("feature_dim", 32)),
            inducing_points=int(config.get("inducing_points", 16)),
            seed=seed,
            load_model_path=load_model_path,
        )
    raise ValueError(f"Unknown model: {name}")


def _score_candidates(
    config: dict,
    model: GPyTorchRegressor | DKLRegressor,
    candidate_idx: np.ndarray,
    features: np.ndarray,
    acquired: list[int],
    target: np.ndarray,
    mean: np.ndarray,
    variance: np.ndarray,
    seed: int,
) -> np.ndarray:
    name = config.get("name", "expected_improvement")
    if name == "expected_improvement":
        return expected_improvement_scores(
            model=model,
            candidate_features=features[candidate_idx],
            best_observed=float(np.max(target[acquired])),
        )
    if name in {"upper_confidence_bound", "ucb", "mu"}:
        return upper_confidence_bound_scores(
            model=model,
            candidate_features=features[candidate_idx],
            beta=float(config.get("beta", 100000.0)),
        )
    if name == "random":
        return random_scores(len(candidate_idx), seed=seed)
    if name == "beacon":
        return beacon_scores(
            candidate_features=features[candidate_idx],
            acquired_features=features[np.asarray(acquired, dtype=int)],
            acquired_values=target[np.asarray(acquired, dtype=int)],
            mean=mean,
            elite_fraction=float(config.get("elite_fraction", 0.1)),
        )
    raise ValueError(f"Unknown acquisition function: {name}")


def _make_dataset(config: dict, seed: int):
    name = config.get("name", "synthetic")
    if name == "synthetic":
        return make_synthetic_dataset(
            grid_shape=tuple(config.get("grid_shape", (24, 24))),
            patch_size=int(config.get("patch_size", 8)),
            noise=float(config.get("noise", 0.03)),
            seed=seed,
        )
    if name in {"stem_h5", "dtmic_stem_h5"}:
        return load_stem_h5_dataset(
            path=config.get("path", "data/raw/test_stem.h5"),
            patch_size=int(config.get("patch_size", 8)),
            reward=config.get("reward", "dipole"),
            normalize_reward=bool(config.get("normalize_reward", True)),
        )
    raise ValueError(f"Unknown dataset: {name}")
