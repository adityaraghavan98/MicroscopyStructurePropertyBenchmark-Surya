from __future__ import annotations

import json
import pickle
import os
from datetime import datetime
from pathlib import Path

import numpy as np


class OutputWriter:
    """Write benchmark artifacts compatible with the notebook workflow."""

    def __init__(self, config: dict, seed: int, method_name: str):
        self.enabled = bool(config.get("enabled", False))
        self.save_step_plots = bool(config.get("save_step_plots", True))
        self.save_step_pickles = bool(config.get("save_step_pickles", True))
        self.save_trajectory_plot = bool(config.get("save_trajectory_plot", True))
        self.run_dir: Path | None = None

        if self.enabled:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = config.get("run_name", method_name)
            self.run_dir = Path(config.get("dir", "outputs")) / f"{run_name}_seed{seed}_{timestamp}"
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self._write_text("run.log", f"[{datetime.now().isoformat()}] created run directory {self.run_dir}\n")

    def log_run_start(self, config: dict, dataset_summary: dict, feature_summary: dict) -> None:
        if not self.enabled or self.run_dir is None:
            return
        payload = {
            "event": "run_start",
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "dataset": dataset_summary,
            "features": feature_summary,
        }
        self._append_jsonl("training_log.jsonl", payload)
        self._append_log_lines(
            [
                "run_start",
                f"dataset={dataset_summary}",
                f"features={feature_summary}",
                f"config={config}",
            ]
        )

    def save_step(
        self,
        *,
        step: int,
        image: np.ndarray,
        coords: np.ndarray,
        target: np.ndarray,
        mean: np.ndarray,
        variance: np.ndarray,
        selected_index: int,
        seed_indices: list[int],
        acquired_order: list[int],
        mse_value: float,
        mae_value: float,
        nlpd_value: float,
        coverage_value: float,
        mean_prediction: float,
        mean_variance: float,
        model_diagnostics: dict,
    ) -> None:
        if not self.enabled or self.run_dir is None:
            return

        true_img = _map_to_image(target, coords, image.shape)
        mean_img = _map_to_image(mean, coords, image.shape)
        variance_img = _map_to_image(variance, coords, image.shape)

        step_data = {
            "true_scalarizer_img": true_img,
            "y_pred_mean_img": mean_img,
            "y_pred_var_img": variance_img,
            "mse": mse_value,
            "mae": mae_value,
            "nlpd": nlpd_value,
            "coverage": coverage_value,
            "selected_index": selected_index,
            "seed_indices": np.asarray(seed_indices, dtype=int),
            "acquired_order": np.asarray(acquired_order, dtype=int),
            "indices_all": coords,
            "y_true": target,
            "y_pred_mean": mean,
            "y_pred_variance": variance,
            "mean_prediction": mean_prediction,
            "mean_variance": mean_variance,
            "model_diagnostics": model_diagnostics,
        }

        self._append_jsonl(
            "training_log.jsonl",
            {
                "event": "bo_step",
                "timestamp": datetime.now().isoformat(),
                "step": step,
                "selected_index": selected_index,
                "selected_coord": coords[selected_index].astype(int).tolist(),
                "n_acquired": len(seed_indices) + len(acquired_order),
                "mse": mse_value,
                "mae": mae_value,
                "nlpd": nlpd_value,
                "coverage": coverage_value,
                "mean_prediction": mean_prediction,
                "mean_variance": mean_variance,
                "target_selected": float(target[selected_index]),
                "prediction_selected": float(mean[selected_index]),
                "variance_selected": float(variance[selected_index]),
                "model_diagnostics": model_diagnostics,
            },
        )
        self._append_log_lines(
            [
                f"step={step} selected_index={selected_index} selected_coord={coords[selected_index].astype(int).tolist()}",
                f"metrics mse={mse_value:.6g} mae={mae_value:.6g} nlpd={nlpd_value:.6g} coverage={coverage_value:.6g}",
                f"prediction mean={mean_prediction:.6g} mean_variance={mean_variance:.6g}",
                (
                    f"loss initial={model_diagnostics.get('loss_initial')} "
                    f"final={model_diagnostics.get('loss_final')} "
                    f"steps={model_diagnostics.get('training_steps')}"
                ),
            ]
        )

        if self.save_step_pickles:
            with (self.run_dir / f"predictions_BO_step{step}.pkl").open("wb") as f:
                pickle.dump(step_data, f)

        if self.save_step_plots:
            _save_step_plot(
                self.run_dir / f"predictions_BO_step{step}.png",
                image=image,
                coords=coords,
                selected_index=selected_index,
                true_img=true_img,
                mean_img=mean_img,
                variance_img=variance_img,
                mae_value=mae_value,
                nlpd_value=nlpd_value,
            )

    def save_final(
        self,
        *,
        image: np.ndarray,
        features: np.ndarray,
        coords: np.ndarray,
        seed_indices: list[int],
        acquired_order: list[int],
        unacquired_indices: list[int],
        mean_trace: list[float],
        variance_trace: list[float],
        mse_trace: list[float],
        mae_trace: list[float],
        nlpd_trace: list[float],
        coverage_trace: list[float],
    ) -> None:
        if not self.enabled or self.run_dir is None:
            return

        stats = {
            "acquired_order": np.asarray(acquired_order, dtype=int),
            "img": image,
            "features": features,
            "indices_all": coords,
            "seed_indices": np.asarray(seed_indices, dtype=int),
            "unacquired_indices": np.asarray(unacquired_indices, dtype=int),
            "mean_y_pred_mean_al": np.asarray(mean_trace),
            "mean_y_pred_variance_al": np.asarray(variance_trace),
            "mse": np.asarray(mse_trace),
            "mae": np.asarray(mae_trace),
            "nlpd": np.asarray(nlpd_trace),
            "coverage": np.asarray(coverage_trace),
        }

        with (self.run_dir / "Active_learning_statistics.pkl").open("wb") as f:
            pickle.dump(stats, f)

        self._append_jsonl(
            "training_log.jsonl",
            {
                "event": "run_final",
                "timestamp": datetime.now().isoformat(),
                "n_steps": len(acquired_order),
                "final_mse": mse_trace[-1] if mse_trace else None,
                "final_mae": mae_trace[-1] if mae_trace else None,
                "final_nlpd": nlpd_trace[-1] if nlpd_trace else None,
                "final_coverage": coverage_trace[-1] if coverage_trace else None,
                "acquired_order": acquired_order,
                "seed_indices": seed_indices,
            },
        )
        self._append_log_lines(
            [
                "run_final",
                f"n_steps={len(acquired_order)}",
                f"final_mse={mse_trace[-1] if mse_trace else None}",
                f"final_mae={mae_trace[-1] if mae_trace else None}",
                f"final_nlpd={nlpd_trace[-1] if nlpd_trace else None}",
                f"final_coverage={coverage_trace[-1] if coverage_trace else None}",
            ]
        )

        if self.save_trajectory_plot:
            _save_trajectory_plot(
                self.run_dir / "AL_traj.png",
                image=image,
                coords=coords,
                seed_indices=seed_indices,
                acquired_order=acquired_order,
            )

    def save_model_checkpoint(self, *, step: int, payload: dict) -> Path | None:
        if not self.enabled or self.run_dir is None:
            return None
        import torch

        checkpoint_dir = self.run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = checkpoint_dir / f"model_step{step:03d}.pt"
        latest_path = checkpoint_dir / "latest.pt"
        torch.save(payload, path)
        torch.save(payload, latest_path)
        self._append_jsonl(
            "training_log.jsonl",
            {
                "event": "model_checkpoint",
                "timestamp": datetime.now().isoformat(),
                "step": step,
                "path": str(path),
                "latest_path": str(latest_path),
            },
        )
        self._append_log_lines([f"model_checkpoint step={step} path={path} latest={latest_path}"])
        return path

    def _append_jsonl(self, filename: str, payload: dict) -> None:
        if self.run_dir is None:
            return
        with (self.run_dir / filename).open("a", encoding="utf-8") as f:
            f.write(json.dumps(_json_safe(payload), sort_keys=True) + "\n")

    def _append_log_lines(self, lines: list[str]) -> None:
        timestamp = datetime.now().isoformat()
        text = "".join(f"[{timestamp}] {line}\n" for line in lines)
        self._write_text("run.log", text, mode="a")

    def _write_text(self, filename: str, text: str, mode: str = "a") -> None:
        if self.run_dir is None:
            return
        with (self.run_dir / filename).open(mode, encoding="utf-8") as f:
            f.write(text)


def _map_to_image(values: np.ndarray, coords: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    img = np.zeros(shape, dtype=np.float32)
    for idx, (row, col) in enumerate(coords):
        img[int(row), int(col)] = float(values[idx])
    return img


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _save_step_plot(
    path: Path,
    *,
    image: np.ndarray,
    coords: np.ndarray,
    selected_index: int,
    true_img: np.ndarray,
    mean_img: np.ndarray,
    variance_img: np.ndarray,
    mae_value: float,
    nlpd_value: float,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    selected_coord = coords[selected_index]

    im0 = axs[0, 0].imshow(image, cmap="gray")
    axs[0, 0].set_title("Original Image with next point selection")
    axs[0, 0].scatter([selected_coord[1]], [selected_coord[0]], color="yellow", marker="x")
    fig.colorbar(im0, ax=axs[0, 0])

    im1 = axs[0, 1].imshow(mean_img, cmap="viridis")
    axs[0, 1].set_title("Predicted Mean")
    fig.colorbar(im1, ax=axs[0, 1])

    im2 = axs[1, 0].imshow(variance_img, cmap="viridis")
    axs[1, 0].set_title("Predicted Variance")
    fig.colorbar(im2, ax=axs[1, 0])

    im3 = axs[1, 1].imshow(true_img, cmap="viridis")
    axs[1, 1].set_title("True Scalarizer")
    fig.colorbar(im3, ax=axs[1, 1])

    for ax in axs.flat:
        ax.axis("off")

    fig.suptitle(f"MAE: {mae_value:.4f}, NLPD: {nlpd_value:.4f}", fontsize=14)
    plt.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_trajectory_plot(
    path: Path,
    *,
    image: np.ndarray,
    coords: np.ndarray,
    seed_indices: list[int],
    acquired_order: list[int],
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seed_coords = coords[np.asarray(seed_indices, dtype=int)]
    acquired_coords = coords[np.asarray(acquired_order, dtype=int)] if acquired_order else np.empty((0, 2))
    time_order = np.arange(len(acquired_coords))

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(image, cmap="gray", origin="upper")
    ax.scatter(seed_coords[:, 1], seed_coords[:, 0], c="b", label="Seed Points", marker="o")
    scatter = ax.scatter(
        acquired_coords[:, 1],
        acquired_coords[:, 0],
        c=time_order,
        cmap="bwr",
        label="Acquired Points",
        marker="x",
    )
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_title("Active Learning Trajectory")
    ax.legend()
    ax.grid(True)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Steps")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
