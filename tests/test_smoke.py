import numpy as np

from MicroscopyStructurePropertyBenchmark.datasets import make_synthetic_dataset
from MicroscopyStructurePropertyBenchmark.metrics import observed_value_coverage
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


def test_pca_gp_benchmark_smoke():
    config = {
        "seed": 3,
        "dataset": {"name": "synthetic", "grid_shape": [8, 8], "patch_size": 4, "noise": 0.01},
        "representation": {"name": "pca", "n_components": 4},
        "model": {"name": "gpytorch_gp", "training_steps": 2, "learning_rate": 0.05},
        "acquisition": {"name": "expected_improvement"},
        "benchmark": {"initial_points": 6, "steps": 2},
    }
    result = run_benchmark(config)
    assert len(result.seed_indices) == 6
    assert len(result.acquired_order) == 2
    assert len(result.mse_trace) == 2


def test_dkl_beacon_benchmark_smoke():
    config = {
        "seed": 4,
        "dataset": {"name": "synthetic", "grid_shape": [8, 8], "patch_size": 4, "noise": 0.01},
        "representation": {"name": "patches"},
        "model": {
            "name": "dkl",
            "training_steps": 1,
            "learning_rate": 0.01,
            "feature_dim": 4,
            "inducing_points": 4,
        },
        "acquisition": {"name": "beacon", "elite_fraction": 0.25},
        "benchmark": {"initial_points": 6, "steps": 1},
    }
    result = run_benchmark(config)
    assert len(result.seed_indices) == 6
    assert len(result.acquired_order) == 1
    assert len(result.coverage_trace) == 1


def test_dkl_mu_benchmark_smoke():
    config = {
        "seed": 8,
        "dataset": {"name": "synthetic", "grid_shape": [8, 8], "patch_size": 4, "noise": 0.01},
        "representation": {"name": "patches"},
        "model": {
            "name": "dkl",
            "training_steps": 1,
            "learning_rate": 0.01,
            "feature_dim": 4,
            "inducing_points": 4,
        },
        "acquisition": {"name": "upper_confidence_bound", "beta": 100000},
        "benchmark": {"initial_points": 6, "steps": 1},
    }
    result = run_benchmark(config)
    assert len(result.seed_indices) == 6
    assert len(result.acquired_order) == 1


def test_output_writer_creates_artifacts(tmp_path):
    config = {
        "seed": 5,
        "dataset": {"name": "synthetic", "grid_shape": [6, 6], "patch_size": 4, "noise": 0.01},
        "representation": {"name": "pca", "n_components": 3},
        "model": {"name": "gpytorch_gp", "training_steps": 1, "learning_rate": 0.05},
        "acquisition": {"name": "random"},
        "benchmark": {"initial_points": 5, "steps": 1},
        "output": {
            "enabled": True,
            "dir": str(tmp_path),
            "run_name": "smoke",
            "save_step_plots": False,
            "save_step_pickles": True,
            "save_trajectory_plot": False,
        },
        "checkpoint": {"save_model": True, "load_model_path": None},
    }
    result = run_benchmark(config)
    assert result.output_dir is not None
    run_dir = next(tmp_path.iterdir())
    assert (run_dir / "predictions_BO_step0.pkl").exists()
    assert (run_dir / "Active_learning_statistics.pkl").exists()
    assert (run_dir / "run.log").exists()
    assert (run_dir / "training_log.jsonl").exists()
    checkpoint_path = run_dir / "checkpoints" / "model_step000.pt"
    assert checkpoint_path.exists()

    load_config = {
        **config,
        "output": {
            "enabled": True,
            "dir": str(tmp_path),
            "run_name": "smoke_load",
            "save_step_plots": False,
            "save_step_pickles": False,
            "save_trajectory_plot": False,
        },
        "checkpoint": {"save_model": False, "load_model_path": str(checkpoint_path)},
    }
    loaded_result = run_benchmark(load_config)
    assert len(loaded_result.acquired_order) == 1


def test_sweep_writes_one_row_per_method_step(tmp_path):
    csv_path = tmp_path / "sweep.csv"
    log_path = tmp_path / "sweep.log"
    jsonl_path = tmp_path / "sweep_log.jsonl"
    config = {
        "seed": 6,
        "dataset": {"name": "synthetic", "grid_shape": [6, 6], "patch_size": 4, "noise": 0.01},
        "benchmark": {"initial_points": 5, "steps": 2},
        "output": {"csv": str(csv_path), "log": str(log_path), "jsonl_log": str(jsonl_path), "save_artifacts": False},
        "methods": [
            {
                "name": "pca_gp_random",
                "representation": {"name": "pca", "n_components": 3},
                "model": {"name": "gpytorch_gp", "training_steps": 1, "learning_rate": 0.05},
                "acquisition": {"name": "random"},
            },
            {
                "name": "dkl_beacon",
                "representation": {"name": "patches"},
                "model": {
                    "name": "dkl",
                    "training_steps": 1,
                    "learning_rate": 0.01,
                    "feature_dim": 4,
                    "inducing_points": 4,
                },
                "acquisition": {"name": "beacon", "elite_fraction": 0.25},
            },
        ],
    }
    rows = run_sweep(config)
    assert len(rows) == 4
    assert csv_path.exists()
    assert log_path.exists()
    assert jsonl_path.exists()
