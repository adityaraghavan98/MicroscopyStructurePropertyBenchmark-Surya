from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from MicroscopyStructurePropertyBenchmark.runners.active_learning import BenchmarkResult, run_benchmark


def run_sweep(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Run multiple methods from one YAML and write one CSV row per BO step."""

    rows: list[dict[str, Any]] = []
    output_cfg = config.get("output", {})
    csv_path = Path(output_cfg.get("csv", "outputs/sweep_results.csv"))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = Path(output_cfg.get("log", csv_path.with_suffix(".log")))
    jsonl_path = Path(output_cfg.get("jsonl_log", csv_path.with_name(f"{csv_path.stem}_log.jsonl")))

    _append_log(log_path, "sweep_start")
    _append_jsonl(jsonl_path, {"event": "sweep_start", "timestamp": _timestamp(), "config": config})

    for reward in _reward_configs(config):
        reward_name = reward["name"]
        _append_log(log_path, f"reward_start name={reward_name}")
        _append_jsonl(jsonl_path, {"event": "reward_start", "timestamp": _timestamp(), "reward": reward_name})

        for method in config.get("methods", []):
            method_name = method["name"]
            run_label = f"{reward_name}__{method_name}" if config.get("rewards") else method_name
            run_config = _method_config(config, method, reward)
            _append_log(log_path, f"method_start reward={reward_name} name={method_name}")
            _append_jsonl(
                jsonl_path,
                {
                    "event": "method_start",
                    "timestamp": _timestamp(),
                    "reward": reward_name,
                    "method": method_name,
                    "run_label": run_label,
                    "config": run_config,
                },
            )
            try:
                result = run_benchmark(run_config)
            except Exception as exc:
                _append_log(log_path, f"method_error reward={reward_name} name={method_name} error={exc!r}")
                _append_jsonl(
                    jsonl_path,
                    {
                        "event": "method_error",
                        "timestamp": _timestamp(),
                        "reward": reward_name,
                        "method": method_name,
                        "run_label": run_label,
                        "error": repr(exc),
                    },
                )
                raise

            method_rows = _result_rows(run_label, result, reward_name=reward_name)
            rows.extend(method_rows)
            summary = {
                "event": "method_end",
                "timestamp": _timestamp(),
                "reward": reward_name,
                "method": method_name,
                "run_label": run_label,
                "rows": len(method_rows),
                "final_mse": result.mse_trace[-1] if result.mse_trace else None,
                "final_mae": result.mae_trace[-1] if result.mae_trace else None,
                "final_nlpd": result.nlpd_trace[-1] if result.nlpd_trace else None,
                "final_coverage": result.coverage_trace[-1] if result.coverage_trace else None,
                "output_dir": result.output_dir,
            }
            _append_log(
                log_path,
                (
                    f"method_end reward={reward_name} name={method_name} rows={len(method_rows)} "
                    f"final_mse={summary['final_mse']} final_mae={summary['final_mae']} "
                    f"final_nlpd={summary['final_nlpd']} final_coverage={summary['final_coverage']}"
                ),
            )
            _append_jsonl(jsonl_path, summary)

        _append_log(log_path, f"reward_end name={reward_name}")
        _append_jsonl(jsonl_path, {"event": "reward_end", "timestamp": _timestamp(), "reward": reward_name})

    _write_csv(csv_path, rows)
    _append_log(log_path, f"sweep_end rows={len(rows)} csv={csv_path}")
    _append_jsonl(
        jsonl_path,
        {"event": "sweep_end", "timestamp": _timestamp(), "rows": len(rows), "csv": str(csv_path)},
    )
    return rows


def _reward_configs(config: dict[str, Any]) -> list[dict[str, Any]]:
    rewards = config.get("rewards")
    if not rewards:
        dataset_reward = config.get("dataset", {}).get("reward", "dipole")
        return [{"name": dataset_reward}]

    reward_configs = []
    for reward in rewards:
        if isinstance(reward, str):
            reward_configs.append({"name": reward})
        elif isinstance(reward, dict) and "name" in reward:
            reward_configs.append(dict(reward))
        else:
            raise ValueError("Each reward must be a name string or a mapping with a 'name' field.")
    return reward_configs


def _method_config(config: dict[str, Any], method: dict[str, Any], reward: dict[str, Any]) -> dict[str, Any]:
    save_artifacts = bool(config.get("output", {}).get("save_artifacts", False))
    dataset_config = dict(config["dataset"])
    dataset_config["reward"] = reward["name"]
    if "reward_energy_range" in reward:
        dataset_config["reward_energy_range"] = reward["reward_energy_range"]
    elif "reward_energy_range" in dataset_config and config.get("rewards"):
        dataset_config.pop("reward_energy_range")

    return {
        "seed": config.get("seed", 0),
        "dataset": dataset_config,
        "benchmark": config.get("benchmark", {}),
        "representation": method.get("representation", {}),
        "model": method.get("model", {}),
        "acquisition": method.get("acquisition", {}),
        "output": {
            "enabled": save_artifacts,
            "dir": config.get("output", {}).get("dir", "outputs"),
            "run_name": f"{reward['name']}__{method['name']}" if config.get("rewards") else method["name"],
            "save_step_plots": save_artifacts,
            "save_step_pickles": save_artifacts,
            "save_trajectory_plot": save_artifacts,
        },
        "checkpoint": method.get("checkpoint", config.get("checkpoint", {})),
    }


def _result_rows(method_name: str, result: BenchmarkResult, reward_name: str | None = None) -> list[dict[str, Any]]:
    rows = []
    for step, selected_index in enumerate(result.acquired_order):
        rows.append(
            {
                "reward": reward_name,
                "method": method_name,
                "step": step,
                "selected_index": selected_index,
                "mse": result.mse_trace[step],
                "mae": result.mae_trace[step],
                "nlpd": result.nlpd_trace[step],
                "coverage": result.coverage_trace[step],
                "mean_prediction": result.mean_prediction_trace[step],
                "mean_variance": result.mean_variance_trace[step],
                "loss_initial": result.loss_initial_trace[step],
                "loss_final": result.loss_final_trace[step],
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "reward",
        "method",
        "step",
        "selected_index",
        "mse",
        "mae",
        "nlpd",
        "coverage",
        "mean_prediction",
        "mean_variance",
        "loss_initial",
        "loss_final",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"[{_timestamp()}] {message}\n")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_json_safe(payload), sort_keys=True) + "\n")


def _timestamp() -> str:
    return datetime.now().isoformat()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value