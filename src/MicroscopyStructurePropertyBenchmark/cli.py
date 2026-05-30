from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from MicroscopyStructurePropertyBenchmark.runners.active_learning import run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an EM active-learning benchmark.")
    parser.add_argument("--config", required=True, help="Path to a YAML benchmark config.")
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    result = run_benchmark(config)
    print(json.dumps(result.summary(), indent=2))


if __name__ == "__main__":
    main()
