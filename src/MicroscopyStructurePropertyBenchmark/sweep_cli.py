from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from MicroscopyStructurePropertyBenchmark.runners.sweep import run_sweep


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an EM benchmark method sweep.")
    parser.add_argument("--config", required=True, help="Path to a YAML sweep config.")
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    rows = run_sweep(config)
    print(json.dumps({"rows": len(rows), "csv": config.get("output", {}).get("csv")}, indent=2))


if __name__ == "__main__":
    main()
