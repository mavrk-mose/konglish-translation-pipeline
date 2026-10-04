from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common import (
    ROOT,
    dataset_manifest,
    load_config,
    new_experiment_dir,
    project_path,
    write_experiment_metadata,
    write_json,
)
from scripts.evaluate import evaluate_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run translation and model-load benchmarks.")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    records = evaluate_run(config, save_results=False)
    benchmark = [
        {
            "model": row["model"],
            "direction": row["direction"],
            "dataset": row["dataset"],
            "device": row["device"],
            "model_loading_seconds": row["model_loading_seconds"],
            "mean_latency_ms": row["mean_latency_ms"],
            "median_latency_ms": row["median_latency_ms"],
            "p95_latency_ms": row["p95_latency_ms"],
            "evaluated_examples": row["evaluated_examples"],
        }
        for row in records
    ]
    destination = project_path("results/benchmark.json")
    write_json(destination, benchmark)
    experiment = new_experiment_dir("benchmark")
    write_experiment_metadata(
        experiment,
        args.config.resolve(),
        config["training"]["seed"],
        "Available translation artifacts",
        dataset_manifest(
            project_path(config["data"]["processed_dir"]) / "test.jsonl",
            project_path(config["data"]["konglish_eval"]),
        ),
        {"evaluation": config["evaluation"]},
    )
    write_json(experiment / "metrics.json", benchmark)
    print(f"Benchmark written to {destination}")


if __name__ == "__main__":
    main()
