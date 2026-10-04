from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common import (
    ROOT,
    assert_eval_disjoint,
    load_config,
    project_path,
    read_jsonl,
)


def split_pairs(
    rows: list[dict[str, str]],
    seed: int,
    ratios: tuple[float, float, float],
    requested_counts: tuple[int, int, int] | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    if len(rows) < 3:
        raise ValueError("At least three unique parallel pairs are required to split data")
    if any(ratio < 0 for ratio in ratios) or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("Split ratios must be non-negative and sum to 1")

    unique: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        pair = (row["ko"].strip(), row["en"].strip())
        if not all(pair):
            raise ValueError("Parallel pairs must contain non-empty Korean and English")
        if pair not in seen:
            seen.add(pair)
            unique.append({"ko": pair[0], "en": pair[1]})
    if len(unique) < 3:
        raise ValueError("At least three unique parallel pairs are required to split data")

    random.Random(seed).shuffle(unique)
    if requested_counts and sum(requested_counts) <= len(unique):
        train_count, validation_count, test_count = requested_counts
        train_count += len(unique) - sum(requested_counts)
    else:
        train_count = round(len(unique) * ratios[0])
        validation_count = round(len(unique) * ratios[1])
        test_count = len(unique) - train_count - validation_count
        if validation_count == 0:
            validation_count = 1
            train_count -= 1
        if test_count == 0:
            test_count = 1
            train_count -= 1

    train_end = train_count
    validation_end = train_end + validation_count
    return (
        unique[:train_end],
        unique[train_end:validation_end],
        unique[validation_end : validation_end + test_count],
    )


def write_split(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic pair-level data splits.")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    source = project_path(config["data"]["human_pairs"])
    output = project_path(config["data"]["processed_dir"])

    rows = read_jsonl(source)
    assert_eval_disjoint(rows, read_jsonl(project_path(config["data"]["konglish_eval"])))
    splits = split_pairs(
        rows,
        seed=config["training"]["seed"],
        ratios=(
            config["data"]["train_ratio"],
            config["data"]["validation_ratio"],
            config["data"]["test_ratio"],
        ),
        requested_counts=(
            config["data"]["train_pairs"],
            config["data"]["validation_pairs"],
            config["data"]["test_pairs"],
        ),
    )
    for name, split in zip(("train", "validation", "test"), splits, strict=True):
        write_split(output / f"{name}.jsonl", split)
        print(f"{name}: {len(split)} pairs")
    print(f"Unique source pairs: {sum(map(len, splits))}; seed: {config['training']['seed']}")


if __name__ == "__main__":
    main()
