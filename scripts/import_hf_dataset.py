from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoTokenizer

from scripts.common import ROOT, load_config, project_path


DATASET_NAME = "lemon-mint/korean_english_parallel_wiki_augmented_v1"
MIN_WORDS = 3
SAMPLE_SEED = 42


def filter_candidate_rows(
    rows,
    min_score: float,
) -> tuple[list[dict[str, str | float]], int]:
    candidates = []
    seen: set[tuple[str, str]] = set()
    duplicates_removed = 0

    for row in rows:
        korean = row.get("korean")
        english = row.get("english")
        score = row.get("score")
        if not isinstance(korean, str) or not isinstance(english, str) or score is None:
            continue

        korean = korean.strip()
        english = english.strip()
        if not korean or not english:
            continue
        if len(korean.split()) < MIN_WORDS or len(english.split()) < MIN_WORDS:
            continue
        try:
            numeric_score = float(score)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(numeric_score) or numeric_score < min_score:
            continue

        pair = (korean, english)
        if pair in seen:
            duplicates_removed += 1
            continue
        seen.add(pair)
        candidates.append({"ko": korean, "en": english, "score": numeric_score})

    return candidates, duplicates_removed


def filter_by_token_length(
    rows: list[dict[str, str | float]],
    tokenizer,
    max_tokens: int,
    batch_size: int = 256,
) -> list[dict[str, str | float]]:
    keep = [True] * len(rows)
    for key, language in (("ko", "kor_Hang"), ("en", "eng_Latn")):
        tokenizer.src_lang = language
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            encoded = tokenizer(
                [row[key] for row in batch],
                add_special_tokens=True,
                truncation=False,
                padding=False,
            )
            for offset, token_ids in enumerate(encoded["input_ids"]):
                if len(token_ids) > max_tokens:
                    keep[start + offset] = False

    return [row for row, should_keep in zip(rows, keep, strict=True) if should_keep]


def select_rows(
    rows: list[dict[str, str | float]],
    limit: int,
    seed: int = SAMPLE_SEED,
) -> list[dict[str, str | float]]:
    if limit < 1:
        raise ValueError("--limit must be a positive integer")
    if len(rows) <= limit:
        return list(rows)
    return random.Random(seed).sample(rows, limit)


def write_pairs_with_backup(
    output_path: Path,
    backup_path: Path,
    rows: list[dict[str, str | float]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not backup_path.exists():
        shutil.copy2(output_path, backup_path)

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output_path.parent,
        delete=False,
    ) as stream:
        temporary_path = Path(stream.name)
        for row in rows:
            stream.write(
                json.dumps({"ko": row["ko"], "en": row["en"]}, ensure_ascii=False) + "\n"
            )
    temporary_path.replace(output_path)


def main() -> None:
    from datasets import load_dataset

    parser = argparse.ArgumentParser(
        description="Import scored Korean-English pairs from the Hugging Face wiki dataset."
    )
    parser.add_argument("--limit", type=int, default=10_000)
    parser.add_argument("--min-score", type=float, default=0.85)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be a positive integer")
    if not math.isfinite(args.min_score):
        parser.error("--min-score must be finite")

    config = load_config(args.config)
    dataset = load_dataset(DATASET_NAME, split="train")
    original_size = len(dataset)
    candidates, duplicates_removed = filter_candidate_rows(dataset, args.min_score)

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["teacher"])
    max_tokens = min(
        config["training"]["max_source_length"],
        config["training"]["max_target_length"],
    )
    filtered = filter_by_token_length(candidates, tokenizer, max_tokens)
    selected = select_rows(filtered, args.limit)

    output_path = project_path(config["data"]["human_pairs"])
    backup_path = output_path.with_name("human_parallel_seed_backup.jsonl")
    write_pairs_with_backup(output_path, backup_path, selected)

    scores = [float(row["score"]) for row in selected]
    print(f"Original dataset size: {original_size}")
    print(f"Remaining after filtering: {len(filtered)}")
    print(f"Selected: {len(selected)}")
    print(f"Duplicates removed: {duplicates_removed}")
    if scores:
        print(
            f"Selected score min/max/mean: {min(scores):.6f}/"
            f"{max(scores):.6f}/{sum(scores) / len(scores):.6f}"
        )
    else:
        print("Selected score min/max/mean: unavailable (no pairs selected)")
    print(f"Output: {output_path}")
    if backup_path.exists():
        print(f"Seed backup: {backup_path}")
    print("Sample Korean / English pairs:")
    for row in selected[:5]:
        print(f"KO: {row['ko']}")
        print(f"EN: {row['en']}\n")


if __name__ == "__main__":
    main()
