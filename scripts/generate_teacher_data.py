from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from scripts.common import (
    DIRECTIONS,
    ROOT,
    assert_no_test_sources,
    dataset_manifest,
    latest_artifact,
    load_config,
    project_path,
    read_jsonl,
    require_lora_improvement,
    select_device,
    set_seed,
    write_experiment_metadata,
)


def read_sentences(path: Path, limit: int) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Monolingual source file not found: {path}. Add one sentence per line; "
            "test-set sentences must not be placed here."
        )
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()][
        :limit
    ]


def generate_direction(
    sentences: list[str],
    direction: str,
    model,
    tokenizer,
    config,
    device: torch.device,
) -> list[dict[str, str]]:
    _, _, config_key = DIRECTIONS[direction]
    language = config["translation"][config_key]
    tokenizer.src_lang = language["source_lang"]
    target_id = tokenizer.convert_tokens_to_ids(language["target_lang"])
    rows = []
    batch_size = config["distillation"]["generation_batch_size"]
    for start in range(0, len(sentences), batch_size):
        batch = sentences[start : start + batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config["training"]["max_source_length"],
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            tokens = model.generate(
                **encoded,
                forced_bos_token_id=target_id,
                max_new_tokens=config["distillation"]["max_new_tokens"],
                num_beams=2,
            )
        translations = tokenizer.batch_decode(tokens, skip_special_tokens=True)
        for source, target in zip(batch, translations, strict=True):
            source_text, target_text = source.strip(), target.strip()
            if source_text and target_text:
                pair = {"ko": source_text, "en": target_text}
                if direction == "en_ko":
                    pair = {"ko": target_text, "en": source_text}
                rows.append(pair)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate separate, labeled teacher data.")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--adapter", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    set_seed(config["training"]["seed"])
    config_path = args.config.resolve()
    if config["distillation"]["require_lora_chrf_improvement"]:
        require_lora_improvement(project_path("results/comparison.json"))
    adapter = args.adapter or latest_artifact("models/lora/*")
    if adapter is None:
        raise FileNotFoundError("Train a LoRA adapter before generating teacher data.")
    adapter = adapter.resolve()

    model_name = config["model"]["teacher"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model = PeftModel.from_pretrained(model, adapter)
    device = select_device()
    model.to(device).eval()

    limit = config["distillation"]["max_examples_per_direction"]
    sources = {
        "ko_en": read_sentences(project_path(config["data"]["monolingual_ko"]), limit),
        "en_ko": read_sentences(project_path(config["data"]["monolingual_en"]), limit),
    }
    test_rows = read_jsonl(project_path(config["data"]["processed_dir"]) / "test.jsonl")
    assert_no_test_sources(sources["ko_en"], sources["en_ko"], test_rows)
    assert_no_test_sources(
        sources["ko_en"],
        sources["en_ko"],
        read_jsonl(project_path(config["data"]["konglish_eval"])),
    )

    stamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S_%f")
    output_root = ROOT / "data" / "synthetic" / f"{stamp}_teacher_generation"
    output_root.mkdir(parents=True, exist_ok=False)
    experiment = ROOT / "experiments" / f"{stamp}_teacher_generation"
    experiment.mkdir(parents=True, exist_ok=False)
    result_summary = {}
    for direction, sentences in sources.items():
        rows = generate_direction(sentences, direction, model, tokenizer, config, device)
        output = output_root / f"synthetic_{direction}.jsonl"
        with output.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        result_summary[direction] = {
            "file": str(output.relative_to(ROOT)),
            "examples": len(rows),
            "source_type": "teacher_generated_synthetic",
            "test_data_used": False,
            "monolingual_source": str(
                project_path(config["data"]["monolingual_ko" if direction == "ko_en" else "monolingual_en"])
            ),
        }
        print(f"{direction}: wrote {len(rows)} examples to {output}")
    write_experiment_metadata(
        experiment,
        config_path,
        config["training"]["seed"],
        model_name,
        {
            **dataset_manifest(
                project_path(config["data"]["monolingual_ko"]),
                project_path(config["data"]["monolingual_en"]),
                project_path(config["data"]["processed_dir"]) / "test.jsonl",
                project_path(config["data"]["konglish_eval"]),
            ),
            "teacher_adapter": str(adapter),
        },
        result_summary,
    )
    (experiment / "metrics.json").write_text(
        json.dumps({"generation": result_summary}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
