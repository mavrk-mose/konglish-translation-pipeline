from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from scripts.common import (
    DIRECTIONS,
    ROOT,
    dataset_manifest,
    latest_complete_directory,
    load_config,
    new_experiment_dir,
    project_path,
    read_jsonl,
    select_device,
    set_seed,
    write_json,
    write_experiment_metadata,
    parameter_count,
)


def direction_data(
    human_rows: list[dict[str, str]],
    synthetic_rows: list[dict[str, str]],
    direction: str,
    synthetic_ratio: float,
    seed: int,
) -> list[dict[str, str]]:
    source_key, target_key, _ = DIRECTIONS[direction]
    randomizer = random.Random(seed)
    max_synthetic = int(len(human_rows) * synthetic_ratio)
    if max_synthetic < len(synthetic_rows):
        synthetic_rows = randomizer.sample(synthetic_rows, max_synthetic)
    rows = [
        {"source": row[source_key], "target": row[target_key], "origin": "human"}
        for row in human_rows
    ]
    rows.extend(
        {"source": row[source_key], "target": row[target_key], "origin": "synthetic"}
        for row in synthetic_rows
    )
    randomizer.shuffle(rows)
    return rows


def tokenize_rows(rows, tokenizer, max_source_length: int, max_target_length: int) -> Dataset:
    dataset = Dataset.from_list(rows)

    def tokenize(example):
        encoded = tokenizer(
            example["source"],
            max_length=max_source_length,
            truncation=True,
        )
        target = tokenizer(
            text_target=example["target"],
            max_length=max_target_length,
            truncation=True,
        )
        encoded["labels"] = target["input_ids"]
        return encoded

    return dataset.map(tokenize, remove_columns=dataset.column_names)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train direction-specific Marian student models.")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    set_seed(config["training"]["seed"])
    device = select_device()
    processed = project_path(config["data"]["processed_dir"])
    human_rows = read_jsonl(processed / "train.jsonl")
    validation_rows = read_jsonl(processed / "validation.jsonl")
    synthetic_run = latest_complete_directory(
        "data/synthetic/*",
        ("synthetic_ko_en.jsonl", "synthetic_en_ko.jsonl"),
    )
    experiment = new_experiment_dir("student")
    training = config["training"]
    run_info = {}
    print(f"Device: {device}")
    print(f"Training configuration: {training}")

    for direction, model_key in (
        ("ko_en", "ko_en_model"),
        ("en_ko", "en_ko_model"),
    ):
        synthetic_path = synthetic_run / f"synthetic_{direction}.jsonl" if synthetic_run else None
        synthetic_rows = read_jsonl(synthetic_path) if synthetic_path else []
        train_rows = direction_data(
            human_rows,
            synthetic_rows,
            direction,
            config["distillation"]["synthetic_to_human_ratio"],
            training["seed"],
        )
        model_name = config["student"][model_key]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model.config.use_cache = False
        parameters = parameter_count(model)
        print(f"{direction}: model={model_name}, parameters={parameters:,}")
        artifact = ROOT / "models" / "student" / experiment.name / direction
        checkpoint_dir = experiment / "checkpoints" / direction
        arguments = Seq2SeqTrainingArguments(
            output_dir=str(checkpoint_dir),
            learning_rate=training["learning_rate"],
            num_train_epochs=training["epochs"],
            per_device_train_batch_size=training["batch_size"],
            per_device_eval_batch_size=training["eval_batch_size"],
            gradient_accumulation_steps=training["gradient_accumulation_steps"],
            weight_decay=training["weight_decay"],
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            logging_steps=training["logging_steps"],
            report_to="none",
            fp16=False,
            use_cpu=device.type == "cpu",
            gradient_checkpointing=True,
            seed=training["seed"],
            data_seed=training["seed"],
        )
        source_key, target_key, _ = DIRECTIONS[direction]
        trainer = Seq2SeqTrainer(
            model=model,
            args=arguments,
            train_dataset=tokenize_rows(
                train_rows,
                tokenizer,
                training["max_source_length"],
                training["max_target_length"],
            ),
            eval_dataset=tokenize_rows(
                [
                    {"source": row[source_key], "target": row[target_key]}
                    for row in validation_rows
                ],
                tokenizer,
                training["max_source_length"],
                training["max_target_length"],
            ),
            data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
            processing_class=tokenizer,
        )
        trainer.train()
        model.config.use_cache = True
        trainer.save_model(str(artifact))
        tokenizer.save_pretrained(artifact)
        shutil.copy2(config_path, artifact / "config.yaml")
        run_info[direction] = {
            "base_model": model_name,
            "artifact": str(artifact.relative_to(ROOT)),
            "human_training_pairs": len(human_rows),
            "synthetic_training_pairs": len(train_rows) - len(human_rows),
            "validation_pairs": len(validation_rows),
            "parameters": parameters,
            "test_data_used": False,
        }
        print(f"{direction}: model saved to {artifact}")

    write_experiment_metadata(
        experiment,
        config_path,
        training["seed"],
        "direction-specific Marian student models",
        dataset_manifest(
            processed / "train.jsonl",
            processed / "validation.jsonl",
            *([synthetic_run / "synthetic_ko_en.jsonl", synthetic_run / "synthetic_en_ko.jsonl"] if synthetic_run else []),
        ),
        {
            **training,
            "directions": run_info,
            "architecture": config["student"]["architecture"],
        },
    )
    write_json(experiment / "metrics.json", {"training_runs": run_info})
    (experiment / "README.md").write_text(
        "# Student training run\n\nHuman train pairs are retained once; synthetic pairs are capped "
        "by the configured ratio. The held-out test set is never read.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
