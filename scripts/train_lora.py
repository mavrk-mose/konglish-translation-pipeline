from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
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
    directional_rows,
    load_config,
    new_experiment_dir,
    project_path,
    read_jsonl,
    select_device,
    set_seed,
    write_experiment_metadata,
)


def tokenize_rows(rows, tokenizer, config) -> Dataset:
    dataset = Dataset.from_list(rows)

    def tokenize(example):
        _, _, direction = DIRECTIONS[example["direction"]]
        language_pair = config["translation"][direction]
        tokenizer.src_lang = language_pair["source_lang"]
        tokenizer.tgt_lang = language_pair["target_lang"]
        encoded = tokenizer(
            example["source"],
            max_length=config["training"]["max_source_length"],
            truncation=True,
        )
        target = tokenizer(
            text_target=example["target"],
            max_length=config["training"]["max_target_length"],
            truncation=True,
        )
        encoded["labels"] = target["input_ids"]
        return encoded

    return dataset.map(tokenize, remove_columns=dataset.column_names)


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tune NLLB for conversational translation.")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    set_seed(config["training"]["seed"])
    device = select_device()

    train_path = project_path(config["data"]["processed_dir"]) / "train.jsonl"
    validation_path = project_path(config["data"]["processed_dir"]) / "validation.jsonl"
    train_rows = directional_rows(read_jsonl(train_path))
    validation_rows = directional_rows(read_jsonl(validation_path))
    model_name = config["model"]["teacher"]

    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print(f"Training examples (both directions): {len(train_rows)}")
    print(f"Validation examples (both directions): {len(validation_rows)}")
    print(f"Training configuration: {config['training']}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_load_kwargs = {"attn_implementation": "eager"} if device.type == "mps" else {}
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **model_load_kwargs)
    model.config.use_cache = False
    lora = config["training"]["lora"]
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=lora["rank"],
            lora_alpha=lora["alpha"],
            lora_dropout=lora["dropout"],
            target_modules=["q_proj", "v_proj"],
        ),
    )
    model.print_trainable_parameters()
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(
        f"Parameters: trainable={trainable_parameters:,}, total={total_parameters:,}, "
        f"trainable_percent={100 * trainable_parameters / total_parameters:.4f}%"
    )

    experiment = new_experiment_dir("nllb_lora")
    artifact = ROOT / "models" / "lora" / experiment.name
    training = config["training"]
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(experiment / "checkpoints"),
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
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_pin_memory=device.type != "mps",
        seed=training["seed"],
        data_seed=training["seed"],
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenize_rows(train_rows, tokenizer, config),
        eval_dataset=tokenize_rows(validation_rows, tokenizer, config),
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100),
        processing_class=tokenizer,
    )
    train_result = trainer.train()
    training_logs = [
        entry
        for entry in trainer.state.log_history
        if "loss" in entry and "grad_norm" in entry
    ]
    if not training_logs:
        raise RuntimeError("Training completed without logged loss and gradient norm values.")
    print(
        f"Optimization steps: {trainer.state.global_step}/{trainer.state.max_steps}"
    )
    print(f"Training loss: {train_result.metrics['train_loss']}")
    print(f"Gradient norm: {training_logs[-1]['grad_norm']}")
    model.config.use_cache = True
    artifact.mkdir(parents=True, exist_ok=False)
    trainer.save_model(str(artifact))
    tokenizer.save_pretrained(artifact)
    shutil.copy2(config_path, artifact / "config.yaml")
    (artifact / "README.txt").write_text(
        "NLLB LoRA adapter; load on top of the configured base teacher. "
        "This is not a merged or replacement teacher model.\n",
        encoding="utf-8",
    )
    write_experiment_metadata(
        experiment,
        config_path,
        training["seed"],
        model_name,
        dataset_manifest(train_path, validation_path),
        {
            **training,
            "device": str(device),
            "train_examples_bidirectional": len(train_rows),
            "validation_examples_bidirectional": len(validation_rows),
            "trainable_parameters": trainable_parameters,
            "total_parameters": total_parameters,
        },
    )
    (experiment / "metrics.json").write_text(
        json.dumps({"log_history": trainer.state.log_history}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"LoRA adapter saved to: {artifact}")


if __name__ == "__main__":
    main()
