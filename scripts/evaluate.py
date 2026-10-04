from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from sacrebleu.metrics import BLEU, CHRF
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from scripts.common import (
    DIRECTIONS,
    ROOT,
    benchmark_translation,
    dataset_manifest,
    latest_artifact,
    latest_directional_run,
    load_config,
    load_nllb,
    model_directory_size,
    new_experiment_dir,
    parameter_count,
    project_path,
    read_jsonl,
    select_device,
    translate_nllb,
    write_json,
    write_experiment_metadata,
)


def load_evaluable(kind: str, path: Path | None, config: dict):
    start = time.perf_counter()
    device = select_device()
    if kind in ("baseline", "lora"):
        tokenizer, model = load_nllb(config, path if kind == "lora" else None)
        model.to(device).eval()
        return tokenizer, model, device, time.perf_counter() - start

    tokenizer = AutoTokenizer.from_pretrained(path)
    if kind == "student_int8":
        from optimum.onnxruntime import ORTModelForSeq2SeqLM

        model = ORTModelForSeq2SeqLM.from_pretrained(
            path,
            provider=config["quantization"]["provider"],
        )
        device = torch.device("cpu")
    else:
        requested_device = config["evaluation"]["student_fp32_device"]
        if requested_device == "auto":
            device = select_device()
        elif requested_device == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError("evaluation.student_fp32_device=mps, but MPS is unavailable")
            device = torch.device("mps")
        elif requested_device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("evaluation.student_fp32_device=cuda, but CUDA is unavailable")
            device = torch.device("cuda")
        elif requested_device == "cpu":
            device = torch.device("cpu")
        else:
            raise ValueError(
                "evaluation.student_fp32_device must be one of: auto, cpu, mps, cuda"
            )
        model = AutoModelForSeq2SeqLM.from_pretrained(path)
        model.to(device).eval()
    return tokenizer, model, device, time.perf_counter() - start


def make_translator(kind: str, tokenizer, model, direction: str, config, device):
    if kind in ("baseline", "lora"):
        return lambda text: translate_nllb(text, tokenizer, model, direction, config, device)

    def translate(text: str) -> str:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=config["training"]["max_source_length"],
        )
        if device.type != "cpu":
            inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=config["evaluation"]["max_new_tokens"],
                num_beams=config["evaluation"]["num_beams"],
            )
        return tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip()

    return translate


def metric_record(
    kind: str,
    model_label: str,
    model_path: Path | None,
    direction: str,
    dataset_name: str,
    references: list[str],
    predictions: list[str],
    latency: dict[str, float],
    load_seconds: float,
    parameters: int | None,
    size_bytes: int | None,
) -> dict:
    return {
        "model": model_label,
        "model_type": kind,
        "model_path": str(model_path) if model_path else model_label,
        "direction": direction,
        "dataset": dataset_name,
        "device": str(device),
        "precision": "INT8" if kind == "student_int8" else "FP32",
        "parameters": parameters,
        "model_size_mb": round(size_bytes / (1024 * 1024), 3) if size_bytes is not None else None,
        "bleu": round(BLEU().corpus_score(predictions, [references]).score, 4),
        "chrf": round(CHRF().corpus_score(predictions, [references]).score, 4),
        "comet": None,
        "mean_latency_ms": round(latency["mean_latency_ms"], 3),
        "median_latency_ms": round(latency["median_latency_ms"], 3),
        "p95_latency_ms": round(latency["p95_latency_ms"], 3),
        "model_loading_seconds": round(load_seconds, 3),
        "evaluated_examples": len(references),
        "notes": "COMET omitted; no COMET model/dependency configured.",
    }


def evaluate_dataset(
    kind: str,
    label: str,
    path: Path | None,
    tokenizer,
    model,
    device,
    load_seconds: float,
    direction: str,
    dataset_name: str,
    rows: list[dict[str, str]],
    config: dict,
) -> dict:
    source_key, target_key, _ = DIRECTIONS[direction]
    texts = [row[source_key] for row in rows]
    references = [row[target_key] for row in rows]
    translate = make_translator(kind, tokenizer, model, direction, config, device)
    predictions, latency = benchmark_translation(
        translate,
        texts,
        config["evaluation"]["latency_warmup"],
        config["evaluation"]["latency_repeats"],
        device,
    )
    length_benchmarks = {}
    buckets = (("1-10_words", 1, 10), ("11-20_words", 11, 20), ("21-40_words", 21, 40))
    for bucket_name, minimum, maximum in buckets:
        bucket_texts = [
            text for text in texts if minimum <= len(text.split()) <= maximum
        ]
        if bucket_texts:
            _, bucket_latency = benchmark_translation(
                translate,
                bucket_texts,
                0,
                config["evaluation"]["latency_repeats"],
                device,
            )
            length_benchmarks[bucket_name] = {
                **bucket_latency,
                "examples": len(bucket_texts),
            }
    if path is not None:
        parameter_total = parameter_count(model) if hasattr(model, "parameters") else None
        if parameter_total is None and kind == "student_int8":
            from transformers import AutoModelForSeq2SeqLM

            source_run = latest_directional_run("models/student")
            source = source_run / direction if source_run else None
            if source and source.exists():
                reference_model = AutoModelForSeq2SeqLM.from_pretrained(source)
                parameter_total = parameter_count(reference_model)
                del reference_model
        size = model_directory_size(path)
        if kind == "lora" and parameter_total is not None:
            size += parameter_total * 4
    else:
        parameter_total = parameter_count(model)
        size = parameter_total * 4
    record = metric_record(
        kind,
        label,
        path,
        direction,
        dataset_name,
        references,
        predictions,
        latency,
        load_seconds,
        parameter_total,
        size,
    )
    record["latency_by_word_count"] = length_benchmarks
    return record


def evaluate_run(
    config: dict,
    save_results: bool = True,
    config_path: Path | None = None,
) -> list[dict]:
    processed = project_path(config["data"]["processed_dir"])
    datasets = {
        "validation": read_jsonl(processed / "validation.jsonl"),
        "test": read_jsonl(processed / "test.jsonl"),
    }
    konglish_path = project_path(config["data"]["konglish_eval"])
    if konglish_path.exists():
        datasets["konglish_eval"] = read_jsonl(konglish_path)

    student_run = latest_directional_run("models/student")
    int8_run = latest_directional_run("models/student_int8")
    artifacts: list[tuple[str, str, Path | None, str | None]] = [
        ("baseline", "NLLB-600M", None, None),
    ]
    adapter = latest_artifact("models/lora/*")
    artifacts.append(("lora", "NLLB-600M + LoRA", adapter, None))
    for direction in config["evaluation"]["directions"]:
        if student_run:
            artifacts.append(("student", "Marian student", student_run / direction, direction))
        if int8_run:
            artifacts.append(
                ("student_int8", "Marian student INT8", int8_run / direction, direction)
            )

    output: list[dict] = []
    for kind, label, artifact_path, only_direction in artifacts:
        if kind == "lora" and artifact_path is None:
            print("Skipping LoRA: no adapter artifact found.")
            continue
        for direction in config["evaluation"]["directions"]:
            if only_direction and direction != only_direction:
                continue
            print(f"Evaluating {label}, {direction}...")
            tokenizer, model, device, load_seconds = load_evaluable(
                kind, artifact_path, config
            )
            for dataset_name, rows in datasets.items():
                record = evaluate_dataset(
                    kind,
                    label,
                    artifact_path,
                    tokenizer,
                    model,
                    device,
                    load_seconds,
                    direction,
                    dataset_name,
                    rows,
                    config,
                )
                output.append(record)
                print(
                    f"  {dataset_name}: BLEU {record['bleu']:.2f}, "
                    f"chrF {record['chrf']:.2f}, "
                    f"mean/median/p95 {record['mean_latency_ms']:.1f}/"
                    f"{record['median_latency_ms']:.1f}/{record['p95_latency_ms']:.1f} ms"
                )
            del tokenizer, model
            if device.type == "mps":
                torch.mps.empty_cache()
            elif device.type == "cuda":
                torch.cuda.empty_cache()

    if save_results:
        results_dir = project_path("results")
        results_dir.mkdir(parents=True, exist_ok=True)
        for name, kind in (("baseline.json", "baseline"), ("lora.json", "lora")):
            write_json(results_dir / name, [record for record in output if record["model_type"] == kind])
        write_json(results_dir / "comparison.json", output)
        columns = [
            "model", "direction", "dataset", "device", "precision", "parameters", "model_size_mb", "bleu", "chrf",
            "comet", "model_loading_seconds", "mean_latency_ms", "median_latency_ms",
            "p95_latency_ms",
        ]
        with (results_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows({column: row.get(column) for column in columns} for row in output)
        with (results_dir / "latency_by_length.csv").open(
            "w",
            encoding="utf-8",
            newline="",
        ) as stream:
            latency_columns = [
                "model", "direction", "dataset", "device", "word_bucket", "examples",
                "mean_latency_ms", "median_latency_ms", "p95_latency_ms",
            ]
            writer = csv.DictWriter(stream, fieldnames=latency_columns)
            writer.writeheader()
            for record in output:
                for bucket, latency in record["latency_by_word_count"].items():
                    writer.writerow({
                        "model": record["model"],
                        "direction": record["direction"],
                        "dataset": record["dataset"],
                        "device": record["device"],
                        "word_bucket": bucket,
                        **latency,
                    })
        grouped: dict[tuple[str, str], dict[str, dict]] = {}
        for row in output:
            if row["dataset"] == "test":
                grouped.setdefault((row["model"], row["dataset"]), {})[row["direction"]] = row
        with (results_dir / "tradeoff.csv").open("w", encoding="utf-8", newline="") as stream:
            tradeoff_columns = [
                "model", "dataset", "device", "parameters", "precision", "model_size_mb",
                "ko_en_bleu", "ko_en_chrf", "en_ko_bleu", "en_ko_chrf",
                "ko_en_mean_latency_ms", "ko_en_median_latency_ms", "ko_en_p95_latency_ms",
                "en_ko_mean_latency_ms", "en_ko_median_latency_ms", "en_ko_p95_latency_ms",
                "mean_latency_ms", "median_latency_ms", "p95_latency_ms",
            ]
            writer = csv.DictWriter(stream, fieldnames=tradeoff_columns)
            writer.writeheader()
            for (model_label, dataset_name), directions in grouped.items():
                first = next(iter(directions.values()))
                ko_en = directions.get("ko_en", {})
                en_ko = directions.get("en_ko", {})
                if first["model_type"] in ("student", "student_int8"):
                    parameter_total = sum(row["parameters"] or 0 for row in directions.values())
                    size_total = sum(row["model_size_mb"] or 0 for row in directions.values())
                else:
                    parameter_total = first["parameters"]
                    size_total = first["model_size_mb"]
                writer.writerow({
                    "model": model_label,
                    "dataset": dataset_name,
                    "device": first["device"],
                    "parameters": parameter_total,
                    "precision": first["precision"],
                    "model_size_mb": round(size_total, 3) if size_total is not None else None,
                    "ko_en_bleu": ko_en.get("bleu"),
                    "ko_en_chrf": ko_en.get("chrf"),
                    "en_ko_bleu": en_ko.get("bleu"),
                    "en_ko_chrf": en_ko.get("chrf"),
                    "ko_en_mean_latency_ms": ko_en.get("mean_latency_ms"),
                    "ko_en_median_latency_ms": ko_en.get("median_latency_ms"),
                    "ko_en_p95_latency_ms": ko_en.get("p95_latency_ms"),
                    "en_ko_mean_latency_ms": en_ko.get("mean_latency_ms"),
                    "en_ko_median_latency_ms": en_ko.get("median_latency_ms"),
                    "en_ko_p95_latency_ms": en_ko.get("p95_latency_ms"),
                    "mean_latency_ms": round(
                        sum(row["mean_latency_ms"] for row in directions.values()) / len(directions),
                        3,
                    ),
                    "median_latency_ms": round(
                        sum(row["median_latency_ms"] for row in directions.values()) / len(directions),
                        3,
                    ),
                    "p95_latency_ms": max(row["p95_latency_ms"] for row in directions.values()),
                })
        experiment = new_experiment_dir("evaluation")
        write_experiment_metadata(
            experiment,
            config_path or ROOT / "config.yaml",
            config["training"]["seed"],
            "Available NLLB, LoRA, and Marian artifacts",
            dataset_manifest(
                processed / "test.jsonl",
                project_path(config["data"]["konglish_eval"]),
            ),
            {"evaluation": config["evaluation"]},
        )
        write_json(experiment / "metrics.json", output)

    print("\nHuman test split — BLEU / chrF")
    print(f"{'Model':32} {'Direction':10} {'BLEU':>8} {'chrF':>8}")
    for record in output:
        if record["dataset"] == "test":
            print(
                f"{record['model']:32} {record['direction']:10} "
                f"{record['bleu']:8.2f} {record['chrf']:8.2f}"
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all available translation artifacts.")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    args = parser.parse_args()
    evaluate_run(load_config(args.config), config_path=args.config.resolve())


if __name__ == "__main__":
    main()
