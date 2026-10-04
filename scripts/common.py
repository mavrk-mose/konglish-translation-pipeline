from __future__ import annotations

import json
import hashlib
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
DIRECTIONS = {
    "ko_en": ("ko", "en", "ko_to_en"),
    "en_ko": ("en", "ko", "en_to_ko"),
}


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else ROOT / "config.yaml"
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {config_path}")
    return config


def project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path: str | Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected non-empty 'ko' and 'en'")
            korean = str(row.get("ko", "")).strip()
            english = str(row.get("en", "")).strip()
            if not korean or not english:
                raise ValueError(f"{path}:{line_number}: expected non-empty 'ko' and 'en'")
            rows.append({"ko": korean, "en": english})
    return rows


def directional_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    expanded = []
    for row in rows:
        expanded.append({"source": row["ko"], "target": row["en"], "direction": "ko_en"})
        expanded.append({"source": row["en"], "target": row["ko"], "direction": "en_ko"})
    return expanded


def assert_no_test_sources(
    korean_sentences: list[str],
    english_sentences: list[str],
    test_rows: list[dict[str, str]],
) -> None:
    test_korean = {row["ko"].strip() for row in test_rows}
    test_english = {row["en"].strip().casefold() for row in test_rows}
    korean_overlap = test_korean.intersection(sentence.strip() for sentence in korean_sentences)
    english_overlap = test_english.intersection(
        sentence.strip().casefold() for sentence in english_sentences
    )
    if korean_overlap or english_overlap:
        raise ValueError(
            "Monolingual generation sources overlap the held-out test set "
            f"({len(korean_overlap)} Korean, {len(english_overlap)} English sentences). "
            "Remove those lines before generating teacher data."
        )


def assert_eval_disjoint(
    human_rows: list[dict[str, str]],
    evaluation_rows: list[dict[str, str]],
) -> None:
    human_korean = {row["ko"].strip() for row in human_rows}
    human_english = {row["en"].strip().casefold() for row in human_rows}
    overlap_korean = {row["ko"].strip() for row in evaluation_rows} & human_korean
    overlap_english = {
        row["en"].strip().casefold() for row in evaluation_rows
    } & human_english
    if overlap_korean or overlap_english:
        raise ValueError(
            "The held-out Konglish evaluation set overlaps human parallel sources "
            f"({len(overlap_korean)} Korean, {len(overlap_english)} English). "
            "Remove evaluation sentences from the training corpus."
        )


def require_lora_improvement(results_path: Path) -> None:
    if not results_path.is_file():
        raise FileNotFoundError(
            "Run python scripts/evaluate.py after LoRA training before teacher generation."
        )
    results = json.loads(results_path.read_text(encoding="utf-8"))
    validation = [
        row for row in results
        if row["dataset"] == "validation" and row["model_type"] in ("baseline", "lora")
    ]
    baseline = {
        row["direction"]: row["chrf"] for row in validation if row["model_type"] == "baseline"
    }
    lora = {row["direction"]: row["chrf"] for row in validation if row["model_type"] == "lora"}
    if not all(direction in baseline and direction in lora for direction in ("ko_en", "en_ko")):
        raise ValueError(
            "Validation results for NLLB and NLLB+LoRA in both directions are required "
            "before synthetic generation."
        )
    baseline_mean = sum(baseline.values()) / 2
    lora_mean = sum(lora.values()) / 2
    if lora_mean <= baseline_mean:
        raise ValueError(
            "LoRA did not improve mean validation chrF in both-direction evaluation "
            f"(baseline={baseline_mean:.3f}, LoRA={lora_mean:.3f}); synthetic generation is gated."
        )


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def new_experiment_dir(kind: str) -> Path:
    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d_%H-%M-%S_%f")
    destination = ROOT / "experiments" / f"{stamp}_{kind}"
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def model_directory_size(path: str | Path) -> int:
    return sum(file.stat().st_size for file in Path(path).rglob("*") if file.is_file())


def dataset_manifest(*paths: Path) -> dict[str, str]:
    manifest = {}
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        try:
            label = str(path.relative_to(ROOT))
        except ValueError:
            label = str(path.resolve())
        manifest[label] = digest
    return manifest


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def latest_artifact(pattern: str) -> Path | None:
    matches = sorted(
        (path for path in ROOT.glob(pattern) if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def latest_directional_run(root: str | Path) -> Path | None:
    artifact_root = project_path(root)
    if not artifact_root.exists():
        return None
    runs = sorted(
        (path for path in artifact_root.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return next(
        (path for path in runs if all((path / direction).is_dir() for direction in DIRECTIONS)),
        None,
    )


def latest_complete_directory(pattern: str, required_files: tuple[str, ...]) -> Path | None:
    matches = sorted(
        (path for path in ROOT.glob(pattern) if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return next(
        (path for path in matches if all((path / name).is_file() for name in required_files)),
        None,
    )


def load_nllb(config: dict[str, Any], adapter_path: Path | None = None):
    from peft import PeftModel
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model_name = config["model"]["teacher"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    return tokenizer, model


def translate_nllb(
    text: str,
    tokenizer,
    model,
    direction: str,
    config: dict[str, Any],
    device: torch.device,
) -> str:
    _, _, direction_config = DIRECTIONS[direction]
    language_config = config["translation"][direction_config]
    tokenizer.src_lang = language_config["source_lang"]
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=config["training"]["max_source_length"],
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    target_id = tokenizer.convert_tokens_to_ids(language_config["target_lang"])
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            forced_bos_token_id=target_id,
            max_new_tokens=config["evaluation"]["max_new_tokens"],
            num_beams=config["evaluation"]["num_beams"],
        )
    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()


def benchmark_translation(
    translate,
    texts: list[str],
    warmup: int,
    repeats: int,
    device: torch.device | None = None,
) -> tuple[list[str], dict[str, float]]:
    device = device or torch.device("cpu")
    for text in texts[:warmup]:
        translate(text)
        sync_device(device)
    timings: list[float] = []
    predictions: list[str] = []
    for text in texts:
        result = ""
        sync_device(device)
        start = time.perf_counter()
        for _ in range(repeats):
            result = translate(text)
        sync_device(device)
        timings.append((time.perf_counter() - start) * 1000 / repeats)
        predictions.append(result)
    ordered = sorted(timings)
    p95_index = max(0, min(len(ordered) - 1, int((0.95 * len(ordered) + 0.999999)) - 1))
    return predictions, {
        "mean_latency_ms": sum(timings) / len(timings) if timings else 0.0,
        "median_latency_ms": statistics.median(ordered) if ordered else 0.0,
        "p95_latency_ms": ordered[p95_index] if ordered else 0.0,
    }


def write_experiment_metadata(
    experiment_dir: Path,
    config_path: Path,
    seed: int,
    model_name: str,
    dataset_version: Any,
    training: dict[str, Any],
) -> None:
    import shutil

    shutil.copy2(config_path, experiment_dir / "config.yaml")
    write_json(
        experiment_dir / "model_info.json",
        {
            "model": model_name,
            "seed": seed,
            "dataset_version": dataset_version,
            "training": training,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    readme = experiment_dir / "README.md"
    if not readme.exists():
        readme.write_text(
            "# Experiment\n\nConfiguration and model metadata are stored alongside "
            "`metrics.json`. Model artifacts are kept separately under `models/`.\n",
            encoding="utf-8",
        )


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
