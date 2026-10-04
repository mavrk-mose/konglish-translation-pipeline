from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common import (
    ROOT,
    latest_directional_run,
    load_config,
    model_directory_size,
    new_experiment_dir,
    write_experiment_metadata,
)
from scripts.evaluate import evaluate_run


def quantize_student(source: Path, destination: Path, provider: str) -> tuple[int, int]:
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from optimum.onnxruntime import ORTModelForSeq2SeqLM
    from transformers import AutoTokenizer

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}_export_") as temporary:
        export_dir = Path(temporary)
        model = ORTModelForSeq2SeqLM.from_pretrained(
            source,
            export=True,
            provider=provider,
        )
        model.save_pretrained(export_dir)
        AutoTokenizer.from_pretrained(source).save_pretrained(export_dir)
        size_before = model_directory_size(export_dir)
        for item in export_dir.iterdir():
            if item.is_dir():
                shutil.copytree(item, destination / item.name)
            elif item.suffix != ".onnx":
                shutil.copy2(item, destination / item.name)
    onnx_files = sorted(destination.rglob("*.onnx"))
    if not onnx_files:
        raise RuntimeError(f"ONNX export produced no graph files in {export_dir}")
    for model_file in onnx_files:
        quantized_file = model_file.with_name(model_file.stem + ".int8.tmp.onnx")
        quantize_dynamic(
            str(model_file),
            str(quantized_file),
            weight_type=QuantType.QInt8,
            per_channel=True,
        )
        quantized_file.replace(model_file)
    return size_before, model_directory_size(destination)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export student models to ONNX and apply ONNX Runtime dynamic INT8 quantization."
    )
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    if not config["quantization"]["enabled"] or config["quantization"]["type"].lower() != "int8":
        raise ValueError("Enable quantization.type=int8 in config.yaml before running this script.")

    student_run = latest_directional_run("models/student")
    if student_run is None:
        raise FileNotFoundError(
            "No complete bidirectional FP32 student run found; train both directions first."
        )
    experiment = new_experiment_dir("student_int8")
    artifact_root = ROOT / "models" / "student_int8" / experiment.name
    artifact_root.mkdir(parents=True, exist_ok=False)
    summary = {}
    for direction in config["evaluation"]["directions"]:
        source = student_run / direction
        if not source.is_dir():
            raise FileNotFoundError(
                f"No trained FP32 student found for {direction}; train the student first."
            )
        destination = artifact_root / direction
        fp32_bytes, int8_bytes = quantize_student(
            source,
            destination,
            config["quantization"]["provider"],
        )
        summary[direction] = {
            "source_model": str(source.relative_to(ROOT)),
            "quantized_model": str(destination.relative_to(ROOT)),
            "onnx_fp32_size_mb": round(fp32_bytes / (1024 * 1024), 3),
            "onnx_int8_size_mb": round(int8_bytes / (1024 * 1024), 3),
            "mobile_runtime": "ONNX Runtime Mobile candidate; Android validation required",
        }
        print(
            f"{direction}: ONNX FP32 {summary[direction]['onnx_fp32_size_mb']} MB -> "
            f"INT8 {summary[direction]['onnx_int8_size_mb']} MB"
        )

    write_experiment_metadata(
        experiment,
        config_path,
        config["training"]["seed"],
        "Marian student, ONNX Runtime dynamic INT8",
        "student FP32 artifact; test set not used for quantization",
        summary,
    )
    evaluation = evaluate_run(config, config_path=config_path)
    summary["evaluation"] = [
        row
        for row in evaluation
        if row["model_type"] == "student_int8" and row["dataset"] == "test"
    ]
    (experiment / "metrics.json").write_text(
        json.dumps({"quantization": summary}, indent=2) + "\n",
        encoding="utf-8",
    )
    print("INT8 quality and latency comparison recorded against NLLB-600M.")


if __name__ == "__main__":
    main()
