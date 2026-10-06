# Konglish translation model experiments

This is a standalone Python pipeline for conversational Korean ↔ English text translation. It does not modify or replace Android Speech Recognition, and it does not send audio or text to a cloud translation service. The eventual app flow remains Android ASR → recognized text → translation → existing UI/TTS.

## Scope and current data

The project starts with a small, manually curated conversational parallel seed set so the pipeline can be exercised. It is **not** the requested 5,000/500/500 experiment yet: no larger human-parallel corpus was present in the project. Add licensed, reviewed pairs to `data/raw/human_parallel.jsonl` to reach that size. At 6,000 or more unique pairs, the configured splitter creates 5,000 train, 500 validation, and 500 test pairs; with fewer pairs it uses the configured deterministic ratios. It never duplicates pairs across splits.

Each human pair is stored once:

```json
{"ko":"오늘 뭐 해?","en":"What are you doing today?"}
```

`data/raw/konglish_eval.jsonl` is a separate curated application-style evaluation set. The splitter rejects overlap between its source sentences and the human corpus. It is not included in training or synthetic generation. The test split and this Konglish evaluation set are only read by evaluation; teacher generation explicitly rejects monolingual inputs that overlap either held-out set. Add one sentence per line to `data` and `data` before generating synthetic data.

## Environment setup

Python 3.11 is the supported setup. PyTorch wheels and some ML dependencies may not yet support every Python release; use a Python 3.11 virtual environment, especially on the M2 Mac.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The scripts select CUDA, then Apple Silicon MPS, then CPU. Training uses conservative batch size 1, gradient accumulation, and full precision by default. The first model downloads require network access; subsequent inference can use cached/downloaded artifacts offline.

## Data preparation and baseline

Import a deterministic, score-filtered sample from the Hugging Face
[`lemon-mint/korean_english_parallel_wiki_augmented_v1`](https://huggingface.co/datasets/lemon-mint/korean_english_parallel_wiki_augmented_v1)
dataset before preparing splits. Rows require a score of at least `0.85`, at least
three whitespace-separated words per language, and no more than the configured
128-token sequence limit. The original seed corpus is preserved on first import
at `data/raw/human_parallel_seed_backup.jsonl`.

```bash
python scripts/import_hf_dataset.py --limit 10000 --min-score 0.85
python scripts/prepare_data.py
python scripts/evaluate.py
```

Use `--limit 50000` to import a larger sample; sampling uses seed 42.

The split script prints actual split counts. Keep the upstream license and provenance for every added corpus, deduplicate before adding it, and do not put test examples into the monolingual generation files. The baseline evaluation measures `facebook/nllb-200-distilled-600M` separately in both directions, with `kor_Hang` and `eng_Latn` explicitly configured.

## LoRA fine-tuning

```bash
python scripts/train_lora.py
python scripts/evaluate.py
```

The NLLB teacher remains unchanged. Each run creates a unique `experiments` record and a separate adapter at `models`. The adapter trains on both directions from each training pair only. Validation is used during training; the test set is not.

Evaluation writes `results`, `results`, `results`, `results`, `results`, and `results`. Available artifacts are evaluated; absent student/LoRA artifacts are reported as skipped rather than represented as successful results. Metrics include validation and held-out test scores, BLEU, chrF, parameter count, model size, load time, mean/median/p95 translation latency, short/medium/long input buckets when those lengths exist in the data, and a separate `konglish_eval` result. COMET is left null because no COMET checkpoint or dependency is configured; BLEU is not treated as the only quality measure. Use validation results for model decisions; test and Konglish scores are reporting-only.

## Teacher generation and student training

Teacher generation is gated on mean validation chrF improving over the baseline across both directions; this gate never reads test or Konglish metrics. Only after reviewing the LoRA validation comparison, provide monolingual conversational text and generate up to the configured limit per direction:

```bash
python scripts/generate_teacher_data.py
python scripts/train_student.py
python scripts/evaluate.py
```

Teacher outputs are stored separately under a unique `data` directory as `synthetic_ko_en.jsonl` and `synthetic_en_ko.jsonl`, with a matching experiment record identifying them as synthetic. Human data is kept separate and takes priority: synthetic examples are capped at the configured ratio to human training pairs. The held-out test set is never read by student training.

The initial student candidate combines Helsinki-NLP Marian checkpoints: [`opus-mt-ko-en`](https://huggingface.co/Helsinki-NLP/opus-mt-ko-en) (Apache-2.0) and [`opus-mt-tc-big-en-ko`](https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-en-ko) (CC-BY-4.0). The standard `opus-mt-en-ko` checkpoint was not available in the public model catalog, so the 209M-parameter reverse-direction checkpoint is used; combined with the roughly 77M-class Korean-to-English model, this is expected to fit the 100M–300M aggregate target. Preserve CC-BY attribution and review each checkpoint's model/data terms before redistribution. Marian is supported by Transformers and the Optimum ONNX Runtime exporter. These translation-specific models are evaluated as one bidirectional candidate, with per-direction results reported independently; the upstream EN→KO model card reports BLEU 13.7 / chrF 36.4 on FLORES101 devtest, so it must not be presumed adequate (those upstream scores are not directly comparable to this conversational test set). Measured parameter counts, quality, export compatibility, licenses, and file sizes determine whether this candidate proceeds.

## INT8 quantization, export, and benchmark

Do not quantize until the FP32 student has been trained and evaluated. The quantizer exports each Marian model to ONNX and applies ONNX Runtime dynamic INT8 quantization, storing it separately from FP32:

```bash
python scripts/quantize.py
python scripts/evaluate.py
python scripts/benchmark.py
```

`quantize.py` also runs the same test and Konglish evaluations after export and records INT8 measurements. ONNX Runtime is selected as the first mobile route because Marian has an ONNX export path and ONNX Runtime Mobile targets Android. This is an export candidate, **not** proof that the graph is ready for an Android app: validate required operators, memory, cold-start, and latency on a real target device and build an Android-compatible ORT package before integration. No React Native or speech-recognition code is changed here.

`results` gives FP32/INT8 size, parameter counts, separate KO→EN / EN→KO BLEU and chrF, and latency. Student FP32 and INT8 latency default to CPU for a same-runtime student comparison; NLLB/LoRA use the detected device (MPS on supported Apple Silicon). Results record their device, so compare cross-model latency only after rerunning with matching device settings. This is not an Android device benchmark. Use measured quality/latency to decide whether the student is close enough to the NLLB baseline; this repository contains no claim that distillation or INT8 meets that bar before the experiments run.

## Reproducibility and commands

Configuration, seed, model, data description, and training parameters are copied into every run under `experiments`. Model artifacts are stored separately in `models`, `models`, `models`, and `models`; the original teacher is only loaded as a base and never overwritten. Timestamped runs are not overwritten.

```bash
python scripts/prepare_data.py
python scripts/train_lora.py
python scripts/generate_teacher_data.py
python scripts/train_student.py
python scripts/evaluate.py
python scripts/quantize.py
python scripts/benchmark.py
python -m unittest discover -s tests
```

Run evaluation before and after each model stage to populate comparisons as artifacts become available. `python scripts/evaluate.py` prints a compact KO→EN / EN→KO BLEU/chrF table and persists full per-dataset metrics in `results`.
