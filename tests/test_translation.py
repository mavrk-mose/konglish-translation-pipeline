import tempfile
import unittest
from pathlib import Path

from scripts.common import (
    assert_eval_disjoint,
    assert_no_test_sources,
    directional_rows,
    load_config,
    project_path,
    read_jsonl,
    require_lora_improvement,
)
from scripts.import_hf_dataset import (
    filter_by_token_length,
    filter_candidate_rows,
    select_rows,
    write_pairs_with_backup,
)
from scripts.prepare_data import split_pairs, write_split


class HuggingFaceImporterTests(unittest.TestCase):
    class WhitespaceTokenizer:
        src_lang = None

        def __call__(self, texts, **kwargs):
            return {"input_ids": [[0, *text.split(), 1] for text in texts]}

    def test_filters_invalid_short_low_score_and_duplicate_rows(self):
        rows = [
            {"korean": None, "english": "three English words", "score": 0.9},
            {"korean": "짧음", "english": "three English words", "score": 0.9},
            {"korean": "세 개의 한국어", "english": "two words", "score": 0.9},
            {"korean": "세 개의 한국어", "english": "three English words", "score": 0.84},
            {"korean": "세 개의 한국어", "english": "three English words", "score": 0.9},
            {"korean": "세 개의 한국어", "english": "three English words", "score": 0.95},
        ]

        filtered, duplicates_removed = filter_candidate_rows(rows, min_score=0.85)

        self.assertEqual(duplicates_removed, 1)
        self.assertEqual(filtered, [{
            "ko": "세 개의 한국어",
            "en": "three English words",
            "score": 0.9,
        }])

    def test_filters_text_exceeding_model_token_limit(self):
        rows = [
            {"ko": "하나 둘 셋", "en": "one two three", "score": 0.9},
            {"ko": "하나 둘 셋 넷 다섯", "en": "one two three", "score": 0.9},
        ]

        filtered = filter_by_token_length(rows, self.WhitespaceTokenizer(), max_tokens=5)

        self.assertEqual(filtered, rows[:1])

    def test_sampling_is_seeded_and_respects_limit(self):
        rows = [{"ko": str(index), "en": str(index), "score": 0.9} for index in range(20)]

        first = select_rows(rows, limit=5, seed=42)
        second = select_rows(rows, limit=5, seed=42)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)

    def test_existing_source_is_backed_up_before_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "human_parallel.jsonl"
            backup = Path(directory) / "human_parallel_seed_backup.jsonl"
            output.write_text('{"ko":"seed","en":"pair"}\n', encoding="utf-8")

            write_pairs_with_backup(
                output,
                backup,
                [{"ko": "새 문장 세 개", "en": "three new words", "score": 0.9}],
            )

            self.assertEqual(backup.read_text(encoding="utf-8"), '{"ko":"seed","en":"pair"}\n')
            self.assertEqual(
                output.read_text(encoding="utf-8"),
                '{"ko": "새 문장 세 개", "en": "three new words"}\n',
            )


class TranslationPipelineTests(unittest.TestCase):
    def test_split_is_deterministic_disjoint_and_pair_level(self):
        rows = [{"ko": f"한국어 문장 {index}", "en": f"English sentence {index}"} for index in range(20)]
        first = split_pairs(rows, 42, (0.8, 0.1, 0.1))
        second = split_pairs(rows, 42, (0.8, 0.1, 0.1))
        self.assertEqual(first, second)
        pair_sets = [{(row["ko"], row["en"]) for row in group} for group in first]
        self.assertFalse(pair_sets[0] & pair_sets[1])
        self.assertFalse(pair_sets[0] & pair_sets[2])
        self.assertFalse(pair_sets[1] & pair_sets[2])
        self.assertEqual(sum(map(len, first)), 20)

    def test_requested_pair_counts_are_used_when_available(self):
        rows = [{"ko": f"가 {index}", "en": f"b {index}"} for index in range(12)]
        parts = split_pairs(rows, 42, (0.8, 0.1, 0.1), (8, 2, 2))
        self.assertEqual([len(part) for part in parts], [8, 2, 2])

    def test_configured_experiment_split_is_exact_at_target_size(self):
        rows = [
            {"ko": f"한국어 {index}", "en": f"English {index}"}
            for index in range(6000)
        ]
        parts = split_pairs(rows, 42, (0.8, 0.1, 0.1), (5000, 500, 500))
        self.assertEqual([len(part) for part in parts], [5000, 500, 500])

    def test_each_human_pair_becomes_both_translation_directions(self):
        pairs = [{"ko": "밥 먹었어?", "en": "Have you eaten?"}]
        self.assertEqual(
            directional_rows(pairs),
            [
                {"source": "밥 먹었어?", "target": "Have you eaten?", "direction": "ko_en"},
                {"source": "Have you eaten?", "target": "밥 먹었어?", "direction": "en_ko"},
            ],
        )

    def test_split_jsonl_round_trips_quotes_and_unicode(self):
        rows = [{"ko": "진짜야?", "en": "I'm on my way."}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.jsonl"
            write_split(path, rows)
            self.assertEqual(load_config()["data"]["train_pairs"], 5000)
            self.assertIn('"en": "I\'m on my way."', path.read_text(encoding="utf-8"))

    def test_synthetic_generation_rejects_test_sources(self):
        test = [{"ko": "오늘 뭐 해?", "en": "What are you doing today?"}]
        with self.assertRaises(ValueError):
            assert_no_test_sources(["오늘 뭐 해?"], [], test)
        with self.assertRaises(ValueError):
            assert_no_test_sources([], ["what are you doing today?"], test)

    def test_teacher_generation_requires_validation_improvement(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comparison.json"
            path.write_text(
                '[{"dataset":"validation","model_type":"baseline","direction":"ko_en","chrf":40},'
                '{"dataset":"validation","model_type":"baseline","direction":"en_ko","chrf":42},'
                '{"dataset":"validation","model_type":"lora","direction":"ko_en","chrf":41},'
                '{"dataset":"validation","model_type":"lora","direction":"en_ko","chrf":43}]',
                encoding="utf-8",
            )
            require_lora_improvement(path)
            path.write_text(
                '[{"dataset":"test","model_type":"baseline","direction":"ko_en","chrf":10}]',
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                require_lora_improvement(path)

    def test_curated_evaluation_is_disjoint_from_training_corpus(self):
        config = load_config()
        human = read_jsonl(project_path(config["data"]["human_pairs"]))
        evaluation = read_jsonl(project_path(config["data"]["konglish_eval"]))
        assert_eval_disjoint(human, evaluation)


if __name__ == "__main__":
    unittest.main()
