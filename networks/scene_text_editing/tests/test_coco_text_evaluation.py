import random
import unittest

from networks.scene_text_editing.evaluate_coco_text import (
    choose_target_text,
    expanded_crop_box,
    levenshtein_distance,
    text_metrics,
)


class CocoTextEvaluationTests(unittest.TestCase):
    def test_levenshtein_distance(self):
        self.assertEqual(levenshtein_distance("kitten", "sitting"), 3)
        self.assertEqual(levenshtein_distance("", "abc"), 3)

    def test_metrics_are_case_insensitive_for_charset_36_ocr(self):
        self.assertEqual(
            text_metrics("HELLO", "hello"),
            {"ACC": 1, "NED": 1.0, "CER": 0.0},
        )

    def test_metrics_use_standard_normalizations(self):
        metrics = text_metrics("abcde", "abxdey")
        self.assertEqual(metrics["ACC"], 0)
        self.assertAlmostEqual(metrics["NED"], 1.0 - 2.0 / 6.0)
        self.assertAlmostEqual(metrics["CER"], 2.0 / 5.0)

    def test_crop_box_expands_and_clips(self):
        self.assertEqual(
            expanded_crop_box([2, 3, 20, 10], (100, 80), 0.25, 4),
            (0, 0, 27, 17),
        )

    def test_target_is_five_characters_and_differs_from_source(self):
        target = choose_target_text(
            "Photo",
            ["Photo", "PHOTO", "hello", "world"],
            random.Random(42),
        )
        self.assertIn(target, {"hello", "world"})
        self.assertNotEqual(target.casefold(), "photo")


if __name__ == "__main__":
    unittest.main()
