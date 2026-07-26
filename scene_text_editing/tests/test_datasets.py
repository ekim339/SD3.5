from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scene_text_editing.datasets import DatasetError, load_dataset


class DatasetTests(unittest.TestCase):
    def test_jsonl_loader_resolves_paths_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "images").mkdir()
            (root / "images/source.png").write_bytes(b"fixture")
            record = {
                "id": "one",
                "source_image": "images/source.png",
                "source_text": "Old words",
                "target_text": "New words",
                "scene": "storefront",
            }
            (root / "manifest.jsonl").write_text(
                json.dumps(record) + "\n",
                encoding="utf-8",
            )
            dataset = load_dataset(
                {
                    "format": "jsonl",
                    "root_dir": str(root),
                    "manifest": "manifest.jsonl",
                    "strict": True,
                }
            )
            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset[0].source_text, "Old words")
            self.assertEqual(dataset[0].metadata["scene"], "storefront")
            self.assertTrue(dataset[0].source_image.is_absolute())

    def test_textctrl_loader_accepts_multiword_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "i_s").mkdir()
            (root / "i_s/example.png").write_bytes(b"fixture")
            (root / "i_s.txt").write_text(
                "example.png source words\n",
                encoding="utf-8",
            )
            (root / "i_t.txt").write_text(
                "example.png replacement phrase\n",
                encoding="utf-8",
            )
            dataset = load_dataset(
                {
                    "format": "textctrl",
                    "root_dir": str(root),
                    "source_dir": "i_s",
                    "source_labels": "i_s.txt",
                    "target_labels": "i_t.txt",
                    "strict": True,
                }
            )
            self.assertEqual(dataset[0].source_text, "source words")
            self.assertEqual(dataset[0].target_text, "replacement phrase")

    def test_textctrl_loader_aligns_by_filename_not_line_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "i_s").mkdir()
            for filename in ("a.png", "b.png"):
                (root / "i_s" / filename).write_bytes(b"fixture")
            (root / "i_s.txt").write_text(
                "a.png alpha\nb.png beta\n",
                encoding="utf-8",
            )
            (root / "i_t.txt").write_text(
                "b.png second\na.png first\n",
                encoding="utf-8",
            )
            dataset = load_dataset(
                {
                    "format": "textctrl",
                    "root_dir": str(root),
                    "strict": True,
                }
            )
            by_name = {sample.sample_id: sample for sample in dataset}
            self.assertEqual(by_name["a.png"].target_text, "first")
            self.assertEqual(by_name["b.png"].target_text, "second")

    def test_textctrl_loader_rejects_misaligned_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "i_s").mkdir()
            (root / "i_s.txt").write_text("a.png alpha\n", encoding="utf-8")
            (root / "i_t.txt").write_text("b.png beta\n", encoding="utf-8")
            with self.assertRaisesRegex(DatasetError, "not aligned"):
                load_dataset(
                    {
                        "format": "textctrl",
                        "root_dir": str(root),
                        "strict": True,
                    }
                )

    def test_duplicate_jsonl_ids_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.png"
            source.write_bytes(b"fixture")
            record = {
                "id": "duplicate",
                "source_image": "source.png",
                "source_text": "A",
                "target_text": "B",
            }
            (root / "manifest.jsonl").write_text(
                json.dumps(record) + "\n" + json.dumps(record) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(DatasetError, "Duplicate sample id"):
                load_dataset(
                    {
                        "format": "jsonl",
                        "root_dir": str(root),
                        "manifest": "manifest.jsonl",
                        "strict": True,
                    }
                )


if __name__ == "__main__":
    unittest.main()
