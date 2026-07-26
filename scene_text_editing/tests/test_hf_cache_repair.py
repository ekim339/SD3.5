from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scene_text_editing.checkpoints import _quarantine_invalid_hub_tree_cache


class HubTreeCacheRepairTests(unittest.TestCase):
    def test_redacted_hashes_are_quarantined(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            trees = cache / "models--owner--gated-model" / "trees"
            trees.mkdir(parents=True)
            tree = trees / f"{'a' * 40}.json"
            tree.write_text(
                json.dumps(
                    {
                        "format_version": 1,
                        "files": {
                            "model.safetensors": {
                                "size": 123,
                                "blob_id": "b" * 40,
                                "lfs_sha256": "*" * 64,
                                "lfs_size": 123,
                                "xet_hash": "*" * 64,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            quarantined = _quarantine_invalid_hub_tree_cache(
                "owner/gated-model",
                cache,
            )

            self.assertFalse(tree.exists())
            self.assertEqual(len(quarantined), 1)
            self.assertTrue(quarantined[0].is_file())
            self.assertIn(".json.invalid", quarantined[0].name)

    def test_valid_hashes_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            trees = cache / "models--owner--model" / "trees"
            trees.mkdir(parents=True)
            tree = trees / f"{'a' * 40}.json"
            tree.write_text(
                json.dumps(
                    {
                        "format_version": 1,
                        "files": {
                            "model.safetensors": {
                                "size": 123,
                                "blob_id": "b" * 40,
                                "lfs_sha256": "c" * 64,
                                "lfs_size": 123,
                                "xet_hash": "d" * 64,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            quarantined = _quarantine_invalid_hub_tree_cache(
                "owner/model",
                cache,
            )

            self.assertEqual(quarantined, [])
            self.assertTrue(tree.is_file())


if __name__ == "__main__":
    unittest.main()
