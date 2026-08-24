import hashlib
import unittest
from pathlib import Path

import numpy as np
import torch

from networks.sd15_residual_extractor_finetune.data import (
    NeutralGlyphRenderer,
    SRNetResidualEditingDataset,
)


class ComponentTests(unittest.TestCase):
    def test_renderer_matches_residual_training_implementation(self):
        font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
        actual = np.asarray(NeutralGlyphRenderer(font, (256, 256))("Hello"))
        digest = hashlib.sha256(actual.tobytes()).hexdigest()
        # Generated once from encoders.dataset.NeutralGlyphRenderer.
        self.assertEqual(digest, "6214d22e1c4ac07eb68f45d161a0448a23471ba91b55aa664e11d2bc02653a6a")

    def test_dataset_contract(self):
        from PIL import Image
        import tempfile
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, target = root / "source.png", root / "target.png"
            Image.new("RGB", (64, 32), "red").save(source)
            Image.new("RGB", (64, 32), "blue").save(target)
            dataset = SRNetResidualEditingDataset(
                [(source, target, "OLD", "NEW")],
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                condition_dropout=0.0,
            )
            sample = dataset[0]
            for key in ("img", "hint", "source_residual",
                        "source_glyph_residual", "target_glyph"):
                self.assertEqual(tuple(sample[key].shape), (3, 256, 256))
            self.assertGreaterEqual(float(sample["img"].min()), 0.0)
            self.assertLessEqual(float(sample["img"].max()), 1.0)
            self.assertGreaterEqual(float(sample["source_residual"].min()), -1.0)
            self.assertEqual(sample["cond"], "NEW")


if __name__ == "__main__":
    unittest.main()
