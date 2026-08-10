from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from networks.scene_text_editing.networks.factory import (
    BackendError,
    SD3ControlNetInpaintBackend,
    TextCtrlSubprocessBackend,
)


class NetworkUtilityTests(unittest.TestCase):
    def test_control_images_are_padded_and_masked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.png"
            Image.new("RGB", (64, 32), color="white").save(source)
            control, mask, box, original_size = (
                SD3ControlNetInpaintBackend._prepare_control_images(
                    source,
                    None,
                    width=128,
                    height=128,
                    full_mask_when_missing=True,
                )
            )
            self.assertEqual(control.size, (128, 128))
            self.assertEqual(mask.size, (128, 128))
            self.assertEqual(box, (0, 32, 128, 96))
            self.assertEqual(original_size, (64, 32))
            self.assertEqual(mask.getpixel((64, 64)), 255)
            self.assertEqual(mask.getpixel((64, 10)), 0)

    def test_legacy_textctrl_rejects_multiword_labels(self) -> None:
        with self.assertRaisesRegex(BackendError, "single-token"):
            TextCtrlSubprocessBackend._validate_label(
                "two words",
                "target_text",
                "sample",
            )


if __name__ == "__main__":
    unittest.main()
