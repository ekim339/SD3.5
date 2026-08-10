from __future__ import annotations

import unittest

from networks.scene_text_editing.checkpoints import (
    SD3_CONTROLNET_ALLOW_PATTERNS,
    SD3_PIPELINE_ALLOW_PATTERNS,
    TEXTCTRL_SD15_ALLOW_PATTERNS,
)


class DownloadPatternTests(unittest.TestCase):
    def test_sd3_patterns_select_one_weight_variant(self) -> None:
        joined = "\n".join(SD3_PIPELINE_ALLOW_PATTERNS)
        self.assertIn("transformer/diffusion_pytorch_model.safetensors", joined)
        self.assertNotIn("fp16", joined)
        self.assertNotIn(".bin", joined)

    def test_textctrl_patterns_keep_legacy_bin_files(self) -> None:
        self.assertIn(
            "unet/diffusion_pytorch_model.bin",
            TEXTCTRL_SD15_ALLOW_PATTERNS,
        )
        self.assertIn(
            "vae/diffusion_pytorch_model.bin",
            TEXTCTRL_SD15_ALLOW_PATTERNS,
        )

    def test_controlnet_patterns_skip_demo_images(self) -> None:
        self.assertEqual(
            SD3_CONTROLNET_ALLOW_PATTERNS,
            ["config.json", "diffusion_pytorch_model.safetensors"],
        )


if __name__ == "__main__":
    unittest.main()
