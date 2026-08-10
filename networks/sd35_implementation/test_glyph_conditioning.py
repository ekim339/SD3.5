from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from .adapters import AuxiliaryConditionProjector, SD3InpaintingPatchProjection
from .dataset import SRNetEditingDataset
from .encoders import FrozenTextCtrlGlyphEncoder
from .model import masked_flow_matching_loss


ROOT = Path(__file__).resolve().parent.parent.parent


class GlyphConditioningTests(unittest.TestCase):
    def test_projected_condition_has_dual_prompt_tokens(self) -> None:
        projector = AuxiliaryConditionProjector()
        editing_prompt = torch.randn(2, 333, 4096)
        target_text = torch.randn(2, 256, 4096)
        result = projector(
            editing_prompt,
            target_text,
            torch.randn(2, 24, 768),
            torch.randn(2, 256, 768),
        )
        self.assertEqual(result.shape, (2, 869, 4096))
        torch.testing.assert_close(result[:, 333:589], target_text)
        self.assertGreater(torch.count_nonzero(result[:, 589:613]), 0)
        self.assertGreater(torch.count_nonzero(result[:, 613:]), 0)
        self.assertAlmostEqual(projector.glyph_gate.item(), 0.1)
        self.assertAlmostEqual(projector.style_gate.item(), 0.1)

    def test_inpainting_projection_preserves_pretrained_path_at_initialization(self) -> None:
        base = torch.nn.Conv2d(4, 8, kernel_size=2, stride=2)
        projection = SD3InpaintingPatchProjection(base, condition_channels=5)
        noisy = torch.randn(2, 4, 8, 8)
        condition = torch.randn(2, 5, 8, 8)
        expected = base(noisy)
        actual = projection(torch.cat((noisy, condition), dim=1))
        torch.testing.assert_close(actual, expected)
        self.assertFalse(
            any(parameter.requires_grad for parameter in projection.base.parameters())
        )
        self.assertTrue(projection.conditioning.weight.requires_grad)


    def test_mask_weighted_loss_emphasizes_edit_region(self) -> None:
        prediction = torch.zeros(1, 1, 2, 2)
        target = torch.zeros_like(prediction)
        target[..., 0, 0] = 1.0
        mask = torch.zeros(1, 1, 2, 2)
        mask[..., 0, 0] = 1.0
        weighted = masked_flow_matching_loss(prediction, target, mask, 5.0, 1.0)
        unweighted = masked_flow_matching_loss(prediction, target, mask, 1.0, 1.0)
        self.assertGreater(weighted.item(), unweighted.item())
        self.assertAlmostEqual(weighted.item(), 5.0 / 6.0)

    def test_dataset_does_not_require_rendered_target_glyphs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for directory in ("i_s", "mask_s", "t_f"):
                (root / directory).mkdir()
            (root / "i_s.txt").write_text("000.png OPEN\n", encoding="utf-8")
            (root / "i_t.txt").write_text("000.png CLOSED\n", encoding="utf-8")
            Image.new("RGB", (32, 16), "white").save(root / "i_s/000.png")
            Image.new("L", (32, 16), 255).save(root / "mask_s/000.png")
            Image.new("RGB", (32, 16), "white").save(root / "t_f/000.png")
            sample = SRNetEditingDataset(root, resolution=64, style_resolution=32)[0]
            self.assertEqual(sample["target_text"], "CLOSED")
            self.assertNotIn("target_glyph", sample)
            source_foreground = torch.nonzero(sample["source_image"][0] > 0)
            mask_foreground = torch.nonzero(sample["source_mask"][0] > 0)
            self.assertEqual(tuple(source_foreground.min(dim=0).values), (16, 0))
            self.assertEqual(tuple(source_foreground.max(dim=0).values), (47, 63))
            torch.testing.assert_close(source_foreground, mask_foreground)
            style_foreground = torch.nonzero(sample["style_image"][0] > 0)
            self.assertEqual(tuple(style_foreground.min(dim=0).values), (8, 0))
            self.assertEqual(tuple(style_foreground.max(dim=0).values), (23, 31))

    def test_real_textctrl_glyph_checkpoint_shape_and_freeze(self) -> None:
        repository = ROOT / "networks/external/TextCtrl"
        encoder = FrozenTextCtrlGlyphEncoder(
            repository, repository / "weights/text_encoder.pth"
        )
        features = encoder(["CLOSED", "OPEN"])
        self.assertEqual(features.shape, (2, 24, 768))
        self.assertFalse(any(parameter.requires_grad for parameter in encoder.parameters()))


if __name__ == "__main__":
    unittest.main()
