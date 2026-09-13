from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from CODEX.style_encoder_supercon.dataset import (
    SRNetStyleDataset,
    StyleBatchSampler,
    collate_style_pairs,
)
from CODEX.style_encoder_supercon.loss import SupervisedContrastiveLoss
from CODEX.style_encoder_supercon.model import TextCtrlGlyphStyleEncoder


class SupConLossTests(unittest.TestCase):
    def test_clustered_styles_have_lower_loss_and_gradients(self):
        labels = torch.tensor([0, 0, 1, 1])
        clustered = torch.tensor(
            [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], requires_grad=True
        )
        mixed = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1], [0.1, 0.9]]
        )
        criterion = SupervisedContrastiveLoss(temperature=0.1)
        good_loss = criterion(clustered, labels)
        self.assertLess(float(good_loss), float(criterion(mixed, labels)))
        good_loss.backward()
        self.assertIsNotNone(clustered.grad)
        self.assertTrue(torch.isfinite(clustered.grad).all())

    def test_rejects_anchor_without_positive(self):
        with self.assertRaisesRegex(ValueError, "without positives"):
            SupervisedContrastiveLoss()(torch.randn(3, 4), torch.arange(3))


class DatasetAndSamplerTests(unittest.TestCase):
    def test_batch_has_two_views_per_style_and_hard_negative(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow is unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "i_s").mkdir()
            (root / "t_f").mkdir()
            sources = ["HELLO", "HELLO", "FOO", "BAZ"]
            targets = ["WORLD", "THERE", "BAR", "QUX"]
            fonts = ["Arial", "Times", "Mono", "Serif"]
            for index in range(4):
                name = "{}.png".format(index)
                Image.new("RGB", (20, 12), (index * 20, 0, 0)).save(root / "i_s" / name)
                Image.new("RGB", (20, 12), (0, index * 20, 0)).save(root / "t_f" / name)
            (root / "i_s.txt").write_text(
                "".join("{}.png {}\n".format(i, value) for i, value in enumerate(sources)),
                encoding="utf-8",
            )
            (root / "i_t.txt").write_text(
                "".join("{}.png {}\n".format(i, value) for i, value in enumerate(targets)),
                encoding="utf-8",
            )
            (root / "font.txt").write_text(
                "".join("{}.png {}\n".format(i, value) for i, value in enumerate(fonts)),
                encoding="utf-8",
            )
            dataset = SRNetStyleDataset([root], image_size=16)
            sampler = StyleBatchSampler(
                dataset, range(4), styles_per_batch=4,
                hard_negative_probability=1.0, shuffle=False,
            )
            indices = next(iter(sampler))
            self.assertIn(0, indices)
            self.assertIn(1, indices)
            batch = collate_style_pairs([dataset[index] for index in indices])
            self.assertEqual(tuple(batch["images"].shape), (8, 3, 16, 16))
            self.assertEqual(torch.unique(batch["style_labels"], return_counts=True)[1].tolist(),
                             [2, 2, 2, 2])


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 4)

    def forward(self, images, mask=None):
        pooled = images.mean(dim=(2, 3))
        return self.projection(pooled).unsqueeze(1).expand(-1, 2, -1)


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(4, 4)

    def forward(self, values):
        return self.projection(values)


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = TinyBackbone()
        self.spatial_attn_block = nn.ModuleList([TinyBlock()])
        self.glyph_attn_block = nn.ModuleList([TinyBlock()])


class FreezingTests(unittest.TestCase):
    def test_only_glyph_branch_updates_and_spatial_is_unused(self):
        model = TextCtrlGlyphStyleEncoder.__new__(TextCtrlGlyphStyleEncoder)
        nn.Module.__init__(model)
        model.encoder = TinyEncoder()
        model.pooling = "mean"
        model.freeze_non_glyph_parameters()
        before_backbone = [value.clone() for value in model.backbone.state_dict().values()]
        before_spatial = [value.clone() for value in model.spatial_head.state_dict().values()]
        optimizer = torch.optim.SGD(model.trainable_parameters(), lr=0.1)
        labels = torch.tensor([0, 0, 1, 1])
        loss = SupervisedContrastiveLoss()(model(torch.randn(4, 3, 8, 8)), labels)
        loss.backward()
        optimizer.step()
        self.assertTrue(all(parameter.grad is None for parameter in model.backbone.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in model.spatial_head.parameters()))
        self.assertTrue(any(parameter.grad is not None for parameter in model.glyph_head.parameters()))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(
            before_backbone, model.backbone.state_dict().values())))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(
            before_spatial, model.spatial_head.state_dict().values())))


if __name__ == "__main__":
    unittest.main()
