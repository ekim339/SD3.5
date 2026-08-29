import torch
from PIL import Image
from torch import nn

from .dataset import build_style_prompt, prepare_conditions, render_glyph
from .model import expand_sd3_input_projection


def test_training_uses_source_reconstruction():
    source = Image.new("RGB", (100, 50), "red")
    mask = Image.new("L", source.size, 0)
    for x in range(20, 60):
        for y in range(10, 30):
            mask.putpixel((x, y), 255)
    sample = prepare_conditions(source, mask, "source", 64)
    assert torch.equal(sample["source_image"], sample["target_image"])
    assert sample["masked_image"][:, 20:40, 20:40].min() == -1
    assert build_style_prompt(source, mask, (64, 64)).size == (64, 64)
    assert render_glyph("target", (128, 64)).getbbox() is not None


def test_projection_is_65_channels_and_preserves_base():
    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.pos_embed = nn.Module()
            self.pos_embed.proj = nn.Conv2d(16, 32, 2, 2)
            self.config = type("Config", (), {"in_channels": 16})()
        def register_to_config(self, **values):
            for key, value in values.items():
                setattr(self.config, key, value)
    transformer = Transformer()
    original = transformer.pos_embed.proj.weight.detach().clone()
    projection = expand_sd3_input_projection(transformer, 49)
    assert projection.in_channels == 65
    assert torch.equal(projection.weight[:, :16], original)
    assert torch.count_nonzero(projection.weight[:, 16:]) == 0
