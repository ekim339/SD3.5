from pathlib import Path

import torch
from PIL import Image
from torch import nn

from .dataset import build_style_prompt, render_glyph
from .model import expand_sd3_input_projection


def test_prompts():
    image = Image.new("RGB", (100, 50), "red")
    mask = Image.new("L", image.size, 0)
    for x in range(20, 61):
        for y in range(10, 31):
            mask.putpixel((x, y), 255)
    style = build_style_prompt(image, mask, (64, 64))
    glyph = render_glyph("Hello", (128, 64))
    assert style.size == (64, 64)
    assert glyph.size == (128, 64) and glyph.getbbox() is not None


def test_projection_preserves_pretrained_slice():
    class Config:
        in_channels = 16
    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.pos_embed = nn.Module()
            self.pos_embed.proj = nn.Conv2d(16, 32, 2, 2)
            self.config = Config()
        def register_to_config(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self.config, key, value)
    transformer = Transformer()
    old = transformer.pos_embed.proj.weight.detach().clone()
    projection = expand_sd3_input_projection(transformer, 49)
    assert projection.in_channels == 65
    assert torch.equal(projection.weight[:, :16], old)
    assert torch.count_nonzero(projection.weight[:, 16:]) == 0
