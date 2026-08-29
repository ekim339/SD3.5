import tempfile

import torch
from diffusers import SD3Transformer2DModel
from peft.utils import get_peft_model_state_dict
from PIL import Image
from torch import nn

from .dataset import build_style_prompt, render_glyph
from .model import (
    add_sd3_lora_adapter,
    expand_sd3_input_projection,
    load_sd3_lora_weights,
    save_sd3_lora_weights,
)


def tiny_sd3_transformer() -> SD3Transformer2DModel:
    return SD3Transformer2DModel(
        sample_size=8,
        patch_size=2,
        in_channels=4,
        num_layers=2,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=32,
        caption_projection_dim=16,
        pooled_projection_dim=32,
        out_channels=4,
        pos_embed_max_size=8,
    )


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


def test_lora_freezes_base_and_covers_condition_projection():
    transformer = tiny_sd3_transformer()
    expand_sd3_input_projection(transformer, 13)
    add_sd3_lora_adapter(
        transformer,
        rank=2,
        alpha=2,
        target_modules=("pos_embed.proj", "attn.to_q"),
    )

    trainable_names = [name for name, parameter in transformer.named_parameters() if parameter.requires_grad]
    assert trainable_names
    assert all(".lora_" in name for name in trainable_names)
    assert any("pos_embed.proj.lora_A" in name for name in trainable_names)
    assert not transformer.pos_embed.proj.base_layer.weight.requires_grad

    adapter_state = get_peft_model_state_dict(transformer)
    assert any(key.startswith("pos_embed.proj.lora_") for key in adapter_state)
    assert all("lora_" in key for key in adapter_state)


def test_condition_projection_lora_receives_gradient():
    torch.manual_seed(0)
    transformer = tiny_sd3_transformer()
    expand_sd3_input_projection(transformer, 13)
    add_sd3_lora_adapter(
        transformer,
        rank=2,
        alpha=2,
        target_modules=("pos_embed.proj", "attn.to_q"),
    )

    hidden = torch.randn(1, 17, 8, 8)
    hidden[:, :4].zero_()
    prediction = transformer(
        hidden_states=hidden,
        encoder_hidden_states=torch.randn(1, 3, 32),
        pooled_projections=torch.randn(1, 32),
        timestep=torch.tensor([1]),
    ).sample
    prediction.square().mean().backward()

    projection_b = next(
        parameter
        for name, parameter in transformer.named_parameters()
        if name == "pos_embed.proj.lora_B.default.weight"
    )
    assert projection_b.grad is not None
    assert torch.count_nonzero(projection_b.grad) > 0


def test_lora_checkpoint_round_trip():
    source = tiny_sd3_transformer()
    expand_sd3_input_projection(source, 13)
    add_sd3_lora_adapter(
        source,
        rank=2,
        alpha=2,
        target_modules=("pos_embed.proj", "attn.to_q"),
    )
    with torch.no_grad():
        for name, parameter in source.named_parameters():
            if ".lora_B." in name:
                parameter.fill_(0.125)

    target = tiny_sd3_transformer()
    expand_sd3_input_projection(target, 13)
    add_sd3_lora_adapter(
        target,
        rank=2,
        alpha=2,
        target_modules=("pos_embed.proj", "attn.to_q"),
    )

    with tempfile.TemporaryDirectory() as directory:
        save_sd3_lora_weights(source, directory)
        load_sd3_lora_weights(target, directory)

    expected = get_peft_model_state_dict(source)
    actual = get_peft_model_state_dict(target)
    assert expected.keys() == actual.keys()
    assert all(torch.equal(expected[key], actual[key]) for key in expected)
