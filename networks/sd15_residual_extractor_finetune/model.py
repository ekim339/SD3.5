"""Residual-style and rendered-glyph spatial controls for TextCtrl SD1.5."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
import torch.nn as nn

from src.trainer.CtrlBase import (
    ControlBase,
    ControlUNetModel,
    UNet2DConditionOutput,
)


def _zero(module):
    for parameter in module.parameters():
        nn.init.zeros_(parameter)
    return module


class ResidualChannelAdapter(nn.Module):
    def __init__(self, input_channels=256, output_channels=768):
        super().__init__()
        self.projection = nn.Conv2d(input_channels, output_channels, 1)

    def forward(self, value):
        if tuple(value.shape[1:]) != (256, 16, 16):
            raise RuntimeError("Expected residual [B,256,16,16], got %s" % (tuple(value.shape),))
        return self.projection(value)


class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1):
        super().__init__()
        groups = min(32, output_channels)
        while output_channels % groups:
            groups -= 1
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, stride=stride, padding=1),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(),
        )

    def forward(self, value):
        return self.block(value)


class GlyphSpatialAdapter(nn.Module):
    """CNN target-glyph branch with 13 zero-initialized ControlNet outputs."""
    output_channels = (320, 320, 320, 320, 640, 640, 640,
                       1280, 1280, 1280, 1280, 1280, 1280)

    def __init__(self, widths=(32, 64, 128, 256, 384, 512)):
        super().__init__()
        self.to_128 = ConvBlock(3, widths[0], stride=2)
        self.to_64 = ConvBlock(widths[0], widths[1], stride=2)
        self.to_32 = ConvBlock(widths[1], widths[2], stride=2)
        self.to_16 = ConvBlock(widths[2], widths[3], stride=2)
        self.to_8 = ConvBlock(widths[3], widths[4], stride=2)
        self.to_4 = ConvBlock(widths[4], widths[5], stride=2)
        source_widths = (widths[2],) * 3 + (widths[3],) * 3 + \
                        (widths[4],) * 3 + (widths[5],) * 4
        self.outputs = nn.ModuleList([
            _zero(nn.Conv2d(source, target, 1))
            for source, target in zip(source_widths, self.output_channels)
        ])

    def forward(self, glyph):
        if tuple(glyph.shape[1:]) != (3, 256, 256):
            raise RuntimeError("Expected target glyph [B,3,256,256], got %s" % (tuple(glyph.shape),))
        value = self.to_128(glyph)
        value = self.to_64(value)
        feature32 = self.to_32(value)
        feature16 = self.to_16(feature32)
        feature8 = self.to_8(feature16)
        feature4 = self.to_4(feature8)
        sources = [feature32] * 3 + [feature16] * 3 + [feature8] * 3 + [feature4] * 4
        return [projection(feature) for projection, feature in zip(self.outputs, sources)]


class FullyTrainableControlUNet(ControlUNetModel):
    """TextCtrl U-Net without the upstream no-grad wrapper around its down path."""

    def forward(self, x, timestep=None, encoder_hidden_states=None, control=None,
                return_dict=True, **kwargs):
        overall = 2 ** self.num_upsamplers
        forward_upsample_size = any(size % overall != 0 for size in x.shape[-2:])
        upsample_size = None
        if self.config.center_input_sample:
            x = 2 * x - 1.0
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=x.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(x.device)
        timesteps = timesteps.expand(x.shape[0])
        embedding = self.time_embedding(self.time_proj(timesteps).to(dtype=self.dtype))
        x = self.conv_in(x)
        skips = (x,)
        for block in self.down_blocks:
            if hasattr(block, "attentions") and block.attentions is not None:
                x, residuals = block(hidden_states=x, temb=embedding,
                                     encoder_hidden_states=encoder_hidden_states)
            else:
                x, residuals = block(hidden_states=x, temb=embedding)
            skips += residuals
        x = self.mid_block(x, embedding, encoder_hidden_states)
        controls = list(control)
        x = x + controls.pop()
        for index, block in enumerate(self.up_blocks):
            final = index == len(self.up_blocks) - 1
            residuals = skips[-len(block.resnets):]
            skips = skips[:-len(block.resnets)]
            added = tuple(item + controls.pop() for item in residuals[::-1])[::-1]
            if not final and forward_upsample_size:
                upsample_size = skips[-1].shape[2:]
            arguments = dict(hidden_states=x, temb=embedding,
                             res_hidden_states_tuple=added, upsample_size=upsample_size)
            if hasattr(block, "attentions") and block.attentions is not None:
                arguments["encoder_hidden_states"] = encoder_hidden_states
            x = block(**arguments)
        x = self.conv_out(self.conv_act(self.conv_norm_out(x)))
        if not return_dict:
            return (x,)
        return UNet2DConditionOutput(sample=x)


def _load_residual_extractor(checkpoint):
    model_path = Path(__file__).resolve().parents[2] / "encoders" / "model.py"
    spec = importlib.util.spec_from_file_location("sd15_residual_style_model", model_path)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load residual extractor model from %s" % model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    payload = torch.load(checkpoint, map_location="cpu")
    model = module.ResidualStyleAutoencoder(**payload.get("config", {}).get("model", {}))
    model.load_state_dict(payload["model"], strict=True)
    model.eval().requires_grad_(False)
    return model


class ResidualGlyphControlTrainer(ControlBase):
    """TextCtrl trainer with independent residual-style and target-glyph controls."""

    def __init__(self, control_config, base_config, residual_checkpoint,
                 adapter_checkpoint=None, glyph_widths=(32, 64, 128, 256, 384, 512),
                 style_scale=1.0, glyph_scale=1.0):
        super().__init__(control_config, base_config)
        # The released pyramid remains; only its ViT is no longer used.
        self.control_model.vit = nn.Identity()
        self.residual_extractor = _load_residual_extractor(residual_checkpoint)
        self.residual_adapter = ResidualChannelAdapter()
        if adapter_checkpoint:
            payload = torch.load(adapter_checkpoint, map_location="cpu")
            self.residual_adapter.load_state_dict(payload.get("adapter", payload), strict=True)
        self.glyph_spatial_adapter = GlyphSpatialAdapter(tuple(glyph_widths))
        self.style_scales = nn.Parameter(torch.full((13,), float(style_scale)))
        self.glyph_scales = nn.Parameter(torch.full((13,), float(glyph_scale)))
        self.text_encoder.eval().requires_grad_(False)
        self.vae.eval().requires_grad_(False)

    def on_train_start(self):
        # The upstream hook assumes a WandB/TensorBoard logger; this trainer can run logger-free.
        print("Starting residual + glyph-spatial TextCtrl fine-tuning")

    def train(self, mode=True):
        super().train(mode)
        self.residual_extractor.eval()
        self.text_encoder.eval()
        self.vae.eval()
        return self

    @torch.no_grad()
    def prepare_input(self, batch):
        values = super().prepare_input(batch)
        source = batch["source_residual"].to(self.device, non_blocking=True)
        glyph = batch["source_glyph_residual"].to(self.device, non_blocking=True)
        values["residual_style"] = self.residual_extractor.extract_style(source, glyph)
        values["target_glyph"] = batch["target_glyph"].to(self.device, non_blocking=True)
        return values

    def _style_controls(self, residual):
        grid = self.residual_adapter(residual)
        outputs = []
        for index, stage in enumerate(self.control_model.stages):
            feature = stage(grid)
            for projection in self.control_model.zero_convs[3 * index:3 * index + 3]:
                outputs.append(projection(feature))
        middle = grid
        for block in self.control_model.middle_block:
            middle = block(input_tensor=middle, temb=None)
        outputs.append(self.control_model.middle_block_out(middle))
        return outputs

    def apply_model(self, values):
        style = self._style_controls(values["residual_style"])
        glyph = self.glyph_spatial_adapter(values["target_glyph"])
        controls = [self.style_scales[i] * left + self.glyph_scales[i] * right
                    for i, (left, right) in enumerate(zip(style, glyph))]
        return self.unet(x=values["latent"], timestep=values["timestep"],
                         encoder_hidden_states=values["cond"], control=controls).sample

    def configure_optimizers(self):
        parameters = [
            {"params": self.unet.parameters()},
            {"params": self.control_model.parameters()},
            {"params": self.residual_adapter.parameters()},
            {"params": self.glyph_spatial_adapter.parameters()},
            {"params": [self.style_scales, self.glyph_scales]},
        ]
        return torch.optim.AdamW(parameters, lr=self.learning_rate,
                                 weight_decay=self.config.weight_decay,
                                 eps=self.config.adam_epsilon)
