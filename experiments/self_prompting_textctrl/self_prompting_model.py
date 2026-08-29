"""Inference-only model definition matching the trained 17-channel checkpoint."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from torch import nn
from transformers import CLIPTextModel, CLIPTokenizer

LATENT_CHANNELS = 4
CONDITIONING_CHANNELS = 17


class SelfPromptingSD15(nn.Module):
    def __init__(self, vae_path: str, unet_path: str, scheduler_path: str,
                 text_model_path: str, revision: str | None = None,
                 max_text_length: int = 77) -> None:
        super().__init__()
        self.max_text_length = max_text_length
        self.vae = AutoencoderKL.from_pretrained(vae_path)
        self.unet = UNet2DConditionModel.from_pretrained(unet_path)
        self.noise_scheduler = DDPMScheduler.from_pretrained(scheduler_path)
        self.tokenizer = CLIPTokenizer.from_pretrained(
            text_model_path, subfolder="tokenizer", revision=revision
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            text_model_path, subfolder="text_encoder", revision=revision
        )
        self.vae.requires_grad_(False).eval()
        self.text_encoder.requires_grad_(False).eval()
        self._expand_unet_input()

    @property
    def device(self):
        return next(self.unet.parameters()).device

    @property
    def dtype(self):
        return next(self.unet.parameters()).dtype

    def _expand_unet_input(self):
        old = self.unet.conv_in
        if old.in_channels == CONDITIONING_CHANNELS:
            return
        if old.in_channels != LATENT_CHANNELS:
            raise ValueError(f"Expected 4-channel SD1.5 UNet, got {old.in_channels}")
        expanded = nn.Conv2d(CONDITIONING_CHANNELS, old.out_channels,
                             old.kernel_size, old.stride, old.padding,
                             dilation=old.dilation, groups=old.groups,
                             bias=old.bias is not None, padding_mode=old.padding_mode)
        with torch.no_grad():
            expanded.weight.zero_()
            expanded.weight[:, :LATENT_CHANNELS].copy_(old.weight)
            if old.bias is not None:
                expanded.bias.copy_(old.bias)
        self.unet.conv_in = expanded
        self.unet.register_to_config(in_channels=CONDITIONING_CHANNELS)

    def load_checkpoint(self, checkpoint: str | Path):
        checkpoint = Path(checkpoint)
        directory = checkpoint / "unet" if (checkpoint / "unet").is_dir() else checkpoint
        loaded = UNet2DConditionModel.from_pretrained(directory)
        if int(loaded.config.in_channels) != CONDITIONING_CHANNELS:
            raise ValueError(f"Checkpoint must have {CONDITIONING_CHANNELS} input channels")
        self.unet = loaded

    @torch.no_grad()
    def encode_images(self, images):
        return (self.vae.encode(images.to(dtype=self.vae.dtype)).latent_dist.sample()
                * self.vae.config.scaling_factor)

    @torch.no_grad()
    def encode_prompts(self, texts: Sequence[str]):
        tokens = self.tokenizer(list(texts), padding="max_length", truncation=True,
                                max_length=self.max_text_length, return_tensors="pt")
        return self.text_encoder(input_ids=tokens.input_ids.to(self.device),
                                 attention_mask=tokens.attention_mask.to(self.device)).last_hidden_state

    @torch.no_grad()
    def visual_conditions(self, masked_source, glyph, style, mask):
        masked = self.encode_images(masked_source)
        glyph_latent = self.encode_images(glyph)
        style_latent = self.encode_images(style)
        latent_mask = F.interpolate(mask.float(), size=masked.shape[-2:], mode="nearest")
        return masked, glyph_latent, style_latent, latent_mask

    def predict_noise(self, noisy, timestep, embeddings, conditions):
        model_input = torch.cat([noisy, *[item.to(noisy.dtype) for item in conditions]], dim=1)
        return self.unet(model_input, timestep,
                         encoder_hidden_states=embeddings.to(self.dtype)).sample
