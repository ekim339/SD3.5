"""Frozen SD3.5 MM-DiT with PEFT LoRA self-prompt adaptation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from diffusers import StableDiffusion3Pipeline
from diffusers.utils import convert_unet_state_dict_to_peft
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from torch import nn


DEFAULT_LORA_TARGETS = (
    "pos_embed.proj",
    "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
    "attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0",
)


def expand_sd3_input_projection(transformer: nn.Module, condition_channels: int) -> nn.Conv2d:
    old = transformer.pos_embed.proj
    if not isinstance(old, nn.Conv2d) or old.groups != 1:
        raise TypeError("SD3.5 patch projection must be an ungrouped Conv2d")
    new = nn.Conv2d(
        old.in_channels + int(condition_channels), old.out_channels,
        old.kernel_size, old.stride, old.padding, old.dilation, old.groups,
        old.bias is not None, old.padding_mode, device=old.weight.device, dtype=old.weight.dtype,
    )
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, : old.in_channels].copy_(old.weight)
        if old.bias is not None:
            new.bias.copy_(old.bias)
    transformer.pos_embed.proj = new
    transformer.register_to_config(in_channels=new.in_channels)
    return new


class SelfPromptingSD35(nn.Module):
    """65-channel SD3.5 whose only trainable parameters are LoRA tensors."""

    def __init__(
        self,
        pipeline,
        foreground_weight: float = 5.0,
        background_weight: float = 1.0,
        lora_rank: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        lora_target_modules: Iterable[str] = DEFAULT_LORA_TARGETS,
    ) -> None:
        super().__init__()
        self.transformer, self.vae, self.scheduler = pipeline.transformer, pipeline.vae, pipeline.scheduler
        self.latent_channels = int(self.transformer.config.out_channels)
        expand_sd3_input_projection(self.transformer, 3 * self.latent_channels + 1)
        targets = list(lora_target_modules)
        if "pos_embed.proj" not in targets:
            raise ValueError("LoRA targets must include pos_embed.proj")
        self.transformer.requires_grad_(False)
        self.transformer.add_adapter(LoraConfig(
            r=int(lora_rank), lora_alpha=int(lora_alpha), lora_dropout=float(lora_dropout),
            target_modules=targets, init_lora_weights="gaussian", bias="none",
        ))
        self.vae.requires_grad_(False).eval()
        for name in ("text_encoder", "text_encoder_2", "text_encoder_3"):
            encoder = getattr(pipeline, name, None)
            if encoder is not None:
                encoder.requires_grad_(False).eval()
        self.foreground_weight, self.background_weight = float(foreground_weight), float(background_weight)
        trainable = [name for name, value in self.transformer.named_parameters() if value.requires_grad]
        if not trainable or any("lora_" not in name for name in trainable):
            raise RuntimeError("Only PEFT LoRA tensors may be trainable")

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [value for value in self.transformer.parameters() if value.requires_grad]

    def train(self, mode: bool = True):
        super().train(mode)
        self.vae.eval()
        return self

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor, sample: bool = True) -> torch.Tensor:
        distribution = self.vae.encode(images).latent_dist
        latents = distribution.sample() if sample else distribution.mode()
        shift = getattr(self.vae.config, "shift_factor", 0.0) or 0.0
        return (latents - shift) * self.vae.config.scaling_factor

    def composite_input(
        self, noisy: torch.Tensor, masked: torch.Tensor, glyph: torch.Tensor,
        style: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = F.interpolate(mask, noisy.shape[-2:], mode="nearest")
        hidden = torch.cat((noisy, masked, glyph, style, mask), dim=1)
        expected = 4 * self.latent_channels + 1
        if hidden.shape[1] != expected:
            raise ValueError(f"Expected {expected} channels, received {hidden.shape[1]}")
        return hidden

    def forward(
        self, target_image, masked_image, glyph_image, style_image, mask,
        prompt_embeds, pooled_prompt_embeds,
    ) -> torch.Tensor:
        with torch.no_grad():
            target = self.encode_images(target_image)
            masked = self.encode_images(masked_image)
            glyph = self.encode_images(glyph_image)
            style = self.encode_images(style_image)
        indices = torch.randint(0, self.scheduler.config.num_train_timesteps, (target.shape[0],), device=target.device)
        timesteps = self.scheduler.timesteps.to(target.device)[indices]
        sigmas = self.scheduler.sigmas.to(target.device, target.dtype)[indices]
        sigma = sigmas.view(-1, *([1] * (target.ndim - 1)))
        noise = torch.randn_like(target)
        noisy = (1.0 - sigma) * target + sigma * noise
        latent_mask = F.interpolate(mask, target.shape[-2:], mode="nearest")
        prediction = self.transformer(
            hidden_states=self.composite_input(noisy, masked, glyph, style, mask),
            timestep=timesteps, encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds, return_dict=True,
        ).sample
        error = (prediction.float() - (noise - target).float()).square()
        foreground = latent_mask.expand_as(error)
        background = 1.0 - foreground
        fg_loss = (error * foreground).sum() / foreground.sum().clamp_min(1.0)
        bg_loss = (error * background).sum() / background.sum().clamp_min(1.0)
        return (self.foreground_weight * fg_loss + self.background_weight * bg_loss) / (
            self.foreground_weight + self.background_weight
        )

    def save_lora_weights(self, directory: str | Path) -> None:
        StableDiffusion3Pipeline.save_lora_weights(
            directory,
            transformer_lora_layers=get_peft_model_state_dict(self.transformer),
            transformer_lora_adapter_metadata=self.transformer.peft_config["default"].to_dict(),
            safe_serialization=True,
        )

    def load_lora_weights(self, directory: str | Path) -> None:
        state_dict = StableDiffusion3Pipeline.lora_state_dict(directory)
        transformer_state = {
            key.removeprefix("transformer."): value
            for key, value in state_dict.items()
            if key.startswith("transformer.")
        }
        if not transformer_state:
            raise ValueError(f"No transformer LoRA weights found in {directory}")
        peft_state = convert_unet_state_dict_to_peft(transformer_state)
        incompatible = set_peft_model_state_dict(
            self.transformer, peft_state, adapter_name="default"
        )
        missing = [
            key for key in getattr(incompatible, "missing_keys", ())
            if ".lora_" in key
        ]
        if missing:
            raise ValueError(f"Missing LoRA checkpoint keys: {missing}")
        unexpected = getattr(incompatible, "unexpected_keys", ())
        if unexpected:
            raise ValueError(f"Unexpected LoRA checkpoint keys: {unexpected}")
