"""SD3.5 MM-DiT adaptation for latent self prompts."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def expand_sd3_input_projection(transformer: nn.Module, condition_channels: int) -> nn.Conv2d:
    """Expand the patch projection and preserve the pretrained noisy-latent slice."""
    old = transformer.pos_embed.proj
    if not isinstance(old, nn.Conv2d) or old.groups != 1:
        raise TypeError("transformer.pos_embed.proj must be an ungrouped Conv2d")
    new = nn.Conv2d(
        old.in_channels + int(condition_channels), old.out_channels,
        old.kernel_size, old.stride, old.padding, old.dilation, old.groups,
        old.bias is not None, old.padding_mode,
        device=old.weight.device, dtype=old.weight.dtype,
    )
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, : old.in_channels].copy_(old.weight)
        if old.bias is not None:
            new.bias.copy_(old.bias)
    transformer.pos_embed.proj = new
    # Diffusers validates hidden-state channels against this config value.
    transformer.register_to_config(in_channels=new.in_channels)
    return new


class SelfPromptingSD35(nn.Module):
    """Full-parameter SD3.5 backbone with frozen VAE and text encoders."""

    def __init__(self, pipeline, foreground_weight: float = 5.0, background_weight: float = 1.0):
        super().__init__()
        self.transformer, self.vae, self.scheduler = pipeline.transformer, pipeline.vae, pipeline.scheduler
        self.latent_channels = int(self.transformer.config.out_channels)
        expand_sd3_input_projection(self.transformer, 3 * self.latent_channels + 1)
        self.vae.requires_grad_(False).eval()
        for encoder_name in ("text_encoder", "text_encoder_2", "text_encoder_3"):
            encoder = getattr(pipeline, encoder_name, None)
            if encoder is not None:
                encoder.requires_grad_(False).eval()
        self.transformer.requires_grad_(True)
        self.foreground_weight = float(foreground_weight)
        self.background_weight = float(background_weight)

    def train(self, mode: bool = True):
        super().train(mode)
        self.vae.eval()
        return self

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        distribution = self.vae.encode(images).latent_dist
        latents = distribution.sample()
        shift = getattr(self.vae.config, "shift_factor", 0.0) or 0.0
        return (latents - shift) * self.vae.config.scaling_factor

    def forward(
        self,
        target_image: torch.Tensor,
        masked_image: torch.Tensor,
        glyph_image: torch.Tensor,
        style_image: torch.Tensor,
        mask: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            target = self.encode_images(target_image)
            masked = self.encode_images(masked_image)
            glyph = self.encode_images(glyph_image)
            style = self.encode_images(style_image)
        mask_latent = F.interpolate(mask, target.shape[-2:], mode="nearest")
        noise = torch.randn_like(target)
        indices = torch.randint(0, len(self.scheduler.timesteps), (target.shape[0],), device=target.device)
        timesteps = self.scheduler.timesteps.to(target.device)[indices]
        sigmas = self.scheduler.sigmas.to(device=target.device, dtype=target.dtype)[indices]
        sigma = sigmas.view(-1, *([1] * (target.ndim - 1)))
        noisy = (1.0 - sigma) * target + sigma * noise
        hidden = torch.cat((noisy, masked, glyph, style, mask_latent), dim=1)
        prediction = self.transformer(
            hidden_states=hidden,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            return_dict=True,
        ).sample
        flow_target = noise - target
        error = (prediction.float() - flow_target.float()).square()
        fg = mask_latent.expand_as(error)
        bg = 1.0 - fg
        fg_loss = (error * fg).sum() / fg.sum().clamp_min(1.0)
        bg_loss = (error * bg).sum() / bg.sum().clamp_min(1.0)
        return (self.foreground_weight * fg_loss + self.background_weight * bg_loss) / (
            self.foreground_weight + self.background_weight
        )
