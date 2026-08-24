from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from torch import nn
from transformers import CLIPTextModel, CLIPTokenizer


LATENT_CHANNELS = 4
CONDITIONING_CHANNELS = 4 + 4 + 4 + 4 + 1


class SelfPromptingSD15(nn.Module):
    """SD1.5 UNet conditioned on masked, glyph, style, and mask latents."""

    def __init__(
        self,
        vae_path: str,
        unet_path: str,
        scheduler_path: str,
        text_model_path: str,
        revision: str | None = None,
        max_text_length: int = 77,
        conditioning_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.max_text_length = max_text_length
        self.conditioning_dropout = conditioning_dropout
        self.vae = AutoencoderKL.from_pretrained(vae_path)
        self.unet = UNet2DConditionModel.from_pretrained(unet_path)
        self.noise_scheduler = DDPMScheduler.from_pretrained(scheduler_path)
        self.tokenizer = CLIPTokenizer.from_pretrained(
            text_model_path, subfolder="tokenizer", revision=revision
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            text_model_path, subfolder="text_encoder", revision=revision
        )
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.vae.eval()
        self.text_encoder.eval()
        self._expand_unet_input()

    @property
    def device(self) -> torch.device:
        return next(self.unet.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.unet.parameters()).dtype

    def _expand_unet_input(self) -> None:
        old = self.unet.conv_in
        if old.in_channels == CONDITIONING_CHANNELS:
            return
        if old.in_channels != LATENT_CHANNELS:
            raise ValueError(
                f"Expected a {LATENT_CHANNELS}-channel SD1.5 UNet, got {old.in_channels}"
            )
        expanded = nn.Conv2d(
            CONDITIONING_CHANNELS,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            dilation=old.dilation,
            groups=old.groups,
            bias=old.bias is not None,
            padding_mode=old.padding_mode,
        )
        with torch.no_grad():
            expanded.weight.zero_()
            expanded.weight[:, :LATENT_CHANNELS].copy_(old.weight)
            if old.bias is not None:
                expanded.bias.copy_(old.bias)
        self.unet.conv_in = expanded
        self.unet.register_to_config(in_channels=CONDITIONING_CHANNELS)

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        posterior = self.vae.encode(images.to(dtype=self.vae.dtype)).latent_dist
        return posterior.sample() * self.vae.config.scaling_factor

    @torch.no_grad()
    def encode_prompts(self, texts: Sequence[str]) -> torch.Tensor:
        tokens = self.tokenizer(
            list(texts),
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.to(self.device)
        attention_mask = tokens.attention_mask.to(self.device)
        return self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state

    @torch.no_grad()
    def encode_visual_conditions(
        self,
        masked_source: torch.Tensor,
        glyph: torch.Tensor,
        style: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        masked_latents = self.encode_images(masked_source)
        glyph_latents = self.encode_images(glyph)
        style_latents = self.encode_images(style)
        latent_mask = F.interpolate(mask.float(), size=masked_latents.shape[-2:], mode="nearest")
        return masked_latents, glyph_latents, style_latents, latent_mask

    def compose_unet_input(
        self,
        noisy_latents: torch.Tensor,
        visual_conditions: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        parts = (noisy_latents, *visual_conditions)
        return torch.cat([part.to(dtype=noisy_latents.dtype) for part in parts], dim=1)

    def predict_noise(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeddings: torch.Tensor,
        visual_conditions: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        model_input = self.compose_unet_input(noisy_latents, visual_conditions)
        return self.unet(
            model_input,
            timesteps,
            encoder_hidden_states=prompt_embeddings.to(dtype=self.dtype),
        ).sample

    def _drop_text_conditions(self, texts: Sequence[str]) -> list[str]:
        if not self.training or self.conditioning_dropout <= 0:
            return list(texts)
        dropped: list[str] = []
        for text in texts:
            use_empty = torch.rand((), device=self.device) < self.conditioning_dropout
            dropped.append("" if bool(use_empty) else text)
        return dropped

    def forward(self, batch: dict[str, torch.Tensor | Sequence[str]]) -> torch.Tensor:
        target = batch["target"]
        if not isinstance(target, torch.Tensor):
            raise TypeError("batch['target'] must be a tensor")
        target_latents = self.encode_images(target)
        noise = torch.randn_like(target_latents)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (target_latents.shape[0],),
            device=target_latents.device,
            dtype=torch.long,
        )
        noisy_latents = self.noise_scheduler.add_noise(target_latents, noise, timesteps)
        visual_conditions = self.encode_visual_conditions(
            batch["masked_source"], batch["glyph"], batch["style"], batch["mask"]
        )
        texts = self._drop_text_conditions(batch["text"])
        prompt_embeddings = self.encode_prompts(texts)
        prediction = self.predict_noise(
            noisy_latents, timesteps, prompt_embeddings, visual_conditions
        )
        if self.noise_scheduler.config.prediction_type == "v_prediction":
            target_prediction = self.noise_scheduler.get_velocity(
                target_latents, noise, timesteps
            )
        else:
            target_prediction = noise
        return F.mse_loss(prediction.float(), target_prediction.float())

    def save_unet(self, output_dir: str | Path) -> None:
        self.unet.save_pretrained(Path(output_dir) / "unet")

    def load_unet_checkpoint(self, checkpoint: str | Path) -> None:
        checkpoint = Path(checkpoint)
        unet_dir = checkpoint / "unet" if (checkpoint / "unet").is_dir() else checkpoint
        loaded = UNet2DConditionModel.from_pretrained(unet_dir)
        if loaded.config.in_channels != CONDITIONING_CHANNELS:
            raise ValueError(f"Checkpoint does not use {CONDITIONING_CHANNELS} input channels")
        self.unet = loaded
