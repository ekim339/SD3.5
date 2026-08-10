from __future__ import annotations

from typing import Iterator, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from .adapters import AuxiliaryConditionProjector, SD3InpaintingPatchProjection


def masked_flow_matching_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    foreground_weight: float,
    background_weight: float,
) -> torch.Tensor:
    """Combine region-normalized edit and background flow-matching losses."""
    if foreground_weight <= 0 or background_weight <= 0:
        raise ValueError("Flow-matching loss weights must be positive")
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target must have equal shapes")
    if mask.shape[0] != prediction.shape[0] or mask.shape[-2:] != prediction.shape[-2:]:
        raise ValueError("Mask must match prediction batch and spatial dimensions")
    squared_error = (prediction.float() - target.float()).square()
    foreground = mask.float().expand_as(squared_error)
    background = 1.0 - foreground
    foreground_count = foreground.sum()
    background_count = background.sum()
    foreground_loss = (
        (squared_error * foreground).sum() / foreground_count.clamp_min(1.0)
    )
    background_loss = (
        (squared_error * background).sum() / background_count.clamp_min(1.0)
    )
    foreground_active = (foreground_count > 0).to(squared_error.dtype)
    background_active = (background_count > 0).to(squared_error.dtype)
    normalizer = (
        foreground_weight * foreground_active
        + background_weight * background_active
    ).clamp_min(1.0)
    return (
        foreground_weight * foreground_loss
        + background_weight * background_loss
    ) / normalizer


class SD35SceneTextEditor(nn.Module):
    """Glyph/style-conditioned SD3.5 with an explicit inpainting input."""

    def __init__(
        self,
        pipeline,
        style_encoder: nn.Module,
        glyph_encoder: nn.Module,
        training_mode: str = "lora",
        lora_config: Mapping | None = None,
        fill_value: float = 0.5,
        glyph_gate_init: float = 0.1,
        style_gate_init: float = 0.1,
        foreground_loss_weight: float = 5.0,
        background_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.transformer = pipeline.transformer
        self.vae = pipeline.vae
        self.scheduler = pipeline.scheduler
        self.style_encoder = style_encoder
        self.glyph_encoder = glyph_encoder
        self.normalized_fill_value = 2.0 * fill_value - 1.0
        self.projector = AuxiliaryConditionProjector(
            glyph_gate_init=glyph_gate_init,
            style_gate_init=style_gate_init,
        )
        self.foreground_loss_weight = float(foreground_loss_weight)
        self.background_loss_weight = float(background_loss_weight)
        if self.foreground_loss_weight <= 0 or self.background_loss_weight <= 0:
            raise ValueError("Flow-matching loss weights must be positive")

        latent_channels = int(self.transformer.config.out_channels)
        base_projection = self.transformer.pos_embed.proj
        if not isinstance(base_projection, nn.Conv2d):
            raise TypeError("SD3 patch embedding projection must be a Conv2d")
        if base_projection.in_channels != latent_channels:
            raise ValueError("SD3 base input and output latent channels must match")
        self.latent_channels = latent_channels
        self.transformer.pos_embed.proj = SD3InpaintingPatchProjection(
            base_projection,
            condition_channels=latent_channels + 1,
        )
        self.vae.requires_grad_(False)
        self.transformer.requires_grad_(False)
        if training_mode not in {"frozen", "lora", "full"}:
            raise ValueError("training_mode must be one of: frozen, lora, full")
        self.training_mode = training_mode
        if training_mode == "full":
            self.transformer.requires_grad_(True)
        elif training_mode == "lora":
            if lora_config is None:
                raise ValueError("lora_config is required when training_mode=lora")
            from peft import LoraConfig

            self.transformer.add_adapter(
                LoraConfig(
                    r=int(lora_config["rank"]),
                    lora_alpha=int(lora_config["alpha"]),
                    lora_dropout=float(lora_config.get("dropout", 0.0)),
                    init_lora_weights=True,
                    target_modules=list(lora_config["target_modules"]),
                )
            )
        self.transformer.pos_embed.proj.conditioning.requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        self.vae.eval()
        self.style_encoder.eval()
        self.glyph_encoder.eval()
        if self.training_mode == "frozen":
            self.transformer.eval()
        return self

    @property
    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def _encode_vae(self, images: torch.Tensor) -> torch.Tensor:
        latents = self.vae.encode(images).latent_dist.sample()
        shift = getattr(self.vae.config, "shift_factor", 0.0) or 0.0
        return (latents - shift) * self.vae.config.scaling_factor

    def forward(
        self,
        target_images: torch.Tensor,
        source_images: torch.Tensor,
        source_masks: torch.Tensor,
        style_images: torch.Tensor,
        target_texts: list[str] | tuple[str, ...],
        prompt_embeddings: torch.Tensor,
        target_text_embeddings: torch.Tensor,
        pooled_prompt_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            target_latents = self._encode_vae(target_images)
            masked_source = (
                source_images * (1 - source_masks)
                + source_masks * self.normalized_fill_value
            )
            source_latents = self._encode_vae(masked_source)
            style_tokens = self.style_encoder(style_images)
            glyph_tokens = self.glyph_encoder(target_texts)

        mask_latent = F.interpolate(source_masks, source_latents.shape[-2:], mode="nearest")
        condition = self.projector(
            prompt_embeddings, target_text_embeddings, glyph_tokens, style_tokens
        )

        noise = torch.randn_like(target_latents)
        indices = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (target_latents.shape[0],),
            device=target_latents.device,
        )
        timesteps = self.scheduler.timesteps.to(target_latents.device)[indices]
        sigmas = self.scheduler.sigmas.to(target_latents.device, target_latents.dtype)[indices]
        sigma = sigmas.view(-1, *([1] * (target_latents.ndim - 1)))
        noisy_latents = (1.0 - sigma) * target_latents + sigma * noise

        inpainting_input = torch.cat(
            (noisy_latents, source_latents, mask_latent), dim=1
        )
        prediction = self.transformer(
            hidden_states=inpainting_input,
            timestep=timesteps,
            encoder_hidden_states=condition,
            pooled_projections=pooled_prompt_embeddings,
            return_dict=True,
        ).sample
        target = noise - target_latents
        return masked_flow_matching_loss(
            prediction,
            target,
            mask_latent,
            self.foreground_loss_weight,
            self.background_loss_weight,
        )
