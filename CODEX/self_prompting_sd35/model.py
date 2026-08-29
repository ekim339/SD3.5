"""SD3.5 MM-DiT adaptation for latent self prompts."""

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


LORA_WEIGHT_NAME = "pytorch_lora_weights.safetensors"
DEFAULT_LORA_TARGET_MODULES = (
    # The custom projection must be adapted so the 49 added condition channels
    # can affect the otherwise frozen transformer.
    "pos_embed.proj",
    # Diffusers' recommended SD3 joint-attention targets.
    "attn.add_k_proj",
    "attn.add_q_proj",
    "attn.add_v_proj",
    "attn.to_add_out",
    "attn.to_k",
    "attn.to_out.0",
    "attn.to_q",
    "attn.to_v",
)


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


def add_sd3_lora_adapter(
    transformer: nn.Module,
    rank: int = 16,
    alpha: int = 16,
    dropout: float = 0.0,
    target_modules: Iterable[str] = DEFAULT_LORA_TARGET_MODULES,
) -> LoraConfig:
    """Freeze the SD3.5 backbone and attach the trainable PEFT adapter."""
    rank, alpha, dropout = int(rank), int(alpha), float(dropout)
    targets = tuple(str(target).strip() for target in target_modules)
    if rank <= 0 or alpha <= 0:
        raise ValueError("LoRA rank and alpha must be positive")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("LoRA dropout must be in [0, 1)")
    if not targets or any(not target for target in targets):
        raise ValueError("LoRA target_modules must contain non-empty module names")
    if "pos_embed.proj" not in targets:
        raise ValueError(
            "LoRA target_modules must include 'pos_embed.proj' so self-prompt channels are trainable"
        )

    transformer.requires_grad_(False)
    adapter_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        init_lora_weights="gaussian",
        bias="none",
        target_modules=list(targets),
    )
    transformer.add_adapter(adapter_config)

    trainable_names = [name for name, parameter in transformer.named_parameters() if parameter.requires_grad]
    if not trainable_names:
        raise RuntimeError("PEFT did not create any trainable LoRA parameters")
    unexpected = [name for name in trainable_names if ".lora_" not in name]
    if unexpected:
        raise RuntimeError(f"Non-LoRA transformer parameters are trainable: {unexpected}")
    missing = [target for target in targets if not any(f"{target}.lora_" in name for name in trainable_names)]
    if missing:
        raise ValueError(f"LoRA targets did not match SD3.5 modules: {missing}")
    return adapter_config


def save_sd3_lora_weights(transformer: nn.Module, directory: str | Path) -> None:
    """Save only transformer adapter tensors in Diffusers' LoRA format."""
    adapter_name = "default"
    if adapter_name not in getattr(transformer, "peft_config", {}):
        raise ValueError("The transformer does not contain the default LoRA adapter")
    StableDiffusion3Pipeline.save_lora_weights(
        save_directory=directory,
        transformer_lora_layers=get_peft_model_state_dict(transformer, adapter_name=adapter_name),
        transformer_lora_adapter_metadata=transformer.peft_config[adapter_name].to_dict(),
        safe_serialization=True,
    )


def load_sd3_lora_weights(transformer: nn.Module, directory: str | Path) -> None:
    """Restore adapter tensors into an already-expanded, adapter-equipped transformer."""
    loaded = StableDiffusion3Pipeline.lora_state_dict(
        directory,
        weight_name=LORA_WEIGHT_NAME,
        use_safetensors=True,
    )
    state_dict = loaded[0] if isinstance(loaded, tuple) else loaded
    transformer_state = {
        key.removeprefix("transformer."): value
        for key, value in state_dict.items()
        if key.startswith("transformer.")
    }
    if not transformer_state:
        raise ValueError(f"No transformer LoRA weights found in {directory}")
    transformer_state = convert_unet_state_dict_to_peft(transformer_state)
    incompatible = set_peft_model_state_dict(transformer, transformer_state, adapter_name="default")
    unexpected = getattr(incompatible, "unexpected_keys", None) if incompatible is not None else None
    if unexpected:
        raise ValueError(f"Unexpected transformer LoRA keys in {directory}: {unexpected}")


class SelfPromptingSD35(nn.Module):
    """LoRA-adapted SD3.5 backbone with frozen base model and encoders."""

    def __init__(
        self,
        pipeline,
        foreground_weight: float = 5.0,
        background_weight: float = 1.0,
        lora_rank: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        lora_target_modules: Iterable[str] = DEFAULT_LORA_TARGET_MODULES,
    ):
        super().__init__()
        self.transformer, self.vae, self.scheduler = pipeline.transformer, pipeline.vae, pipeline.scheduler
        self.latent_channels = int(self.transformer.config.out_channels)
        expand_sd3_input_projection(self.transformer, 3 * self.latent_channels + 1)
        add_sd3_lora_adapter(
            self.transformer,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            target_modules=lora_target_modules,
        )
        self.vae.requires_grad_(False).eval()
        for encoder_name in ("text_encoder", "text_encoder_2", "text_encoder_3"):
            encoder = getattr(pipeline, encoder_name, None)
            if encoder is not None:
                encoder.requires_grad_(False).eval()
        self.foreground_weight = float(foreground_weight)
        self.background_weight = float(background_weight)

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Return the LoRA tensors that are safe to optimize."""
        return [parameter for parameter in self.transformer.parameters() if parameter.requires_grad]

    def save_lora_weights(self, directory: str | Path) -> None:
        save_sd3_lora_weights(self.transformer, directory)

    def load_lora_weights(self, directory: str | Path) -> None:
        load_sd3_lora_weights(self.transformer, directory)

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
