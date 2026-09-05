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
from safetensors.torch import load_file, save_file
from torch import nn


DEFAULT_LORA_TARGETS = (
    "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
    "attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0",
)

FLOW_OBJECTIVES = ("self_reconstruction", "cooldown")


def build_flow_matching_path(
    target: torch.Tensor,
    sigma: torch.Tensor,
    *,
    objective: str,
    source: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the interpolated latent and its velocity supervision.

    Self-reconstruction uses SD3's ordinary clean-to-Gaussian rectified-flow
    path. Cooldown follows Self-Prompting DiT Eq. (3)--(4) exactly: sigma=0 is
    the paired source, sigma=1 is the edited target, and no Gaussian endpoint
    participates in that objective.
    """
    if objective not in FLOW_OBJECTIVES:
        raise ValueError(
            f"Unsupported flow objective {objective!r}; expected one of {FLOW_OBJECTIVES}"
        )
    if objective == "self_reconstruction":
        if source is not None:
            raise ValueError("self_reconstruction must not receive a source latent")
        noise = torch.randn_like(target) if noise is None else noise
        if noise.shape != target.shape:
            raise ValueError("Noise and target latents must have identical shapes")
        return (1.0 - sigma) * target + sigma * noise, noise - target

    if source is None:
        raise ValueError("cooldown requires the paired source latent")
    if noise is not None:
        raise ValueError("cooldown must not receive a Gaussian-noise endpoint")
    if source.shape != target.shape:
        raise ValueError("Source and target latents must have identical shapes")
    return (1.0 - sigma) * source + sigma * target, target - source


def flow_matching_mse(
    prediction: torch.Tensor, velocity_target: torch.Tensor
) -> torch.Tensor:
    """Paper-faithful full-latent squared-error reduction."""
    if prediction.shape != velocity_target.shape:
        raise ValueError("Prediction and velocity target must have identical shapes")
    return (prediction.float() - velocity_target.float()).square().mean()


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
    """65-channel SD3.5 with a full input projection and attention LoRA."""

    INPUT_PROJECTION_NAME = "input_projection.safetensors"

    def __init__(
        self,
        pipeline,
        lora_rank: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        lora_target_modules: Iterable[str] = DEFAULT_LORA_TARGETS,
    ) -> None:
        super().__init__()
        self.transformer, self.vae, self.scheduler = pipeline.transformer, pipeline.vae, pipeline.scheduler
        self.latent_channels = int(self.transformer.config.out_channels)
        projection = expand_sd3_input_projection(
            self.transformer, 3 * self.latent_channels + 1
        )
        targets = list(lora_target_modules)
        if "pos_embed.proj" in targets:
            raise ValueError(
                "pos_embed.proj is trained in full and must not be a LoRA target"
            )
        self.transformer.requires_grad_(False)
        self.transformer.add_adapter(LoraConfig(
            r=int(lora_rank), lora_alpha=int(lora_alpha), lora_dropout=float(lora_dropout),
            target_modules=targets, init_lora_weights="gaussian", bias="none",
        ))
        projection.requires_grad_(True)
        self.vae.requires_grad_(False).eval()
        for name in ("text_encoder", "text_encoder_2", "text_encoder_3"):
            encoder = getattr(pipeline, name, None)
            if encoder is not None:
                encoder.requires_grad_(False).eval()
        trainable = [name for name, value in self.transformer.named_parameters() if value.requires_grad]
        allowed = lambda name: "lora_" in name or name.startswith("pos_embed.proj.")
        if not trainable or any(not allowed(name) for name in trainable):
            raise RuntimeError(
                "Only the full input projection and PEFT LoRA tensors may be trainable"
            )

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
        style: torch.Tensor | None, mask: torch.Tensor,
    ) -> torch.Tensor:
        # Keep the 65-channel architecture identical across stages so a
        # self-reconstruction checkpoint can initialize cooldown training. The
        # absent style condition is an exact zero latent; VAE-encoding a blank
        # image would produce a nonzero latent and would not mean "no style".
        if style is None:
            style = torch.zeros_like(noisy)
        mask = F.interpolate(mask, noisy.shape[-2:], mode="nearest")
        hidden = torch.cat((noisy, masked, glyph, style, mask), dim=1)
        expected = 4 * self.latent_channels + 1
        if hidden.shape[1] != expected:
            raise ValueError(f"Expected {expected} channels, received {hidden.shape[1]}")
        return hidden

    def forward(
        self, target_image, masked_image, glyph_image, style_image, mask,
        prompt_embeds, pooled_prompt_embeds, *,
        objective: str = "self_reconstruction", source_image=None,
    ) -> torch.Tensor:
        if objective not in FLOW_OBJECTIVES:
            raise ValueError(
                f"Unsupported flow objective {objective!r}; expected one of {FLOW_OBJECTIVES}"
            )
        if objective == "cooldown":
            if source_image is None:
                raise ValueError("cooldown requires source_image")
            if style_image is None:
                raise ValueError("cooldown requires the visual style prompt")
        elif source_image is not None:
            raise ValueError("self_reconstruction must not receive source_image")
        with torch.no_grad():
            target = self.encode_images(target_image)
            source = (
                self.encode_images(source_image) if objective == "cooldown" else None
            )
            masked = self.encode_images(masked_image)
            glyph = self.encode_images(glyph_image)
            style = None if style_image is None else self.encode_images(style_image)
        indices = torch.randint(0, self.scheduler.config.num_train_timesteps, (target.shape[0],), device=target.device)
        timesteps = self.scheduler.timesteps.to(target.device)[indices]
        sigmas = self.scheduler.sigmas.to(target.device, target.dtype)[indices]
        sigma = sigmas.view(-1, *([1] * (target.ndim - 1)))
        noisy, velocity_target = build_flow_matching_path(
            target, sigma, objective=objective, source=source
        )
        prediction = self.transformer(
            hidden_states=self.composite_input(noisy, masked, glyph, style, mask),
            timestep=timesteps, encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds, return_dict=True,
        ).sample
        return flow_matching_mse(prediction, velocity_target)

    def save_lora_weights(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        StableDiffusion3Pipeline.save_lora_weights(
            directory,
            transformer_lora_layers=get_peft_model_state_dict(self.transformer),
            transformer_lora_adapter_metadata=self.transformer.peft_config["default"].to_dict(),
            safe_serialization=True,
        )
        projection = self.transformer.pos_embed.proj
        save_file(
            {
                key: value.detach().cpu().contiguous()
                for key, value in projection.state_dict().items()
            },
            str(directory / self.INPUT_PROJECTION_NAME),
        )

    def load_lora_weights(self, directory: str | Path) -> None:
        directory = Path(directory)
        projection_path = directory / self.INPUT_PROJECTION_NAME
        if not projection_path.is_file():
            raise FileNotFoundError(
                f"Missing expanded input projection checkpoint: {projection_path}"
            )
        self.transformer.pos_embed.proj.load_state_dict(load_file(str(projection_path)))
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
