from __future__ import annotations

import torch
from torch import nn


class AuxiliaryConditionProjector(nn.Module):
    """Project TextCtrl glyph/style tokens into SD3.5's context width."""

    def __init__(
        self,
        glyph_dim: int = 768,
        style_dim: int = 768,
        output_dim: int = 4096,
        glyph_gate_init: float = 0.1,
        style_gate_init: float = 0.1,
    ):
        super().__init__()
        self.glyph = nn.Sequential(nn.LayerNorm(glyph_dim), nn.Linear(glyph_dim, output_dim))
        self.style = nn.Sequential(nn.LayerNorm(style_dim), nn.Linear(style_dim, output_dim))
        self.glyph_gate = nn.Parameter(torch.tensor(float(glyph_gate_init)))
        self.style_gate = nn.Parameter(torch.tensor(float(style_gate_init)))

    def forward(
        self,
        prompt_tokens: torch.Tensor,
        target_text_tokens: torch.Tensor,
        glyph_tokens: torch.Tensor,
        style_tokens: torch.Tensor,
    ) -> torch.Tensor:
        conditions = (prompt_tokens, target_text_tokens, glyph_tokens, style_tokens)
        if any(condition.ndim != 3 for condition in conditions):
            raise ValueError("All condition tensors must have shape [batch, tokens, channels]")
        if len({condition.shape[0] for condition in conditions}) != 1:
            raise ValueError("Condition tensors must have the same batch size")
        if target_text_tokens.shape[-1] != prompt_tokens.shape[-1]:
            raise ValueError("Prompt and target-text T5 tokens must have equal width")
        target_text = target_text_tokens.to(prompt_tokens.dtype)
        glyph = self.glyph(glyph_tokens).to(prompt_tokens.dtype) * self.glyph_gate
        style = self.style(style_tokens).to(prompt_tokens.dtype) * self.style_gate
        return torch.cat((prompt_tokens, target_text, glyph, style), dim=1)




class SD3InpaintingPatchProjection(nn.Module):
    """Extend SD3's patch projection with masked-image and mask channels."""

    def __init__(self, base_projection: nn.Conv2d, condition_channels: int):
        super().__init__()
        if base_projection.groups != 1:
            raise ValueError("SD3 inpainting projection requires an ungrouped Conv2d")
        self.base_channels = base_projection.in_channels
        self.condition_channels = int(condition_channels)
        self.base = base_projection.requires_grad_(False)
        self.conditioning = nn.Conv2d(
            self.condition_channels,
            base_projection.out_channels,
            kernel_size=base_projection.kernel_size,
            stride=base_projection.stride,
            padding=base_projection.padding,
            dilation=base_projection.dilation,
            bias=False,
            padding_mode=base_projection.padding_mode,
        )
        nn.init.zeros_(self.conditioning.weight)

    @property
    def in_channels(self) -> int:
        return self.base_channels + self.condition_channels

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} inpainting channels, got {latent.shape[1]}"
            )
        noisy, condition = latent.split(
            (self.base_channels, self.condition_channels), dim=1
        )
        return self.base(noisy) + self.conditioning(condition)
