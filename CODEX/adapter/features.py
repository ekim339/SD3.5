"""Frozen residual-student input and TextCtrl teacher feature extraction."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from encoders.extract import load_extractor
from networks.sd35_implementation.encoders import FrozenTextCtrlStyleEncoder


def freeze(module: nn.Module) -> nn.Module:
    module.eval()
    module.requires_grad_(False)
    return module


def load_frozen_extractors(residual_checkpoint: str | Path,
                           textctrl_repository: str | Path,
                           style_checkpoint: str | Path,
                           device: torch.device):
    residual = freeze(load_extractor(residual_checkpoint, device))
    teacher = freeze(FrozenTextCtrlStyleEncoder(textctrl_repository, style_checkpoint).to(device))
    return residual, teacher


@torch.no_grad()
def residual_features(extractor, image: torch.Tensor, glyph: torch.Tensor) -> torch.Tensor:
    """Extract learned `R(E(I),E(G))` features from inputs normalized to [-1,1]."""
    result = extractor.extract_style(image, glyph)
    if tuple(result.shape[1:]) != (256, 16, 16):
        raise RuntimeError(f"Expected residual features [B,256,16,16], got {tuple(result.shape)}")
    return result


@torch.no_grad()
def textctrl_style_grid(teacher, image: torch.Tensor) -> torch.Tensor:
    """Extract and reshape TextCtrl features; teacher input is normalized to [0,1]."""
    textctrl_image = ((image + 1.0) * 0.5).clamp(0.0, 1.0)
    tokens = teacher(textctrl_image)
    grid = tokens.transpose(1, 2).reshape(tokens.shape[0], 768, 16, 16)
    if tuple(grid.shape[1:]) != (768, 16, 16):
        raise RuntimeError(f"Expected TextCtrl grid [B,768,16,16], got {tuple(grid.shape)}")
    return grid
