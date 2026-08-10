from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
from torch import nn


def _freeze(module: nn.Module) -> nn.Module:
    module.eval()
    module.requires_grad_(False)
    return module


class FrozenTextCtrlStyleEncoder(nn.Module):
    """TextCtrl ViT style backbone returning 256 tokens of width 768."""

    def __init__(self, repository: str | Path, checkpoint: str | Path) -> None:
        super().__init__()
        prestyle_path = Path(repository).resolve() / "prestyle"
        if str(prestyle_path) not in sys.path:
            sys.path.append(str(prestyle_path))
        spec = importlib.util.spec_from_file_location(
            "textctrl_prestyle_model", prestyle_path / "model.py"
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import TextCtrl style model from {prestyle_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        encoder = module.VisionTransformerEncoder(
            img_size=256,
            patch_size=16,
            in_chans=3,
            num_classes=0,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=module.partial(nn.LayerNorm, eps=1e-6),
            init_values=0,
            use_learnable_pos_emb=False,
        )
        encoder.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
        )
        self.encoder = _freeze(encoder)

    def train(self, mode: bool = True):
        super().train(False)
        self.encoder.eval()
        return self

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images, mask=None)
        if features.shape[1:] != (256, 768):
            raise RuntimeError(f"Unexpected TextCtrl style shape: {tuple(features.shape)}")
        return features


class FrozenTextCtrlGlyphEncoder(nn.Module):
    """TextCtrl LabelEncoder returning 24 target-text tokens of width 768."""

    def __init__(self, repository: str | Path, checkpoint: str | Path) -> None:
        super().__init__()
        repository = Path(repository).resolve()
        if str(repository) not in sys.path:
            sys.path.append(str(repository))
        from src.module.textencoder.modules import LabelEncoder

        encoder = LabelEncoder(max_len=24, emb_dim=768, ckpt_path=None)
        encoder.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
        )
        self.encoder = _freeze(encoder)

    def train(self, mode: bool = True):
        super().train(False)
        self.encoder.eval()
        return self

    @torch.no_grad()
    def forward(self, target_texts: list[str] | tuple[str, ...]) -> torch.Tensor:
        features = self.encoder(list(target_texts))
        if features.shape[1:] != (24, 768):
            raise RuntimeError(f"Unexpected TextCtrl glyph shape: {tuple(features.shape)}")
        return features
