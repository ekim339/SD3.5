"""TextCtrl dual-branch style encoder loading and selective fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
from typing import Dict, List, Mapping, Set, Union
import warnings

import torch
from torch import nn


@dataclass(frozen=True)
class LoadReport:
    checkpoint: str
    backbone_tensors: int
    glyph_head_tensors: int
    spatial_head_tensors: int
    ignored_tensors: int

    @property
    def has_pretrained_glyph_head(self) -> bool:
        return self.glyph_head_tensors > 0


def _load_textctrl_model_module(repository: Union[str, Path]):
    prestyle = Path(repository).expanduser().resolve() / "prestyle"
    model_path = prestyle / "model.py"
    if not model_path.is_file():
        raise FileNotFoundError("TextCtrl prestyle model not found: {}".format(model_path))
    if str(prestyle) not in sys.path:
        sys.path.insert(0, str(prestyle))
    spec = importlib.util.spec_from_file_location("textctrl_supcon_prestyle_model", model_path)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot import TextCtrl style model from {}".format(model_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _unwrap_state_dict(payload: object) -> Mapping[str, torch.Tensor]:
    if not isinstance(payload, Mapping):
        raise ValueError("Checkpoint must contain a state-dict mapping")
    current = payload
    for key in ("state_dict", "model", "style_encoder", "encoder"):
        candidate = current.get(key)
        if isinstance(candidate, Mapping) and any(
            torch.is_tensor(value) for value in candidate.values()
        ):
            current = candidate
            break
    tensors = {str(key): value for key, value in current.items() if torch.is_tensor(value)}
    if not tensors:
        raise ValueError("Checkpoint contains no tensors")
    return tensors


def _key_candidates(key: str) -> List[str]:
    """Return progressively de-wrapped state keys, preserving inner modules."""

    pieces = key.split(".")
    candidates = [key]
    wrappers = {"module", "model", "state_dict", "style_encoder", "encoder"}
    start = 0
    while start < len(pieces) and pieces[start] in wrappers:
        start += 1
        candidates.append(".".join(pieces[start:]))
    for marker in ("encoder", "style_encoder"):
        if marker in pieces:
            candidates.append(".".join(pieces[pieces.index(marker) + 1 :]))
    # The released TextCtrl file is a bare VisionTransformerEncoder state dict.
    candidates.extend("vit." + candidate for candidate in list(candidates))
    return list(dict.fromkeys(candidate for candidate in candidates if candidate))


def load_pretrained_style_encoder(
    encoder: nn.Module,
    checkpoint: Union[str, Path],
    require_glyph_head: bool = False,
) -> LoadReport:
    """Load full pretraining, Lightning, or released backbone-only checkpoints."""

    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError("Style encoder checkpoint not found: {}".format(checkpoint_path))
    payload = torch.load(str(checkpoint_path), map_location="cpu")
    source = _unwrap_state_dict(payload)
    target = encoder.state_dict()
    selected: Dict[str, torch.Tensor] = {}
    used_source: Set[str] = set()
    for source_key, value in source.items():
        for candidate in _key_candidates(source_key):
            if candidate in target and tuple(value.shape) == tuple(target[candidate].shape):
                if candidate in selected:
                    continue
                selected[candidate] = value
                used_source.add(source_key)
                break

    backbone_keys = [key for key in target if key.startswith("vit.")]
    missing_backbone = [key for key in backbone_keys if key not in selected]
    if missing_backbone:
        preview = ", ".join(missing_backbone[:5])
        raise ValueError(
            "Checkpoint is missing {} shared-backbone tensors (for example: {})".format(
                len(missing_backbone), preview
            )
        )

    glyph_keys = [key for key in target if key.startswith("glyph_attn_block.")]
    loaded_glyph = [key for key in glyph_keys if key in selected]
    if loaded_glyph and len(loaded_glyph) != len(glyph_keys):
        raise ValueError(
            "Checkpoint contains only {}/{} glyph-head tensors".format(
                len(loaded_glyph), len(glyph_keys)
            )
        )
    if require_glyph_head and not loaded_glyph:
        raise ValueError(
            "The checkpoint contains a pretrained ViT backbone but no glyph/style head. "
            "Provide a full prestyle StyleNet checkpoint or set "
            "checkpoint.require_glyph_head_weights=false."
        )
    if not loaded_glyph:
        warnings.warn(
            "The checkpoint has no glyph_attn_block weights. The released TextCtrl "
            "style_encoder.pth is backbone-only, so the glyph/style head starts from "
            "its TextCtrl initialization.",
            RuntimeWarning,
        )

    encoder.load_state_dict(selected, strict=False)
    return LoadReport(
        checkpoint=str(checkpoint_path),
        backbone_tensors=sum(key.startswith("vit.") for key in selected),
        glyph_head_tensors=sum(key.startswith("glyph_attn_block.") for key in selected),
        spatial_head_tensors=sum(key.startswith("spatial_attn_block.") for key in selected),
        ignored_tensors=len(source) - len(used_source),
    )


class TextCtrlGlyphStyleEncoder(nn.Module):
    """Run the frozen ViT backbone and trainable glyph/style attention branch.

    The spatial branch is kept as part of the checkpoint-compatible module but
    is frozen and never evaluated by ``forward``.
    """

    def __init__(
        self,
        repository: Union[str, Path],
        checkpoint: Union[str, Path],
        image_size: int = 128,
        patch_size: int = 16,
        embed_dim: int = 768,
        pooling: str = "mean",
        require_glyph_head_weights: bool = False,
    ) -> None:
        super().__init__()
        if pooling not in {"mean", "flatten"}:
            raise ValueError("pooling must be 'mean' or 'flatten'")
        module = _load_textctrl_model_module(repository)
        self.encoder = module.StyleEncoder(
            image_size=int(image_size),
            patch_size=int(patch_size),
            in_chans=3,
            embed_dim=int(embed_dim),
        )
        self.pooling = pooling
        self.load_report = load_pretrained_style_encoder(
            self.encoder,
            checkpoint,
            require_glyph_head=bool(require_glyph_head_weights),
        )
        self.freeze_non_glyph_parameters()

    @property
    def glyph_head(self) -> nn.Module:
        return self.encoder.glyph_attn_block

    @property
    def spatial_head(self) -> nn.Module:
        return self.encoder.spatial_attn_block

    @property
    def backbone(self) -> nn.Module:
        return self.encoder.vit

    def freeze_non_glyph_parameters(self) -> None:
        self.encoder.requires_grad_(False)
        self.glyph_head.requires_grad_(True)
        self.backbone.eval()
        self.spatial_head.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # Frozen modules must remain in evaluation mode even if dropout or batch
        # normalization is added to a future upstream TextCtrl implementation.
        self.backbone.eval()
        self.spatial_head.eval()
        self.glyph_head.train(mode)
        return self

    def forward_tokens(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            shared = self.backbone(images, mask=None)
        glyph = shared.detach()
        for block in self.glyph_head:
            glyph = block(glyph)
        return glyph

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.forward_tokens(images)
        if self.pooling == "mean":
            return tokens.mean(dim=1)
        return tokens.flatten(start_dim=1)

    def trainable_parameters(self):
        return (parameter for parameter in self.glyph_head.parameters() if parameter.requires_grad)


def parameter_counts(model: TextCtrlGlyphStyleEncoder) -> Dict[str, int]:
    return {
        "backbone": sum(parameter.numel() for parameter in model.backbone.parameters()),
        "spatial_head": sum(parameter.numel() for parameter in model.spatial_head.parameters()),
        "glyph_head": sum(parameter.numel() for parameter in model.glyph_head.parameters()),
        "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }
