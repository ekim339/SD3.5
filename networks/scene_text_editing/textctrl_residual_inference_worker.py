"""TextCtrl inference using residual-extractor style conditioning.

This worker runs inside TextCtrl's pinned Python 3.8 environment. It replaces
only the style ViT output; the pretrained style pyramid and the rest of the
released SD1.5/TextCtrl inference pipeline are left unchanged.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path


def parse_args(argv: Sequence[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--starting-layer", type=int, required=True)
    parser.add_argument("--num-inference-steps", type=int, required=True)
    parser.add_argument("--guidance-scale", type=float, required=True)
    parser.add_argument("--residual-checkpoint", required=True)
    parser.add_argument("--adapter-checkpoint", required=True)
    parser.add_argument("--canonical-font", required=True)
    parser.add_argument("--residual-resolution", type=int, default=256)
    return parser.parse_args(argv)


def read_labels(path: Path):
    labels = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.strip().split(maxsplit=1)
        if not parts:
            continue
        if len(parts) != 2:
            raise ValueError("Expected filename and text at %s:%d" % (path, number))
        labels[parts[0]] = parts[1]
    return labels


def load_residual_extractor(checkpoint, device):
    import importlib.util
    import torch

    model_path = Path(__file__).resolve().parents[2] / "encoders" / "model.py"
    spec = importlib.util.spec_from_file_location("residual_style_model", model_path)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load residual model from %s" % model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ResidualStyleAutoencoder = module.ResidualStyleAutoencoder
    payload = torch.load(checkpoint, map_location="cpu")
    model_config = payload.get("config", {}).get("model", {})
    model = ResidualStyleAutoencoder(**model_config)
    model.load_state_dict(payload["model"], strict=True)
    model.eval().requires_grad_(False)
    return model.to(device)


def load_channel_adapter(checkpoint, device):
    import torch
    from adapter.model import ChannelAdapter

    payload = torch.load(checkpoint, map_location="cpu")
    model = ChannelAdapter(**payload.get("model_config", {}))
    model.load_state_dict(payload.get("adapter", payload), strict=True)
    model.eval().requires_grad_(False)
    return model.to(device)


def image_to_tensor(image):
    """Exact residual-extractor training normalization: RGB [0,255] to [-1,1]."""
    import numpy as np
    import torch
    array = np.asarray(image, dtype="float32") / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


class NeutralGlyphRenderer:
    """Exact Pillow renderer used by `encoders.dataset` during training."""
    def __init__(self, font_path, canvas_size=(256, 256), padding=12):
        self.font_path = Path(font_path).expanduser().resolve()
        self.canvas_size = canvas_size
        self.padding = padding

    def __call__(self, text):
        from PIL import Image, ImageDraw, ImageFont
        width, height = self.canvas_size
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        max_width = width - 2 * self.padding
        max_height = height - 2 * self.padding
        size = max(8, height - 2 * self.padding)
        while size >= 8:
            font = ImageFont.truetype(str(self.font_path), size=size)
            box = draw.textbbox((0, 0), text, font=font)
            if box[2] - box[0] <= max_width and box[3] - box[1] <= max_height:
                break
            size -= 2
        if size < 8:
            font = ImageFont.truetype(str(self.font_path), size=8)
            box = draw.textbbox((0, 0), text, font=font)
        x = (width - (box[2] - box[0])) // 2 - box[0]
        y = (height - (box[3] - box[1])) // 2 - box[1]
        draw.text((x, y), text, font=font, fill="black")
        return image


def adapted_style_tokens(source_path, source_text, renderer, resolution,
                         residual_extractor, channel_adapter, device):
    import torch
    from PIL import Image

    with Image.open(source_path) as opened:
        source = opened.convert("RGB").resize(
            (resolution, resolution), Image.Resampling.LANCZOS)
    glyph = renderer(source_text)
    source_tensor = image_to_tensor(source).unsqueeze(0).to(device)
    glyph_tensor = image_to_tensor(glyph).unsqueeze(0).to(device)
    with torch.no_grad():
        residual = residual_extractor.extract_style(source_tensor, glyph_tensor)
        if tuple(residual.shape[1:]) != (256, 16, 16):
            raise RuntimeError(
                "Expected residual shape [B,256,16,16], got %s" % (tuple(residual.shape),)
            )
        grid = channel_adapter(residual)
        if tuple(grid.shape[1:]) != (768, 16, 16):
            raise RuntimeError(
                "Expected adapted grid [B,768,16,16], got %s" % (tuple(grid.shape),)
            )
        return grid.flatten(2).transpose(1, 2).contiguous()


def main(argv: Sequence[str] = None) -> None:
    args = parse_args(argv)
    if args.residual_resolution != 256:
        raise ValueError("The trained residual extractor requires resolution 256")
    repository = Path(args.repository).expanduser().resolve()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    sys.path.insert(0, str(repository))
    os.chdir(repository)

    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from PIL import Image
    from pytorch_lightning import seed_everything
    from tqdm import tqdm
    from inference import load_image
    from src.MuSA.GaMuSA import GaMuSA
    from src.MuSA.GaMuSA_app import text_editing
    from utils import create_model, load_state_dict

    if not torch.cuda.is_available():
        raise RuntimeError("TextCtrl residual inference requires a CUDA GPU")
    device = torch.device("cuda")
    source_labels = read_labels(dataset_dir / "i_s.txt")
    target_labels = read_labels(dataset_dir / "i_t.txt")
    if source_labels.keys() != target_labels.keys():
        raise ValueError("Source and target label filenames differ")
    output_dir.mkdir(parents=True, exist_ok=True)

    textctrl = create_model("configs/inference.yaml").cuda()
    textctrl.load_state_dict(load_state_dict(args.checkpoint), strict=False)
    textctrl.eval()
    monitor = {
        "max_length": 25,
        "loss_weight": 1.0,
        "attention": "position",
        "backbone": "transformer",
        "backbone_ln": 3,
        "checkpoint": "weights/vision_model.pth",
        "charset_path": "src/module/abinet/data/charset_36.txt",
    }
    pipeline = GaMuSA(textctrl, monitor)
    residual_extractor = load_residual_extractor(args.residual_checkpoint, device)
    channel_adapter = load_channel_adapter(args.adapter_checkpoint, device)
    renderer = NeutralGlyphRenderer(
        Path(args.canonical_font).expanduser().resolve(),
        (args.residual_resolution, args.residual_resolution),
    )
    seed_everything(args.seed)

    class FixedStyleTokens(torch.nn.Module):
        def __init__(self, tokens):
            super().__init__()
            self.register_buffer("tokens", tokens)

        def forward(self, hint, mask=None):
            count = hint.shape[0]
            if self.tokens.shape[0] == count:
                return self.tokens
            if self.tokens.shape[0] != 1:
                raise RuntimeError("Cannot expand fixed style batch to %d" % count)
            return self.tokens.expand(count, -1, -1)

    for filename in tqdm(list(source_labels), desc="Residual-conditioned TextCtrl"):
        source_path = dataset_dir / "i_s" / filename
        source_text = source_labels[filename]
        target_text = target_labels[filename]
        tokens = adapted_style_tokens(
            source_path, source_text, renderer, args.residual_resolution,
            residual_extractor, channel_adapter, device)
        source_image = load_image(str(source_path))
        with Image.open(source_path) as opened:
            original_size = opened.size
        original_vit = pipeline.control_model.vit
        pipeline.control_model.vit = FixedStyleTokens(tokens)
        try:
            _, generated = text_editing(
                pipeline, source_image, source_image, source_text, target_text,
                starting_layer=args.starting_layer,
                ddim_steps=args.num_inference_steps,
                scale=args.guidance_scale,
            )
        finally:
            pipeline.control_model.vit = original_vit
        image = Image.fromarray(np.clip(generated * 255, 0, 255).astype(np.uint8))
        image.resize(original_size, Image.Resampling.BICUBIC).save(output_dir / filename)


if __name__ == "__main__":
    main()
