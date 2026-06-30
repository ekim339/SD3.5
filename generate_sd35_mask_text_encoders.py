#!/usr/bin/env python3
"""Generate SD3.5 images while masking one text encoder at a time.

For each seed, this writes one collage containing:

    original, clip-l, clip-g, t5

Masking is done at the prompt embedding level:
    - clip-l: zero CLIP-L channels in token positions 0..76 and pooled channels 0..767
    - clip-g: zero CLIP-G channels in token positions 0..76 and pooled channels 768..2047
    - t5: zero T5 token positions after the 77 CLIP positions

CUDA_VISIBLE_DEVICES=0 python3 generate_sd35_mask_text_encoders.py \
  --device cuda \
  --prompt "A cinematic photo of a glass greenhouse on a rainy evening" \
  --seeds 123 456 \
  --output-dir outputs/masked_greenhouse
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"
CONDITIONS = ("original", "clip-l", "clip-g", "t5")
CLIP_SEQUENCE_LENGTH = 77
CLIP_L_CHANNELS = (0, 768)
CLIP_G_CHANNELS = (768, 2048)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SD3.5 images with CLIP-L, CLIP-G, and T5 masked one at a time."
    )
    parser.add_argument("--prompt", required=True, help="Text prompt used for generation.")
    parser.add_argument(
        "--negative-prompt",
        default=None,
        help="Optional negative prompt.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        required=True,
        help="One or more integer seeds, for example: --seeds 11 22 33",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs_masked",
        help="Directory where collages and metadata are saved.",
    )
    parser.add_argument(
        "--metadata-file",
        default="metadata.json",
        help="Metadata filename written inside output-dir before generation starts.",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=("cuda", "mps", "cpu"),
        help="Device to use. Defaults to cuda, then mps, then cpu.",
    )
    parser.add_argument("--steps", type=int, default=28, help="Number of denoising steps.")
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=3.5,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument("--width", type=int, default=1024, help="Generated image width.")
    parser.add_argument("--height", type=int, default=1024, help="Generated image height.")
    return parser.parse_args()


def missing_dependency(name: str) -> None:
    print(
        f"Missing dependency: {name}\n\n"
        "Install the runtime packages first, for example:\n"
        "  pip install torch diffusers transformers accelerate sentencepiece protobuf\n",
        file=sys.stderr,
    )
    raise SystemExit(1)


def import_dependencies():
    try:
        import torch
    except ImportError:
        missing_dependency("torch")

    try:
        from diffusers import StableDiffusion3Pipeline
    except ImportError:
        missing_dependency("diffusers")

    return torch, StableDiffusion3Pipeline


def choose_device(torch, requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def choose_dtype(torch, device: str):
    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


def dtype_name(torch, device: str) -> str:
    dtype = choose_dtype(torch, device)
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.float32:
        return "float32"
    return str(dtype)


def expected_output_files(seeds: list[int]) -> list[str]:
    return [f"seed_{seed}_collage.png" for seed in seeds]


def write_metadata(args: argparse.Namespace, output_dir: Path, device: str, dtype: str) -> Path:
    metadata_path = output_dir / args.metadata_file
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "seeds": args.seeds,
        "output_dir": str(output_dir),
        "conditions": list(CONDITIONS),
        "masking": {
            "clip-l": {
                "prompt_embeds_tokens": [0, CLIP_SEQUENCE_LENGTH - 1],
                "prompt_embeds_channels": list(CLIP_L_CHANNELS),
                "pooled_prompt_embeds_channels": list(CLIP_L_CHANNELS),
            },
            "clip-g": {
                "prompt_embeds_tokens": [0, CLIP_SEQUENCE_LENGTH - 1],
                "prompt_embeds_channels": list(CLIP_G_CHANNELS),
                "pooled_prompt_embeds_channels": list(CLIP_G_CHANNELS),
            },
            "t5": {
                "prompt_embeds_tokens": [CLIP_SEQUENCE_LENGTH, "end"],
                "prompt_embeds_channels": "all",
                "pooled_prompt_embeds_channels": None,
            },
        },
        "mask_negative_embeddings_too": True,
        "device": device,
        "requested_device": args.device,
        "dtype": dtype,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "width": args.width,
        "height": args.height,
        "metadata_file": args.metadata_file,
        "output_files": expected_output_files(args.seeds),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved metadata: {metadata_path}")
    return metadata_path


def load_pipeline(torch, StableDiffusion3Pipeline, device: str):
    dtype = choose_dtype(torch, device)
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    token_arg = token if token else True

    print(f"Loading {MODEL_ID} on {device} with dtype={dtype}...")
    try:
        pipe = StableDiffusion3Pipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            token=token_arg,
        )
    except Exception as exc:
        print(
            "\nCould not load the model from Hugging Face.\n"
            "Make sure you have accepted the gated model terms at:\n"
            f"  https://huggingface.co/{MODEL_ID}\n\n"
            "Then authenticate with `huggingface-cli login` or set HF_TOKEN.\n"
            f"\nOriginal error: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    if device == "cuda":
        pipe.enable_model_cpu_offload()
        return pipe

    return pipe.to(device)


def make_generator(torch, seed: int):
    return torch.Generator(device="cpu").manual_seed(seed)


def encode_prompt_embeddings(torch, pipe, args: argparse.Namespace):
    execution_device = getattr(pipe, "_execution_device", pipe.device)
    do_cfg = args.guidance_scale > 1.0

    with torch.no_grad():
        return pipe.encode_prompt(
            prompt=args.prompt,
            prompt_2=None,
            prompt_3=None,
            negative_prompt=args.negative_prompt,
            negative_prompt_2=None,
            negative_prompt_3=None,
            do_classifier_free_guidance=do_cfg,
            device=execution_device,
            num_images_per_prompt=1,
        )


def clone_embeddings(embeddings: tuple):
    return tuple(tensor.clone() if tensor is not None else None for tensor in embeddings)


def zero_clip_slice(prompt_embeds, pooled_prompt_embeds, channel_slice: tuple[int, int]) -> None:
    if prompt_embeds is None:
        return
    start, end = channel_slice
    prompt_embeds[:, :CLIP_SEQUENCE_LENGTH, start:end] = 0
    if pooled_prompt_embeds is not None:
        pooled_prompt_embeds[:, start:end] = 0


def apply_mask(embeddings: tuple, condition: str) -> tuple:
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = clone_embeddings(embeddings)

    if condition == "original":
        return (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )
    if condition == "clip-l":
        zero_clip_slice(prompt_embeds, pooled_prompt_embeds, CLIP_L_CHANNELS)
        zero_clip_slice(negative_prompt_embeds, negative_pooled_prompt_embeds, CLIP_L_CHANNELS)
    elif condition == "clip-g":
        zero_clip_slice(prompt_embeds, pooled_prompt_embeds, CLIP_G_CHANNELS)
        zero_clip_slice(negative_prompt_embeds, negative_pooled_prompt_embeds, CLIP_G_CHANNELS)
    elif condition == "t5":
        prompt_embeds[:, CLIP_SEQUENCE_LENGTH:, :] = 0
        if negative_prompt_embeds is not None:
            negative_prompt_embeds[:, CLIP_SEQUENCE_LENGTH:, :] = 0
    else:
        raise ValueError(f"Unknown condition: {condition}")

    return (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    )


def generate_for_seed(
    torch,
    pipe,
    args: argparse.Namespace,
    seed: int,
    condition: str,
    embeddings: tuple,
):
    generator = make_generator(torch, seed)
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = apply_mask(embeddings, condition)

    print(f"Generating {condition} seed {seed}...")
    image = pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        width=args.width,
        height=args.height,
        generator=generator,
    ).images[0]
    return image


def require_pillow():
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:
        raise RuntimeError(
            "Creating collages requires Pillow. Install it with `python -m pip install pillow`."
        ) from exc
    return Image, ImageDraw, ImageFont


def load_label_font(ImageFont, image_width: int):
    font_size = max(24, image_width // 28)
    for font_path in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(font_path, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_centered_label(draw, box_width: int, y: int, label: str, font) -> None:
    try:
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
    except AttributeError:
        text_width = draw.textlength(label, font=font)
    x = (box_width - text_width) / 2
    draw.text((x, y), label, fill=(20, 20, 20), font=font)


def make_collage(condition_images: dict):
    Image, ImageDraw, ImageFont = require_pillow()
    images = [condition_images[condition].convert("RGB") for condition in CONDITIONS]
    image_width, image_height = images[0].size
    label_height = max(48, image_height // 14)
    padding = max(12, image_width // 80)
    collage_width = image_width * 2 + padding * 3
    collage_height = (image_height + label_height) * 2 + padding * 3
    collage = Image.new("RGB", (collage_width, collage_height), "white")
    draw = ImageDraw.Draw(collage)
    font = load_label_font(ImageFont, image_width)

    positions = {
        "original": (padding, padding),
        "clip-l": (image_width + padding * 2, padding),
        "clip-g": (padding, image_height + label_height + padding * 2),
        "t5": (image_width + padding * 2, image_height + label_height + padding * 2),
    }
    labels = {
        "original": "original",
        "clip-l": "clip-l masked",
        "clip-g": "clip-g masked",
        "t5": "t5 masked",
    }

    for condition, image in zip(CONDITIONS, images):
        x, y = positions[condition]
        draw_centered_label(
            draw,
            image_width,
            y + max(4, padding // 2),
            labels[condition],
            font,
        )
        collage.paste(image, (x, y + label_height))

    return collage


def generate_collage_for_seed(
    torch,
    pipe,
    args: argparse.Namespace,
    output_dir: Path,
    seed: int,
    embeddings: tuple,
) -> None:
    condition_images = {}
    for condition in CONDITIONS:
        condition_images[condition] = generate_for_seed(
            torch,
            pipe,
            args,
            seed,
            condition,
            embeddings,
        )

    output_path = output_dir / f"seed_{seed}_collage.png"
    collage = make_collage(condition_images)
    collage.save(output_path)
    print(f"Saved collage {output_path}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    torch, StableDiffusion3Pipeline = import_dependencies()
    device = choose_device(torch, args.device)
    write_metadata(args, output_dir, device, dtype_name(torch, device))
    pipe = load_pipeline(torch, StableDiffusion3Pipeline, device)
    embeddings = encode_prompt_embeddings(torch, pipe, args)

    for seed in args.seeds:
        generate_collage_for_seed(torch, pipe, args, output_dir, seed, embeddings)


if __name__ == "__main__":
    main()
