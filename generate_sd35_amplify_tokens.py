#!/usr/bin/env python3
"""Generate SD3.5 token-signal multiplier sweeps as collages.

Edit these constants to choose the tokens and multiplier sweep:

    CLIP_TOKEN_INDICES = [0, 1, 2, 3, 7, 10, 16]
    T5_TOKEN_INDICES = [0, 4, 6, 7, 8]
    MULTIPLIER_VALUES = [0, 0.5, 1, 2, 3, 5, 10]

Set CLIP_TOKEN_INDICES = [] to scale every CLIP token position.
Set T5_TOKEN_INDICES = [] to scale every T5 token position.

For each seed, this creates one collage:

    row 1: CLIP-L + CLIP-G + T5 token rows scaled together
    row 2: CLIP-L token channels only
    row 3: CLIP-G token channels only
    row 4: T5 token rows only

Each column corresponds to one multiplier value.

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 generate_sd35_amplify_tokens.py \
      --prompt "A bathroom mat that says 'hello' in bold capital letters, photorealistic" \
      --seeds 0 1 2 3 4 5 6 7 8 9 \
      --output-dir outputs/amplify_sweep \
      --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"
CLIP_SEQUENCE_LENGTH = 77
T5_SEQUENCE_LENGTH = 256
CLIP_L_CHANNELS = (0, 768)
CLIP_G_CHANNELS = (768, 2048)
CLIP_BOTH_CHANNELS = (0, 2048)

CLIP_TOKEN_INDICES = [0, 1, 2, 3, 7, 10, 16]
T5_TOKEN_INDICES = [0, 4, 6, 7, 8]
MULTIPLIER_VALUES = [0, 0.5, 1, 2, 3, 5, 10]

ROW_SPECS = (
    ("all", "clip-l + clip-g + t5", "all", None),
    ("clip-l", "clip-l only", "clip-l", CLIP_TOKEN_INDICES),
    ("clip-g", "clip-g only", "clip-g", CLIP_TOKEN_INDICES),
    ("t5", "t5 only", "t5", T5_TOKEN_INDICES),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SD3.5 token multiplier sweep collages."
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
        default="outputs_amplified",
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
        "  pip install torch diffusers transformers accelerate sentencepiece protobuf pillow\n",
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


def validate_args(args: argparse.Namespace) -> None:
    all_indices = CLIP_TOKEN_INDICES + T5_TOKEN_INDICES
    if any(token_index < 0 for token_index in all_indices):
        raise ValueError("Token index constants must be non-negative.")
    if any(token_index >= CLIP_SEQUENCE_LENGTH for token_index in CLIP_TOKEN_INDICES):
        raise ValueError(f"CLIP token indices must be < {CLIP_SEQUENCE_LENGTH}.")
    if any(token_index >= T5_SEQUENCE_LENGTH for token_index in T5_TOKEN_INDICES):
        raise ValueError(f"T5 token indices must be < {T5_SEQUENCE_LENGTH}.")
    if not MULTIPLIER_VALUES:
        raise ValueError("MULTIPLIER_VALUES must not be empty.")
    if args.steps <= 0:
        raise ValueError("--steps must be > 0.")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be > 0.")


def selected_clip_token_indices() -> list[int]:
    if CLIP_TOKEN_INDICES:
        return CLIP_TOKEN_INDICES
    return list(range(CLIP_SEQUENCE_LENGTH))


def selected_t5_token_indices() -> list[int]:
    if T5_TOKEN_INDICES:
        return T5_TOKEN_INDICES
    return list(range(T5_SEQUENCE_LENGTH))


def expected_output_files(seeds: list[int]) -> list[str]:
    return [f"seed_{seed}_sweep.png" for seed in seeds]


def write_metadata(
    args: argparse.Namespace,
    output_dir: Path,
    device: str,
    dtype: str,
    token_summary: dict,
) -> Path:
    metadata_path = output_dir / args.metadata_file
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "seeds": args.seeds,
        "clip_token_indices": CLIP_TOKEN_INDICES,
        "t5_token_indices": T5_TOKEN_INDICES,
        "clip_token_selection": "all" if not CLIP_TOKEN_INDICES else "custom",
        "t5_token_selection": "all" if not T5_TOKEN_INDICES else "custom",
        "effective_clip_token_indices": selected_clip_token_indices(),
        "effective_t5_token_indices": selected_t5_token_indices(),
        "multiplier_values": MULTIPLIER_VALUES,
        "rows": [
            {"key": key, "label": label, "mode": mode, "token_indices": token_indices}
            for key, label, mode, token_indices in ROW_SPECS
        ],
        "token_summary": token_summary,
        "token_mapping": {
            "all": "scales CLIP_TOKEN_INDICES in clip-l and clip-g channels plus T5_TOKEN_INDICES in T5 rows",
            "clip": "token index i scales prompt_embeds[:, i, selected_clip_channels]",
            "t5": "token index i scales prompt_embeds[:, 77 + i, :]",
            "pooled_prompt_embeds": "unchanged because pooled embeddings are not token-specific",
            "negative_prompt_embeds": "unchanged",
        },
        "output_dir": str(output_dir),
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
            max_sequence_length=T5_SEQUENCE_LENGTH,
        )


def clone_embeddings(embeddings: tuple):
    return tuple(tensor.clone() if tensor is not None else None for tensor in embeddings)


def clip_channel_slice(mode: str) -> tuple[int, int]:
    if mode == "clip":
        return CLIP_BOTH_CHANNELS
    if mode == "clip-l":
        return CLIP_L_CHANNELS
    if mode == "clip-g":
        return CLIP_G_CHANNELS
    raise ValueError(f"Unsupported CLIP mode: {mode}")


def scale_clip_tokens(prompt_embeds, token_indices: list[int], multiplier: float, mode: str) -> None:
    sequence_length = prompt_embeds.shape[1]
    start, end = clip_channel_slice(mode)
    effective_indices = token_indices if token_indices else selected_clip_token_indices()
    for token_index in effective_indices:
        if token_index < CLIP_SEQUENCE_LENGTH and token_index < sequence_length:
            prompt_embeds[:, token_index, start:end] *= multiplier


def scale_t5_tokens(prompt_embeds, token_indices: list[int], multiplier: float) -> None:
    sequence_length = prompt_embeds.shape[1]
    effective_indices = token_indices if token_indices else selected_t5_token_indices()
    for token_index in effective_indices:
        t5_position = CLIP_SEQUENCE_LENGTH + token_index
        if t5_position < sequence_length:
            prompt_embeds[:, t5_position, :] *= multiplier


def scaled_prompt_embeddings(
    embeddings: tuple,
    mode: str,
    token_indices: list[int] | None,
    multiplier: float,
) -> tuple:
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = clone_embeddings(embeddings)

    if mode == "all":
        scale_clip_tokens(prompt_embeds, selected_clip_token_indices(), multiplier, "clip")
        scale_t5_tokens(prompt_embeds, selected_t5_token_indices(), multiplier)
    elif mode in {"clip", "clip-l", "clip-g"}:
        if token_indices is None:
            raise ValueError(f"Token indices are required for mode: {mode}")
        scale_clip_tokens(prompt_embeds, token_indices, multiplier, mode)
    elif mode == "t5":
        if token_indices is None:
            raise ValueError("Token indices are required for mode: t5")
        scale_t5_tokens(prompt_embeds, token_indices, multiplier)
    else:
        raise ValueError(f"Unknown scaling mode: {mode}")

    return (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    )


def generate_image(
    torch,
    pipe,
    args: argparse.Namespace,
    seed: int,
    embeddings: tuple,
    mode: str,
    token_indices: list[int],
    multiplier: float,
):
    generator = make_generator(torch, seed)
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = scaled_prompt_embeddings(embeddings, mode, token_indices, multiplier)

    print(f"Generating seed {seed}: {mode} x {multiplier}...")
    return pipe(
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


def decode_token(tokenizer, token_id: int) -> str:
    decoded = tokenizer.decode([token_id], skip_special_tokens=False).strip()
    return decoded or f"id:{token_id}"


def tokenizer_labels(tokenizer, prompt: str, token_indices: list[int], max_length: int) -> dict[int, str]:
    encoded = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    ids = encoded.input_ids[0].tolist()
    labels = {}
    for token_index in token_indices:
        labels[token_index] = decode_token(tokenizer, ids[token_index])
    return labels


def build_token_summary(pipe, prompt: str) -> dict:
    clip_indices = selected_clip_token_indices()
    t5_indices = selected_t5_token_indices()
    clip_l_labels = tokenizer_labels(
        pipe.tokenizer,
        prompt,
        clip_indices,
        pipe.tokenizer.model_max_length,
    )
    clip_g_labels = tokenizer_labels(
        pipe.tokenizer_2,
        prompt,
        clip_indices,
        pipe.tokenizer_2.model_max_length,
    )
    t5_labels = tokenizer_labels(pipe.tokenizer_3, prompt, t5_indices, T5_SEQUENCE_LENGTH)

    clip_labels = {}
    for token_index in clip_indices:
        clip_l = clip_l_labels[token_index]
        clip_g = clip_g_labels[token_index]
        if clip_l == clip_g:
            clip_labels[token_index] = clip_l
        else:
            clip_labels[token_index] = f"clip-l={clip_l!r}; clip-g={clip_g!r}"

    return {
        "clip": clip_labels,
        "clip_l": clip_l_labels,
        "clip_g": clip_g_labels,
        "t5": t5_labels,
    }


def format_token_summary(token_summary: dict) -> tuple[str, str]:
    if CLIP_TOKEN_INDICES:
        clip_text = ", ".join(
            f"{index}:{token_summary['clip'][index]!r}" for index in CLIP_TOKEN_INDICES
        )
    else:
        clip_text = f"ALL {CLIP_SEQUENCE_LENGTH} positions"

    if T5_TOKEN_INDICES:
        t5_text = ", ".join(
            f"{index}:{token_summary['t5'][index]!r}" for index in T5_TOKEN_INDICES
        )
    else:
        t5_text = f"ALL {T5_SEQUENCE_LENGTH} positions"

    return f"CLIP tokens: {clip_text}", f"T5 tokens: {t5_text}"


def require_pillow():
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:
        raise RuntimeError(
            "Creating collages requires Pillow. Install it with `python -m pip install pillow`."
        ) from exc
    return Image, ImageDraw, ImageFont


def load_font(ImageFont, image_width: int, scale: int, minimum: int, bold: bool = False):
    font_size = max(minimum, image_width // scale)
    paths = (
        (
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        )
        if bold else
        (
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        )
    )
    for font_path in paths:
        try:
            return ImageFont.truetype(font_path, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def text_dimensions(draw, text: str, font) -> tuple[int, int, int]:
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1], bbox[1]
    except AttributeError:
        return int(draw.textlength(text, font=font)), 0, 0


def wrap_text_to_width(draw, text: str, font, max_width: int) -> list[str]:
    lines = []
    for paragraph in text.splitlines() or [""]:
        current = ""
        for word in paragraph.split():
            candidate = f"{current} {word}".strip()
            width, _, _ = text_dimensions(draw, candidate, font)
            if current and width > max_width:
                lines.append(current)
                current = word
            else:
                current = candidate
        lines.append(current)
    return lines


def draw_centered_text(draw, x: int, y: int, width: int, height: int, text: str, font) -> None:
    text_width, text_height, y_offset = text_dimensions(draw, text, font)
    text_x = x + (width - text_width) / 2
    text_y = y + max(0, (height - text_height) / 2) - y_offset
    draw.text((text_x, text_y), text, fill=(20, 20, 20), font=font)


def draw_wrapped_centered_text(draw, x: int, y: int, width: int, lines: list[str], font, line_height: int) -> None:
    for line_index, line in enumerate(lines):
        text_width, _, y_offset = text_dimensions(draw, line, font)
        text_x = x + (width - text_width) / 2
        text_y = y + line_index * line_height - y_offset
        draw.text((text_x, text_y), line, fill=(20, 20, 20), font=font)


def multiplier_label(value: float) -> str:
    return f"x{value:g}"


def make_collage(seed: int, generated_images: dict, prompt: str, token_summary: dict):
    Image, ImageDraw, ImageFont = require_pillow()
    sample = next(iter(generated_images.values())).convert("RGB")
    image_width, image_height = sample.size
    padding = max(12, image_width // 80)
    row_label_width = max(160, image_width // 5)
    column_header_height = max(52, image_height // 16)
    row_header_height = max(44, image_height // 20)
    prompt_line_height = max(28, image_height // 34)

    collage_width = row_label_width + image_width * len(MULTIPLIER_VALUES) + padding * (len(MULTIPLIER_VALUES) + 2)
    scratch = Image.new("RGB", (collage_width, 1), "white")
    scratch_draw = ImageDraw.Draw(scratch)
    prompt_font = load_font(ImageFont, image_width, scale=38, minimum=20)
    label_font = load_font(ImageFont, image_width, scale=32, minimum=20, bold=True)
    small_font = load_font(ImageFont, image_width, scale=42, minimum=16)

    clip_summary, t5_summary = format_token_summary(token_summary)
    header_lines = []
    for header_text in (f"Prompt: {prompt}", clip_summary, t5_summary):
        header_lines.extend(
            wrap_text_to_width(
                scratch_draw,
                header_text,
                prompt_font,
                collage_width - padding * 2,
            )
        )
    header_height = prompt_line_height * len(header_lines) + padding * 2

    collage_height = (
        header_height
        + column_header_height
        + (row_header_height + image_height) * len(ROW_SPECS)
        + padding * (len(ROW_SPECS) + 2)
    )
    collage = Image.new("RGB", (collage_width, collage_height), "white")
    draw = ImageDraw.Draw(collage)
    draw_wrapped_centered_text(
        draw,
        padding,
        padding,
        collage_width - padding * 2,
        header_lines,
        prompt_font,
        prompt_line_height,
    )

    grid_x = row_label_width + padding * 2
    column_y = header_height + padding
    for col_index, value in enumerate(MULTIPLIER_VALUES):
        x = grid_x + col_index * (image_width + padding)
        draw_centered_text(
            draw,
            x,
            column_y,
            image_width,
            column_header_height,
            multiplier_label(value),
            label_font,
        )

    row_start_y = column_y + column_header_height + padding
    for row_index, (row_key, row_label, _mode, _token_indices) in enumerate(ROW_SPECS):
        y = row_start_y + row_index * (row_header_height + image_height + padding)
        draw_centered_text(
            draw,
            padding,
            y + row_header_height,
            row_label_width,
            image_height,
            row_label,
            small_font,
        )
        for col_index, value in enumerate(MULTIPLIER_VALUES):
            x = grid_x + col_index * (image_width + padding)
            image = generated_images[(row_key, value)].convert("RGB")
            collage.paste(image, (x, y + row_header_height))

    draw.text((padding, collage_height - padding - 18), f"seed: {seed}", fill=(60, 60, 60), font=small_font)
    return collage


def generate_sweep_for_seed(
    torch,
    pipe,
    args: argparse.Namespace,
    output_dir: Path,
    seed: int,
    embeddings: tuple,
    token_summary: dict,
) -> None:
    generated_images = {}
    for row_key, _row_label, mode, token_indices in ROW_SPECS:
        for multiplier in MULTIPLIER_VALUES:
            generated_images[(row_key, multiplier)] = generate_image(
                torch,
                pipe,
                args,
                seed,
                embeddings,
                mode,
                token_indices,
                multiplier,
            )

    output_path = output_dir / f"seed_{seed}_sweep.png"
    collage = make_collage(seed, generated_images, args.prompt, token_summary)
    collage.save(output_path)
    print(f"Saved collage {output_path}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    torch, StableDiffusion3Pipeline = import_dependencies()
    device = choose_device(torch, args.device)
    pipe = load_pipeline(torch, StableDiffusion3Pipeline, device)
    token_summary = build_token_summary(pipe, args.prompt)
    write_metadata(args, output_dir, device, dtype_name(torch, device), token_summary)
    embeddings = encode_prompt_embeddings(torch, pipe, args)

    for seed in args.seeds:
        generate_sweep_for_seed(torch, pipe, args, output_dir, seed, embeddings, token_summary)


if __name__ == "__main__":
    main()
