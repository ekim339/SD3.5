#!/usr/bin/env python3
"""Generate SD3.5 images after scaling selected token embedding rows.

Pass token indices with --tokens. Each index is applied to both token spaces
where present:

    - CLIP token row index i in prompt_embeds[:, i, :]
    - T5 token row index i in prompt_embeds[:, 77 + i, :]

The pooled CLIP embeddings are not modified because they are not token-specific.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SD3.5 images after amplifying or zeroing selected token signals."
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
        "--tokens",
        type=int,
        nargs="+",
        required=True,
        help="Token indices to scale, for example: --tokens 4 5 6",
    )
    parser.add_argument(
        "--multiple",
        type=float,
        default=1.0,
        help="Multiplier for selected token signals, for example 2, 3, or 0.5.",
    )
    parser.add_argument(
        "--zero-out",
        action="store_true",
        help="Set selected token signals to zero. Overrides --multiple.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs_amplified",
        help="Directory where generated images and metadata are saved.",
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


def validate_args(args: argparse.Namespace) -> None:
    if any(token_index < 0 for token_index in args.tokens):
        raise ValueError("--tokens must contain non-negative token indices.")
    if args.steps <= 0:
        raise ValueError("--steps must be > 0.")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be > 0.")


def effective_multiplier(args: argparse.Namespace) -> float:
    return 0.0 if args.zero_out else args.multiple


def expected_output_files(seeds: list[int]) -> list[str]:
    return [f"seed_{seed}.png" for seed in seeds]


def write_metadata(args: argparse.Namespace, output_dir: Path, device: str, dtype: str) -> Path:
    metadata_path = output_dir / args.metadata_file
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "seeds": args.seeds,
        "tokens": args.tokens,
        "zero_out": args.zero_out,
        "requested_multiple": args.multiple,
        "effective_multiple": effective_multiplier(args),
        "token_mapping": {
            "clip": "token index i scales prompt_embeds[:, i, :] when i < 77",
            "t5": "token index i scales prompt_embeds[:, 77 + i, :] when present",
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
        )


def clone_embeddings(embeddings: tuple):
    return tuple(tensor.clone() if tensor is not None else None for tensor in embeddings)


def scaled_prompt_embeddings(embeddings: tuple, token_indices: list[int], multiplier: float) -> tuple:
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = clone_embeddings(embeddings)

    sequence_length = prompt_embeds.shape[1]
    for token_index in token_indices:
        if token_index < CLIP_SEQUENCE_LENGTH and token_index < sequence_length:
            prompt_embeds[:, token_index, :] *= multiplier

        t5_position = CLIP_SEQUENCE_LENGTH + token_index
        if t5_position < sequence_length:
            prompt_embeds[:, t5_position, :] *= multiplier

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
    output_dir: Path,
    seed: int,
    embeddings: tuple,
) -> None:
    output_path = output_dir / f"seed_{seed}.png"
    generator = make_generator(torch, seed)
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = scaled_prompt_embeddings(embeddings, args.tokens, effective_multiplier(args))

    print(f"Generating seed {seed}...")
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
    image.save(output_path)
    print(f"Saved {output_path}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    torch, StableDiffusion3Pipeline = import_dependencies()
    device = choose_device(torch, args.device)
    write_metadata(args, output_dir, device, dtype_name(torch, device))
    pipe = load_pipeline(torch, StableDiffusion3Pipeline, device)
    embeddings = encode_prompt_embeddings(torch, pipe, args)

    for seed in args.seeds:
        generate_for_seed(torch, pipe, args, output_dir, seed, embeddings)


if __name__ == "__main__":
    main()
