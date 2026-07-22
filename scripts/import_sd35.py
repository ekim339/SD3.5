#!/usr/bin/env python3
"""Load Stable Diffusion 3.5 Medium from Hugging Face.

Before running, accept the model terms on Hugging Face and authenticate:

    huggingface-cli login

or set one of:

    HF_TOKEN=...
    HUGGINGFACE_HUB_TOKEN=...
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import/load Stable Diffusion 3.5 Medium with Diffusers."
    )
    parser.add_argument(
        "--prompt",
        default="A cinematic photo of a glass greenhouse on a rainy evening",
        help="Prompt to use when --generate is enabled.",
    )
    parser.add_argument(
        "--output",
        default="sd35_test.png",
        help="Output image path for --generate.",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate a test image after loading the pipeline.",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=("cuda", "mps", "cpu"),
        help="Device to use. Defaults to cuda, then mps, then cpu.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=28,
        help="Number of inference steps for --generate.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=3.5,
        help="Guidance scale for --generate.",
    )
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


def main() -> None:
    args = parse_args()
    torch, StableDiffusion3Pipeline = import_dependencies()
    device = choose_device(torch, args.device)
    pipe = load_pipeline(torch, StableDiffusion3Pipeline, device)

    print("Stable Diffusion 3.5 Medium imported successfully.")

    if not args.generate:
        return

    output_path = Path(args.output).expanduser().resolve()
    print(f"Generating image: {output_path}")
    image = pipe(
        args.prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
    ).images[0]
    image.save(output_path)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
