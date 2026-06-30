#!/usr/bin/env python3
"""Generate images with Stable Diffusion 3.5 Medium.

Before running, accept the model terms on Hugging Face and authenticate:

    huggingface-cli login

or set one of:

    HF_TOKEN=...
    HUGGINGFACE_HUB_TOKEN=...


Usage:
    CUDA_VISIBLE_DEVICES=0 python3 generate_sd35.py \
      --prompt "A bathroom mat that says 'hell' in bold capital letters, photorealistic" \
      --seeds 0 1 2 3 4 5 6 7 8 9 \
      --output-dir outputs/bathroom_hell \
      --trajectory
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate images with Stable Diffusion 3.5 Medium."
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Text prompt used for generation.",
    )
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
        default="outputs",
        help="Directory where generated images are saved.",
    )
    parser.add_argument(
        "--metadata-file",
        default="metadata.json",
        help="Metadata filename written inside output-dir before generation starts.",
    )
    parser.add_argument(
        "--trajectory",
        action="store_true",
        help="Save a decoded snapshot at every denoising step for each seed.",
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
        help="Number of denoising steps.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=3.5,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Generated image width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Generated image height.",
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


def dtype_name(torch, device: str) -> str:
    dtype = choose_dtype(torch, device)
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.float32:
        return "float32"
    return str(dtype)


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
        "trajectory": args.trajectory,
        "device": device,
        "requested_device": args.device,
        "dtype": dtype,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "width": args.width,
        "height": args.height,
        "metadata_file": args.metadata_file,
        "output_files": [
            str(Path(f"seed_{seed}") / "final.png")
            if args.trajectory else f"seed_{seed}.png"
            for seed in args.seeds
        ],
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


def decode_latents_to_pil(torch, pipe, latents):
    vae_config = pipe.vae.config
    scaling_factor = getattr(vae_config, "scaling_factor", 1.0)
    shift_factor = getattr(vae_config, "shift_factor", 0.0)
    vae_dtype = next(pipe.vae.parameters()).dtype

    with torch.no_grad():
        latents = latents.detach().to(dtype=vae_dtype)
        latents = (latents / scaling_factor) + shift_factor
        image = pipe.vae.decode(latents, return_dict=False)[0]
        image = pipe.image_processor.postprocess(image, output_type="pil")

    return image[0]


def make_trajectory_callback(torch, pipe, seed_dir: Path):
    seed_dir.mkdir(parents=True, exist_ok=True)

    def callback_on_step_end(_pipe, step: int, timestep, callback_kwargs):
        latents = callback_kwargs["latents"]
        timestep_value = timestep.item() if hasattr(timestep, "item") else timestep
        snapshot = decode_latents_to_pil(torch, pipe, latents)
        snapshot.save(
            seed_dir / f"step_{step + 1:04d}_timestep_{int(timestep_value):04d}.png"
        )
        return callback_kwargs

    return callback_on_step_end


def generate_for_seed(torch, pipe, args: argparse.Namespace, output_dir: Path, seed: int) -> None:
    seed_name = f"seed_{seed}"
    seed_dir = output_dir / seed_name
    final_path = output_dir / f"{seed_name}.png"
    generator = make_generator(torch, seed)

    generation_kwargs = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "width": args.width,
        "height": args.height,
        "generator": generator,
    }

    if args.trajectory:
        generation_kwargs["callback_on_step_end"] = make_trajectory_callback(
            torch, pipe, seed_dir
        )
        generation_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]

    print(f"Generating seed {seed}...")
    image = pipe(**generation_kwargs).images[0]

    if args.trajectory:
        seed_dir.mkdir(parents=True, exist_ok=True)
        final_path = seed_dir / "final.png"

    image.save(final_path)
    print(f"Saved {final_path}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    torch, StableDiffusion3Pipeline = import_dependencies()
    device = choose_device(torch, args.device)
    write_metadata(args, output_dir, device, dtype_name(torch, device))
    pipe = load_pipeline(torch, StableDiffusion3Pipeline, device)

    for seed in args.seeds:
        generate_for_seed(torch, pipe, args, output_dir, seed)


if __name__ == "__main__":
    main()
