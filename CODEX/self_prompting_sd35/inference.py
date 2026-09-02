"""Edit one text region with a trained self-prompting LoRA."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import StableDiffusion3Pipeline
from PIL import Image

from .conditioning import encode_t5_target_conditioning
from .dataset import prepare_conditions
from .model import SelfPromptingSD35


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--text", required=True, help="Target replacement text")
    parser.add_argument("--lora", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("SD3.5 inference requires a CUDA device")
    device, dtype = torch.device("cuda"), torch.bfloat16
    pipe = StableDiffusion3Pipeline.from_pretrained(args.model, dtype=dtype).to(device)
    model = SelfPromptingSD35(pipe).to(device).eval()
    model.load_lora_weights(args.lora)
    with Image.open(args.image) as opened:
        source = opened.convert("RGB")
    with Image.open(args.mask) as opened:
        mask_image = opened.convert("L")
    prepared = prepare_conditions(source, mask_image, args.text, args.resolution)
    tensors = {key: value.unsqueeze(0).to(device, dtype) for key, value in prepared.items()}
    prompt, pooled = encode_t5_target_conditioning(
        pipe, args.text, device=device, max_sequence_length=256,
    )
    masked = model.encode_images(tensors["masked_image"], sample=False)
    glyph = model.encode_images(tensors["glyph_image"], sample=False)
    style = model.encode_images(tensors["style_image"], sample=False)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    latent = torch.randn(masked.shape, generator=generator, device=device, dtype=dtype)
    pipe.scheduler.set_timesteps(args.steps, device=device)
    latent = latent * getattr(pipe.scheduler, "init_noise_sigma", 1.0)
    for timestep in pipe.scheduler.timesteps:
        prediction = model.transformer(
            hidden_states=model.composite_input(latent, masked, glyph, style, tensors["mask"]),
            timestep=timestep.expand(latent.shape[0]), encoder_hidden_states=prompt,
            pooled_projections=pooled, return_dict=True,
        ).sample
        latent = pipe.scheduler.step(prediction, timestep, latent, return_dict=False)[0]
    shift = getattr(pipe.vae.config, "shift_factor", 0.0) or 0.0
    decoded = pipe.vae.decode(latent / pipe.vae.config.scaling_factor + shift, return_dict=False)[0]
    edited = decoded.add(1).div(2).clamp(0, 1)
    source_tensor = tensors["source_image"].add(1).div(2)
    edited = edited * tensors["mask"] + source_tensor * (1.0 - tensors["mask"])
    image = (edited[0].float().cpu().permute(1, 2, 0).numpy() * 255).round().astype("uint8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(args.output)


if __name__ == "__main__":
    main()
