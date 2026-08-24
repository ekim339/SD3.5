from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import DDIMScheduler
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm

from dataset import GlyphRenderer
from model import SelfPromptingSD15
from utils import load_config, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Edit a cropped text image")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--mask", required=True, help="White pixels indicate the edit region")
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_rgb(path: str, size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = TF.resize(image, [size, size], InterpolationMode.BILINEAR, antialias=True)
    return TF.to_tensor(image).unsqueeze(0)


def load_mask(path: str, size: int) -> torch.Tensor:
    mask = Image.open(path).convert("L")
    mask = TF.resize(mask, [size, size], InterpolationMode.NEAREST)
    return (TF.to_tensor(mask).unsqueeze(0) >= 0.5).float()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model_config = config["model"]
    infer_config = config["inference"]
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    model = SelfPromptingSD15(**{
        key: model_config[key]
        for key in (
            "vae_path", "unet_path", "scheduler_path", "text_model_path",
            "revision", "max_text_length", "conditioning_dropout",
        )
    })
    model.load_unet_checkpoint(args.checkpoint)
    model.to(device=device, dtype=dtype).eval()
    scheduler = DDIMScheduler.from_config(model.noise_scheduler.config)
    scheduler.set_timesteps(infer_config["num_inference_steps"], device=device)

    size = model_config["image_size"]
    source_01 = load_rgb(args.source, size).to(device)
    mask = load_mask(args.mask, size).to(device)
    source = source_01 * 2.0 - 1.0
    masked_source = (source_01 * (1.0 - mask)) * 2.0 - 1.0
    glyph = GlyphRenderer(config["data"]["font_path"], size)(args.text)
    glyph = glyph.unsqueeze(0).to(device) * 2.0 - 1.0
    visual = model.encode_visual_conditions(masked_source, glyph, source, mask)
    conditional = model.encode_prompts([args.text])
    unconditional = model.encode_prompts([""])

    generator = torch.Generator(device=device).manual_seed(args.seed)
    latent_shape = (1, 4, size // 8, size // 8)
    latents = torch.randn(latent_shape, generator=generator, device=device, dtype=dtype)
    latents *= scheduler.init_noise_sigma
    guidance_scale = float(infer_config["guidance_scale"])
    for timestep in tqdm(scheduler.timesteps):
        latent_input = scheduler.scale_model_input(latents, timestep)
        latent_input = torch.cat([latent_input, latent_input])
        visual_input = tuple(torch.cat([item, item]).to(dtype=dtype) for item in visual)
        embeddings = torch.cat([unconditional, conditional])
        noise = model.predict_noise(latent_input, timestep, embeddings, visual_input)
        noise_uncond, noise_cond = noise.chunk(2)
        guided_noise = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        latents = scheduler.step(guided_noise, timestep, latents).prev_sample

    decoded = model.vae.decode(latents / model.vae.config.scaling_factor).sample
    generated = (decoded.float() / 2.0 + 0.5).clamp(0, 1)
    # Preserve source pixels outside the requested edit region exactly.
    result = generated * mask + source_01 * (1.0 - mask)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    TF.to_pil_image(result[0].cpu()).save(output)


if __name__ == "__main__":
    main()
