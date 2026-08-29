"""Generate self-prompting outputs in the modern SD1.5 environment."""
from __future__ import annotations
import argparse, json
from pathlib import Path


def arguments():
    parser = argparse.ArgumentParser()
    for name in ("manifest", "checkpoint", "vae-path", "unet-path", "scheduler-path",
                 "text-model-path", "font-path"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-text-length", type=int, default=77)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def glyph_tensor(text, font_path, size):
    from PIL import Image, ImageDraw, ImageFont
    from torchvision.transforms import functional as TF
    canvas = Image.new("L", (size, size), 0); draw = ImageDraw.Draw(canvas)
    font = None
    for font_size in range(size // 2, 7, -2):
        candidate = ImageFont.truetype(font_path, font_size)
        box = draw.textbbox((0, 0), text, font=candidate)
        if box[2] - box[0] <= int(size * .9) and box[3] - box[1] <= size // 2:
            font = candidate; break
    font = font or ImageFont.truetype(font_path, 8)
    box = draw.textbbox((0, 0), text, font=font)
    position = ((size - (box[2]-box[0]))//2-box[0], (size-(box[3]-box[1]))//2-box[1])
    draw.text(position, text, fill=255, font=font)
    return TF.to_tensor(canvas.convert("RGB"))


def load_inputs(job, size):
    from PIL import Image
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as TF
    source_image = Image.open(job["input_path"]).convert("RGB")
    original_size = source_image.size
    source = TF.to_tensor(TF.resize(source_image, [size, size], InterpolationMode.BILINEAR,
                                    antialias=True)).unsqueeze(0)
    mask_image = Image.open(job["mask_path"]).convert("L")
    mask = (TF.to_tensor(TF.resize(mask_image, [size, size], InterpolationMode.NEAREST))
            .unsqueeze(0) >= .5).float()
    return source, mask, original_size


def main():
    args = arguments()
    import torch
    from diffusers import DDIMScheduler
    from PIL import Image
    from torchvision.transforms import functional as TF
    from tqdm import tqdm
    from experiments.self_prompting_textctrl.self_prompting_model import SelfPromptingSD15
    if not torch.cuda.is_available():
        raise RuntimeError("Self-prompting generation requires CUDA")
    device, dtype = torch.device("cuda"), torch.float16
    model = SelfPromptingSD15(args.vae_path, args.unet_path, args.scheduler_path,
                              args.text_model_path, args.revision, args.max_text_length)
    model.load_checkpoint(args.checkpoint); model.to(device=device, dtype=dtype).eval()
    scheduler = DDIMScheduler.from_config(model.noise_scheduler.config)
    scheduler.set_timesteps(args.num_inference_steps, device=device)
    jobs = [json.loads(line) for line in Path(args.manifest).read_text().splitlines()
            if line.strip() and json.loads(line)["model"] == "self_prompting"]
    with torch.inference_mode():
        for job in tqdm(jobs, desc="Self-prompting TextCtrl"):
            destination = Path(job["output_path"])
            if destination.is_file() and not args.overwrite:
                continue
            source_01, mask, original_size = load_inputs(job, args.image_size)
            source_01, mask = source_01.to(device), mask.to(device)
            source = source_01 * 2 - 1
            masked = source_01 * (1-mask) * 2 - 1
            glyph = glyph_tensor(job["target_text"], args.font_path, args.image_size)
            glyph = glyph.unsqueeze(0).to(device) * 2 - 1
            conditions = model.visual_conditions(masked, glyph, source, mask)
            conditional = model.encode_prompts([job["target_text"]])
            unconditional = model.encode_prompts([""])
            generator = torch.Generator(device=device).manual_seed(int(job["generation_seed"]))
            latents = torch.randn((1, 4, args.image_size//8, args.image_size//8),
                                  generator=generator, device=device, dtype=dtype)
            latents *= scheduler.init_noise_sigma
            for timestep in scheduler.timesteps:
                latent_input = scheduler.scale_model_input(latents, timestep)
                predicted = model.predict_noise(torch.cat([latent_input]*2), timestep,
                    torch.cat([unconditional, conditional]),
                    tuple(torch.cat([item, item]).to(dtype) for item in conditions))
                uncond, cond = predicted.chunk(2)
                latents = scheduler.step(uncond + args.guidance_scale*(cond-uncond),
                                         timestep, latents).prev_sample
            decoded = model.vae.decode(latents / model.vae.config.scaling_factor).sample
            generated = (decoded.float()/2+.5).clamp(0,1)
            result = generated*mask + source_01*(1-mask)
            destination.parent.mkdir(parents=True, exist_ok=True)
            TF.to_pil_image(result[0].cpu()).resize(original_size, Image.Resampling.BICUBIC).save(destination)


if __name__ == "__main__": main()
