"""Generate regular TextCtrl outputs in its pinned legacy environment."""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path


def arguments():
    parser = argparse.ArgumentParser()
    for name in ("repository", "checkpoint", "manifest"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--starting-layer", type=int, default=10)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def generate(pipeline, source, source_text, target_text, seed, steps, layer, guidance):
    import torch
    from src.MuSA.GaMuSA import glyph_cosine_similarity, prepare_label
    from src.MuSA.utils import MuSA_TextCtrl, regiter_attention_editor_diffusers_Edit
    import torchvision.transforms as transforms
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    with torch.no_grad():
        start_code, _ = pipeline.inversion(source, source, [source_text],
                                            guidance_scale=guidance,
                                            num_inference_steps=steps,
                                            return_intermediates=True)
        prompts = [source_text, target_text]
        clean = pipeline.model.get_text_conditioning(prompts)
        uncond = pipeline.model.get_text_conditioning(["", ""])
        embeddings = torch.cat([uncond, clean], dim=0)
        latents = start_code.expand(2, -1, -1, -1).clone()
        pipeline.scheduler.set_timesteps(steps)
        labels, _ = prepare_label(prompts, pipeline.charset, pipeline.max_length, pipeline.device)
        controller = MuSA_TextCtrl(24, layer)
        regiter_attention_editor_diffusers_Edit(pipeline.unet, controller)
        controller.start_ctrl()
        try:
            for index, timestep in enumerate(pipeline.scheduler.timesteps):
                latent_input = torch.cat([latents] * 2)
                hint = torch.cat([source.expand(2, -1, -1, -1)] * 2)
                control = pipeline.control_model(hint, latent_input, timestep, embeddings)
                predicted = pipeline.unet(x=latent_input, timestep=timestep,
                                          encoder_hidden_states=embeddings,
                                          control=control).sample
                unconditional, conditional = predicted.chunk(2)
                predicted = unconditional + guidance * (conditional - unconditional)
                latents, _ = pipeline.step(predicted, timestep, latents)
                if (index + 1) % 5 == 0:
                    monitor = transforms.Resize([32, 128])(pipeline.latent2image_grad(latents))
                    controller.reset_alpha(glyph_cosine_similarity(pipeline.monitor(monitor), labels))
        finally:
            controller.reset_ctrl(); controller.reset()
        return pipeline.latent2image_grad(latents)[1].clamp(0, 1)


def main():
    args = arguments(); repository = Path(args.repository).resolve()
    os.chdir(str(repository)); sys.path.insert(0, str(repository))
    import numpy as np, torch
    from PIL import Image
    from inference import load_image
    from src.MuSA.GaMuSA import GaMuSA
    from utils import create_model, load_state_dict
    if not torch.cuda.is_available():
        raise RuntimeError("Regular TextCtrl generation requires CUDA")
    jobs = [json.loads(line) for line in Path(args.manifest).read_text().splitlines()
            if line.strip() and json.loads(line)["model"] == "regular"]
    model = create_model("configs/inference.yaml").cuda()
    model.load_state_dict(load_state_dict(args.checkpoint), strict=False); model.eval()
    pipeline = GaMuSA(model, {"max_length": 25, "loss_weight": 1.0,
        "attention": "position", "backbone": "transformer", "backbone_ln": 3,
        "checkpoint": "weights/vision_model.pth",
        "charset_path": "src/module/abinet/data/charset_36.txt"})
    from tqdm import tqdm
    for job in tqdm(jobs, desc="Regular TextCtrl"):
        destination = Path(job["output_path"])
        if destination.is_file() and not args.overwrite:
            continue
        generated = generate(pipeline, load_image(job["input_path"]), job["source_text"],
                             job["target_text"], int(job["generation_seed"]),
                             args.num_inference_steps, args.starting_layer, args.guidance_scale)
        array = (generated.cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        with Image.open(job["input_path"]) as opened: size = opened.size
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(array).resize(size, Image.Resampling.BICUBIC).save(destination)


if __name__ == "__main__": main()
