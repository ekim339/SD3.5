"""CUDA worker executed in TextCtrl's pinned legacy environment."""
from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    for name in ("repository", "checkpoint", "ocr-checkpoint", "manifest", "predictions"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--starting-layer", type=int, default=10)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def masked_style_features(features, job):
    """Zero complete style tokens, preserving the tensor's [B,256,768] shape."""
    import torch

    if features.ndim != 3 or tuple(features.shape[1:]) != (256, 768):
        raise RuntimeError("Expected TextCtrl style features [B,256,768], got %s" %
                           (tuple(features.shape),))
    result = features.clone()
    explicit = [int(value) for value in job.get("masked_token_indices", [])]
    if explicit:
        result[:, explicit, :] = 0
        return result
    proportion = float(job.get("masking_proportion", 0.0))
    count = round(proportion * result.shape[1])
    if count:
        generator = torch.Generator(device=result.device)
        generator.manual_seed(int(job["style_mask_seed"]))
        indices = torch.randperm(result.shape[1], generator=generator,
                                 device=result.device)[:count]
        result[:, indices, :] = 0
    return result


@contextmanager
def style_mask(control_model, job):
    """Temporarily mask the ViT output before TextCtrl builds its feature pyramid."""
    original = control_model.vit.forward

    def forward(*args, **kwargs):
        return masked_style_features(original(*args, **kwargs), job)

    control_model.vit.forward = forward
    try:
        yield
    finally:
        control_model.vit.forward = original


def generate(pipe, image, source, target, job, steps, layer, guidance):
    import torch
    import torchvision.transforms as transforms
    from src.MuSA.GaMuSA import glyph_cosine_similarity, prepare_label
    from src.MuSA.utils import MuSA_TextCtrl, regiter_attention_editor_diffusers_Edit

    with torch.no_grad():
        seed = int(job["generation_seed"])
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Invert the noisy source with intact style features. The intervention applies only
        # while editing, so each condition starts from the identical inverted latent.
        start, _ = pipe.inversion(image, image, [source], guidance_scale=guidance,
                                  num_inference_steps=steps, return_intermediates=True)
        prompts = [source, target]
        clean = pipe.model.get_text_conditioning(prompts)
        embeddings = torch.cat([pipe.model.get_text_conditioning(["", ""]), clean], 0)
        latents = start.expand(2, -1, -1, -1).clone()
        pipe.scheduler.set_timesteps(steps)
        ids, _ = prepare_label(prompts, pipe.charset, pipe.max_length, pipe.device)
        controller = MuSA_TextCtrl(24, layer)
        regiter_attention_editor_diffusers_Edit(pipe.unet, controller)
        controller.start_ctrl()
        try:
            with style_mask(pipe.control_model, job):
                for index, timestep in enumerate(pipe.scheduler.timesteps):
                    model_input = torch.cat([latents] * 2)
                    hint = torch.cat([image.expand(2, -1, -1, -1)] * 2)
                    control = pipe.control_model(hint, model_input, timestep, embeddings)
                    output = pipe.unet(x=model_input, timestep=timestep,
                                       encoder_hidden_states=embeddings, control=control).sample
                    unconditional, conditional = output.chunk(2)
                    latents, _ = pipe.step(unconditional + guidance * (conditional - unconditional),
                                           timestep, latents)
                    if (index + 1) % 5 == 0:
                        monitored = transforms.Resize([32, 128])(pipe.latent2image_grad(latents))
                        controller.reset_alpha(glyph_cosine_similarity(pipe.monitor(monitored), ids))
        finally:
            controller.reset_ctrl()
            controller.reset()
        return pipe.latent2image_grad(latents)[1].clamp(0, 1)


def main(argv=None):
    args = parse_args(argv)
    repository = Path(args.repository).resolve()
    os.chdir(repository)
    sys.path.insert(0, str(repository))
    import numpy as np
    import torch
    import torchvision.transforms as transforms
    from omegaconf import OmegaConf
    from PIL import Image
    from tqdm import tqdm
    from inference import load_image
    from src.MuSA.GaMuSA import GaMuSA
    from src.module.abinet import ABINetIterModel, CharsetMapper, postprocess
    from utils import create_model, load_state_dict

    if not torch.cuda.is_available():
        raise RuntimeError("TextCtrl style masking requires a CUDA GPU")
    jobs = read_jsonl(args.manifest)
    predictions_path = Path(args.predictions)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    completed = {} if args.overwrite or not predictions_path.is_file() else {
        int(row["index"]): row for row in read_jsonl(predictions_path)}
    if args.overwrite:
        predictions_path.write_text("", encoding="utf-8")

    model = create_model("configs/inference.yaml").cuda()
    model.load_state_dict(load_state_dict(args.checkpoint), strict=False)
    model.eval()
    pipe = GaMuSA(model, {"max_length": 25, "loss_weight": 1., "attention": "position",
                          "backbone": "transformer", "backbone_ln": 3,
                          "checkpoint": "weights/vision_model.pth",
                          "charset_path": "src/module/abinet/data/charset_36.txt"})
    cfg = OmegaConf.load("configs/inference.yaml").model.params.base_config.ocr_model
    ocr = ABINetIterModel(cfg).cuda()
    ocr.load_state_dict(torch.load(args.ocr_checkpoint, map_location="cpu"))
    ocr.eval()
    charset = CharsetMapper(filename=str(cfg.charset_path), max_length=int(cfg.max_length) + 1)
    resize, to_tensor = transforms.Resize([int(cfg.height), int(cfg.width)]), transforms.ToTensor()

    with predictions_path.open("a", encoding="utf-8", buffering=1) as output:
        for job in tqdm(jobs, desc="Style token masking"):
            index, destination = int(job["index"]), Path(job["output_path"])
            if args.overwrite or not destination.is_file():
                generated = generate(pipe, load_image(job["input_path"]), job["source_text"],
                                     job["target_text"], job, args.num_inference_steps,
                                     args.starting_layer, args.guidance_scale)
                array = np.clip(generated.cpu().permute(1, 2, 0).numpy() * 255, 0, 255).astype(np.uint8)
                with Image.open(job["input_path"]) as opened:
                    original_size = opened.size
                destination.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(array).resize(original_size, Image.Resampling.BICUBIC).save(destination)
            if not args.overwrite and index in completed:
                continue
            with Image.open(destination) as opened:
                ocr_input = resize(to_tensor(opened.convert("RGB"))).unsqueeze(0).cuda()
            with torch.no_grad():
                prediction = postprocess(ocr(ocr_input, mode="test"), charset, "alignment")[0][0]
            row = {"index": index, "ocr_predicted_text": prediction,
                   "output_path": str(destination)}
            output.write(json.dumps(row, sort_keys=True) + "\n")
            completed[index] = row


if __name__ == "__main__":
    main()
