"""Generate TextCtrl outputs in TextCtrl's pinned legacy CUDA environment."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


MODEL_NAME = "textctrl"
VALID_MODELS = {MODEL_NAME, "self_prompting_sd35"}
REQUIRED_JOB_FIELDS = {
    "index",
    "model",
    "input_path",
    "output_path",
    "source_text",
    "target_text",
    "generation_seed",
}


def _positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _nonnegative_int(value):
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return parsed


def _nonnegative_float(value):
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate manifest jobs labelled 'textctrl'."
    )
    parser.add_argument("--repository", required=True, help="TextCtrl repository root")
    parser.add_argument("--checkpoint", required=True, help="TextCtrl model.pth")
    parser.add_argument("--manifest", required=True, help="JSONL job manifest")
    parser.add_argument("--starting-layer", type=_nonnegative_int, default=10)
    parser.add_argument("--num-inference-steps", type=_positive_int, default=50)
    parser.add_argument("--guidance-scale", type=_nonnegative_float, default=2.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _resolve(value, base):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _require_file(path, description):
    if not path.is_file():
        raise FileNotFoundError("{} does not exist or is not a file: {}".format(description, path))
    if path.stat().st_size == 0:
        raise ValueError("{} is empty: {}".format(description, path))


def _read_jobs(manifest, path_base):
    jobs = []
    seen_indices = set()
    seen_outputs = set()
    for line_number, raw in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not raw.strip():
            continue
        try:
            job = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Invalid JSON in manifest {} at line {}: {}".format(
                    manifest, line_number, exc
                )
            ) from exc
        if not isinstance(job, dict):
            raise TypeError(
                "Manifest {} line {} must contain a JSON object".format(
                    manifest, line_number
                )
            )
        model_name = str(job.get("model", ""))
        if model_name not in VALID_MODELS:
            raise ValueError(
                "Manifest {} line {} has unsupported model {!r}; expected one of {}".format(
                    manifest, line_number, model_name, sorted(VALID_MODELS)
                )
            )
        if model_name != MODEL_NAME:
            continue
        missing = sorted(REQUIRED_JOB_FIELDS.difference(job))
        if missing:
            raise ValueError(
                "Manifest {} line {} is missing fields: {}".format(
                    manifest, line_number, ", ".join(missing)
                )
            )
        try:
            index = int(job["index"])
            seed = int(job["generation_seed"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Manifest {} line {} has a non-integer index or generation_seed".format(
                    manifest, line_number
                )
            ) from exc
        if index in seen_indices:
            raise ValueError("Duplicate TextCtrl job index {} in {}".format(index, manifest))
        input_path = _resolve(job["input_path"], path_base)
        output_path = _resolve(job["output_path"], path_base)
        if output_path in seen_outputs:
            raise ValueError(
                "Multiple TextCtrl jobs write the same output: {}".format(output_path)
            )
        if not str(job["source_text"]):
            raise ValueError("TextCtrl job {} has an empty source_text".format(index))
        if not str(job["target_text"]):
            raise ValueError("TextCtrl job {} has an empty target_text".format(index))
        if any(character.isspace() for character in str(job["source_text"])):
            raise ValueError("TextCtrl job {} source_text contains whitespace".format(index))
        if any(character.isspace() for character in str(job["target_text"])):
            raise ValueError("TextCtrl job {} target_text contains whitespace".format(index))
        seen_indices.add(index)
        seen_outputs.add(output_path)
        job = dict(job)
        job["index"] = index
        job["generation_seed"] = seed
        job["_input_path"] = input_path
        job["_output_path"] = output_path
        jobs.append(job)
    return jobs


def generate(pipeline, source, source_text, target_text, seed, steps, layer, guidance):
    """Generate one paired TextCtrl edit and always release its controller."""
    import torch
    import torchvision.transforms as transforms
    from src.MuSA.GaMuSA import glyph_cosine_similarity, prepare_label
    from src.MuSA.utils import MuSA_TextCtrl, regiter_attention_editor_diffusers_Edit

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.no_grad():
        start_code, _ = pipeline.inversion(
            source,
            source,
            [source_text],
            guidance_scale=guidance,
            num_inference_steps=steps,
            return_intermediates=True,
        )
        prompts = [source_text, target_text]
        clean = pipeline.model.get_text_conditioning(prompts)
        unconditional = pipeline.model.get_text_conditioning(["", ""])
        embeddings = torch.cat([unconditional, clean], dim=0)
        latents = start_code.expand(2, -1, -1, -1).clone()
        pipeline.scheduler.set_timesteps(steps)
        labels, _ = prepare_label(
            prompts, pipeline.charset, pipeline.max_length, pipeline.device
        )

        controller = MuSA_TextCtrl(24, layer)
        try:
            regiter_attention_editor_diffusers_Edit(pipeline.unet, controller)
            controller.start_ctrl()
            resize_for_monitor = transforms.Resize([32, 128])
            for step_index, timestep in enumerate(pipeline.scheduler.timesteps):
                latent_input = torch.cat([latents] * 2)
                hint = torch.cat([source.expand(2, -1, -1, -1)] * 2)
                control = pipeline.control_model(
                    hint, latent_input, timestep, embeddings
                )
                predicted = pipeline.unet(
                    x=latent_input,
                    timestep=timestep,
                    encoder_hidden_states=embeddings,
                    control=control,
                ).sample
                unconditioned, conditioned = predicted.chunk(2)
                predicted = unconditioned + guidance * (
                    conditioned - unconditioned
                )
                latents, _ = pipeline.step(predicted, timestep, latents)
                if (step_index + 1) % 5 == 0:
                    monitor_input = resize_for_monitor(
                        pipeline.latent2image_grad(latents)
                    )
                    score = glyph_cosine_similarity(
                        pipeline.monitor(monitor_input), labels
                    )
                    controller.reset_alpha(score)
        finally:
            controller.reset_ctrl()
            controller.reset()

        return pipeline.latent2image_grad(latents)[1].clamp(0, 1)


def main(argv=None):
    args = arguments(argv)
    invocation_dir = Path.cwd()
    repository = _resolve(args.repository, invocation_dir)
    checkpoint = _resolve(args.checkpoint, invocation_dir)
    manifest = _resolve(args.manifest, invocation_dir)

    _require_file(repository / "inference.py", "TextCtrl inference module")
    _require_file(repository / "configs" / "inference.yaml", "TextCtrl inference config")
    _require_file(repository / "weights" / "vision_model.pth", "TextCtrl vision monitor")
    _require_file(
        repository / "src" / "module" / "abinet" / "data" / "charset_36.txt",
        "TextCtrl monitor charset",
    )
    _require_file(checkpoint, "TextCtrl checkpoint")
    _require_file(manifest, "job manifest")

    jobs = _read_jobs(manifest, manifest.parent)
    runnable = [
        job
        for job in jobs
        if args.overwrite or not job["_output_path"].is_file()
    ]
    if not runnable:
        print("No pending TextCtrl jobs ({} already complete).".format(len(jobs)))
        return
    for job in runnable:
        _require_file(job["_input_path"], "input for job {}".format(job["index"]))

    os.chdir(str(repository))
    sys.path.insert(0, str(repository))

    import numpy as np
    import torch
    from PIL import Image
    from tqdm import tqdm
    from inference import load_image
    from src.MuSA.GaMuSA import GaMuSA
    from utils import create_model, load_state_dict

    if not torch.cuda.is_available():
        raise RuntimeError(
            "TextCtrl generation requires a CUDA GPU and a CUDA-enabled PyTorch "
            "installation; run this worker with the pinned TextCtrl environment."
        )
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    model = create_model("configs/inference.yaml").cuda()
    model.load_state_dict(load_state_dict(str(checkpoint)), strict=False)
    model.eval()
    pipeline = GaMuSA(
        model,
        {
            "max_length": 25,
            "loss_weight": 1.0,
            "attention": "position",
            "backbone": "transformer",
            "backbone_ln": 3,
            "checkpoint": "weights/vision_model.pth",
            "charset_path": "src/module/abinet/data/charset_36.txt",
        },
    )
    pipeline.monitor.eval()

    for job in tqdm(runnable, desc="TextCtrl"):
        source_path = job["_input_path"]
        destination = job["_output_path"]
        try:
            with Image.open(source_path) as opened:
                original_size = opened.size
            source = load_image(str(source_path))
            generated = generate(
                pipeline,
                source,
                str(job["source_text"]),
                str(job["target_text"]),
                int(job["generation_seed"]),
                args.num_inference_steps,
                args.starting_layer,
                args.guidance_scale,
            )
            pixels = (
                generated.detach().cpu().permute(1, 2, 0).numpy() * 255.0
            ).clip(0, 255).astype(np.uint8)
            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(pixels).resize(
                original_size, Image.Resampling.BICUBIC
            ).save(destination)
        except Exception as exc:
            raise RuntimeError(
                "TextCtrl job {} failed for input {} -> {}".format(
                    job["index"], source_path, destination
                )
            ) from exc


if __name__ == "__main__":
    main()
