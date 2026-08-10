"""Run TextCtrl generation and ABINet OCR in the pinned Python 3.8 runtime."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ocr-checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--starting-layer", type=int, required=True)
    parser.add_argument("--num-inference-steps", type=int, required=True)
    parser.add_argument("--guidance-scale", type=float, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _read_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _load_completed(path):
    completed = {}
    if not path.is_file():
        return completed
    for record in _read_jsonl(path):
        completed[int(record["index"])] = record
    return completed


def main(argv=None):
    args = parse_args(argv)
    repository = Path(args.repository).expanduser().resolve()
    os.chdir(str(repository))
    sys.path.insert(0, str(repository))

    import numpy as np
    import torch
    import torchvision.transforms as transforms
    from omegaconf import OmegaConf
    from PIL import Image
    from pytorch_lightning import seed_everything
    from tqdm import tqdm

    from inference import load_image
    from src.MuSA.GaMuSA import GaMuSA
    from src.MuSA.GaMuSA_app import text_editing
    from src.module.abinet import ABINetIterModel, CharsetMapper, postprocess
    from utils import create_model, load_state_dict

    if not torch.cuda.is_available():
        raise RuntimeError("TextCtrl evaluation requires a CUDA GPU.")
    records = _read_jsonl(Path(args.manifest))
    predictions_path = Path(args.predictions)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    completed = {} if args.overwrite else _load_completed(predictions_path)
    if args.overwrite:
        predictions_path.write_text("", encoding="utf-8")

    pending_generation = [
        record
        for record in records
        if args.overwrite or not Path(record["output_path"]).is_file()
    ]
    model = None
    pipeline = None
    if pending_generation:
        model = create_model("configs/inference.yaml").cuda()
        model.load_state_dict(load_state_dict(args.checkpoint), strict=False)
        model.eval()
        monitor_config = {
            "max_length": 25,
            "loss_weight": 1.0,
            "attention": "position",
            "backbone": "transformer",
            "backbone_ln": 3,
            "checkpoint": "weights/vision_model.pth",
            "charset_path": "src/module/abinet/data/charset_36.txt",
        }
        pipeline = GaMuSA(model, monitor_config)

    ocr_config = OmegaConf.load("configs/inference.yaml").model.params.base_config.ocr_model
    ocr_model = ABINetIterModel(ocr_config).cuda()
    ocr_state = torch.load(args.ocr_checkpoint, map_location="cpu")
    ocr_model.load_state_dict(ocr_state)
    ocr_model.eval()
    charset = CharsetMapper(
        filename=str(ocr_config.charset_path),
        max_length=int(ocr_config.max_length) + 1,
    )
    ocr_resize = transforms.Resize(
        [int(ocr_config.height), int(ocr_config.width)]
    )
    to_tensor = transforms.ToTensor()
    seed_everything(args.seed)

    with predictions_path.open("a", encoding="utf-8", buffering=1) as prediction_file:
        for record in tqdm(records, desc="TextCtrl + ABINet"):
            index = int(record["index"])
            output_path = Path(record["output_path"])
            if args.overwrite or not output_path.is_file():
                source_image = load_image(record["input_path"])
                result = text_editing(
                    pipeline,
                    source_image,
                    source_image,
                    record["source_text"],
                    record["target_text"],
                    starting_layer=args.starting_layer,
                    ddim_steps=args.num_inference_steps,
                    scale=args.guidance_scale,
                )
                with Image.open(record["input_path"]) as input_image:
                    original_size = input_image.size
                generated = Image.fromarray(
                    (result[1] * 255).clip(0, 255).astype(np.uint8)
                ).resize(original_size, Image.Resampling.BICUBIC)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                generated.save(output_path)

            if not args.overwrite and index in completed:
                continue
            with Image.open(output_path) as generated_image:
                ocr_input = ocr_resize(
                    to_tensor(generated_image.convert("RGB"))
                ).unsqueeze(0).cuda()
            with torch.no_grad():
                ocr_output = ocr_model(ocr_input, mode="test")
            predicted_text = postprocess(
                ocr_output,
                charset,
                "alignment",
            )[0][0]
            prediction = {
                "index": index,
                "image_title": record["image_title"],
                "ground_truth_text": record["target_text"],
                "ocr_predicted_text": predicted_text,
                "output_path": str(output_path),
            }
            prediction_file.write(json.dumps(prediction, sort_keys=True) + "\n")
            completed[index] = prediction


if __name__ == "__main__":
    main()

