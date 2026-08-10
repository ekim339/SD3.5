"""Run the pretrained TextCtrl ABINet OCR in its pinned Python 3.8 runtime."""

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
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _read_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main(argv=None):
    args = parse_args(argv)
    repository = Path(args.repository).expanduser().resolve()
    os.chdir(str(repository))
    sys.path.insert(0, str(repository))

    import torch
    import torchvision.transforms as transforms
    from omegaconf import OmegaConf
    from PIL import Image
    from tqdm import tqdm

    from src.module.abinet import ABINetIterModel, CharsetMapper, postprocess

    if not torch.cuda.is_available():
        raise RuntimeError("ABINet OCR evaluation requires a CUDA GPU.")
    records = _read_jsonl(Path(args.manifest))
    predictions_path = Path(args.predictions)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    completed = {}
    if predictions_path.is_file() and not args.overwrite:
        completed = {
            int(record["index"]): record
            for record in _read_jsonl(predictions_path)
        }
    if args.overwrite:
        predictions_path.write_text("", encoding="utf-8")

    ocr_config = OmegaConf.load("configs/inference.yaml").model.params.base_config.ocr_model
    model = ABINetIterModel(ocr_config).cuda()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()
    charset = CharsetMapper(
        filename=str(ocr_config.charset_path),
        max_length=int(ocr_config.max_length) + 1,
    )
    resize = transforms.Resize([int(ocr_config.height), int(ocr_config.width)])
    to_tensor = transforms.ToTensor()

    with predictions_path.open("a", encoding="utf-8", buffering=1) as handle:
        for record in tqdm(records, desc="ABINet OCR"):
            index = int(record["index"])
            if not args.overwrite and index in completed:
                continue
            output_path = Path(record["output_path"])
            if not output_path.is_file():
                raise FileNotFoundError(f"Missing generated crop: {output_path}")
            with Image.open(output_path) as image:
                tensor = resize(to_tensor(image.convert("RGB"))).unsqueeze(0).cuda()
            with torch.no_grad():
                output = model(tensor, mode="test")
            predicted = postprocess(output, charset, "alignment")[0][0]
            prediction = {
                "index": index,
                "image_title": record["image_title"],
                "ground_truth_text": record["target_text"],
                "ocr_predicted_text": predicted,
                "output_path": str(output_path),
            }
            handle.write(json.dumps(prediction, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

