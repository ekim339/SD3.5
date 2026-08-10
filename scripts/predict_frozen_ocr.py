"""Run TextCtrl's frozen ABINet OCR checkpoint over a directory of images."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--textctrl-dir", type=Path, default=Path("networks/external/TextCtrl"))
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    textctrl_dir = args.textctrl_dir.resolve()
    sys.path.insert(0, str(textctrl_dir))

    from src.module.abinet.abinet_base import postprocess, preprocess
    from src.module.abinet.modules.model_abinet_iter import ABINetIterModel
    from src.module.abinet.utils import CharsetMapper

    config = OmegaConf.load(textctrl_dir / "configs" / "inference.yaml")
    ocr_config = config.model.params.base_config.ocr_model
    charset_path = textctrl_dir / "src/module/abinet/data/charset_36.txt"
    checkpoint_path = textctrl_dir / "weights/ocr_model.pth"
    for branch in ("vision", "language", "alignment"):
        ocr_config[branch].charset_path = str(charset_path)
    ocr_config.charset_path = str(charset_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ABINetIterModel(ocr_config)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval().requires_grad_(False).to(device)
    charset = CharsetMapper(
        filename=str(charset_path), max_length=int(ocr_config.max_length) + 1
    )

    image_paths = sorted(
        path
        for path in args.image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    predictions: list[tuple[str, str]] = []
    with torch.inference_mode():
        for start in range(0, len(image_paths), args.batch_size):
            batch_paths = image_paths[start : start + args.batch_size]
            tensors = []
            for path in batch_paths:
                with Image.open(path) as image:
                    tensors.append(
                        preprocess(
                            image.convert("RGB"),
                            width=int(ocr_config.width),
                            height=int(ocr_config.height),
                        ).squeeze(0)
                    )
            outputs = model(torch.stack(tensors).to(device), mode="eval")
            texts, _scores, _lengths = postprocess(outputs, charset, "alignment")
            predictions.extend((path.name, text) for path, text in zip(batch_paths, texts))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("image_title", "ocr_prediction"))
        writer.writerows(predictions)
    print(f"Wrote {len(predictions)} predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
