#!/usr/bin/env python3
"""Classify an image with a checkpoint from train_image_classifier.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from train_image_classifier import SmallCNN, choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify one image with a trained classifier.")
    parser.add_argument("--checkpoint", required=True, help="Path to classifier.pt.")
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument(
        "--device",
        choices=("cuda", "mps", "cpu"),
        default=None,
        help="Device to use. Defaults to cuda, then mps, then cpu.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of plain text.",
    )
    return parser.parse_args()


def load_image(path: Path, image_size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
    tensor = torch.tensor(list(image.getdata()), dtype=torch.float32)
    tensor = tensor.view(image_size, image_size, 3).permute(2, 0, 1) / 255.0
    tensor = (tensor - 0.5) / 0.5
    return tensor.unsqueeze(0)


def probability_dict(labels: list[str], probabilities: torch.Tensor) -> dict[str, float]:
    return {
        label: round(float(probability), 6)
        for label, probability in zip(labels, probabilities.detach().cpu().tolist())
    }


def load_checkpoint(path: Path, device: str) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    image_path = Path(args.image).expanduser().resolve()
    device = choose_device(args.device)

    checkpoint = load_checkpoint(checkpoint_path, device)
    class_names = checkpoint["class_names"]
    image_size = int(checkpoint.get("image_size", 224))

    model = SmallCNN(num_text_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image = load_image(image_path, image_size).to(device)
    with torch.no_grad():
        outputs = model(image)

    object_probs = torch.softmax(outputs["object_logits"][0], dim=0)
    text_probs = torch.softmax(outputs["text_logits"][0], dim=0)
    text_class_probs = torch.softmax(outputs["text_class_logits"][0], dim=0)

    object_labels = ["object_wrong", "object_correct"]
    text_labels = ["text_wrong", "text_correct"]
    object_prediction = object_labels[int(object_probs.argmax().item())]
    text_prediction = text_labels[int(text_probs.argmax().item())]
    text_class_prediction = class_names[int(text_class_probs.argmax().item())]

    result = {
        "image": str(image_path),
        "checkpoint": str(checkpoint_path),
        "object_prediction": object_prediction,
        "text_prediction": text_prediction,
        "text_class_prediction": text_class_prediction,
        "object_probabilities": probability_dict(object_labels, object_probs),
        "text_probabilities": probability_dict(text_labels, text_probs),
        "text_class_probabilities": probability_dict(class_names, text_class_probs),
    }

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print(f"image: {result['image']}")
    print(f"object: {object_prediction}")
    for label, probability in result["object_probabilities"].items():
        print(f"  {label}: {probability:.4f}")
    print(f"text accurate: {text_prediction}")
    for label, probability in result["text_probabilities"].items():
        print(f"  {label}: {probability:.4f}")
    print(f"text class: {text_class_prediction}")
    for label, probability in result["text_class_probabilities"].items():
        print(f"  {label}: {probability:.4f}")


if __name__ == "__main__":
    main()
