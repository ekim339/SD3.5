#!/usr/bin/env python3
"""Train a multitask image classifier from folder-level labels.

The config file assigns folders to binary object/text labels and optional
rendered-text classes. Images are loaded from all listed folders, shuffled during
training, and optimized with three heads:

    object accuracy: correct / wrong
    text accuracy: correct / wrong
    text class: hello / hello! / HELLO / helllo / ...

Text-class loss is used only for samples that have a text-class label.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IGNORE_INDEX = -100


@dataclass(frozen=True)
class Sample:
    path: Path
    object_label: int
    text_label: int
    text_class_label: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train folder-labeled multitask image classifier.")
    parser.add_argument("--config", required=True, help="Path to dataset JSON config.")
    parser.add_argument("--output-dir", default="classifier_runs", help="Directory for checkpoints.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default=None)
    return parser.parse_args()


def choose_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def read_config(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    try:
        config = json.loads(text)
    except json.JSONDecodeError as json_error:
        try:
            config = json.loads(strip_json_comments_and_trailing_commas(text))
        except json.JSONDecodeError:
            try:
                config = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                line = text.splitlines()[json_error.lineno - 1] if text.splitlines() else ""
                pointer = " " * max(0, json_error.colno - 1) + "^"
                raise ValueError(
                    f"Could not parse config file: {path}\n"
                    f"JSON error at line {json_error.lineno}, column {json_error.colno}: "
                    f"{json_error.msg}\n\n"
                    f"{line}\n"
                    f"{pointer}\n\n"
                    "Use valid JSON with double-quoted keys/strings, or a Python dict literal.\n"
                    "Common JSON mistakes: single quotes, unquoted keys, and trailing commas."
                ) from json_error

    if not isinstance(config, dict):
        raise ValueError(f"Config must be a JSON object/dict, got {type(config).__name__}.")
    return config


def strip_json_comments_and_trailing_commas(text: str) -> str:
    stripped_lines = []
    in_block_comment = False
    for line in text.splitlines():
        cleaned = []
        index = 0
        in_string = False
        quote = ""
        while index < len(line):
            char = line[index]
            next_char = line[index + 1] if index + 1 < len(line) else ""
            if in_block_comment:
                if char == "*" and next_char == "/":
                    in_block_comment = False
                    index += 2
                else:
                    index += 1
                continue
            if not in_string and char == "/" and next_char == "*":
                in_block_comment = True
                index += 2
                continue
            if not in_string and char == "/" and next_char == "/":
                break
            if char in {'"', "'"} and (index == 0 or line[index - 1] != "\\"):
                if in_string and char == quote:
                    in_string = False
                    quote = ""
                elif not in_string:
                    in_string = True
                    quote = char
            cleaned.append(char)
            index += 1
        stripped_lines.append("".join(cleaned))

    without_comments = "\n".join(stripped_lines)
    return re.sub(r",\s*([}\]])", r"\1", without_comments)


def resolve_folder_list(config: dict, key: str, base_dir: Path) -> list[Path]:
    folders = []
    for folder in config.get(key, []):
        folder_path = Path(folder).expanduser()
        if not folder_path.is_absolute():
            folder_path = base_dir / folder_path
        folders.append(folder_path.resolve())
    return folders


def image_paths(folder: Path) -> list[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Expected folder, got file: {folder}")
    return sorted(path for path in folder.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def add_folder_labels(
    label_by_path: dict[Path, dict],
    folders: list[Path],
    label_key: str,
    label_value: int,
) -> None:
    for folder in folders:
        for path in image_paths(folder):
            labels = label_by_path.setdefault(path, {})
            if label_key in labels and labels[label_key] != label_value:
                raise ValueError(f"Conflicting {label_key} labels for image: {path}")
            labels[label_key] = label_value


def build_samples(config: dict, config_path: Path) -> tuple[list[Sample], list[str]]:
    base_dir = config_path.parent.resolve()
    label_by_path: dict[Path, dict] = {}

    object_correct = resolve_folder_list(config, "object_correct_folders", base_dir)
    object_wrong = resolve_folder_list(config, "object_wrong_folders", base_dir)
    text_correct = resolve_folder_list(config, "text_correct_folders", base_dir)
    text_wrong = resolve_folder_list(config, "text_wrong_folders", base_dir)

    text_classes = config.get("text_classes", {})
    class_names = list(text_classes.keys())
    for class_index, class_name in enumerate(class_names):
        folders = []
        for folder in text_classes[class_name]:
            folder_path = Path(folder).expanduser()
            if not folder_path.is_absolute():
                folder_path = base_dir / folder_path
            folders.append(folder_path.resolve())
        add_folder_labels(label_by_path, folders, "text_class_label", class_index)

    add_folder_labels(label_by_path, object_correct, "object_label", 1)
    add_folder_labels(label_by_path, object_wrong, "object_label", 0)
    add_folder_labels(label_by_path, text_correct, "text_label", 1)
    add_folder_labels(label_by_path, text_wrong, "text_label", 0)

    samples = []
    for path, labels in sorted(label_by_path.items()):
        if "object_label" not in labels:
            raise ValueError(f"Missing object label for image: {path}")
        if "text_label" not in labels:
            raise ValueError(f"Missing text label for image: {path}")
        samples.append(
            Sample(
                path=path,
                object_label=labels["object_label"],
                text_label=labels["text_label"],
                text_class_label=labels.get("text_class_label", IGNORE_INDEX),
            )
        )

    if not samples:
        raise ValueError("No images found from the folders in the config.")
    return samples, class_names


class FolderImageDataset(Dataset):
    def __init__(self, samples: list[Sample], image_size: int, train: bool) -> None:
        self.samples = samples
        self.image_size = image_size
        self.train = train

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        image = Image.open(sample.path).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        tensor = torch.tensor(list(image.getdata()), dtype=torch.float32)
        tensor = tensor.view(self.image_size, self.image_size, 3).permute(2, 0, 1) / 255.0
        tensor = (tensor - 0.5) / 0.5
        if self.train and random.random() < 0.5:
            tensor = torch.flip(tensor, dims=(2,))
        return {
            "image": tensor,
            "object_label": torch.tensor(sample.object_label, dtype=torch.long),
            "text_label": torch.tensor(sample.text_label, dtype=torch.long),
            "text_class_label": torch.tensor(sample.text_class_label, dtype=torch.long),
            "path": str(sample.path),
        }


class SmallCNN(nn.Module):
    def __init__(self, num_text_classes: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            self.block(3, 32),
            self.block(32, 64),
            self.block(64, 128),
            self.block(128, 256),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.object_head = nn.Linear(256, 2)
        self.text_head = nn.Linear(256, 2)
        self.text_class_head = nn.Linear(256, num_text_classes)

    @staticmethod
    def block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, image: torch.Tensor) -> dict:
        features = self.backbone(image)
        return {
            "object_logits": self.object_head(features),
            "text_logits": self.text_head(features),
            "text_class_logits": self.text_class_head(features),
        }


def accuracy(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int | None = None) -> tuple[int, int]:
    if ignore_index is not None:
        mask = labels != ignore_index
        if not mask.any():
            return 0, 0
        logits = logits[mask]
        labels = labels[mask]
    predictions = logits.argmax(dim=1)
    return int((predictions == labels).sum().item()), int(labels.numel())


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: str,
    text_class_weight: float,
) -> dict:
    training = optimizer is not None
    model.train(training)
    loss_fn = nn.CrossEntropyLoss()
    class_loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    total_loss = 0.0
    total_batches = 0
    metrics = {
        "object_correct": 0,
        "object_total": 0,
        "text_correct": 0,
        "text_total": 0,
        "class_correct": 0,
        "class_total": 0,
    }

    for batch in loader:
        images = batch["image"].to(device)
        object_labels = batch["object_label"].to(device)
        text_labels = batch["text_label"].to(device)
        text_class_labels = batch["text_class_label"].to(device)

        with torch.set_grad_enabled(training):
            outputs = model(images)
            loss = loss_fn(outputs["object_logits"], object_labels)
            loss = loss + loss_fn(outputs["text_logits"], text_labels)
            if (text_class_labels != IGNORE_INDEX).any():
                loss = loss + text_class_weight * class_loss_fn(
                    outputs["text_class_logits"], text_class_labels
                )

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        total_loss += float(loss.item())
        total_batches += 1

        correct, total = accuracy(outputs["object_logits"], object_labels)
        metrics["object_correct"] += correct
        metrics["object_total"] += total
        correct, total = accuracy(outputs["text_logits"], text_labels)
        metrics["text_correct"] += correct
        metrics["text_total"] += total
        correct, total = accuracy(outputs["text_class_logits"], text_class_labels, IGNORE_INDEX)
        metrics["class_correct"] += correct
        metrics["class_total"] += total

    return {
        "loss": total_loss / max(1, total_batches),
        "object_acc": metrics["object_correct"] / max(1, metrics["object_total"]),
        "text_acc": metrics["text_correct"] / max(1, metrics["text_total"]),
        "text_class_acc": metrics["class_correct"] / max(1, metrics["class_total"]),
    }


def save_checkpoint(
    output_dir: Path,
    model: nn.Module,
    config: dict,
    class_names: list[str],
    image_size: int,
    metrics: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "class_names": class_names,
            "image_size": image_size,
            "metrics": metrics,
        },
        output_dir / "classifier.pt",
    )
    (output_dir / "class_names.json").write_text(
        json.dumps(class_names, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    config_path = Path(args.config).expanduser().resolve()
    config = read_config(config_path)
    image_size = int(config.get("image_size", 224))
    text_class_weight = float(config.get("text_class_loss_weight", 1.0))
    samples, class_names = build_samples(config, config_path)
    if not class_names:
        raise ValueError("Config must define at least one text class in `text_classes`.")

    val_count = max(1, math.floor(len(samples) * args.val_fraction))
    train_count = len(samples) - val_count
    if train_count <= 0:
        raise ValueError("Need at least two images for train/validation split.")

    shuffled_samples = samples[:]
    random.shuffle(shuffled_samples)
    train_samples = shuffled_samples[:train_count]
    val_samples = shuffled_samples[train_count:]
    train_dataset = FolderImageDataset(train_samples, image_size=image_size, train=True)
    val_dataset = FolderImageDataset(val_samples, image_size=image_size, train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = choose_device(args.device)
    model = SmallCNN(num_text_classes=len(class_names)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"Loaded {len(samples)} images.")
    print(f"Text classes: {class_names}")
    print(f"Training on {device}.")

    best_val_loss = float("inf")
    best_metrics = {}
    output_dir = Path(args.output_dir).expanduser().resolve()
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, text_class_weight)
        val_metrics = run_epoch(model, val_loader, None, device, text_class_weight)
        print(
            f"epoch {epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"object_acc={val_metrics['object_acc']:.3f} "
            f"text_acc={val_metrics['text_acc']:.3f} "
            f"text_class_acc={val_metrics['text_class_acc']:.3f}"
        )
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_metrics = val_metrics
            save_checkpoint(output_dir, model, config, class_names, image_size, best_metrics)

    print(f"Saved best checkpoint to {output_dir / 'classifier.pt'}")


if __name__ == "__main__":
    main()
