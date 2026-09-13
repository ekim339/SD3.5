"""Fine-tune only TextCtrl's glyph/style head with supervised contrastive loss."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import random
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import yaml

try:
    from .dataset import SRNetStyleDataset, StyleBatchSampler, collate_style_pairs
    from .loss import SupervisedContrastiveLoss
    from .model import TextCtrlGlyphStyleEncoder, parameter_counts
except ImportError:  # Also support `python train.py`.
    from dataset import SRNetStyleDataset, StyleBatchSampler, collate_style_pairs
    from loss import SupervisedContrastiveLoss
    from model import TextCtrlGlyphStyleEncoder, parameter_counts


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("Configuration root must be a mapping")
    return value


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def split_indices(length: int, validation_records: int, seed: int) -> Tuple[Sequence[int], Sequence[int]]:
    if validation_records < 2 or validation_records >= length - 1:
        raise ValueError("validation_records must be at least 2 and leave 2 training records")
    order = torch.randperm(length, generator=torch.Generator().manual_seed(seed)).tolist()
    return order[validation_records:], order[:validation_records]


def make_loader(dataset, indices, config, device, training: bool) -> DataLoader:
    sampler = StyleBatchSampler(
        dataset,
        indices,
        styles_per_batch=int(config["styles_per_batch"]),
        hard_negative_probability=float(config["hard_negative_probability"]) if training else 1.0,
        shuffle=training,
        drop_last=True,
        seed=int(config["batch_seed"]) + (0 if training else 1_000_000),
    )
    workers = int(config["num_workers"])
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        collate_fn=collate_style_pairs,
    )


def amp_settings(device: torch.device, precision: str):
    if precision not in {"float32", "float16", "bfloat16"}:
        raise ValueError("precision must be float32, float16, or bfloat16")
    enabled = device.type == "cuda" and precision != "float32"
    dtype = torch.bfloat16 if precision == "bfloat16" else torch.float16
    return enabled, dtype


def autocast_context(enabled: bool, dtype: torch.dtype):
    return torch.cuda.amp.autocast(dtype=dtype) if enabled else nullcontext()


def similarity_metrics(features: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    normalized = F.normalize(features.detach().float(), dim=1)
    similarity = normalized @ normalized.transpose(0, 1)
    self_mask = torch.eye(labels.shape[0], dtype=torch.bool, device=labels.device)
    positive = labels[:, None].eq(labels[None, :]) & ~self_mask
    negative = ~labels[:, None].eq(labels[None, :])
    return {
        "positive_cosine": float(similarity[positive].mean().cpu()),
        "negative_cosine": float(similarity[negative].mean().cpu()),
    }


@torch.no_grad()
def validate(model, criterion, loader, device, use_amp, amp_dtype) -> Dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "positive_cosine": 0.0, "negative_cosine": 0.0}
    examples = 0
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        labels = batch["style_labels"].to(device, non_blocking=True)
        with autocast_context(use_amp, amp_dtype):
            features = model(images)
            loss = criterion(features, labels)
        metrics = similarity_metrics(features, labels)
        count = int(images.shape[0])
        examples += count
        totals["loss"] += float(loss.cpu()) * count
        totals["positive_cosine"] += metrics["positive_cosine"] * count
        totals["negative_cosine"] += metrics["negative_cosine"] * count
    model.train()
    if not examples:
        raise RuntimeError("Validation loader produced no samples")
    return {name: total / examples for name, total in totals.items()}


def checkpoint_payload(step, model, optimizer, scaler, config, validation):
    return {
        "format": "textctrl-glyph-style-supcon-v1",
        "step": step,
        # Store only the trained branch; the immutable base checkpoint is named below.
        "glyph_attn_block": model.glyph_head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "source_checkpoint": model.load_report.checkpoint,
        "load_report": model.load_report.__dict__,
        "config": dict(config),
        "validation": dict(validation) if validation is not None else None,
    }


def resume_training(path: Path, model, optimizer, scaler) -> int:
    payload = torch.load(str(path), map_location="cpu")
    if not isinstance(payload, dict) or payload.get("format") != "textctrl-glyph-style-supcon-v1":
        raise ValueError("Not a glyph/style SupCon checkpoint: {}".format(path))
    model.glyph_head.load_state_dict(payload["glyph_attn_block"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    if "scaler" in payload:
        scaler.load_state_dict(payload["scaler"])
    return int(payload["step"])


def train(config: Mapping[str, Any]) -> None:
    seed_everything(int(config["seed"]))
    device = choose_device(str(config["device"]))
    data_config = config["dataset"]
    training_config = config["training"]
    checkpoint_config = config["checkpoint"]
    model_config = config["model"]

    dataset = SRNetStyleDataset(
        [resolve_path(str(path)) for path in data_config["roots"]],
        image_size=int(model_config["image_size"]),
        limit=data_config.get("limit"),
        require_different_text=bool(data_config["require_different_text"]),
        strict=bool(data_config["strict"]),
    )
    train_indices, validation_indices = split_indices(
        len(dataset), int(data_config["validation_records"]), int(data_config["split_seed"])
    )
    train_loader = make_loader(dataset, train_indices, training_config, device, True)
    validation_loader = make_loader(dataset, validation_indices, training_config, device, False)

    model = TextCtrlGlyphStyleEncoder(
        repository=resolve_path(str(checkpoint_config["textctrl_repository"])),
        checkpoint=resolve_path(str(checkpoint_config["pretrained"])),
        image_size=int(model_config["image_size"]),
        patch_size=int(model_config["patch_size"]),
        embed_dim=int(model_config["embed_dim"]),
        pooling=str(model_config["pooling"]),
        require_glyph_head_weights=bool(checkpoint_config["require_glyph_head_weights"]),
    ).to(device)
    counts = parameter_counts(model)
    if counts["trainable"] != counts["glyph_head"]:
        raise RuntimeError("Only the glyph/style head may be trainable")
    style_classes = len({record.style_id for record in dataset.records})
    print(json.dumps({
        "device": str(device), "records": len(dataset),
        "style_classes": style_classes, **counts,
    }, sort_keys=True))
    print(json.dumps({"checkpoint_load": model.load_report.__dict__}, sort_keys=True))

    criterion = SupervisedContrastiveLoss(float(training_config["temperature"]))
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    use_amp, amp_dtype = amp_settings(device, str(training_config["precision"]))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)
    output_dir = resolve_path(str(training_config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(config), handle, sort_keys=False)

    step = 0
    if training_config.get("resume_from"):
        resume_path = resolve_path(str(training_config["resume_from"]))
        step = resume_training(resume_path, model, optimizer, scaler)
        print(json.dumps({"resumed_from": str(resume_path), "step": step}, sort_keys=True))
    maximum = int(training_config["max_steps"])
    if step >= maximum:
        raise ValueError("max_steps must exceed the resumed step")

    model.train()
    last_validation = None
    while step < maximum:
        for batch in train_loader:
            images = batch["images"].to(device, non_blocking=True)
            labels = batch["style_labels"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(use_amp, amp_dtype):
                features = model(images)
                loss = criterion(features, labels)
            scaler.scale(loss).backward()
            max_norm = float(training_config["max_grad_norm"])
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.glyph_head.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
            step += 1

            if step % int(training_config["log_every"]) == 0:
                print(json.dumps({"step": step, "loss": float(loss.detach().cpu()),
                                  **similarity_metrics(features, labels)}, sort_keys=True))
            if step % int(training_config["validate_every"]) == 0 or step == maximum:
                last_validation = validate(
                    model, criterion, validation_loader, device, use_amp, amp_dtype
                )
                print(json.dumps({"step": step, **{
                    "validation_" + key: value for key, value in last_validation.items()
                }}, sort_keys=True))
            if step % int(training_config["save_every"]) == 0 or step == maximum:
                payload = checkpoint_payload(step, model, optimizer, scaler, config, last_validation)
                torch.save(payload, str(output_dir / "glyph-style-supcon-{:06d}.pt".format(step)))
                torch.save(payload, str(output_dir / "glyph-style-supcon-latest.pt"))
            if step >= maximum:
                break


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).resolve().parent / "config.yaml")
    args = parser.parse_args()
    train(load_config(args.config.expanduser().resolve()))


if __name__ == "__main__":
    main()
