"""Train a channel adapter by distilling TextCtrl style features."""
from __future__ import annotations

import json
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, random_split

from encoders.dataset import SRNetStylePairDataset

from .features import load_frozen_extractors, residual_features, textctrl_style_grid
from .loss import adapter_loss
from .model import ChannelAdapter

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve(path) -> Path:
    value = Path(str(path)).expanduser()
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def paired_inputs(batch, device):
    """Use both same-style SRNet strings as independent image/glyph examples."""
    images = torch.cat((batch["image_a"], batch["image_b"]), dim=0).to(
        device, non_blocking=True)
    glyphs = torch.cat((batch["glyph_a"], batch["glyph_b"]), dim=0).to(
        device, non_blocking=True)
    return images, glyphs


def split_dataset(dataset, validation_samples: int, seed: int):
    if validation_samples <= 0 or validation_samples >= len(dataset):
        raise ValueError("validation_samples must be between 1 and dataset_size - 1")
    lengths = [len(dataset) - validation_samples, validation_samples]
    return random_split(dataset, lengths, generator=torch.Generator().manual_seed(seed))


@torch.no_grad()
def validate(adapter, residual_extractor, teacher, loader, device, config,
             use_amp, amp_dtype):
    adapter.eval()
    totals = {"total": 0.0, "mse": 0.0, "cosine_distance": 0.0}
    examples = 0
    for batch in loader:
        images, glyphs = paired_inputs(batch, device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            residual = residual_features(residual_extractor, images, glyphs)
            target = textctrl_style_grid(teacher, images)
            losses = adapter_loss(adapter(residual), target, **config)
        count = images.shape[0]
        examples += count
        for name in totals:
            totals[name] += float(losses[name]) * count
    adapter.train()
    return {name: value / examples for name, value in totals.items()}


def save_checkpoint(path, step, adapter, optimizer, config, validation=None):
    payload = {
        "step": step,
        "adapter": adapter.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_config": OmegaConf.to_container(config.model, resolve=True),
        "config": OmegaConf.to_container(config, resolve=True),
        "validation": validation,
        "format": "residual-to-textctrl-channel-adapter-v1",
    }
    torch.save(payload, path)


def train(config: DictConfig):
    torch.manual_seed(int(config.seed))
    device = choose_device(str(config.device))
    dataset = SRNetStylePairDataset(
        [resolve(path) for path in config.dataset.roots],
        font_path=resolve(config.dataset.canonical_font),
        resolution=int(config.dataset.resolution),
        limit=config.dataset.limit,
        strict_different_text=True,
    )
    training_set, validation_set = split_dataset(
        dataset, int(config.dataset.validation_samples), int(config.dataset.split_seed))
    loader_options = {
        "batch_size": int(config.training.batch_size),
        "num_workers": int(config.training.num_workers),
        "pin_memory": device.type == "cuda",
    }
    training_loader = DataLoader(training_set, shuffle=True, drop_last=True, **loader_options)
    validation_loader = DataLoader(validation_set, shuffle=False, drop_last=False,
                                   **loader_options)
    residual_extractor, teacher = load_frozen_extractors(
        resolve(config.residual_checkpoint), resolve(config.textctrl_repository),
        resolve(config.textctrl_style_checkpoint), device)
    adapter = ChannelAdapter(**OmegaConf.to_container(config.model, resolve=True)).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=float(config.training.learning_rate),
                                  weight_decay=float(config.training.weight_decay))
    step = 0
    if config.training.resume_from:
        payload = torch.load(resolve(config.training.resume_from), map_location=device,
                             weights_only=False)
        adapter.load_state_dict(payload["adapter"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        step = int(payload["step"])

    precision = str(config.training.precision)
    use_amp = device.type == "cuda" and precision in {"float16", "bfloat16"}
    amp_dtype = torch.bfloat16 if precision == "bfloat16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    output_dir = resolve(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output_dir / "config.yaml")
    loss_config = OmegaConf.to_container(config.loss, resolve=True)
    adapter.train()
    last_validation = None
    maximum = int(config.training.max_steps)
    while step < maximum:
        for batch in training_loader:
            images, glyphs = paired_inputs(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                residual = residual_features(residual_extractor, images, glyphs)
                target = textctrl_style_grid(teacher, images)
                losses = adapter_loss(adapter(residual), target, **loss_config)
            scaler.scale(losses["total"]).backward()
            if float(config.training.max_grad_norm) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(adapter.parameters(),
                                               float(config.training.max_grad_norm))
            scaler.step(optimizer)
            scaler.update()
            step += 1
            if step % int(config.training.log_every) == 0:
                print(json.dumps({"step": step, **{
                    name: round(float(value.detach()), 6) for name, value in losses.items()}},
                    sort_keys=True))
            if step % int(config.training.validate_every) == 0 or step == maximum:
                last_validation = validate(adapter, residual_extractor, teacher,
                                           validation_loader, device, loss_config,
                                           use_amp, amp_dtype)
                print(json.dumps({"step": step, **{
                    f"validation_{key}": round(value, 6)
                    for key, value in last_validation.items()}}, sort_keys=True))
            if step % int(config.training.save_every) == 0 or step == maximum:
                numbered = output_dir / f"channel-adapter-{step:06d}.pt"
                save_checkpoint(numbered, step, adapter, optimizer, config, last_validation)
                save_checkpoint(output_dir / "channel-adapter-latest.pt", step, adapter,
                                optimizer, config, last_validation)
            if step >= maximum:
                break


@hydra.main(version_base="1.3", config_path="configs", config_name="train")
def main(config: DictConfig):
    train(config)


if __name__ == "__main__":
    main()
