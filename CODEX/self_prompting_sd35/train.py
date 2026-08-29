"""Accelerate training entry point for self-prompted SD3.5."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import yaml
from accelerate import Accelerator
from diffusers import StableDiffusion3Pipeline
from torch.utils.data import DataLoader

from .dataset import SRNetSelfPromptDataset
from .model import SelfPromptingSD35


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def dtype_for(name: str):
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def main() -> None:
    args, cfg = parse_args(), None
    cfg = load_config(args.config)
    train_cfg = cfg["training"]
    accelerator = Accelerator(
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        mixed_precision=train_cfg["mixed_precision"],
    )
    torch.manual_seed(cfg["seed"])
    dataset = SRNetSelfPromptDataset(**cfg["dataset"])
    loader = DataLoader(
        dataset, batch_size=train_cfg["batch_size"], shuffle=True,
        num_workers=train_cfg["num_workers"], pin_memory=True, drop_last=True,
    )
    pipe = StableDiffusion3Pipeline.from_pretrained(
        cfg["model"]["pretrained_model"], torch_dtype=dtype_for(cfg["model"]["dtype"]),
    )
    model = SelfPromptingSD35(
        pipe, train_cfg["foreground_weight"], train_cfg["background_weight"]
    )
    if cfg["model"]["gradient_checkpointing"]:
        model.transformer.enable_gradient_checkpointing()
    optimizer = torch.optim.AdamW(
        model.transformer.parameters(), lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    # Frozen modules are deliberately outside optimizer but must follow the active device.
    pipe.vae.to(accelerator.device)
    for name in ("text_encoder", "text_encoder_2", "text_encoder_3"):
        encoder = getattr(pipe, name, None)
        if encoder is not None:
            encoder.to(accelerator.device)
    output = Path(train_cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if args.resume:
        accelerator.load_state(args.resume)
    step = 0
    model.train()
    while step < train_cfg["max_steps"]:
        for batch in loader:
            target_strings = list(batch["target_text"])
            with torch.no_grad():
                prompt_embeds, _, pooled, _ = pipe.encode_prompt(
                    prompt=target_strings, prompt_2=target_strings, prompt_3=target_strings,
                    device=accelerator.device, do_classifier_free_guidance=False,
                    max_sequence_length=cfg["model"]["max_sequence_length"],
                )
            with accelerator.accumulate(model):
                loss = model(
                    batch["target_image"], batch["masked_image"], batch["glyph_image"],
                    batch["style_image"], batch["mask"], prompt_embeds, pooled,
                )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), train_cfg["max_grad_norm"])
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                step += 1
                if step % train_cfg["log_every"] == 0:
                    accelerator.print(f"step={step} loss={loss.detach().item():.6f}")
                if step % train_cfg["save_every"] == 0:
                    accelerator.save_state(output / f"checkpoint-{step:06d}")
                    if accelerator.is_main_process:
                        accelerator.unwrap_model(model).transformer.save_pretrained(
                            output / f"transformer-{step:06d}", safe_serialization=True
                        )
                if step >= train_cfg["max_steps"]:
                    break
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.unwrap_model(model).transformer.save_pretrained(output / "transformer-final")


if __name__ == "__main__":
    main()
