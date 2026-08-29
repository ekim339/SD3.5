"""Accelerate entry point for LoRA self-reconstruction training."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from accelerate import Accelerator
from diffusers import StableDiffusion3Pipeline
from torch.utils.data import DataLoader

from .dataset import SRNetSelfPromptDataset
from .model import SelfPromptingSD35


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--resume", type=str)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    with args.config.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    training = cfg["training"]
    accelerator = Accelerator(
        gradient_accumulation_steps=training["gradient_accumulation_steps"],
        mixed_precision=training["mixed_precision"],
    )
    torch.manual_seed(cfg["seed"])
    dataset = SRNetSelfPromptDataset(**cfg["dataset"])
    loader = DataLoader(
        dataset, batch_size=training["batch_size"], shuffle=True,
        num_workers=training["num_workers"], pin_memory=True, drop_last=True,
    )
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[
        cfg["model"]["dtype"]
    ]
    pipe = StableDiffusion3Pipeline.from_pretrained(cfg["model"]["pretrained_model"], torch_dtype=dtype)
    lora = cfg["model"]["lora"]
    model = SelfPromptingSD35(
        pipe, training["foreground_weight"], training["background_weight"],
        lora["rank"], lora["alpha"], lora["dropout"], lora["target_modules"],
    )
    if cfg["model"]["gradient_checkpointing"]:
        model.transformer.enable_gradient_checkpointing()
    parameters = model.trainable_parameters()
    optimizer = torch.optim.AdamW(parameters, lr=training["learning_rate"], weight_decay=training["weight_decay"])
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    for name in ("text_encoder", "text_encoder_2", "text_encoder_3"):
        encoder = getattr(pipe, name, None)
        if encoder is not None:
            encoder.to(accelerator.device)
    output = Path(training["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if args.resume:
        accelerator.load_state(args.resume)
    step = 0
    model.train()
    while step < training["max_steps"]:
        for batch in loader:
            strings = list(batch["source_text"])
            with torch.no_grad():
                prompt, _, pooled, _ = pipe.encode_prompt(
                    prompt=strings, prompt_2=strings, prompt_3=strings,
                    device=accelerator.device, do_classifier_free_guidance=False,
                    max_sequence_length=cfg["model"]["max_sequence_length"],
                )
            with accelerator.accumulate(model):
                loss = model(
                    batch["target_image"], batch["masked_image"], batch["glyph_image"],
                    batch["style_image"], batch["mask"], prompt, pooled,
                )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(parameters, training["max_grad_norm"])
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                step += 1
                if step % training["log_every"] == 0:
                    accelerator.print(f"step={step} loss={loss.detach().item():.6f}")
                if step % training["save_every"] == 0:
                    accelerator.save_state(output / f"state-{step:06d}")
                    if accelerator.is_main_process:
                        accelerator.unwrap_model(model).save_lora_weights(output / f"lora-{step:06d}")
                if step >= training["max_steps"]:
                    break
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.unwrap_model(model).save_lora_weights(output / "lora-final")


if __name__ == "__main__":
    main()
