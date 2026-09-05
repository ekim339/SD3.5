"""Accelerate entry point for self-reconstruction and cooldown training."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
import torch
from accelerate import Accelerator
from diffusers import StableDiffusion3Pipeline
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from .conditioning import encode_t5_target_conditioning
from .dataset import SRNetSelfPromptDataset
from .model import SelfPromptingSD35


def configure_distributed_environment(config: dict) -> None:
    """Apply NCCL workarounds before Accelerate creates the process group."""
    if config.get("distributed", {}).get("nccl_p2p_disable", False):
        os.environ["NCCL_P2P_DISABLE"] = "1"


def checkpoint_step(path: str | Path) -> int:
    match = re.fullmatch(r"checkpoint-(\d+)", Path(path).name)
    if match is None:
        raise ValueError("Resume directory must be named checkpoint-<step>")
    return int(match.group(1))


def resolve_resume_checkpoint(value: str | None, checkpoint_dir: Path) -> Path | None:
    if value in (None, "", "null"):
        return None
    if value == "latest":
        candidates = sorted(
            (path for path in checkpoint_dir.glob("checkpoint-*") if path.is_dir()),
            key=checkpoint_step,
        )
        if not candidates:
            raise FileNotFoundError(
                f"No checkpoint-* directories found in {checkpoint_dir}"
            )
        return candidates[-1]
    path = Path(to_absolute_path(value))
    if not path.is_dir():
        raise FileNotFoundError(f"Resume checkpoint directory does not exist: {path}")
    checkpoint_step(path)
    return path


def unwrap_model(accelerator: Accelerator, model: torch.nn.Module) -> SelfPromptingSD35:
    unwrapped = accelerator.unwrap_model(model)
    return getattr(unwrapped, "_orig_mod", unwrapped)


def register_lora_checkpoint_hooks(accelerator: Accelerator) -> None:
    """Save adapter tensors instead of serializing the frozen backbone."""

    def save_model_hook(models, weights, output_dir):
        for saved_model in models:
            unwrapped = unwrap_model(accelerator, saved_model)
            if not isinstance(unwrapped, SelfPromptingSD35):
                raise TypeError(f"Unexpected model in LoRA checkpoint: {type(unwrapped).__name__}")
            if accelerator.is_main_process:
                unwrapped.save_lora_weights(output_dir)
        while weights:
            weights.pop()

    def load_model_hook(models, input_dir):
        while models:
            unwrapped = unwrap_model(accelerator, models.pop())
            if not isinstance(unwrapped, SelfPromptingSD35):
                raise TypeError(f"Unexpected model in LoRA checkpoint: {type(unwrapped).__name__}")
            unwrapped.load_lora_weights(input_dir)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)


def load_hydra_config(overrides: list[str] | None = None) -> DictConfig:
    """Compose config without Hydra's Python-3.14-incompatible CLI parser."""
    values = list(sys.argv[1:] if overrides is None else overrides)
    config_dir = str(Path(__file__).resolve().parent)
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        return compose(config_name="config", overrides=values)


def main() -> None:
    hydra_config = load_hydra_config()
    cfg = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(cfg, dict):
        raise TypeError("Hydra config must resolve to a mapping")
    configure_distributed_environment(cfg)
    training = cfg["training"]
    accelerator = Accelerator(
        gradient_accumulation_steps=training["gradient_accumulation_steps"],
        mixed_precision=training["mixed_precision"],
    )
    torch.manual_seed(cfg["seed"])
    mode = str(cfg.get("mode", "self_reconstruction"))
    dataset_config = dict(cfg["dataset"])
    dataset_config["roots"] = [to_absolute_path(root) for root in dataset_config["roots"]]
    if dataset_config.get("font_path"):
        dataset_config["font_path"] = to_absolute_path(dataset_config["font_path"])
    dataset = SRNetSelfPromptDataset(mode=mode, **dataset_config)
    loader = DataLoader(
        dataset, batch_size=training["batch_size"], shuffle=True,
        num_workers=training["num_workers"], pin_memory=True, drop_last=True,
    )
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[
        cfg["model"]["dtype"]
    ]
    pipe = StableDiffusion3Pipeline.from_pretrained(cfg["model"]["pretrained_model"], dtype=dtype)
    lora = cfg["model"]["lora"]
    model = SelfPromptingSD35(
        pipe, training["foreground_weight"], training["background_weight"],
        lora["rank"], lora["alpha"], lora["dropout"], lora["target_modules"],
    )
    if cfg["model"]["gradient_checkpointing"]:
        model.transformer.enable_gradient_checkpointing()
    parameters = model.trainable_parameters()
    if not parameters:
        raise RuntimeError("No LoRA parameters were selected for training")
    trainable_count = sum(parameter.numel() for parameter in parameters)
    transformer_count = sum(parameter.numel() for parameter in model.transformer.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=training["learning_rate"], weight_decay=training["weight_decay"])
    register_lora_checkpoint_hooks(accelerator)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    accelerator.print(f"Training mode: {mode}; samples: {len(dataset):,}")
    accelerator.print(
        "Visual style prompt: " + ("enabled" if mode == "cooldown" else "disabled")
    )
    accelerator.print(
        f"Input-projection + LoRA trainable parameters: {trainable_count:,} / {transformer_count:,} "
        f"({100.0 * trainable_count / transformer_count:.3f}%)"
    )
    for name in ("text_encoder", "text_encoder_2", "text_encoder_3"):
        encoder = getattr(pipe, name, None)
        if encoder is not None:
            encoder.to(accelerator.device)
    output = Path(to_absolute_path(training["output_dir"]))
    output.mkdir(parents=True, exist_ok=True)
    resume = resolve_resume_checkpoint(training.get("resume_from_checkpoint"), output)
    if resume is not None:
        accelerator.print(f"Resuming complete training state from {resume}")
        accelerator.load_state(resume)
    step = checkpoint_step(resume) if resume is not None else 0
    model.train()
    while step < training["max_steps"]:
        for batch in loader:
            target_strings = list(batch["target_text"])
            # Self-reconstruction must not receive the readable source crop;
            # cooldown is the only stage that supplies it.
            style_image = batch["style_image"] if mode == "cooldown" else None
            with torch.no_grad():
                prompt, pooled = encode_t5_target_conditioning(
                    pipe,
                    target_strings,
                    device=accelerator.device,
                    max_sequence_length=cfg["model"]["max_sequence_length"],
                )
            with accelerator.accumulate(model):
                loss = model(
                    batch["target_image"], batch["masked_image"], batch["glyph_image"],
                    style_image, batch["mask"], prompt, pooled,
                    loss_mask=batch["loss_mask"],
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
                    accelerator.wait_for_everyone()
                    accelerator.save_state(output / f"checkpoint-{step:06d}")
                    accelerator.wait_for_everyone()
                if step >= training["max_steps"]:
                    break
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrap_model(accelerator, model).save_lora_weights(output / "lora-final")
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
