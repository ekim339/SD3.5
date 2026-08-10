from __future__ import annotations

from pathlib import Path

import hydra
import torch
from accelerate import Accelerator
from diffusers import StableDiffusion3Pipeline
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import ConcatDataset, DataLoader, Subset, WeightedRandomSampler

from .dataset import SRNetEditingDataset
from .encoders import FrozenTextCtrlGlyphEncoder, FrozenTextCtrlStyleEncoder
from .model import SD35SceneTextEditor


def _dtype(name: str):
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def _build_training_dataset(dataset_config: DictConfig, prompt_template: str):
    roots = list(dataset_config.get("roots") or [])
    if not roots and dataset_config.get("root"):
        roots = [dataset_config.root]
    if not roots:
        raise ValueError("dataset.roots must contain at least one SRNet dataset")

    partitions = [
        SRNetEditingDataset(
            root,
            resolution=dataset_config.resolution,
            style_resolution=dataset_config.style_resolution,
            prompt_template=prompt_template,
        )
        for root in roots
    ]
    records = [record for partition in partitions for record in partition.records]
    dataset = partitions[0] if len(partitions) == 1 else ConcatDataset(partitions)
    limit = dataset_config.limit
    if limit is not None:
        count = min(int(limit), len(dataset))
        dataset = Subset(dataset, range(count))
        records = records[:count]
    return dataset, records, roots


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(config: DictConfig) -> None:
    run_training(config)


def run_training(config: DictConfig) -> None:
    """Fine-tune SD3.5 from an already composed configuration."""
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
    )
    torch.manual_seed(config.seed)
    dataset, records, dataset_roots = _build_training_dataset(
        config.dataset, config.prompt.template
    )
    accelerator.print(f"training_partitions={len(dataset_roots)} samples={len(dataset)}")
    uppercase_fraction = float(config.training.uppercase_sampling_fraction)
    if not 0.0 <= uppercase_fraction <= 1.0:
        raise ValueError("training.uppercase_sampling_fraction must be between 0 and 1")
    uppercase = [
        any(character.isupper() for character in record[2]) for record in records
    ]
    uppercase_count = sum(uppercase)
    sampler = None
    if 0.0 < uppercase_fraction < 1.0 and 0 < uppercase_count < len(dataset):
        lowercase_count = len(dataset) - uppercase_count
        weights = [
            uppercase_fraction / uppercase_count
            if is_uppercase
            else (1.0 - uppercase_fraction) / lowercase_count
            for is_uppercase in uppercase
        ]
        sampler = WeightedRandomSampler(weights, len(dataset), replacement=True)
        accelerator.print(
            f"case_balanced_sampling uppercase={uppercase_count}/{len(dataset)} "
            f"target_fraction={uppercase_fraction:.2f}"
        )
    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=config.training.num_workers,
        pin_memory=True,
    )
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        config.model.pretrained_model,
        revision=config.model.revision,
        torch_dtype=_dtype(config.model.dtype),
    )
    style_encoder = FrozenTextCtrlStyleEncoder(
        config.encoders.textctrl_repository,
        config.encoders.style_checkpoint,
    )
    glyph_encoder = FrozenTextCtrlGlyphEncoder(
        config.encoders.textctrl_repository,
        config.encoders.glyph_checkpoint,
    )
    model = SD35SceneTextEditor(
        pipeline,
        style_encoder,
        glyph_encoder,
        training_mode=config.model.training_mode,
        lora_config=OmegaConf.to_container(config.model.lora, resolve=True),
        fill_value=config.dataset.fill_value,
        glyph_gate_init=config.conditioning.glyph_gate_init,
        style_gate_init=config.conditioning.style_gate_init,
        foreground_loss_weight=config.training.foreground_loss_weight,
        background_loss_weight=config.training.background_loss_weight,
    )
    if config.model.gradient_checkpointing and config.model.training_mode != "frozen":
        model.transformer.enable_gradient_checkpointing()
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    accelerator.print(
        f"training_mode={config.model.training_mode} "
        f"trainable_parameters={trainable_count:,}"
    )
    optimizer = torch.optim.AdamW(
        model.trainable_parameters,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    pipeline.to(accelerator.device)
    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if accelerator.is_main_process:
        OmegaConf.save(config, output_dir / "config.yaml")

    step = 0
    while step < config.training.max_steps:
        for batch in loader:
            with torch.no_grad():
                prompt_embeds, _, pooled_embeds, _ = pipeline.encode_prompt(
                    prompt=batch["prompt"],
                    prompt_2=batch["prompt"],
                    prompt_3=batch["prompt"],
                    device=accelerator.device,
                    do_classifier_free_guidance=False,
                    max_sequence_length=config.model.max_prompt_length,
                )
                target_text_embeds = pipeline._get_t5_prompt_embeds(
                    prompt=batch["target_text"],
                    device=accelerator.device,
                    max_sequence_length=config.conditioning.target_text_max_length,
                )
            with accelerator.accumulate(model):
                loss = model(
                    target_images=batch["target_image"],
                    source_images=batch["source_image"],
                    source_masks=batch["source_mask"],
                    style_images=batch["style_image"],
                    target_texts=batch["target_text"],
                    prompt_embeddings=prompt_embeds,
                    target_text_embeddings=target_text_embeds,
                    pooled_prompt_embeddings=pooled_embeds,
                )
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                step += 1
                if step % config.training.log_every == 0:
                    unwrapped = accelerator.unwrap_model(model)
                    glyph_gate = unwrapped.projector.glyph_gate.detach().float().item()
                    style_gate = unwrapped.projector.style_gate.detach().float().item()
                    accelerator.print(
                        f"step={step} loss={loss.detach().item():.6f} "
                        f"glyph_gate={glyph_gate:.6f} style_gate={style_gate:.6f}"
                    )
                if step % config.training.save_every == 0 and accelerator.is_main_process:
                    unwrapped = accelerator.unwrap_model(model)
                    trainable = {
                        name: parameter.detach().cpu()
                        for name, parameter in unwrapped.named_parameters()
                        if parameter.requires_grad
                    }
                    torch.save(trainable, output_dir / f"adapters-{step:06d}.pt")
                if step >= config.training.max_steps:
                    break


if __name__ == "__main__":
    main()
