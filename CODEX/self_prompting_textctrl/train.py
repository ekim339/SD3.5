from __future__ import annotations

import argparse
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset import SRNetSelfPromptDataset
from model import SelfPromptingSD15
from utils import atomic_torch_save, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train self-prompting TextCtrl on SRNet")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--resume", default=None, help="Checkpoint directory to resume from")
    return parser.parse_args()


def move_frozen_modules(model: SelfPromptingSD15, device: torch.device, dtype: torch.dtype) -> None:
    model.vae.to(device=device, dtype=dtype)
    model.text_encoder.to(device=device, dtype=dtype)
    model.vae.eval()
    model.text_encoder.eval()


def save_checkpoint(
    accelerator: Accelerator,
    model: SelfPromptingSD15,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    output_dir: Path,
    step: int,
) -> None:
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    checkpoint = output_dir / f"checkpoint-{step:08d}"
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_unet(checkpoint)
    atomic_torch_save(
        {"global_step": step, "optimizer": optimizer.state_dict(),
         "lr_scheduler": lr_scheduler.state_dict()},
        checkpoint / "training_state.pt",
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model_config = config["model"]
    data_config = config["data"]
    train_config = config["training"]
    ddp_kwargs = DistributedDataParallelKwargs(gradient_as_bucket_view=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
        mixed_precision=train_config["mixed_precision"],
        kwargs_handlers=[ddp_kwargs],
    )
    set_seed(train_config["seed"])
    dataset = SRNetSelfPromptDataset(
        root=data_config["root"], splits=data_config["splits"],
        font_path=data_config["font_path"], image_size=model_config["image_size"],
        limit=data_config["num_samples"],
    )
    dataloader = DataLoader(
        dataset, batch_size=train_config["batch_size"], shuffle=True,
        num_workers=data_config["num_workers"], pin_memory=True,
        persistent_workers=data_config["num_workers"] > 0, drop_last=True,
    )
    model = SelfPromptingSD15(**{
        key: model_config[key] for key in (
            "vae_path", "unet_path", "scheduler_path", "text_model_path",
            "revision", "max_text_length", "conditioning_dropout",
        )
    })
    if args.resume:
        model.load_unet_checkpoint(args.resume)
    if train_config["gradient_checkpointing"]:
        model.unet.enable_gradient_checkpointing()
    model.unet.train()
    optimizer = torch.optim.AdamW(
        model.unet.parameters(), lr=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"],
    )
    lr_scheduler = get_scheduler(
        "constant", optimizer=optimizer, num_warmup_steps=0,
        num_training_steps=train_config["max_steps"],
    )
    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )
    frozen_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        frozen_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        frozen_dtype = torch.bfloat16
    move_frozen_modules(accelerator.unwrap_model(model), accelerator.device, frozen_dtype)

    output_dir = Path(train_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    global_step = 0
    if args.resume:
        state_path = Path(args.resume) / "training_state.pt"
        if state_path.is_file():
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            optimizer.load_state_dict(state["optimizer"])
            lr_scheduler.load_state_dict(state["lr_scheduler"])
            global_step = int(state["global_step"])

    progress = tqdm(total=train_config["max_steps"], initial=global_step,
                    disable=not accelerator.is_local_main_process)
    while global_step < train_config["max_steps"]:
        for batch in dataloader:
            with accelerator.accumulate(model):
                loss = model(batch)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        accelerator.unwrap_model(model).unet.parameters(),
                        train_config["max_grad_norm"],
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)
                if global_step % train_config["log_every"] == 0:
                    progress.set_postfix(loss=f"{loss.detach().item():.4f}")
                if global_step % train_config["checkpoint_every"] == 0:
                    save_checkpoint(
                        accelerator, model, optimizer, lr_scheduler, output_dir, global_step
                    )
                if global_step >= train_config["max_steps"]:
                    break
    save_checkpoint(accelerator, model, optimizer, lr_scheduler, output_dir, global_step)
    progress.close()
    accelerator.end_training()


if __name__ == "__main__":
    main()
