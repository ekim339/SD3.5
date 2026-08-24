"""Legacy Python 3.8/CUDA worker for dual-control TextCtrl fine-tuning."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    from omegaconf import OmegaConf
    config = OmegaConf.load(args.config)
    repository = Path(config.textctrl.repository).resolve()
    sys.path.insert(0, str(repository))
    os.chdir(repository)

    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from src.trainer.utils import instantiate_from_config
    from utils import create_data, load_state_dict

    pl.seed_everything(int(config.seed), workers=True)
    upstream = OmegaConf.load(repository / "configs" / "train.yaml")
    base = upstream.model.params.base_config
    base.scheduler_config = str(config.textctrl.scheduler)
    base.vae.pretrained = str(config.textctrl.vae)
    base.text_encoder.params.ckpt_path = str(config.textctrl.text_encoder_checkpoint)
    base.text_encoder.optimize = False
    base.unet_pretrained = str(config.textctrl.unet)
    base.unet.target = "networks.sd15_residual_extractor_finetune.model.FullyTrainableControlUNet"
    base.font_path = str(config.dataset.canonical_font)
    base.ocr_model.pretrained = str(config.textctrl.ocr_checkpoint)
    base.ocr_model.ocr_supervised = bool(config.loss.ocr_supervised)
    base.reconstruction_loss = bool(config.loss.reconstruction)
    base.ocr_loss_alpha = float(config.loss.ocr_weight)
    base.vgg_weight = (str(config.textctrl.vgg_checkpoint)
                       if bool(config.loss.reconstruction) else False)
    charset = repository / "src" / "module" / "abinet" / "data" / "charset_36.txt"
    base.ocr_model.charset_path = str(charset)
    base.ocr_model.vision.charset_path = str(charset)
    base.ocr_model.language.charset_path = str(charset)
    base.ocr_model.alignment.charset_path = str(charset)
    base.weight_decay = float(config.training.weight_decay)
    base.adam_epsilon = float(config.training.adam_epsilon)

    upstream.model.target = (
        "networks.sd15_residual_extractor_finetune.model.ResidualGlyphControlTrainer"
    )
    upstream.model.params.residual_checkpoint = str(config.residual.checkpoint)
    upstream.model.params.adapter_checkpoint = str(config.residual.adapter_checkpoint)
    upstream.model.params.glyph_widths = list(config.model.glyph_widths)
    upstream.model.params.style_scale = float(config.model.initial_style_scale)
    upstream.model.params.glyph_scale = float(config.model.initial_glyph_scale)
    upstream.data = OmegaConf.create({
        "target": "networks.sd15_residual_extractor_finetune.data.SRNetResidualDataModule",
        "batch_size": int(config.training.batch_size),
        "roots": list(config.dataset.roots),
        "canonical_font": str(config.dataset.canonical_font),
        "resolution": int(config.dataset.resolution),
        "validation_samples": int(config.dataset.validation_samples),
        "split_seed": int(config.dataset.split_seed),
        "condition_dropout": float(config.dataset.condition_dropout),
        "limit": config.dataset.limit,
        "num_workers": int(config.dataset.num_workers),
    })
    output = Path(config.training.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    generated = output / "textctrl_dual_control.generated.yaml"
    OmegaConf.save(upstream, generated)

    model = instantiate_from_config(upstream.model).cpu()
    incompatible = model.load_state_dict(
        load_state_dict(str(config.textctrl.checkpoint), location="cpu"), strict=False)
    unexpected = [key for key in incompatible.unexpected_keys
                  if not key.startswith("control_model.vit.")]
    if unexpected:
        raise RuntimeError("Unexpected pretrained TextCtrl keys: %s" % unexpected)
    model.learning_rate = float(config.training.learning_rate)
    data = create_data(upstream.data)
    data.prepare_data()
    data.setup(stage="fit")
    callbacks = []
    if bool(config.training.enable_checkpointing):
        callbacks.append(ModelCheckpoint(
            dirpath=str(output), filename="step-{step:08d}",
            every_n_train_steps=int(config.training.checkpoint_every_n_steps),
            save_top_k=-1, save_last=True, save_weights_only=False,
        ))
    trainer = pl.Trainer(
        logger=False, callbacks=callbacks,
        enable_checkpointing=bool(config.training.enable_checkpointing),
        default_root_dir=str(output),
        accelerator=str(config.training.accelerator), devices=config.training.devices,
        strategy=(None if str(config.training.strategy) == "auto"
                  else str(config.training.strategy)),
        precision=config.training.precision,
        max_steps=int(config.training.max_steps), max_epochs=int(config.training.max_epochs),
        accumulate_grad_batches=int(config.training.accumulate_grad_batches),
        gradient_clip_val=float(config.training.gradient_clip_val),
        log_every_n_steps=int(config.training.log_every_n_steps),
        val_check_interval=int(config.training.validate_every_n_steps),
        limit_val_batches=config.training.limit_val_batches,
        resume_from_checkpoint=config.training.resume_from,
    )
    trainer.fit(model=model, datamodule=data)


if __name__ == "__main__":
    main()
