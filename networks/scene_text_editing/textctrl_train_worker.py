"""Worker executed with TextCtrl's pinned Python/CUDA environment."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--style-checkpoint", required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--checkpoint-frequency", type=int, required=True)
    parser.add_argument("--precision", type=int, default=32)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--sd-locked", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository = Path(args.repository).expanduser().resolve()
    sys.path.insert(0, str(repository))
    os.chdir(repository)

    import pytorch_lightning as pl
    from omegaconf import OmegaConf
    from pytorch_lightning.callbacks import ModelCheckpoint

    from src.trainer.utils import instantiate_from_config
    from utils import create_data, create_model, load_state_dict
    pl.seed_everything(args.seed, workers=True)

    model = create_model(args.config).cpu()
    model.load_state_dict(
        load_state_dict(args.style_checkpoint, location="cpu"),
        strict=False,
    )
    model.learning_rate = args.learning_rate
    model.sd_locked = args.sd_locked

    composed = OmegaConf.load(args.config)
    data_config = composed.pop("data", OmegaConf.create())
    data = create_data(data_config)
    data.prepare_data()
    data.setup(stage="fit")
    image_logger = instantiate_from_config(composed.image_logger)
    checkpoint = ModelCheckpoint(
        every_n_epochs=args.checkpoint_frequency,
        save_top_k=-1,
    )
    trainer = pl.Trainer(
        precision=args.precision,
        callbacks=[image_logger, checkpoint],
        **composed.lightning,
    )
    trainer.fit(model=model, datamodule=data)


if __name__ == "__main__":
    main()
