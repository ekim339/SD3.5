"""Hydra entrypoint for TextCtrl-compatible fine-tuning."""

from __future__ import annotations

from collections.abc import Mapping

import hydra
from omegaconf import DictConfig, OmegaConf

from scene_text_editing.configuration import validate_config
from scene_text_editing.training import TrainingError, run_textctrl_training


@hydra.main(version_base="1.3", config_path="configs", config_name="train")
def main(config: DictConfig) -> None:
    resolved = OmegaConf.to_container(config, resolve=True)
    if not isinstance(resolved, Mapping):
        raise TrainingError("Hydra did not compose a mapping config.")
    validate_config(resolved)
    generated = run_textctrl_training(resolved)
    if bool(resolved.get("validate_only", False)) or not bool(
        resolved["training"].get("execute", False)
    ):
        print(f"Validated training handoff config: {generated}")
        print(
            "Set validate_only=false training.execute=true after preparing the "
            "TextCtrl environment and synthetic dataset."
        )
    else:
        print(f"Training completed using: {generated}")


if __name__ == "__main__":
    main()
