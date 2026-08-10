"""Hydra entrypoint for explicit pretrained-model acquisition."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from networks.scene_text_editing.checkpoints import CheckpointError, download_models


@hydra.main(version_base="1.3", config_path="configs", config_name="download")
def main(config: DictConfig) -> None:
    resolved = OmegaConf.to_container(config, resolve=True)
    if not isinstance(resolved, Mapping):
        raise CheckpointError("Hydra did not compose a mapping config.")
    paths = download_models(resolved)
    if bool(resolved.get("dry_run", False)):
        return
    print("Model acquisition complete.")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()

