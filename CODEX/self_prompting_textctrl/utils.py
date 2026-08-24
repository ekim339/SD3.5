from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    _resolve_paths(config, path.parent)
    return config


def _resolve_paths(value: Any, base_dir: Path) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if (key == "root" or key.endswith(("_path", "_dir"))) and isinstance(child, str):
                candidate = Path(child).expanduser()
                if not candidate.is_absolute() and not _looks_like_hub_id(child):
                    value[key] = str((base_dir / candidate).resolve())
            else:
                _resolve_paths(child, base_dir)
    elif isinstance(value, list):
        for child in value:
            _resolve_paths(child, base_dir)


def _looks_like_hub_id(value: str) -> bool:
    return "/" in value and not value.startswith((".", "/"))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_torch_save(state: dict[str, Any], destination: str | Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(destination)


def choose_weight_dtype(mixed_precision: str) -> torch.dtype:
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32
