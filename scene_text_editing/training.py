"""Preparation and isolated execution of TextCtrl fine-tuning."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from scene_text_editing.checkpoints import validate_textctrl_installation
from scene_text_editing.configuration import resolve_path


class TrainingError(RuntimeError):
    """Raised when a TextCtrl fine-tuning run cannot be prepared."""


def _read_upstream_labels(path: Path) -> list[tuple[str, str]]:
    """Parse exactly the two space-separated fields TextCtrl expects."""

    if not path.is_file():
        raise TrainingError(f"Missing TextCtrl label file: {path}")
    records: list[tuple[str, str]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\r\n")
            parts = line.split(" ")
            if len(parts) != 2 or not all(parts):
                raise TrainingError(
                    f"TextCtrl requires `filename single_token` at "
                    f"{path}:{line_number}."
                )
            filename, label = parts
            if filename in seen:
                raise TrainingError(
                    f"Duplicate filename {filename!r} at {path}:{line_number}."
                )
            seen.add(filename)
            records.append((filename, label))
    if not records:
        raise TrainingError(f"TextCtrl label file is empty: {path}")
    return records


def validate_textctrl_training_data(dataset: Mapping[str, Any]) -> Path:
    """Preflight the legacy loader's shard, font, label, and image contracts."""

    root = resolve_path(str(dataset["root_dir"]))
    font = root / "fonts" / "arial.ttf"
    if not font.is_file() or font.stat().st_size == 0:
        raise TrainingError(
            f"TextCtrl training requires a non-empty Arial font at {font}."
        )

    source_dir_name = str(dataset.get("source_dir", "i_s"))
    target_dir_name = str(dataset.get("target_dir", "t_f"))
    source_labels_name = str(dataset.get("source_labels", "i_s.txt"))
    target_labels_name = str(dataset.get("target_labels", "i_t.txt"))
    split_patterns = {
        "train": str(dataset.get("train_glob", "train/train-*")),
        "validation": str(dataset.get("validation_glob", "eval/eval-*")),
    }
    for split, pattern in split_patterns.items():
        shards = sorted(path for path in root.glob(pattern) if path.is_dir())
        if not shards:
            raise TrainingError(
                f"No TextCtrl {split} shards match {root / pattern}."
            )
        for shard in shards:
            source_dir = shard / source_dir_name
            target_dir = shard / target_dir_name
            if not source_dir.is_dir():
                raise TrainingError(f"Missing TextCtrl source directory: {source_dir}")
            if not target_dir.is_dir():
                raise TrainingError(f"Missing TextCtrl target directory: {target_dir}")
            source_records = _read_upstream_labels(shard / source_labels_name)
            target_records = _read_upstream_labels(shard / target_labels_name)
            source_names = [name for name, _label in source_records]
            target_names = [name for name, _label in target_records]
            if source_names != target_names:
                raise TrainingError(
                    "TextCtrl requires source and target labels in identical "
                    f"filename order under {shard}."
                )
            for filename in source_names:
                source_image = source_dir / filename
                target_image = target_dir / filename
                if not source_image.is_file():
                    raise TrainingError(
                        f"Missing TextCtrl source training image: {source_image}"
                    )
                if not target_image.is_file():
                    raise TrainingError(
                        f"Missing TextCtrl target training image: {target_image}"
                    )
    return root


def build_textctrl_training_config(
    config: Mapping[str, Any],
    destination: Path,
) -> Path:
    """Translate the Hydra experiment into TextCtrl's legacy OmegaConf schema."""

    network = config["network"]
    dataset = config["task"]["dataset"]
    training = config["training"]
    repository = resolve_path(str(network["repository_dir"]))
    upstream_config = repository / "configs" / "train.yaml"
    if not upstream_config.is_file():
        raise TrainingError(f"Missing upstream TextCtrl config: {upstream_config}")
    if str(dataset.get("format")) != "textctrl_shards":
        raise TrainingError(
            "TextCtrl fine-tuning requires a textctrl_shards dataset config."
        )

    root = resolve_path(str(dataset["root_dir"]))
    fonts = root / "fonts"
    weights = resolve_path(str(network["weights_dir"]))
    composed = OmegaConf.load(upstream_config)
    composed.data.batch_size = int(training["batch_size"])
    composed.data.train.root_dir = str(root / "train")
    composed.data.train.font_dir = str(fonts)
    composed.data.validation.root_dir = str(root / "eval")
    composed.data.validation.font_dir = str(fonts)

    base = composed.model.params.base_config
    base.scheduler_config = str(weights / "sd/scheduler/scheduler_config.json")
    base.vae.pretrained = str(weights / "sd/vae")
    base.text_encoder.params.ckpt_path = str(weights / "text_encoder.pth")
    base.unet_pretrained = str(weights / "sd/unet/diffusion_pytorch_model.bin")
    base.font_path = str(fonts / "arial.ttf")
    base.ocr_model.pretrained = str(weights / "ocr_model.pth")
    charset = repository / "src/module/abinet/data/charset_36.txt"
    base.ocr_model.charset_path = str(charset)
    base.ocr_model.vision.charset_path = str(charset)
    base.ocr_model.language.charset_path = str(charset)
    base.ocr_model.alignment.charset_path = str(charset)
    base.vgg_weight = str(weights / "vgg19.pth")

    composed.lightning.max_epochs = int(training["max_epochs"])
    composed.lightning.accumulate_grad_batches = int(
        training["accumulate_grad_batches"]
    )
    composed.lightning.accelerator = str(training.get("accelerator", "gpu"))
    composed.lightning.strategy = str(training.get("strategy", "auto"))
    composed.lightning.default_root_dir = str(
        resolve_path(str(config["output_dir"]))
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(composed, destination)
    return destination


def run_textctrl_training(config: Mapping[str, Any]) -> Path:
    """Build the legacy config and optionally launch the isolated worker."""

    network = config["network"]
    if str(network["backend"]) != "textctrl_subprocess":
        raise TrainingError("Only the TextCtrl backend currently supports training.")
    missing = validate_textctrl_installation(network)
    weights_dir = resolve_path(str(network["weights_dir"]))
    for relative in network.get("required_training_checkpoints", []):
        path = weights_dir / str(relative)
        if not path.is_file():
            missing.append(path)
    if missing:
        formatted = "\n  - ".join(str(path) for path in missing)
        raise TrainingError(
            "TextCtrl training prerequisites are missing:\n"
            f"  - {formatted}"
        )
    validate_textctrl_training_data(config["task"]["dataset"])

    output_dir = resolve_path(str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_config = build_textctrl_training_config(
        config,
        output_dir / "textctrl_train.generated.yaml",
    )
    if bool(config.get("validate_only", False)) or not bool(
        config["training"].get("execute", False)
    ):
        return generated_config

    repository = resolve_path(str(network["repository_dir"]))
    worker = Path(__file__).with_name("textctrl_train_worker.py").resolve()
    command = [
        str(network.get("python_executable", "python3")),
        str(worker),
        "--repository",
        str(repository),
        "--config",
        str(generated_config),
        "--style-checkpoint",
        str(resolve_path(str(network["weights_dir"])) / "style_encoder.pth"),
        "--learning-rate",
        str(float(config["training"]["learning_rate"])),
        "--checkpoint-frequency",
        str(int(config["training"]["checkpoint_every_n_epochs"])),
        "--precision",
        str(config["training"].get("precision", 32)),
        "--seed",
        str(int(config.get("seed", 42))),
    ]
    if bool(config["training"].get("sd_locked", False)):
        command.append("--sd-locked")
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(repository)
        if not existing_pythonpath
        else os.pathsep.join((str(repository), existing_pythonpath))
    )
    try:
        subprocess.run(command, cwd=repository, env=environment, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise TrainingError(
            "TextCtrl fine-tuning failed in its isolated environment."
        ) from exc
    return generated_config
