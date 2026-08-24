"""Validate configuration and launch isolated legacy TextCtrl fine-tuning."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[2]


def resolve(value):
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def validate(config):
    files = [
        config.textctrl.checkpoint, config.textctrl.text_encoder_checkpoint,
        config.textctrl.ocr_checkpoint, config.textctrl.vgg_checkpoint,
        config.textctrl.scheduler, config.textctrl.unet,
        config.residual.checkpoint, config.residual.adapter_checkpoint,
        config.dataset.canonical_font,
    ]
    missing = [str(resolve(path)) for path in files if not resolve(path).is_file()]
    directories = [config.textctrl.repository, config.textctrl.vae, *config.dataset.roots]
    missing += [str(resolve(path)) for path in directories if not resolve(path).is_dir()]
    if missing:
        raise FileNotFoundError("Missing fine-tuning prerequisites:\n  - " + "\n  - ".join(missing))
    if int(config.dataset.resolution) != 256:
        raise ValueError("TextCtrl and the pretrained residual extractor require resolution 256")
    if int(config.dataset.validation_samples) <= 0:
        raise ValueError("dataset.validation_samples must be positive")
    if int(config.training.max_steps) <= 0:
        raise ValueError("training.max_steps must be positive")


@hydra.main(version_base="1.3", config_path="configs", config_name="train")
def main(config: DictConfig):
    validate(config)
    output = resolve(config.training.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    resolved = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    for section, keys in {
        "textctrl": ("repository", "checkpoint", "text_encoder_checkpoint", "ocr_checkpoint",
                     "vgg_checkpoint", "scheduler", "vae", "unet"),
        "residual": ("checkpoint", "adapter_checkpoint"),
        "dataset": ("canonical_font",),
    }.items():
        for key in keys:
            resolved[section][key] = str(resolve(config[section][key]))
    resolved.dataset.roots = [str(resolve(path)) for path in config.dataset.roots]
    resolved.training.output_dir = str(output)
    if config.training.resume_from:
        resolved.training.resume_from = str(resolve(config.training.resume_from))
    runtime_config = output / "runtime_config.yaml"
    OmegaConf.save(resolved, runtime_config)
    if bool(config.validate_only):
        print(OmegaConf.to_yaml(resolved, resolve=True))
        print("Configuration is valid; training was not launched.")
        return

    repository = resolve(config.textctrl.repository)
    worker = Path(__file__).with_name("worker.py").resolve()
    command = [str(config.textctrl.python_executable), str(worker),
               "--config", str(runtime_config)]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, (
        str(ROOT), str(repository), environment.get("PYTHONPATH", ""))))
    subprocess.run(command, cwd=repository, env=environment, check=True)


if __name__ == "__main__":
    main()
