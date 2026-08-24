"""Single config-driven entry point for SD3.5 and TextCtrl."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from networks.scene_text_editing.configuration import resolve_path


class UnifiedConfigError(ValueError):
    """Raised when base_config selects an incompatible combination."""


def _resolved(config: DictConfig) -> dict[str, Any]:
    value = OmegaConf.to_container(config, resolve=True)
    if not isinstance(value, dict):
        raise UnifiedConfigError("configs/base_config.yaml must compose to a mapping")
    return value


def _validate(config: Mapping[str, Any]) -> None:
    mode = str(config.get("mode", ""))
    if mode not in {"inference", "train"}:
        raise UnifiedConfigError("mode must be either 'inference' or 'train'")
    backend = str(config["network"].get("backend", ""))
    diffusion = str(config["diffusion"].get("name", ""))
    compatible = {
        "sd35_lora": "flow_matching",
        "textctrl_subprocess": "pndm",
    }
    if backend not in compatible:
        raise UnifiedConfigError(f"Unknown network backend: {backend!r}")
    if diffusion != "auto" and diffusion != compatible[backend]:
        raise UnifiedConfigError(
            f"network={config['network']['name']} requires "
            f"diffusion={compatible[backend]}, not {diffusion}"
        )
    checkpoint = str(config.get("model", {}).get("checkpoint", "")).strip()
    if not checkpoint:
        raise UnifiedConfigError("model.checkpoint must be set")
    if mode == "inference" and not resolve_path(checkpoint).is_file():
        raise UnifiedConfigError(f"Model checkpoint does not exist: {resolve_path(checkpoint)}")
    if backend == "textctrl_subprocess":
        style = config.get("textctrl_style", {})
        style_encoder = str(style.get("encoder", "textctrl"))
        if style_encoder not in {"textctrl", "residual"}:
            raise UnifiedConfigError(
                "textctrl_style.encoder must be either 'textctrl' or 'residual'"
            )
        if style_encoder == "residual":
            if mode != "inference":
                raise UnifiedConfigError(
                    "Residual TextCtrl style conditioning currently supports inference only"
                )
            required = {
                "residual checkpoint": style.get("residual_checkpoint", ""),
                "channel-adapter checkpoint": style.get("adapter_checkpoint", ""),
                "canonical glyph font": style.get("canonical_font", ""),
            }
            missing = [
                f"{name}: {resolve_path(str(value))}"
                for name, value in required.items()
                if not str(value).strip() or not resolve_path(str(value)).is_file()
            ]
            if missing:
                raise UnifiedConfigError(
                    "Residual TextCtrl style conditioning is missing:\n  - "
                    + "\n  - ".join(missing)
                )
            if int(style.get("resolution", 0)) != 256:
                raise UnifiedConfigError(
                    "textctrl_style.resolution must be 256 for the trained residual extractor"
                )


def _effective_diffusion(config: Mapping[str, Any]) -> dict[str, Any]:
    diffusion = dict(config["diffusion"])
    if diffusion.get("name") == "auto":
        selected = diffusion.get(str(config["network"]["name"]))
        if not isinstance(selected, Mapping):
            raise UnifiedConfigError("diffusion=auto has no profile for the selected network")
        return dict(selected)
    return diffusion


def _runtime_config(config: Mapping[str, Any]) -> dict[str, Any]:
    network = dict(config["network"])
    network["checkpoint_path"] = str(config["model"]["checkpoint"])
    return {
        "mode": config["mode"],
        "seed": config["seed"],
        "device": config["device"],
        "dtype": config["dtype"],
        "offline": config["offline"],
        "limit": config.get("limit"),
        "overwrite": config.get("overwrite", False),
        "output_dir": config["output_dir"],
        "network": network,
        "diffusion": _effective_diffusion(config),
        "conditioning": dict(network.get("conditioning", {})),
        "textctrl_style": dict(config.get("textctrl_style", {})),
        "task": {
            "name": "text_image_editing",
            "dataset": dict(config["dataset"]),
            "prompts": {"name": "base", **dict(config["prompt"])},
            "generation": dict(config["generation"]),
        },
    }


def _run_inference(config: DictConfig, resolved: Mapping[str, Any]) -> None:
    from networks.scene_text_editing.datasets import load_dataset

    runtime = _runtime_config(resolved)
    samples = list(load_dataset(runtime["task"]["dataset"]))
    limit = runtime.get("limit")
    if limit is not None:
        if int(limit) <= 0:
            raise UnifiedConfigError("limit must be null or a positive integer")
        samples = samples[: int(limit)]
    if not samples:
        raise UnifiedConfigError("The selected dataset contains no samples")

    if runtime["network"]["backend"] == "sd35_lora":
        from networks.sd35_implementation.inference_backend import SD35LoRABackend

        backend = SD35LoRABackend(runtime)
    else:
        from networks.scene_text_editing.networks.factory import TextCtrlSubprocessBackend

        backend = TextCtrlSubprocessBackend(runtime)

    if bool(resolved.get("validate_only", False)):
        print(OmegaConf.to_yaml(config, resolve=True))
        print(f"Configuration is valid; found {len(samples)} input sample(s).")
        return

    output_dir = resolve_path(str(runtime["output_dir"]))
    records = backend.run(samples, output_dir, overwrite=bool(runtime["overwrite"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(
        OmegaConf.to_yaml(config, resolve=True), encoding="utf-8"
    )
    (output_dir / "results.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in records),
        encoding="utf-8",
    )
    generated = sum(item.get("status") == "generated" for item in records)
    print(f"Completed {generated} edit(s). Results: {output_dir}")


def _sd35_training_config(config: Mapping[str, Any]) -> DictConfig:
    network = config["network"]
    training = config["training"]
    dataset = config["dataset"]
    roots = list(dataset.get("training_roots") or [])
    return OmegaConf.create(
        {
            "seed": config["seed"],
            "model": {
                "pretrained_model": network["base_model_id"],
                "revision": network.get("revision"),
                "dtype": "bfloat16" if training["mixed_precision"] == "bf16" else training["mixed_precision"],
                "gradient_checkpointing": network.get("gradient_checkpointing", True),
                "enable_xformers": network.get("enable_xformers", False),
                "training_mode": network.get("training_mode", "lora"),
                "max_prompt_length": network.get("max_prompt_length", 256),
                "lora": network["lora"],
            },
            "dataset": {
                "roots": roots,
                "resolution": dataset.get("resolution", config["generation"]["width"]),
                "style_resolution": dataset.get("style_resolution", 256),
                "fill_value": dataset.get("fill_value", network.get("fill_value", 0.5)),
                "limit": dataset.get("training_limit"),
            },
            "encoders": {
                "textctrl_repository": network["textctrl_repository"],
                "style_checkpoint": network["style_checkpoint"],
                "glyph_checkpoint": network["glyph_checkpoint"],
            },
            "conditioning": network["conditioning"],
            "prompt": {"template": config["prompt"]["template"]},
            "training": {
                **dict(training),
                "output_dir": training["output_dir"],
            },
        }
    )


def _run_training(resolved: Mapping[str, Any]) -> None:
    backend = resolved["network"]["backend"]
    if backend == "sd35_lora":
        if bool(resolved.get("validate_only", False)):
            composed = _sd35_training_config(resolved)
            roots = [resolve_path(path) for path in composed.dataset.roots]
            missing = [path for path in roots if not path.is_dir()]
            if missing:
                raise UnifiedConfigError(f"Missing training dataset roots: {missing}")
            print(OmegaConf.to_yaml(composed, resolve=True))
            print("Configuration is valid; no model was loaded.")
            return
        from networks.sd35_implementation.train import run_training

        run_training(_sd35_training_config(resolved))
        return

    from networks.scene_text_editing.training import run_textctrl_training

    runtime = _runtime_config(resolved)
    runtime["training"] = dict(resolved["training"])
    runtime["output_dir"] = resolved["training"]["output_dir"]
    runtime["validate_only"] = bool(resolved.get("validate_only", False))
    generated = run_textctrl_training(runtime)
    print(f"TextCtrl training configuration: {generated}")


@hydra.main(version_base="1.3", config_path="configs", config_name="base_config")
def main(config: DictConfig) -> None:
    resolved = _resolved(config)
    _validate(resolved)
    if resolved["mode"] == "inference":
        _run_inference(config, resolved)
    else:
        _run_training(resolved)


if __name__ == "__main__":
    main()
