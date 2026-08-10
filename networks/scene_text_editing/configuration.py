"""Configuration validation shared by the Hydra entrypoints."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


class ConfigurationError(ValueError):
    """Raised when composed Hydra groups describe an invalid experiment."""


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _get(mapping: Mapping[str, Any], key: str, default: Any = None) -> Any:
    value = mapping.get(key, default)
    return value


def resolve_path(path: str | Path, *, base_dir: str | Path | None = None) -> Path:
    """Resolve a user/config path without depending on Hydra's run directory."""

    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    base = Path(base_dir).expanduser() if base_dir is not None else PROJECT_ROOT
    return (base / candidate).resolve()


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate cross-group compatibility after Hydra has composed a config."""

    network = _get(config, "network")
    diffusion = _get(config, "diffusion")
    task = _get(config, "task")
    if not isinstance(network, Mapping):
        raise ConfigurationError("The composed config is missing the network group.")
    if not isinstance(diffusion, Mapping):
        raise ConfigurationError("The composed config is missing the diffusion group.")
    if not isinstance(task, Mapping):
        raise ConfigurationError("The composed config is missing the task group.")

    network_name = str(_get(network, "name", ""))
    backend = str(_get(network, "backend", ""))
    diffusion_name = str(_get(diffusion, "name", ""))
    supported = [str(item) for item in _get(network, "supported_diffusions", [])]
    if diffusion_name not in supported:
        supported_text = ", ".join(supported) if supported else "none"
        raise ConfigurationError(
            f"network={network_name!r} does not support diffusion={diffusion_name!r}; "
            f"supported values: {supported_text}."
        )

    if backend == "textctrl_subprocess" and diffusion_name != "pndm":
        raise ConfigurationError(
            "The released TextCtrl SD1.5 checkpoint is epsilon-trained and must use "
            "diffusion=pndm. It cannot be sampled as a flow-matching model."
        )
    if backend == "textctrl_subprocess" and not bool(
        _get(diffusion, "skip_prk_steps", False)
    ):
        raise ConfigurationError(
            "TextCtrl's released scheduler fixes diffusion.skip_prk_steps=true."
        )
    if backend == "sd3_controlnet_inpaint" and diffusion_name != "flow_matching":
        raise ConfigurationError(
            "The SD3 transformer editing backend requires diffusion=flow_matching."
        )
    if network_name == "sd35_medium" and not bool(
        _get(network, "allow_experimental_base_model", False)
    ):
        raise ConfigurationError(
            "The public inpainting ControlNet was trained for SD3 Medium, not SD3.5. "
            "Use network=sd3_inpainting, or explicitly set "
            "network.allow_experimental_base_model=true after accepting the risk."
        )

    mode = str(_get(config, "mode", "inference"))
    capability = "supports_training" if mode == "train" else "supports_inference"
    if not bool(_get(network, capability, False)):
        raise ConfigurationError(
            f"network={network_name!r} does not declare {capability}=true."
        )

    if str(_get(task, "name", "")) != "text_image_editing":
        raise ConfigurationError(
            "This entrypoint requires task.name=text_image_editing."
        )
    dataset = _get(task, "dataset")
    prompts = _get(task, "prompts")
    if not isinstance(dataset, Mapping):
        raise ConfigurationError("The task dataset config was not composed.")
    if not isinstance(prompts, Mapping):
        raise ConfigurationError("The task prompt config was not composed.")

    generation = _get(task, "generation", {})
    if not isinstance(generation, Mapping):
        raise ConfigurationError("task.generation must be a mapping.")
    for dimension in ("width", "height"):
        value = int(_get(generation, dimension, 0))
        if value <= 0 or value % 16:
            raise ConfigurationError(
                f"task.generation.{dimension} must be positive and divisible by 16."
            )

    steps = int(_get(diffusion, "num_inference_steps", 0))
    if steps <= 0:
        raise ConfigurationError("diffusion.num_inference_steps must be positive.")

