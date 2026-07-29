"""Factory for native SD3 and isolated legacy TextCtrl inference backends."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scene_text_editing.checkpoints import (
    CheckpointError,
    validate_textctrl_installation,
)
from scene_text_editing.configuration import resolve_path
from scene_text_editing.tasks import TextImageEditingTask


class BackendError(RuntimeError):
    """Raised when an editing backend cannot be prepared or executed."""


def _safe_stem(value: str, fallback: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return stem or fallback


def _unique_output_stems(samples: Sequence[Any]) -> list[str]:
    """Return safe output stems and reject lossy filename collisions."""

    owners: dict[str, str] = {}
    stems: list[str] = []
    for index, sample in enumerate(samples):
        sample_id = str(sample.sample_id)
        stem = _safe_stem(sample_id, f"sample_{index:06d}")
        previous = owners.get(stem)
        if previous is not None:
            raise BackendError(
                f"Sample ids {previous!r} and {sample_id!r} both map to "
                f"output stem {stem!r}. Rename one sample id."
            )
        owners[stem] = sample_id
        stems.append(stem)
    return stems


def _optional_pretrained_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "local_files_only": bool(config.get("local_files_only", False)),
    }
    for name in ("revision", "variant"):
        value = config.get(name)
        if value:
            kwargs[name] = value
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if token:
        kwargs["token"] = token
    return kwargs


class SD3ControlNetInpaintBackend:
    """Text replacement using the Diffusers SD3 ControlNet inpainting pipeline."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config
        self.network = config["network"]
        self.diffusion = config["diffusion"]
        self.task = TextImageEditingTask(config["task"])
        self.pipe: Any = None
        self.torch: Any = None
        self.device = ""
        self.dtype: Any = None

    @staticmethod
    def _choose_device(torch: Any, requested: str) -> str:
        if requested != "auto":
            return requested
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _choose_dtype(torch: Any, requested: str, device: str) -> Any:
        names = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if requested != "auto":
            try:
                return names[requested]
            except KeyError as exc:
                raise BackendError(
                    "dtype must be auto, float32, float16, or bfloat16."
                ) from exc
        if device == "cuda":
            supports_bfloat16 = getattr(torch.cuda, "is_bf16_supported", lambda: False)
            return torch.bfloat16 if supports_bfloat16() else torch.float16
        if device == "mps":
            return torch.float16
        return torch.float32

    def load(self) -> None:
        if self.pipe is not None:
            return
        try:
            import torch
            from diffusers import (
                FlowMatchEulerDiscreteScheduler,
                SD3ControlNetModel,
                StableDiffusion3ControlNetInpaintingPipeline,
            )
        except ImportError as exc:
            raise BackendError(
                "The SD3 editing backend requires torch and a recent diffusers "
                "release with StableDiffusion3ControlNetInpaintingPipeline."
            ) from exc

        requested_device = str(self.config.get("device", "auto"))
        requested_dtype = str(self.config.get("dtype", "auto"))
        device = self._choose_device(torch, requested_device)
        dtype = self._choose_dtype(torch, requested_dtype, device)
        common_kwargs = _optional_pretrained_kwargs(self.network)

        controlnet_kwargs = dict(common_kwargs)
        controlnet_kwargs.update(
            {
                "torch_dtype": dtype,
                "use_safetensors": bool(
                    self.network.get("use_safetensors", True)
                ),
                "extra_conditioning_channels": int(
                    self.network.get("extra_conditioning_channels", 1)
                ),
            }
        )
        try:
            controlnet = SD3ControlNetModel.from_pretrained(
                str(self.network["controlnet_model_id"]),
                **controlnet_kwargs,
            )
            pipe_kwargs = dict(common_kwargs)
            pipe_kwargs.update(
                {
                    "controlnet": controlnet,
                    "torch_dtype": dtype,
                    "use_safetensors": bool(
                        self.network.get("use_safetensors", True)
                    ),
                }
            )
            pipe = StableDiffusion3ControlNetInpaintingPipeline.from_pretrained(
                str(self.network["base_model_id"]),
                **pipe_kwargs,
            )
        except Exception as exc:
            raise BackendError(
                "Could not load the SD3 base model and inpainting ControlNet. "
                "Run `python -m scene_text_editing.download_models`, accept any "
                "gated Hugging Face terms, and authenticate before retrying. "
                f"Original error: {exc}"
            ) from exc

        scheduler_overrides: dict[str, Any] = {}
        for key in ("shift", "use_dynamic_shifting"):
            value = self.diffusion.get(key)
            if value is not None:
                scheduler_overrides[key] = value
        pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            pipe.scheduler.config,
            **scheduler_overrides,
        )

        lora_path = self.network.get("lora_path")
        if lora_path:
            pipe.load_lora_weights(str(resolve_path(str(lora_path))))

        if device == "cuda" and bool(
            self.network.get("enable_model_cpu_offload", True)
        ):
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)

        self.pipe = pipe
        self.torch = torch
        self.device = device
        self.dtype = dtype

    @staticmethod
    def _prepare_control_images(
        source_path: Path,
        mask_path: Path | None,
        *,
        width: int,
        height: int,
        full_mask_when_missing: bool,
    ) -> tuple[Any, Any, tuple[int, int, int, int], tuple[int, int]]:
        try:
            from PIL import Image
        except ImportError as exc:
            raise BackendError("Pillow is required to load editing images.") from exc

        with Image.open(source_path) as opened:
            source = opened.convert("RGB")
        original_size = source.size
        scale = min(width / source.width, height / source.height)
        fitted_size = (
            max(1, round(source.width * scale)),
            max(1, round(source.height * scale)),
        )
        source = source.resize(fitted_size, Image.Resampling.LANCZOS)
        left = (width - fitted_size[0]) // 2
        top = (height - fitted_size[1]) // 2
        box = (left, top, left + fitted_size[0], top + fitted_size[1])

        control = Image.new("RGB", (width, height), color=(0, 0, 0))
        control.paste(source, (left, top))
        mask = Image.new("L", (width, height), color=0)
        if mask_path is not None:
            with Image.open(mask_path) as opened:
                source_mask = opened.convert("L")
            source_mask = source_mask.resize(fitted_size, Image.Resampling.NEAREST)
            mask.paste(source_mask, (left, top))
        elif full_mask_when_missing:
            mask.paste(Image.new("L", fitted_size, color=255), (left, top))
        else:
            raise BackendError(
                f"No mask was supplied for {source_path}; either add one to the "
                "manifest or enable task.generation.full_crop_mask_when_missing."
            )
        return control, mask, box, original_size

    def run(
        self,
        samples: Sequence[Any],
        output_dir: Path,
        *,
        overwrite: bool,
    ) -> list[dict[str, Any]]:
        self.load()
        output_dir.mkdir(parents=True, exist_ok=True)
        generation = self.task.generation_config
        width = int(generation["width"])
        height = int(generation["height"])
        preserve_size = bool(generation.get("preserve_source_size", True))
        records: list[dict[str, Any]] = []
        image_count = int(generation.get("num_images_per_prompt", 1))
        stems = _unique_output_stems(samples)

        for index, sample in enumerate(samples):
            stem = stems[index]
            planned_paths = (
                [output_dir / f"{stem}.png"]
                if image_count == 1
                else [
                    output_dir / f"{stem}_{item:02d}.png"
                    for item in range(image_count)
                ]
            )
            if not overwrite and all(path.exists() for path in planned_paths):
                for path in planned_paths:
                    records.append(
                        {
                            "sample_id": str(sample.sample_id),
                            "output_path": str(path),
                            "status": "skipped_existing",
                        }
                    )
                continue

            source_path = Path(sample.source_image)
            mask_value = getattr(sample, "mask_image", None)
            mask_path = Path(mask_value) if mask_value is not None else None
            control, mask, content_box, original_size = self._prepare_control_images(
                source_path,
                mask_path,
                width=width,
                height=height,
                full_mask_when_missing=bool(
                    generation.get("full_crop_mask_when_missing", True)
                ),
            )
            prompt = self.task.prompt_for(sample)
            generator = self.torch.Generator(device="cpu").manual_seed(
                int(self.config.get("seed", 42)) + index
            )
            call_kwargs = {
                "prompt": prompt,
                "negative_prompt": self.task.negative_prompt,
                "height": height,
                "width": width,
                "control_image": control,
                "control_mask": mask,
                "num_inference_steps": int(
                    self.diffusion["num_inference_steps"]
                ),
                "guidance_scale": float(self.diffusion["guidance_scale"]),
                "controlnet_conditioning_scale": float(
                    generation["controlnet_conditioning_scale"]
                ),
                "num_images_per_prompt": image_count,
                "max_sequence_length": int(
                    generation.get("max_sequence_length", 256)
                ),
                "generator": generator,
            }
            result = self.pipe(**call_kwargs)
            images = result.images
            for image_index, image in enumerate(images):
                if preserve_size:
                    from PIL import Image

                    image = image.crop(content_box).resize(
                        original_size,
                        Image.Resampling.LANCZOS,
                    )
                current_path = (
                    output_dir / f"{stem}.png"
                    if len(images) == 1
                    else output_dir / f"{stem}_{image_index:02d}.png"
                )
                if current_path.exists() and not overwrite:
                    records.append(
                        {
                            "sample_id": str(sample.sample_id),
                            "output_path": str(current_path),
                            "status": "skipped_existing",
                        }
                    )
                    continue
                image.save(current_path)
                records.append(
                    {
                        "sample_id": str(sample.sample_id),
                        "source_image": str(source_path),
                        "source_text": str(sample.source_text),
                        "target_text": str(sample.target_text),
                        "prompt": prompt,
                        "output_path": str(current_path),
                        "seed": int(self.config.get("seed", 42)) + index,
                        "status": "generated",
                    }
                )
        return records


class TextCtrlSubprocessBackend:
    """Adapter around TextCtrl's released, legacy CUDA/PyTorch environment."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config
        self.network = config["network"]
        self.diffusion = config["diffusion"]

    def _validate_installation(self) -> tuple[Path, Path]:
        repository = resolve_path(str(self.network["repository_dir"]))
        inference_script = repository / "inference.py"
        if not inference_script.is_file():
            raise BackendError(
                f"TextCtrl checkout not found at {repository}. Run the model "
                "downloader with network=textctrl_sd15 first."
            )

        weights_dir = resolve_path(str(self.network["weights_dir"]))
        expected_weights_dir = (repository / "weights").resolve()
        if weights_dir != expected_weights_dir:
            raise BackendError(
                "The released TextCtrl code hardcodes weights/ paths. "
                f"network.weights_dir must resolve to {expected_weights_dir}."
            )
        try:
            missing_paths = validate_textctrl_installation(self.network)
        except CheckpointError as exc:
            raise BackendError(str(exc)) from exc
        missing = [str(path) for path in missing_paths]
        if missing:
            formatted = "\n  - ".join(missing)
            raise BackendError(
                "TextCtrl is missing required source/checkpoints/components:\n"
                f"  - {formatted}\nRun the model downloader and verify its output."
            )
        return repository, resolve_path(str(self.network["checkpoint_path"]))

    @staticmethod
    def _validate_label(label: str, field: str, sample_id: str) -> None:
        if not label or any(character.isspace() for character in label):
            raise BackendError(
                f"TextCtrl's released parser requires a single-token {field}; "
                f"sample {sample_id!r} contains {label!r}. Use the native SD3 "
                "backend for phrases containing whitespace."
            )

    def run(
        self,
        samples: Sequence[Any],
        output_dir: Path,
        *,
        overwrite: bool,
    ) -> list[dict[str, Any]]:
        repository, checkpoint = self._validate_installation()
        output_dir.mkdir(parents=True, exist_ok=True)
        runnable: list[tuple[Any, str, Path]] = []
        records: list[dict[str, Any]] = []
        stems = _unique_output_stems(samples)
        for index, sample in enumerate(samples):
            sample_id = str(sample.sample_id)
            stem = stems[index]
            destination = output_dir / f"{stem}.png"
            if destination.exists() and not overwrite:
                records.append(
                    {
                        "sample_id": sample_id,
                        "output_path": str(destination),
                        "status": "skipped_existing",
                    }
                )
                continue
            self._validate_label(str(sample.source_text), "source_text", sample_id)
            self._validate_label(str(sample.target_text), "target_text", sample_id)
            runnable.append((sample, f"{index:06d}.png", destination))
        if not runnable:
            return records

        with tempfile.TemporaryDirectory(
            prefix="textctrl-input-",
            dir=output_dir,
        ) as temporary:
            staging = Path(temporary)
            source_dir = staging / "i_s"
            raw_output = staging / "result"
            source_dir.mkdir()
            source_lines: list[str] = []
            target_lines: list[str] = []
            try:
                from PIL import Image
            except ImportError as exc:
                raise BackendError("Pillow is required to stage TextCtrl inputs.") from exc

            for sample, staged_name, _destination in runnable:
                with Image.open(sample.source_image) as opened:
                    opened.convert("RGB").save(source_dir / staged_name)
                source_lines.append(f"{staged_name} {sample.source_text}")
                target_lines.append(f"{staged_name} {sample.target_text}")
            (staging / "i_s.txt").write_text(
                "\n".join(source_lines) + "\n",
                encoding="utf-8",
            )
            (staging / "i_t.txt").write_text(
                "\n".join(target_lines) + "\n",
                encoding="utf-8",
            )

            worker = (
                Path(__file__).resolve().parent.parent
                / "textctrl_inference_worker.py"
            )
            command = [
                str(self.network.get("python_executable", "python3")),
                str(worker),
                "--repository",
                str(repository),
                "--checkpoint",
                str(checkpoint),
                "--dataset-dir",
                str(staging),
                "--output-dir",
                str(raw_output),
                "--seed",
                str(int(self.config.get("seed", 42))),
                "--starting-layer",
                str(int(self.network.get("starting_layer", 10))),
                "--num-inference-steps",
                str(int(self.diffusion["num_inference_steps"])),
                "--guidance-scale",
                str(float(self.diffusion["guidance_scale"])),
            ]
            environment = os.environ.copy()
            existing_pythonpath = environment.get("PYTHONPATH")
            environment["PYTHONPATH"] = (
                str(repository)
                if not existing_pythonpath
                else os.pathsep.join((str(repository), existing_pythonpath))
            )
            try:
                subprocess.run(
                    command,
                    cwd=repository,
                    env=environment,
                    check=True,
                )
            except (OSError, subprocess.CalledProcessError) as exc:
                raise BackendError(
                    "The isolated TextCtrl inference process failed. Confirm "
                    "TEXTCTRL_PYTHON points to its Python 3.8/CUDA environment."
                ) from exc

            for sample, staged_name, destination in runnable:
                generated = raw_output / staged_name
                if not generated.is_file():
                    raise BackendError(
                        f"TextCtrl did not produce the expected file {generated}."
                    )
                shutil.copy2(generated, destination)
                records.append(
                    {
                        "sample_id": str(sample.sample_id),
                        "source_image": str(sample.source_image),
                        "source_text": str(sample.source_text),
                        "target_text": str(sample.target_text),
                        "output_path": str(destination),
                        "seed": int(self.config.get("seed", 42)),
                        "status": "generated",
                    }
                )
        return records


def create_backend(config: Mapping[str, Any]) -> Any:
    backend = str(config["network"]["backend"])
    if backend == "sd3_controlnet_inpaint":
        return SD3ControlNetInpaintBackend(config)
    if backend == "textctrl_subprocess":
        return TextCtrlSubprocessBackend(config)
    raise BackendError(f"Unknown editing backend: {backend!r}.")

