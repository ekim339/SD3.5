"""Inference backend for the locally trained SD3.5 scene-text LoRA."""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from typing import Any

from networks.scene_text_editing.configuration import resolve_path
from networks.scene_text_editing.tasks import TextImageEditingTask


class SD35LoRABackend:
    """Run MMDiT LoRA with glyph/style tokens and a 33-channel inpainting input."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config
        self.network = config["network"]
        self.diffusion = config["diffusion"]
        self.task = TextImageEditingTask(config["task"])
        self.pipe: Any = None
        self.model: Any = None
        self.torch: Any = None
        self.device = ""
        self.dtype: Any = None

    def load(self) -> None:
        if self.model is not None:
            return
        try:
            import torch
            from diffusers import FlowMatchEulerDiscreteScheduler, StableDiffusion3Pipeline
        except ImportError as exc:
            raise RuntimeError("SD3.5 LoRA inference requires torch, diffusers, and PEFT") from exc

        from networks.scene_text_editing.networks.factory import (
            BackendError,
            SD3ControlNetInpaintBackend,
            _optional_pretrained_kwargs,
        )

        checkpoint = resolve_path(str(self.network["checkpoint_path"]))
        if not checkpoint.is_file():
            raise BackendError(f"SD3.5 adapter checkpoint does not exist: {checkpoint}")
        from .encoders import FrozenTextCtrlGlyphEncoder, FrozenTextCtrlStyleEncoder
        from .model import SD35SceneTextEditor

        device = SD3ControlNetInpaintBackend._choose_device(
            torch, str(self.config.get("device", "auto"))
        )
        dtype = SD3ControlNetInpaintBackend._choose_dtype(
            torch, str(self.config.get("dtype", "auto")), device
        )
        kwargs = _optional_pretrained_kwargs(self.network)
        kwargs.update({"torch_dtype": dtype, "use_safetensors": True})
        try:
            pipe = StableDiffusion3Pipeline.from_pretrained(
                str(self.network["base_model_id"]), **kwargs
            )
        except Exception as exc:
            raise BackendError(f"Could not load SD3.5 Medium: {exc}") from exc
        pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)

        textctrl = resolve_path(str(self.network["textctrl_repository"]))
        style = FrozenTextCtrlStyleEncoder(
            textctrl, resolve_path(str(self.network["style_checkpoint"]))
        )
        glyph = FrozenTextCtrlGlyphEncoder(
            textctrl, resolve_path(str(self.network["glyph_checkpoint"]))
        )
        model = SD35SceneTextEditor(
            pipe,
            style,
            glyph,
            training_mode="lora",
            lora_config=dict(self.network["lora"]),
            fill_value=float(self.network.get("fill_value", 0.5)),
        )
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        incompatible = model.load_state_dict(state, strict=False)
        if incompatible.unexpected_keys:
            raise BackendError(
                f"Checkpoint has unexpected keys: {incompatible.unexpected_keys[:5]}"
            )
        model.requires_grad_(False).eval()
        pipe.to(device)
        model.to(device)
        self.pipe, self.model, self.torch = pipe, model, torch
        self.device, self.dtype = device, dtype

    @staticmethod
    def _render_glyph(text: str, font_path: Path, *, width: int = 128, height: int = 32):
        from PIL import Image, ImageDraw, ImageFont

        canvas = Image.new("RGB", (width, height), color=(127, 127, 127))
        draw = ImageDraw.Draw(canvas)
        size = max(8, height - 4)
        while size > 7:
            font = ImageFont.truetype(str(font_path), size=size)
            box = draw.textbbox((0, 0), text, font=font)
            if box[2] - box[0] <= width - 4 and box[3] - box[1] <= height - 4:
                break
            size -= 1
        x = (width - (box[2] - box[0])) // 2 - box[0]
        y = (height - (box[3] - box[1])) // 2 - box[1]
        draw.text((x, y), text, font=font, fill=(0, 0, 0))
        return canvas

    def _generate(self, sample: Any, index: int):
        import torch.nn.functional as F
        from PIL import Image
        from torchvision.transforms import functional as TF

        generation = self.task.generation_config
        width = int(self.network.get("width", generation["width"]))
        height = int(self.network.get("height", generation["height"]))
        from networks.scene_text_editing.networks.factory import SD3ControlNetInpaintBackend

        source, mask, content_box, original_size = (
            SD3ControlNetInpaintBackend._prepare_control_images(
                Path(sample.source_image),
                Path(sample.mask_image) if sample.mask_image else None,
                width=width,
                height=height,
                full_mask_when_missing=bool(
                    generation.get("full_crop_mask_when_missing", True)
                ),
                allow_upscale=bool(
                    generation.get("allow_upscale", False)
                ),
            )
        )
        from PIL import ImageFilter
        style_box = mask.getbbox()

        expand = max(0, int(generation.get("mask_expand_pixels", 0)))
        feather = max(0.0, float(generation.get("mask_feather_pixels", 0.0)))
        if expand:
            mask = mask.filter(ImageFilter.MaxFilter(2 * expand + 1))
        composite_mask = (
            mask.filter(ImageFilter.GaussianBlur(feather)) if feather else mask
        )
        prompt = self.task.prompt_for(sample)
        source_tensor = TF.to_tensor(source).unsqueeze(0).to(self.device)
        mask_tensor = TF.to_tensor(mask).unsqueeze(0).to(self.device)
        composite_mask_tensor = (
            TF.to_tensor(composite_mask).unsqueeze(0).to(self.device)
        )
        style_source = source.crop(style_box) if style_box is not None else source
        style_scale = min(256 / style_source.width, 256 / style_source.height)
        style_size = (
            max(1, round(style_source.width * style_scale)),
            max(1, round(style_source.height * style_scale)),
        )
        style_source = style_source.resize(style_size, Image.Resampling.BICUBIC)
        style_canvas = Image.new("RGB", (256, 256), color=(0, 0, 0))
        style_canvas.paste(
            style_source,
            ((256 - style_size[0]) // 2, (256 - style_size[1]) // 2),
        )
        style_tensor = TF.to_tensor(style_canvas).unsqueeze(0).to(self.device)
        normalized_source = source_tensor.mul(2).sub(1)

        torch = self.torch
        generator = torch.Generator(device=self.device).manual_seed(
            int(self.config.get("seed", 42)) + index
        )
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=self.dtype, enabled=self.device == "cuda"
        ):
            negative = self.task.negative_prompt or ""
            prompt_embeds, negative_embeds, pooled, negative_pooled = (
                self.pipe.encode_prompt(
                    prompt=prompt,
                    prompt_2=prompt,
                    prompt_3=prompt,
                    negative_prompt=negative,
                    negative_prompt_2=negative,
                    negative_prompt_3=negative,
                    device=self.device,
                    do_classifier_free_guidance=True,
                    max_sequence_length=int(
                        generation.get("max_sequence_length", 256)
                    ),
                )
            )
            target_text_embeds = self.pipe._get_t5_prompt_embeds(
                prompt=[str(sample.target_text)],
                device=self.device,
                max_sequence_length=int(
                    self.config["conditioning"]["target_text_max_length"]
                ),
            )
            style_tokens = self.model.style_encoder(style_tensor)
            glyph_tokens = self.model.glyph_encoder([str(sample.target_text)])
            condition = self.model.projector(
                prompt_embeds,
                target_text_embeds,
                glyph_tokens,
                style_tokens,
            )
            negative_condition = torch.cat(
                (
                    negative_embeds,
                    torch.zeros_like(condition[:, negative_embeds.shape[1] :]),
                ),
                dim=1,
            )
            masked = (
                normalized_source * (1 - mask_tensor)
                + mask_tensor * self.model.normalized_fill_value
            )
            source_latent = self.model._encode_vae(masked)
            latent_mask = F.interpolate(
                mask_tensor, source_latent.shape[-2:], mode="nearest"
            )
            self.pipe.scheduler.set_timesteps(
                int(self.diffusion["num_inference_steps"]), device=self.device
            )
            latents = torch.randn(
                source_latent.shape,
                generator=generator,
                device=self.device,
                dtype=prompt_embeds.dtype,
            )
            guidance = float(self.diffusion["guidance_scale"])
            for timestep in self.pipe.scheduler.timesteps:
                latent_input = torch.cat((latents, latents))
                source_input = torch.cat((source_latent, source_latent))
                mask_input = torch.cat((latent_mask, latent_mask))
                inpainting_input = torch.cat(
                    (latent_input, source_input, mask_input), dim=1
                )
                context = torch.cat((negative_condition, condition))
                pooled_input = torch.cat((negative_pooled, pooled))
                prediction = self.model.transformer(
                    hidden_states=inpainting_input,
                    timestep=timestep.expand(inpainting_input.shape[0]),
                    encoder_hidden_states=context,
                    pooled_projections=pooled_input,
                    return_dict=True,
                ).sample
                unconditional, conditional = prediction.chunk(2)
                prediction = unconditional + guidance * (conditional - unconditional)
                latents = self.pipe.scheduler.step(
                    prediction, timestep, latents
                ).prev_sample
            shift = getattr(self.pipe.vae.config, "shift_factor", 0.0) or 0.0
            decoded = self.pipe.vae.decode(
                latents / self.pipe.vae.config.scaling_factor + shift,
                return_dict=True,
            ).sample
            generated = decoded.float().clamp(-1, 1).add(1).div(2)
            composite = (
                source_tensor * (1 - composite_mask_tensor)
                + generated * composite_mask_tensor
            )
        image = TF.to_pil_image(composite[0].cpu())
        if bool(generation.get("preserve_source_size", True)):
            image = image.crop(content_box).resize(
                original_size, Image.Resampling.LANCZOS
            )
        return image, prompt

    def run(
        self, samples: Sequence[Any], output_dir: Path, *, overwrite: bool
    ) -> list[dict[str, Any]]:
        from networks.scene_text_editing.networks.factory import _unique_output_stems

        self.load()
        output_dir.mkdir(parents=True, exist_ok=True)
        records: list[dict[str, Any]] = []
        stems = _unique_output_stems(samples)
        for index, sample in enumerate(samples):
            destination = output_dir / f"{stems[index]}.png"
            if destination.exists() and not overwrite:
                records.append(
                    {
                        "sample_id": str(sample.sample_id),
                        "output_path": str(destination),
                        "status": "skipped_existing",
                    }
                )
                continue
            image, prompt = self._generate(sample, index)
            image.save(destination)
            records.append(
                {
                    "sample_id": str(sample.sample_id),
                    "source_image": str(sample.source_image),
                    "source_text": str(sample.source_text),
                    "target_text": str(sample.target_text),
                    "prompt": prompt,
                    "output_path": str(destination),
                    "seed": int(self.config.get("seed", 42)) + index,
                    "status": "generated",
                }
            )
        return records
