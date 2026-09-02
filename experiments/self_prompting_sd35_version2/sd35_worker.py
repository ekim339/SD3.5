"""Run deterministic Self-Prompting SD3.5 version-2 jobs from JSONL.

Only manifest records whose ``model`` is ``self_prompting_sd35_version2`` are
processed. Heavy ML imports are delayed until all pending inputs, masks, and
checkpoint artifacts have passed lightweight validation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    from .generation_provenance import (
        GenerationProvenanceError,
        begin as begin_generation_provenance,
        finish as finish_generation_provenance,
    )
except ImportError:  # Direct execution: the worker directory is on sys.path.
    from generation_provenance import (  # type: ignore[no-redef]
        GenerationProvenanceError,
        begin as begin_generation_provenance,
        finish as finish_generation_provenance,
    )

MODEL_FILTER = "self_prompting_sd35_version2"
DEFAULT_CHECKPOINT = Path(
    "/home/ekim339/project/SD3.5/CODEX/self_prompting_sd35/"
    "checkpoints/version2/checkpoint-030000"
)
DEFAULT_BASE_MODEL = "stabilityai/stable-diffusion-3.5-medium"
LORA_WEIGHT_NAME = "pytorch_lora_weights.safetensors"
INPUT_PROJECTION_NAME = "input_projection.safetensors"
SUPPORTED_OUTPUT_FORMATS = {
    ".bmp": "BMP",
    ".jpeg": "JPEG",
    ".jpg": "JPEG",
    ".png": "PNG",
    ".tif": "TIFF",
    ".tiff": "TIFF",
    ".webp": "WEBP",
}
DTYPE_ALIASES = {
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fp16": "float16",
    "float16": "float16",
    "fp32": "float32",
    "float32": "float32",
}


class WorkerError(RuntimeError):
    """An actionable manifest, configuration, or inference failure."""


@dataclass(frozen=True)
class Job:
    """One validated version-2 inference request."""

    line_number: int
    input_path: Path
    mask_path: Path
    target_text: str
    output_path: Path
    generation_seed: int
    sample_id: str


def _required_string(record: dict[str, Any], key: str, line_number: int) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise WorkerError(
            f"Manifest line {line_number}: {key!r} must be a non-empty string"
        )
    return value


def _resolve_path(value: str, manifest_directory: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_directory / path
    return path.resolve(strict=False)


def _parse_seed(value: Any, line_number: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise WorkerError(
            f"Manifest line {line_number}: 'generation_seed' must be an integer"
        )
    if not 0 <= value < 2**63:
        raise WorkerError(
            f"Manifest line {line_number}: 'generation_seed' must be in [0, 2**63)"
        )
    return value


def load_jobs(manifest: Path) -> tuple[list[Job], int]:
    """Parse matching jobs and return them with the other-model row count."""

    manifest = manifest.expanduser().resolve(strict=False)
    if not manifest.is_file():
        raise WorkerError(f"Manifest is not a readable file: {manifest}")

    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise WorkerError(f"Could not read manifest {manifest}: {exc}") from exc

    jobs: list[Job] = []
    ignored = 0
    manifest_directory = manifest.parent
    for line_number, raw_line in enumerate(lines, 1):
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise WorkerError(
                f"Manifest line {line_number}: invalid JSON "
                f"({exc.msg} at column {exc.colno})"
            ) from exc
        if not isinstance(record, dict):
            raise WorkerError(
                f"Manifest line {line_number}: each JSONL entry must be an object"
            )
        model = record.get("model")
        if not isinstance(model, str) or not model:
            raise WorkerError(
                f"Manifest line {line_number}: 'model' must be a non-empty string"
            )
        if model != MODEL_FILTER:
            ignored += 1
            continue

        input_path = _resolve_path(
            _required_string(record, "input_path", line_number), manifest_directory
        )
        mask_path = _resolve_path(
            _required_string(record, "mask_path", line_number), manifest_directory
        )
        output_path = _resolve_path(
            _required_string(record, "output_path", line_number), manifest_directory
        )
        target_text = _required_string(record, "target_text", line_number)
        generation_seed = _parse_seed(record.get("generation_seed"), line_number)
        sample_id = str(
            record.get(
                "sample_id",
                record.get("sample_index", record.get("source_id", output_path.stem)),
            )
        )

        if output_path in (input_path, mask_path):
            raise WorkerError(
                f"Manifest line {line_number}: output_path must not overwrite its input or mask"
            )
        if output_path.suffix.lower() not in SUPPORTED_OUTPUT_FORMATS:
            supported = ", ".join(sorted(SUPPORTED_OUTPUT_FORMATS))
            raise WorkerError(
                f"Manifest line {line_number}: unsupported output extension "
                f"{output_path.suffix!r}; choose one of {supported}"
            )
        jobs.append(
            Job(
                line_number=line_number,
                input_path=input_path,
                mask_path=mask_path,
                target_text=target_text,
                output_path=output_path,
                generation_seed=generation_seed,
                sample_id=sample_id,
            )
        )

    if not jobs:
        raise WorkerError(
            f"Manifest contains no jobs with model={MODEL_FILTER!r}: {manifest}"
        )

    output_lines: dict[Path, int] = {}
    for job in jobs:
        previous_line = output_lines.get(job.output_path)
        if previous_line is not None:
            raise WorkerError(
                f"Manifest lines {previous_line} and {job.line_number} have the same "
                f"output_path: {job.output_path}"
            )
        output_lines[job.output_path] = job.line_number
    return jobs, ignored


def normalize_dtype_name(value: str) -> str:
    """Return a canonical Torch dtype name without importing Torch."""

    canonical = DTYPE_ALIASES.get(value.lower())
    if canonical is None:
        choices = ", ".join(sorted(DTYPE_ALIASES))
        raise WorkerError(f"Unsupported dtype {value!r}; choose one of {choices}")
    return canonical


def select_pending_jobs(
    jobs: Iterable[Job], overwrite: bool
) -> tuple[list[Job], list[Job]]:
    """Split jobs into pending and complete while rejecting directory outputs."""

    pending: list[Job] = []
    complete: list[Job] = []
    for job in jobs:
        if job.output_path.exists():
            if job.output_path.is_dir():
                raise WorkerError(
                    f"Manifest line {job.line_number}: output_path is a directory: "
                    f"{job.output_path}"
                )
            if not overwrite:
                complete.append(job)
                continue
        pending.append(job)
    return pending, complete


def validate_options(args: argparse.Namespace) -> None:
    if args.resolution <= 0 or args.resolution % 16:
        raise WorkerError("--resolution must be a positive multiple of 16")
    if args.steps <= 0:
        raise WorkerError("--steps must be a positive integer")
    if args.max_sequence_length <= 0:
        raise WorkerError("--max-sequence-length must be a positive integer")
    if not args.base_model:
        raise WorkerError("--base-model must not be empty")
    normalize_dtype_name(args.dtype)

    checkpoint = args.checkpoint.expanduser().resolve(strict=False)
    if not checkpoint.is_dir():
        raise WorkerError(f"Checkpoint directory does not exist: {checkpoint}")
    for artifact in (LORA_WEIGHT_NAME, INPUT_PROJECTION_NAME):
        artifact_path = checkpoint / artifact
        if not artifact_path.is_file():
            raise WorkerError(
                f"Version-2 checkpoint does not contain {artifact}: {checkpoint}"
            )
        if artifact_path.stat().st_size == 0:
            raise WorkerError(
                f"Version-2 checkpoint artifact is empty: {artifact_path}"
            )
    args.checkpoint = checkpoint

    if args.font_path is not None:
        font_path = args.font_path.expanduser().resolve(strict=False)
        if not font_path.is_file():
            raise WorkerError(f"Font file does not exist: {font_path}")
        args.font_path = font_path


def validate_pending_paths(jobs: Iterable[Job]) -> None:
    """Validate filesystem inputs before allocating the model."""

    for job in jobs:
        if not job.input_path.is_file():
            raise WorkerError(
                f"Manifest line {job.line_number}: input image does not exist: "
                f"{job.input_path}"
            )
        if not job.mask_path.is_file():
            raise WorkerError(
                f"Manifest line {job.line_number}: mask image does not exist: "
                f"{job.mask_path}"
            )


def validate_image_pairs(jobs: Iterable[Job], image_module: Any) -> None:
    """Check image decodability and source/mask alignment before model loading."""

    for job in jobs:
        try:
            with image_module.open(job.input_path) as opened:
                source_size = opened.size
                opened.verify()
            with image_module.open(job.mask_path) as opened:
                mask_size = opened.size
                opened.verify()
        except Exception as exc:
            raise WorkerError(
                f"Manifest line {job.line_number}: could not decode input/mask image: {exc}"
            ) from exc
        if source_size != mask_size:
            raise WorkerError(
                f"Manifest line {job.line_number}: source size {source_size} does not "
                f"match mask size {mask_size}"
            )


def fitted_content_box(
    image: Any,
    canvas_size: tuple[int, int],
    resample: Any,
) -> tuple[int, int, int, int]:
    """Return the centered content box produced by the shared ``fit_canvas``."""

    fitted = image.copy()
    fitted.thumbnail(canvas_size, resample)
    left = (canvas_size[0] - fitted.width) // 2
    top = (canvas_size[1] - fitted.height) // 2
    return left, top, left + fitted.width, top + fitted.height


def _ensure_repository_import_path() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    root_string = str(repository_root)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)


def _import_pillow() -> Any:
    try:
        from PIL import Image
    except ImportError as exc:
        raise WorkerError(
            "Pillow is required; install CODEX/self_prompting_sd35/requirements.txt"
        ) from exc
    return Image


def _load_runtime(dtype_name: str) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Import the version-2 runtime lazily and resolve the Torch dtype."""

    _ensure_repository_import_path()
    try:
        import torch
        from diffusers import StableDiffusion3Pipeline

        from CODEX.self_prompting_sd35.conditioning import (
            encode_t5_target_conditioning,
        )
        from CODEX.self_prompting_sd35.dataset import prepare_conditions
        from CODEX.self_prompting_sd35.model import SelfPromptingSD35
    except ImportError as exc:
        raise WorkerError(
            "Self-Prompting runtime dependencies are unavailable; install "
            "CODEX/self_prompting_sd35/requirements.txt. "
            f"Original import error: {exc}"
        ) from exc

    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_name]
    return (
        torch,
        StableDiffusion3Pipeline,
        SelfPromptingSD35,
        prepare_conditions,
        encode_t5_target_conditioning,
        torch_dtype,
    )


def _load_pipeline_and_adapter(
    *,
    torch: Any,
    pipeline_class: Any,
    model_class: Any,
    base_model: str,
    checkpoint: Path,
    torch_dtype: Any,
    device: Any,
    revision: str | None = None,
) -> tuple[Any, Any]:
    """Build the 65-channel model and load its projection plus attention LoRA."""

    del torch  # Retained in this interface to match the worker's injected runtime.
    try:
        load_options = {"dtype": torch_dtype}
        if revision:
            load_options["revision"] = revision
        pipe = pipeline_class.from_pretrained(base_model, **load_options).to(device)
        model = model_class(pipe).to(device)
        # Version 2 stores the fully trained 65-channel input projection beside
        # the attention LoRA; the wrapper loader is the only supported path for
        # restoring both artifacts.
        model.load_lora_weights(checkpoint)
        model.eval()
    except Exception as exc:
        raise WorkerError(
            f"Could not load base model {base_model!r} with version-2 checkpoint "
            f"{checkpoint}: {exc}"
        ) from exc
    return pipe, model


def _atomic_save(image: Any, output_path: Path) -> None:
    """Save atomically so interrupted jobs remain safely resumable."""

    image_format = SUPPORTED_OUTPUT_FORMATS[output_path.suffix.lower()]
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output_path.stem}.",
            suffix=output_path.suffix,
            dir=output_path.parent,
        )
        os.close(descriptor)
    except OSError as exc:
        raise WorkerError(f"Could not prepare output path {output_path}: {exc}") from exc

    temporary_path = Path(temporary_name)
    try:
        image.save(temporary_path, format=image_format)
        os.replace(temporary_path, output_path)
    except Exception as exc:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise WorkerError(f"Could not save output image {output_path}: {exc}") from exc


def _run_job(
    job: Job,
    *,
    torch: Any,
    image_module: Any,
    pipe: Any,
    model: Any,
    prepare_conditions: Any,
    encode_target_conditioning: Any,
    device: Any,
    torch_dtype: Any,
    resolution: int,
    steps: int,
    max_sequence_length: int,
    font_path: Path | None,
) -> None:
    try:
        with image_module.open(job.input_path) as opened:
            source = opened.convert("RGB")
        with image_module.open(job.mask_path) as opened:
            mask_image = opened.convert("L")

        prepared = prepare_conditions(
            source,
            mask_image,
            job.target_text,
            resolution,
            font_path=font_path,
        )
        tensors = {
            key: value.unsqueeze(0).to(device=device, dtype=torch_dtype)
            for key, value in prepared.items()
        }

        with torch.inference_mode():
            # This shared helper gives both CLIP towers empty prompts and sends
            # the target string only through T5 (prompt_3).
            prompt, pooled = encode_target_conditioning(
                pipe,
                job.target_text,
                device=device,
                max_sequence_length=max_sequence_length,
            )
            masked = model.encode_images(tensors["masked_image"], sample=False)
            glyph = model.encode_images(tensors["glyph_image"], sample=False)
            style = model.encode_images(tensors["style_image"], sample=False)

            generator = torch.Generator(device=device).manual_seed(job.generation_seed)
            latent = torch.randn(
                masked.shape,
                generator=generator,
                device=device,
                dtype=torch_dtype,
            )
            pipe.scheduler.set_timesteps(steps, device=device)
            latent = latent * getattr(pipe.scheduler, "init_noise_sigma", 1.0)
            for timestep in pipe.scheduler.timesteps:
                prediction = model.transformer(
                    hidden_states=model.composite_input(
                        latent,
                        masked,
                        glyph,
                        style,
                        tensors["mask"],
                    ),
                    timestep=timestep.expand(latent.shape[0]),
                    encoder_hidden_states=prompt,
                    pooled_projections=pooled,
                    return_dict=True,
                ).sample
                latent = pipe.scheduler.step(
                    prediction, timestep, latent, return_dict=False
                )[0]

            shift = getattr(pipe.vae.config, "shift_factor", 0.0) or 0.0
            decoded = pipe.vae.decode(
                latent / pipe.vae.config.scaling_factor + shift,
                return_dict=False,
            )[0]
            edited = decoded.add(1).div(2).clamp(0, 1)
            source_tensor = tensors["source_image"].add(1).div(2)
            edited = edited * tensors["mask"] + source_tensor * (1.0 - tensors["mask"])
            pixels = (
                edited[0]
                .float()
                .cpu()
                .permute(1, 2, 0)
                .mul(255)
                .round()
                .clamp(0, 255)
                .to(torch.uint8)
                .numpy()
            )

        square_output = image_module.fromarray(pixels, mode="RGB")
        content_box = fitted_content_box(
            source,
            (resolution, resolution),
            image_module.Resampling.BICUBIC,
        )
        restored_output = square_output.crop(content_box).resize(
            source.size,
            image_module.Resampling.BICUBIC,
        )
        _atomic_save(restored_output, job.output_path)
    except WorkerError:
        raise
    except Exception as exc:
        raise WorkerError(
            f"Manifest line {job.line_number} (sample {job.sample_id!r}) failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run resumable Self-Prompting SD3.5 version-2 JSONL jobs."
    )
    parser.add_argument("--manifest", type=Path, required=True, help="JSONL job manifest")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Version-2 checkpoint directory containing LoRA and input projection",
    )
    parser.add_argument(
        "--base-model",
        "--model",
        dest="base_model",
        default=DEFAULT_BASE_MODEL,
        help="Matching SD3.5 base model name or local path",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional pinned base-model commit or revision",
    )
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument(
        "--max-sequence-length",
        "--max_sequence_length",
        dest="max_sequence_length",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--font-path",
        "--font",
        dest="font_path",
        type=Path,
        default=None,
        help="Optional TrueType/OpenType glyph-rendering font",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        help="bfloat16/bf16, float16/fp16, or float32/fp32",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate outputs that already exist instead of resuming past them",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    jobs, ignored = load_jobs(args.manifest)
    pending, complete = select_pending_jobs(jobs, args.overwrite)
    validate_options(args)
    try:
        provenance_path, provenance_signature = begin_generation_provenance(
            args, jobs, Path(__file__)
        )
    except GenerationProvenanceError as exc:
        raise WorkerError(str(exc)) from exc
    if ignored:
        print(f"Ignored {ignored} manifest job(s) for other models.", flush=True)
    if complete:
        print(f"Skipping {len(complete)} existing output(s).", flush=True)
    if not pending:
        try:
            finish_generation_provenance(provenance_path, provenance_signature)
        except GenerationProvenanceError as exc:
            raise WorkerError(str(exc)) from exc
        print(
            f"All {len(jobs)} {MODEL_FILTER} job(s) are already complete.",
            flush=True,
        )
        return 0

    validate_pending_paths(pending)
    image_module = _import_pillow()
    validate_image_pairs(pending, image_module)

    dtype_name = normalize_dtype_name(args.dtype)
    (
        torch,
        pipeline_class,
        model_class,
        prepare_conditions,
        encode_target_conditioning,
        torch_dtype,
    ) = _load_runtime(dtype_name)
    if not torch.cuda.is_available():
        raise WorkerError("Self-Prompting SD3.5 inference requires a CUDA device")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = False
    pipe, model = _load_pipeline_and_adapter(
        torch=torch,
        pipeline_class=pipeline_class,
        model_class=model_class,
        base_model=args.base_model,
        checkpoint=args.checkpoint,
        torch_dtype=torch_dtype,
        device=device,
        revision=args.revision,
    )

    total = len(pending)
    for index, job in enumerate(pending, 1):
        print(
            f"[{index}/{total}] sample={job.sample_id!r} seed={job.generation_seed} "
            f"-> {job.output_path}",
            flush=True,
        )
        _run_job(
            job,
            torch=torch,
            image_module=image_module,
            pipe=pipe,
            model=model,
            prepare_conditions=prepare_conditions,
            encode_target_conditioning=encode_target_conditioning,
            device=device,
            torch_dtype=torch_dtype,
            resolution=args.resolution,
            steps=args.steps,
            max_sequence_length=args.max_sequence_length,
            font_path=args.font_path,
        )

    try:
        finish_generation_provenance(provenance_path, provenance_signature)
    except GenerationProvenanceError as exc:
        raise WorkerError(str(exc)) from exc
    print(f"Completed {total} {MODEL_FILTER} job(s).", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except WorkerError as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
