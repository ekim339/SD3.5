#!/usr/bin/env python3
"""Analyze classifier metrics over SD3.5 token-signal multiplier sweeps.

Edit these constants to choose the tokens and multiplier sweep:

    CLIP_TOKEN_INDICES = [0, 1, 2, 3, 7, 10, 16]
    T5_TOKEN_INDICES = [0, 4, 6, 7, 8]
    MULTIPLIER_VALUES = [0, 0.5, 1, 2, 3, 5, 10]

Set CLIP_TOKEN_INDICES = [] to scale every CLIP token position.
Set T5_TOKEN_INDICES = [] to scale every T5 token position.

For each seed, this produces a metric matrix with shape:

    4 x len(MULTIPLIER_VALUES) x 3

Axis 0: all encoders, clip-l, clip-g, t5
Axis 1: multiplier scale
Axis 2: object_correct probability, text_correct probability, best text-class probability
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


MODEL_ID = "stabilityai/stable-diffusion-3.5-large"
CLIP_SEQUENCE_LENGTH = 77
T5_SEQUENCE_LENGTH = 256
CLIP_L_CHANNELS = (0, 768)
CLIP_G_CHANNELS = (768, 2048)
CLIP_BOTH_CHANNELS = (0, 2048)

CLIP_TOKEN_INDICES = [0, 1, 2, 3, 7, 10, 16]
T5_TOKEN_INDICES = [0, 4, 6, 7, 8]
MULTIPLIER_VALUES = [0, 0.5, 1, 2, 3, 5, 10]

ROW_SPECS = (
    ("all-encoders", "clip-l + clip-g + t5", "all", None),
    ("clip-l", "clip-l only", "clip-l", CLIP_TOKEN_INDICES),
    ("clip-g", "clip-g only", "clip-g", CLIP_TOKEN_INDICES),
    ("t5", "t5 only", "t5", T5_TOKEN_INDICES),
)
METRIC_NAMES = ("object_correct", "text_correct", "best_text_class_probability")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SD3.5 token sweeps, classify generated images, and analyze metrics."
    )
    parser.add_argument("--prompt", required=True, help="Text prompt used for generation.")
    parser.add_argument("--negative-prompt", default=None, help="Optional negative prompt.")
    parser.add_argument("--seeds", type=int, nargs="+", required=True, help="One or more seeds.")
    parser.add_argument(
        "--classifier-checkpoint",
        required=True,
        help="Path to classifier.pt from train_image_classifier.py.",
    )
    parser.add_argument("--output-dir", default="classifier_metric_analysis")
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default=None)
    parser.add_argument(
        "--classifier-device",
        choices=("cuda", "mps", "cpu"),
        default="cpu",
        help="Device for classifier. Defaults to cpu to keep GPU memory free.",
    )
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--matrix-file", default="metrics_matrix.pt")
    parser.add_argument("--json-file", default="metrics_matrix.json")
    parser.add_argument("--report-file", default="classifier_metric_report.txt")
    return parser.parse_args()


def missing_dependency(name: str) -> None:
    print(
        f"Missing dependency: {name}\n\n"
        "Install the runtime packages first, for example:\n"
        "  pip install torch diffusers transformers accelerate sentencepiece protobuf pillow\n",
        file=sys.stderr,
    )
    raise SystemExit(1)


def import_dependencies():
    try:
        import torch
    except ImportError:
        missing_dependency("torch")

    try:
        from diffusers import StableDiffusion3Pipeline
    except ImportError:
        missing_dependency("diffusers")

    return torch, StableDiffusion3Pipeline


def choose_device(torch, requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def choose_dtype(torch, device: str):
    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


def dtype_name(torch, device: str) -> str:
    dtype = choose_dtype(torch, device)
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.float32:
        return "float32"
    return str(dtype)


def validate_args(args: argparse.Namespace) -> None:
    all_indices = CLIP_TOKEN_INDICES + T5_TOKEN_INDICES
    if any(token_index < 0 for token_index in all_indices):
        raise ValueError("Token index constants must be non-negative.")
    if any(token_index >= CLIP_SEQUENCE_LENGTH for token_index in CLIP_TOKEN_INDICES):
        raise ValueError(f"CLIP token indices must be < {CLIP_SEQUENCE_LENGTH}.")
    if any(token_index >= T5_SEQUENCE_LENGTH for token_index in T5_TOKEN_INDICES):
        raise ValueError(f"T5 token indices must be < {T5_SEQUENCE_LENGTH}.")
    if not MULTIPLIER_VALUES:
        raise ValueError("MULTIPLIER_VALUES must not be empty.")
    if args.steps <= 0:
        raise ValueError("--steps must be > 0.")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be > 0.")
    if not Path(args.classifier_checkpoint).expanduser().exists():
        raise FileNotFoundError(f"Classifier checkpoint does not exist: {args.classifier_checkpoint}")


def selected_clip_token_indices() -> list[int]:
    if CLIP_TOKEN_INDICES:
        return CLIP_TOKEN_INDICES
    return list(range(CLIP_SEQUENCE_LENGTH))


def selected_t5_token_indices() -> list[int]:
    if T5_TOKEN_INDICES:
        return T5_TOKEN_INDICES
    return list(range(T5_SEQUENCE_LENGTH))


def load_pipeline(torch, StableDiffusion3Pipeline, device: str):
    dtype = choose_dtype(torch, device)
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    token_arg = token if token else True

    print(f"Loading {MODEL_ID} on {device} with dtype={dtype}...")
    try:
        pipe = StableDiffusion3Pipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            token=token_arg,
        )
    except Exception as exc:
        print(
            "\nCould not load the model from Hugging Face.\n"
            "Make sure you have accepted the gated model terms at:\n"
            f"  https://huggingface.co/{MODEL_ID}\n\n"
            "Then authenticate with `huggingface-cli login` or set HF_TOKEN.\n"
            f"\nOriginal error: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    if device == "cuda":
        pipe.enable_model_cpu_offload()
        return pipe

    return pipe.to(device)


def load_torch_checkpoint(torch, path: Path, device: str) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_classifier(torch, checkpoint_path: str, device: str) -> dict:
    try:
        from train_image_classifier import SmallCNN
    except ImportError as exc:
        raise RuntimeError(
            "Could not import SmallCNN from train_image_classifier.py. "
            "Keep train_image_classifier.py in the same folder."
        ) from exc

    checkpoint = load_torch_checkpoint(torch, Path(checkpoint_path).expanduser().resolve(), device)
    class_names = checkpoint["class_names"]
    image_size = int(checkpoint.get("image_size", 224))
    model = SmallCNN(num_text_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return {
        "model": model,
        "class_names": class_names,
        "image_size": image_size,
        "device": device,
    }


def pil_image_to_classifier_tensor(torch, image, image_size: int):
    image = image.convert("RGB").resize((image_size, image_size))
    tensor = torch.tensor(list(image.getdata()), dtype=torch.float32)
    tensor = tensor.view(image_size, image_size, 3).permute(2, 0, 1) / 255.0
    tensor = (tensor - 0.5) / 0.5
    return tensor.unsqueeze(0)


def classify_generated_image(torch, classifier: dict, image) -> dict:
    model = classifier["model"]
    device = classifier["device"]
    class_names = classifier["class_names"]
    tensor = pil_image_to_classifier_tensor(torch, image, classifier["image_size"]).to(device)
    with torch.no_grad():
        outputs = model(tensor)

    object_probs = torch.softmax(outputs["object_logits"][0], dim=0).detach().cpu()
    text_probs = torch.softmax(outputs["text_logits"][0], dim=0).detach().cpu()
    text_class_probs = torch.softmax(outputs["text_class_logits"][0], dim=0).detach().cpu()
    best_class_index = int(text_class_probs.argmax().item())
    return {
        "object_correct": float(object_probs[1].item()),
        "text_correct": float(text_probs[1].item()),
        "text_class": class_names[best_class_index],
        "text_class_probability": float(text_class_probs[best_class_index].item()),
    }


def make_generator(torch, seed: int):
    return torch.Generator(device="cpu").manual_seed(seed)


def encode_prompt_embeddings(torch, pipe, args: argparse.Namespace):
    execution_device = getattr(pipe, "_execution_device", pipe.device)
    do_cfg = args.guidance_scale > 1.0

    with torch.no_grad():
        return pipe.encode_prompt(
            prompt=args.prompt,
            prompt_2=None,
            prompt_3=None,
            negative_prompt=args.negative_prompt,
            negative_prompt_2=None,
            negative_prompt_3=None,
            do_classifier_free_guidance=do_cfg,
            device=execution_device,
            num_images_per_prompt=1,
            max_sequence_length=T5_SEQUENCE_LENGTH,
        )


def clone_embeddings(embeddings: tuple):
    return tuple(tensor.clone() if tensor is not None else None for tensor in embeddings)


def clip_channel_slice(mode: str) -> tuple[int, int]:
    if mode == "clip":
        return CLIP_BOTH_CHANNELS
    if mode == "clip-l":
        return CLIP_L_CHANNELS
    if mode == "clip-g":
        return CLIP_G_CHANNELS
    raise ValueError(f"Unsupported CLIP mode: {mode}")


def scale_clip_tokens(prompt_embeds, token_indices: list[int], multiplier: float, mode: str) -> None:
    sequence_length = prompt_embeds.shape[1]
    start, end = clip_channel_slice(mode)
    effective_indices = token_indices if token_indices else selected_clip_token_indices()
    for token_index in effective_indices:
        if token_index < CLIP_SEQUENCE_LENGTH and token_index < sequence_length:
            prompt_embeds[:, token_index, start:end] *= multiplier


def scale_t5_tokens(prompt_embeds, token_indices: list[int], multiplier: float) -> None:
    sequence_length = prompt_embeds.shape[1]
    effective_indices = token_indices if token_indices else selected_t5_token_indices()
    for token_index in effective_indices:
        t5_position = CLIP_SEQUENCE_LENGTH + token_index
        if t5_position < sequence_length:
            prompt_embeds[:, t5_position, :] *= multiplier


def scaled_prompt_embeddings(
    embeddings: tuple,
    mode: str,
    token_indices: list[int] | None,
    multiplier: float,
) -> tuple:
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = clone_embeddings(embeddings)

    if mode == "all":
        scale_clip_tokens(prompt_embeds, selected_clip_token_indices(), multiplier, "clip")
        scale_t5_tokens(prompt_embeds, selected_t5_token_indices(), multiplier)
    elif mode in {"clip", "clip-l", "clip-g"}:
        if token_indices is None:
            raise ValueError(f"Token indices are required for mode: {mode}")
        scale_clip_tokens(prompt_embeds, token_indices, multiplier, mode)
    elif mode == "t5":
        if token_indices is None:
            raise ValueError("Token indices are required for mode: t5")
        scale_t5_tokens(prompt_embeds, token_indices, multiplier)
    else:
        raise ValueError(f"Unknown scaling mode: {mode}")

    return (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    )


def generate_image(
    torch,
    pipe,
    args: argparse.Namespace,
    seed: int,
    embeddings: tuple,
    mode: str,
    token_indices: list[int] | None,
    multiplier: float,
):
    generator = make_generator(torch, seed)
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = scaled_prompt_embeddings(embeddings, mode, token_indices, multiplier)

    print(f"Generating seed {seed}: {mode} x {multiplier}...")
    return pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        width=args.width,
        height=args.height,
        generator=generator,
    ).images[0]


def generate_metrics_for_seed(
    torch,
    pipe,
    args: argparse.Namespace,
    seed: int,
    embeddings: tuple,
    classifier: dict,
) -> tuple:
    seed_matrix = []
    seed_text_classes = []
    for _row_key, row_label, mode, token_indices in ROW_SPECS:
        row_values = []
        row_text_classes = []
        for multiplier in MULTIPLIER_VALUES:
            image = generate_image(
                torch,
                pipe,
                args,
                seed,
                embeddings,
                mode,
                token_indices,
                multiplier,
            )
            metrics = classify_generated_image(torch, classifier, image)
            row_values.append(
                [
                    metrics["object_correct"],
                    metrics["text_correct"],
                    metrics["text_class_probability"],
                ]
            )
            row_text_classes.append(metrics["text_class"])
            print(
                f"  {row_label} x {multiplier:g}: "
                f"object={metrics['object_correct']:.3f}, "
                f"text={metrics['text_correct']:.3f}, "
                f"{metrics['text_class']}={metrics['text_class_probability']:.3f}"
            )
        seed_matrix.append(row_values)
        seed_text_classes.append(row_text_classes)
    return seed_matrix, seed_text_classes


def tensor_mean_std(torch, tensor, dim: int):
    mean = tensor.mean(dim=dim)
    if tensor.shape[dim] > 1:
        std = tensor.std(dim=dim, unbiased=True)
    else:
        std = torch.zeros_like(mean)
    return mean, std


def paired_delta_stats(torch, tensor, baseline_index: int):
    baseline = tensor[:, :, baseline_index : baseline_index + 1, :]
    deltas = tensor - baseline
    mean_delta, std_delta = tensor_mean_std(torch, deltas, dim=0)
    n = tensor.shape[0]
    stderr = std_delta / max(1.0, n**0.5)
    t_stat = torch.where(stderr > 0, mean_delta / stderr, torch.zeros_like(mean_delta))
    cohen_d = torch.where(std_delta > 0, mean_delta / std_delta, torch.zeros_like(mean_delta))
    return deltas, mean_delta, std_delta, t_stat, cohen_d


def nearest_baseline_index() -> int:
    if 1 in MULTIPLIER_VALUES:
        return MULTIPLIER_VALUES.index(1)
    return min(range(len(MULTIPLIER_VALUES)), key=lambda index: abs(MULTIPLIER_VALUES[index] - 1))


def summarize_text_classes(text_classes_by_seed: list, row_index: int, scale_index: int) -> str:
    counts = {}
    for seed_classes in text_classes_by_seed:
        label = seed_classes[row_index][scale_index]
        counts[label] = counts.get(label, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return ", ".join(f"{label}:{count}" for label, count in ordered)


def write_report(
    torch,
    args: argparse.Namespace,
    output_dir: Path,
    metrics_tensor,
    text_classes_by_seed: list,
    matrix_path: Path,
    json_path: Path,
) -> Path:
    baseline_index = nearest_baseline_index()
    mean, std = tensor_mean_std(torch, metrics_tensor, dim=0)
    _deltas, mean_delta, std_delta, t_stat, cohen_d = paired_delta_stats(
        torch,
        metrics_tensor,
        baseline_index,
    )

    lines = []
    lines.append("SD3.5 Classifier Metric Sweep Analysis")
    lines.append("=" * 42)
    lines.append(f"created_at_utc: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"model_id: {MODEL_ID}")
    lines.append(f"prompt: {args.prompt}")
    lines.append(f"negative_prompt: {args.negative_prompt}")
    lines.append(f"seeds: {args.seeds}")
    lines.append(f"matrix_shape: {list(metrics_tensor.shape)}  # seeds x rows x scales x metrics")
    lines.append(f"rows: {[row[1] for row in ROW_SPECS]}")
    lines.append(f"scales: {MULTIPLIER_VALUES}")
    lines.append(f"metrics: {list(METRIC_NAMES)}")
    lines.append(f"baseline_scale: {MULTIPLIER_VALUES[baseline_index]}")
    lines.append(f"clip_token_indices: {CLIP_TOKEN_INDICES or 'ALL'}")
    lines.append(f"t5_token_indices: {T5_TOKEN_INDICES or 'ALL'}")
    lines.append(f"matrix_file: {matrix_path}")
    lines.append(f"json_file: {json_path}")
    lines.append("")

    lines.append("Per-Condition Summary")
    lines.append("-" * 21)
    for row_index, (_row_key, row_label, _mode, _token_indices) in enumerate(ROW_SPECS):
        lines.append(f"\n[{row_index}] {row_label}")
        for scale_index, scale in enumerate(MULTIPLIER_VALUES):
            metric_bits = []
            for metric_index, metric_name in enumerate(METRIC_NAMES):
                metric_bits.append(
                    f"{metric_name}={mean[row_index, scale_index, metric_index]:.4f}"
                    f"+/-{std[row_index, scale_index, metric_index]:.4f}"
                )
            class_counts = summarize_text_classes(text_classes_by_seed, row_index, scale_index)
            lines.append(f"  scale {scale:g}: " + ", ".join(metric_bits))
            lines.append(f"    top_text_class_counts: {class_counts}")

    lines.append("")
    lines.append("Paired Delta vs Baseline")
    lines.append("-" * 24)
    lines.append(
        "Each delta is condition_metric - same_seed_baseline_metric. "
        "t_stat and cohen_d are descriptive paired-effect summaries."
    )
    for row_index, (_row_key, row_label, _mode, _token_indices) in enumerate(ROW_SPECS):
        lines.append(f"\n[{row_index}] {row_label}")
        for metric_index, metric_name in enumerate(METRIC_NAMES):
            lines.append(f"  {metric_name}")
            for scale_index, scale in enumerate(MULTIPLIER_VALUES):
                lines.append(
                    f"    scale {scale:g}: "
                    f"mean_delta={mean_delta[row_index, scale_index, metric_index]:+.4f}, "
                    f"std_delta={std_delta[row_index, scale_index, metric_index]:.4f}, "
                    f"t={t_stat[row_index, scale_index, metric_index]:+.3f}, "
                    f"d={cohen_d[row_index, scale_index, metric_index]:+.3f}"
                )

    lines.append("")
    lines.append("Largest Effects")
    lines.append("-" * 15)
    for metric_index, metric_name in enumerate(METRIC_NAMES):
        absolute_delta = mean_delta[:, :, metric_index].abs()
        flat_index = int(absolute_delta.argmax().item())
        row_index = flat_index // len(MULTIPLIER_VALUES)
        scale_index = flat_index % len(MULTIPLIER_VALUES)
        row_label = ROW_SPECS[row_index][1]
        lines.append(
            f"{metric_name}: largest absolute mean delta at row='{row_label}', "
            f"scale={MULTIPLIER_VALUES[scale_index]:g}, "
            f"delta={mean_delta[row_index, scale_index, metric_index]:+.4f}, "
            f"mean={mean[row_index, scale_index, metric_index]:.4f}"
        )

    lines.append("")
    lines.append("Encoder Sensitivity Score")
    lines.append("-" * 25)
    lines.append("Mean absolute paired delta from baseline, averaged over scales and metrics.")
    sensitivity = mean_delta.abs().mean(dim=(1, 2))
    for row_index, (_row_key, row_label, _mode, _token_indices) in enumerate(ROW_SPECS):
        lines.append(f"{row_label}: {sensitivity[row_index]:.4f}")

    report_path = output_dir / args.report_file
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def save_outputs(
    torch,
    args: argparse.Namespace,
    output_dir: Path,
    metrics_tensor,
    text_classes_by_seed: list,
) -> tuple[Path, Path]:
    matrix_path = output_dir / args.matrix_file
    json_path = output_dir / args.json_file
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "seeds": args.seeds,
        "rows": [row[1] for row in ROW_SPECS],
        "row_keys": [row[0] for row in ROW_SPECS],
        "scales": MULTIPLIER_VALUES,
        "metrics": list(METRIC_NAMES),
        "clip_token_indices": CLIP_TOKEN_INDICES,
        "t5_token_indices": T5_TOKEN_INDICES,
        "clip_token_selection": "all" if not CLIP_TOKEN_INDICES else "custom",
        "t5_token_selection": "all" if not T5_TOKEN_INDICES else "custom",
        "effective_clip_token_indices": selected_clip_token_indices(),
        "effective_t5_token_indices": selected_t5_token_indices(),
        "classifier_checkpoint": args.classifier_checkpoint,
        "matrix_shape": list(metrics_tensor.shape),
        "matrix": metrics_tensor.tolist(),
        "text_class_predictions": text_classes_by_seed,
    }
    torch.save(payload, matrix_path)
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return matrix_path, json_path


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    torch, StableDiffusion3Pipeline = import_dependencies()
    device = choose_device(torch, args.device)
    classifier_device = choose_device(torch, args.classifier_device)

    pipe = load_pipeline(torch, StableDiffusion3Pipeline, device)
    classifier = load_classifier(torch, args.classifier_checkpoint, classifier_device)
    print(f"Loaded classifier on {classifier_device}.")
    print(f"Generation dtype: {dtype_name(torch, device)}")

    embeddings = encode_prompt_embeddings(torch, pipe, args)
    matrices = []
    text_classes_by_seed = []
    for seed in args.seeds:
        seed_matrix, seed_text_classes = generate_metrics_for_seed(
            torch,
            pipe,
            args,
            seed,
            embeddings,
            classifier,
        )
        matrices.append(seed_matrix)
        text_classes_by_seed.append(seed_text_classes)

    metrics_tensor = torch.tensor(matrices, dtype=torch.float32)
    matrix_path, json_path = save_outputs(
        torch,
        args,
        output_dir,
        metrics_tensor,
        text_classes_by_seed,
    )
    report_path = write_report(
        torch,
        args,
        output_dir,
        metrics_tensor,
        text_classes_by_seed,
        matrix_path,
        json_path,
    )

    print(f"Saved metrics tensor: {matrix_path}")
    print(f"Saved metrics JSON: {json_path}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
