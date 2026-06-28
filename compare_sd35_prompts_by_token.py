#!/usr/bin/env python3
"""Compare SD3.5 text-encoder changes between two prompts at the token level.

The script compares prompt A and prompt B for each SD3.5 text stream:

    1. CLIP-L
    2. CLIP-G
    3. T5-XXL

For each token position it reports:
    mean_abs_delta      = mean(abs(value_b - value_a))
    mean_norm_abs_delta = mean(abs(value_b - value_a) / std(channel))
    rms_delta           = sqrt(mean((value_b - value_a)^2))

The companion bar plot writes each token under its bar. To keep memory lower for
SD3.5 Large, the encoders are loaded, compared, and released one at a time.

Usage:
    python3 compare_sd35_prompts_by_token.py \
      --device cuda:0 \
      --encoders clip_l clip_g \
      --prompt-a "..." \
      --prompt-b "..." \
      --output-txt text_encoder_inspection/clip_only.txt
"""

from __future__ import annotations

import argparse
import gc
import os
import textwrap
from pathlib import Path


MODEL_ID = "stabilityai/stable-diffusion-3.5-large"

torch = None
CLIPTextModelWithProjection = None
CLIPTokenizer = None
T5EncoderModel = None
T5TokenizerFast = None
plt = None


ENCODER_SPECS = {
    "clip_l": {
        "label": "CLIP-L",
        "tokenizer_subfolder": "tokenizer",
        "encoder_subfolder": "text_encoder",
        "kind": "clip",
        "channels": 768,
    },
    "clip_g": {
        "label": "CLIP-G",
        "tokenizer_subfolder": "tokenizer_2",
        "encoder_subfolder": "text_encoder_2",
        "kind": "clip",
        "channels": 1280,
    },
    "t5": {
        "label": "T5-XXL",
        "tokenizer_subfolder": "tokenizer_3",
        "encoder_subfolder": "text_encoder_3",
        "kind": "t5",
        "channels": 4096,
    },
}

PLOT_ENCODER_ORDER = ["clip_l", "clip_g", "t5"]
METRIC_KEYS = [
    "mean_norm_abs_delta",
    "mean_abs_delta",
    "signed_mean_delta",
    "rms_delta",
    "max_abs_delta",
]


def require_runtime() -> None:
    """Import heavy dependencies lazily so --help works without loading them."""
    global torch, CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
    if torch is not None:
        return
    try:
        import torch as torch_module
        from transformers import CLIPTextModelWithProjection as clip_text_model
        from transformers import CLIPTokenizer as clip_tokenizer
        from transformers import T5EncoderModel as t5_encoder_model
        from transformers import T5TokenizerFast as t5_tokenizer_fast
    except Exception as exc:
        raise RuntimeError(
            "This script requires torch and transformers in the active environment."
        ) from exc
    torch = torch_module
    CLIPTextModelWithProjection = clip_text_model
    CLIPTokenizer = clip_tokenizer
    T5EncoderModel = t5_encoder_model
    T5TokenizerFast = t5_tokenizer_fast


def require_matplotlib() -> None:
    """Import matplotlib lazily so --help works without plotting dependencies."""
    global plt
    if plt is not None:
        return
    try:
        import matplotlib.pyplot as pyplot
    except Exception as exc:
        raise RuntimeError(
            "Saving the barplot requires matplotlib. Install it with "
            "`python -m pip install matplotlib`, or pass --no-plot."
        ) from exc
    plt = pyplot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare SD3.5 text-encoder changes between two prompts by token."
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--prompt-a", required=True)
    parser.add_argument("--prompt-b", required=True)
    parser.add_argument(
        "--output-txt",
        default="text_encoder_inspection/sd35_prompt_token_diff.txt",
    )
    parser.add_argument(
        "--output-plot",
        default=None,
        help="Optional output PNG path. Defaults to output-txt with .png extension.",
    )
    parser.add_argument(
        "--encoders",
        nargs="+",
        choices=["clip_l", "clip_g", "t5"],
        default=["clip_l", "clip_g", "t5"],
        help="Text encoders to compare.",
    )
    parser.add_argument(
        "--clip-hidden-state",
        choices=["penultimate", "last"],
        default="penultimate",
        help="Which CLIP hidden state to compare. penultimate matches Diffusers SD3 conditioning.",
    )
    parser.add_argument(
        "--t5-max-sequence-length",
        type=int,
        default=256,
        help="Maximum token length for the T5 text encoder.",
    )
    parser.add_argument(
        "--sort-by",
        choices=["normalized", "raw", "rms"],
        default="normalized",
        help="Score used to sort tokens inside each report section.",
    )
    parser.add_argument(
        "--std-eps",
        type=float,
        default=1e-8,
        help="Small denominator floor for normalized differences.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on rows printed in each section.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not save the companion barplot PNG.",
    )
    parser.add_argument(
        "--token-label-width",
        type=int,
        default=14,
        help="Maximum characters shown for each token label under the bars.",
    )
    parser.add_argument(
        "--no-t5-device-map",
        action="store_true",
        help="Disable Accelerate device_map='auto' for T5 on CUDA and move it directly to --device.",
    )
    return parser.parse_args()


def dtype_from_name(name: str):
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def token_arg():
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    return token if token else True


def load_tokenizer_and_encoder(spec: dict, dtype, device: str, use_t5_device_map: bool):
    if spec["kind"] == "clip":
        tokenizer_class = CLIPTokenizer
        encoder_class = CLIPTextModelWithProjection
    else:
        tokenizer_class = T5TokenizerFast
        encoder_class = T5EncoderModel

    tokenizer = tokenizer_class.from_pretrained(
        MODEL_ID,
        subfolder=spec["tokenizer_subfolder"],
        token=token_arg(),
    )
    load_kwargs = {
        "subfolder": spec["encoder_subfolder"],
        "torch_dtype": dtype,
        "token": token_arg(),
    }
    if spec["kind"] == "t5" and device.startswith("cuda") and use_t5_device_map:
        load_kwargs["device_map"] = "auto"

    text_encoder = encoder_class.from_pretrained(MODEL_ID, **load_kwargs)
    if "device_map" not in load_kwargs:
        text_encoder = text_encoder.to(device)
    text_encoder.eval()
    return tokenizer, text_encoder


def first_parameter_device(module):
    return next(module.parameters()).device


def encode_clip_prompt(tokenizer, text_encoder, prompt: str, device: str, hidden_state: str):
    encoded = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoded.input_ids.to(device)
    with torch.no_grad():
        outputs = text_encoder(
            input_ids=input_ids,
            output_hidden_states=True,
            return_dict=True,
        )

    hidden_index = -2 if hidden_state == "penultimate" else -1
    embeddings = outputs.hidden_states[hidden_index][0].detach().float().cpu()
    return encoded, embeddings


def encode_t5_prompt(tokenizer, text_encoder, prompt: str, device: str, max_sequence_length: int):
    encoded = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    input_device = first_parameter_device(text_encoder)
    input_ids = encoded.input_ids.to(input_device)
    attention_mask = encoded.attention_mask.to(input_device) if "attention_mask" in encoded else None
    with torch.no_grad():
        outputs = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

    embeddings = outputs.last_hidden_state[0].detach().float().cpu()
    return encoded, embeddings


def decode_labels(tokenizer, input_ids, attention_mask) -> list[str]:
    labels = []
    ids = input_ids[0].tolist()
    mask = attention_mask[0].tolist() if attention_mask is not None else [1] * len(ids)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    eot_seen = False

    for index, (token_id, is_active) in enumerate(zip(ids, mask)):
        decoded = tokenizer.decode([token_id], skip_special_tokens=False).strip()
        if eos_id is not None and token_id == eos_id and not eot_seen:
            labels.append("EOT")
            eot_seen = True
        elif pad_id is not None and token_id == pad_id:
            labels.append("PAD")
        elif not is_active:
            labels.append(f"MASKED:{decoded or 'blank'}")
        else:
            labels.append(decoded or f"token_{index}")
    return labels


def compare_encoder(
    encoder_name: str,
    prompt_a: str,
    prompt_b: str,
    dtype,
    device: str,
    clip_hidden_state: str,
    t5_max_sequence_length: int,
    std_eps: float,
    sort_by: str,
    use_t5_device_map: bool,
) -> dict:
    spec = ENCODER_SPECS[encoder_name]
    print(f"Loading {spec['label']} from {MODEL_ID}...")
    tokenizer, text_encoder = load_tokenizer_and_encoder(
        spec,
        dtype,
        device,
        use_t5_device_map,
    )

    if spec["kind"] == "clip":
        encoded_a, embeddings_a = encode_clip_prompt(
            tokenizer, text_encoder, prompt_a, device, clip_hidden_state
        )
        encoded_b, embeddings_b = encode_clip_prompt(
            tokenizer, text_encoder, prompt_b, device, clip_hidden_state
        )
    else:
        encoded_a, embeddings_a = encode_t5_prompt(
            tokenizer, text_encoder, prompt_a, device, t5_max_sequence_length
        )
        encoded_b, embeddings_b = encode_t5_prompt(
            tokenizer, text_encoder, prompt_b, device, t5_max_sequence_length
        )

    labels_a = decode_labels(tokenizer, encoded_a.input_ids, encoded_a.attention_mask)
    labels_b = decode_labels(tokenizer, encoded_b.input_ids, encoded_b.attention_mask)
    rows = collect_token_rows(
        encoder_name,
        spec["label"],
        embeddings_a,
        embeddings_b,
        labels_a,
        labels_b,
        std_eps,
        sort_by,
    )

    del text_encoder
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "name": encoder_name,
        "label": spec["label"],
        "channels": embeddings_a.shape[-1],
        "sequence_length": embeddings_a.shape[0],
        "labels_a": labels_a,
        "labels_b": labels_b,
        "rows": rows,
    }


def collect_token_rows(
    encoder_name: str,
    encoder_label: str,
    embeddings_a,
    embeddings_b,
    labels_a: list[str],
    labels_b: list[str],
    std_eps: float,
    sort_by: str,
) -> list[dict]:
    delta = embeddings_b - embeddings_a
    calibration = torch.cat([embeddings_a, embeddings_b], dim=0)
    channel_std = calibration.std(dim=0, unbiased=False).clamp_min(std_eps)
    normalized_abs_delta = delta.abs() / channel_std

    sort_key_by_name = {
        "normalized": "mean_norm_abs_delta",
        "raw": "mean_abs_delta",
        "rms": "rms_delta",
    }
    sort_key = sort_key_by_name[sort_by]

    rows = []
    for token_index in range(delta.shape[0]):
        token_delta = delta[token_index]
        token_abs_delta = token_delta.abs()
        token_norm_abs_delta = normalized_abs_delta[token_index]
        rows.append({
            "encoder": encoder_name,
            "description": encoder_label,
            "token_index": token_index,
            "token_a": labels_a[token_index],
            "token_b": labels_b[token_index],
            "signed_mean_delta": float(token_delta.mean().item()),
            "mean_abs_delta": float(token_abs_delta.mean().item()),
            "mean_norm_abs_delta": float(token_norm_abs_delta.mean().item()),
            "rms_delta": float(torch.sqrt((token_delta ** 2).mean()).item()),
            "max_abs_delta": float(token_abs_delta.max().item()),
        })
    rows.sort(key=lambda row: row[sort_key], reverse=True)
    return rows


def format_table(rows: list[dict], max_rows: int | None = None) -> list[str]:
    rows_to_print = rows[:max_rows] if max_rows is not None else rows
    headers = [
        "rank",
        "token",
        "token_a",
        "token_b",
        "mean_norm_abs_delta",
        "mean_abs_delta",
        "signed_mean_delta",
        "rms_delta",
        "max_abs_delta",
    ]
    table_rows = []
    for rank, row in enumerate(rows_to_print, start=1):
        table_rows.append([
            str(rank),
            f"{row['token_index']:03d}",
            repr(row["token_a"]),
            repr(row["token_b"]),
            f"{row['mean_norm_abs_delta']:.8f}",
            f"{row['mean_abs_delta']:.8f}",
            f"{row['signed_mean_delta']:+.8f}",
            f"{row['rms_delta']:.8f}",
            f"{row['max_abs_delta']:.8f}",
        ])

    widths = [
        max(len(headers[col]), *(len(table_row[col]) for table_row in table_rows))
        if table_rows else len(headers[col])
        for col in range(len(headers))
    ]
    lines = []
    lines.append(" | ".join(header.ljust(widths[col]) for col, header in enumerate(headers)))
    lines.append("-+-".join("-" * width for width in widths))
    for table_row in table_rows:
        lines.append(" | ".join(table_row[col].ljust(widths[col]) for col in range(len(headers))))
    return lines


def rows_by_token_index(rows: list[dict]) -> dict[int, dict]:
    return {row["token_index"]: row for row in rows}


def sort_key_for_name(sort_by: str) -> str:
    return {
        "normalized": "mean_norm_abs_delta",
        "raw": "mean_abs_delta",
        "rms": "rms_delta",
    }[sort_by]


def build_averaged_rows(results: dict[str, dict], sort_by: str) -> list[dict]:
    by_encoder = {
        name: rows_by_token_index(result["rows"])
        for name, result in results.items()
    }
    max_sequence_length = max(result["sequence_length"] for result in results.values())
    rows = []
    for token_index in range(max_sequence_length):
        available_rows = [
            by_encoder[name][token_index]
            for name in PLOT_ENCODER_ORDER
            if name in by_encoder and token_index in by_encoder[name]
        ]
        if not available_rows:
            continue

        averaged = {
            "encoder": "averaged",
            "description": "Averaged encoders",
            "token_index": token_index,
            "token_a": "; ".join(
                f"{ENCODER_SPECS[row['encoder']]['label']}={row['token_a']!r}"
                for row in available_rows
            ),
            "token_b": "; ".join(
                f"{ENCODER_SPECS[row['encoder']]['label']}={row['token_b']!r}"
                for row in available_rows
            ),
        }
        for metric in METRIC_KEYS:
            averaged[metric] = sum(row[metric] for row in available_rows) / len(available_rows)
        rows.append(averaged)

    rows.sort(key=lambda row: row[sort_key_for_name(sort_by)], reverse=True)
    return rows


def short_token_label(label: str, max_length: int) -> str:
    label = label.replace("\n", "\\n")
    return textwrap.shorten(label, width=max_length, placeholder="...")


def barplot_tick_label(index: int, row: dict | None, max_length: int) -> str:
    if row is None:
        return str(index)
    token_a = short_token_label(row["token_a"], max_length=max_length)
    token_b = short_token_label(row["token_b"], max_length=max_length)
    if token_a == token_b:
        return f"{index}:{token_a}"
    return f"{index}:{token_a}/{token_b}"


def preferred_label_row(index: int, rows_by_encoder: dict[str, dict[int, dict]]) -> dict | None:
    for encoder_name in ("t5", "clip_l", "clip_g"):
        if index in rows_by_encoder.get(encoder_name, {}):
            return rows_by_encoder[encoder_name][index]
    return None


def save_barplot(
    results: dict[str, dict],
    output_path: Path,
    prompt_a: str,
    prompt_b: str,
    metric: str,
    token_label_width: int,
) -> None:
    require_matplotlib()

    rows_by_encoder = {
        name: rows_by_token_index(result["rows"])
        for name, result in results.items()
    }
    averaged_rows = rows_by_token_index(build_averaged_rows(results, "normalized"))
    max_sequence_length = max(result["sequence_length"] for result in results.values())
    token_indices = list(range(max_sequence_length))

    figure_width = max(18, max_sequence_length * 0.16)
    figure, axis = plt.subplots(figsize=(figure_width, 7.8), constrained_layout=False)

    series = [
        ("clip_l", "CLIP-L", "#4C78A8"),
        ("clip_g", "CLIP-G", "#F58518"),
        ("t5", "T5-XXL", "#54A24B"),
        ("averaged", "Averaged", "#B279A2"),
    ]
    width = 0.2
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    for (series_name, series_label, color), offset in zip(series, offsets):
        values = []
        for index in token_indices:
            if series_name == "averaged":
                row = averaged_rows.get(index)
            else:
                row = rows_by_encoder.get(series_name, {}).get(index)
            values.append(row[metric] if row is not None else float("nan"))
        axis.bar(
            [index + offset for index in token_indices],
            values,
            width=width,
            label=series_label,
            color=color,
        )

    tick_labels = [
        barplot_tick_label(
            index,
            preferred_label_row(index, rows_by_encoder),
            token_label_width,
        )
        for index in token_indices
    ]
    axis.set_xticks(token_indices)
    axis.set_xticklabels(tick_labels, rotation=90, fontsize=5)
    axis.grid(axis="y", alpha=0.25)
    axis.margins(x=0.005)
    axis.legend(ncol=4, loc="upper right")

    output_title = output_path.stem
    figure.text(
        0.012,
        0.56,
        output_title,
        rotation=90,
        va="center",
        ha="left",
        fontsize=18,
        fontweight="bold",
    )
    figure.text(
        0.035,
        0.56,
        metric,
        rotation=90,
        va="center",
        ha="left",
        fontsize=11,
    )
    title = (
        "SD3.5 prompt difference by token\n"
        f"Prompt A: {textwrap.shorten(prompt_a, width=150, placeholder='...')}\n"
        f"Prompt B: {textwrap.shorten(prompt_b, width=150, placeholder='...')}"
    )
    figure.suptitle(title, fontsize=11, y=0.985)
    figure.supxlabel(
        "Token position: prompt A token / prompt B token. Labels use T5 when present, otherwise CLIP.",
        fontsize=10,
    )
    figure.tight_layout(rect=(0.06, 0.04, 1, 0.92))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def write_report(
    results: dict[str, dict],
    output_path: Path,
    prompt_a: str,
    prompt_b: str,
    args: argparse.Namespace,
) -> None:
    lines = []
    lines.append("SD3.5 text-encoder prompt differences by token")
    lines.append(f"Model: {MODEL_ID}")
    lines.append(f"Prompt A: {prompt_a}")
    lines.append(f"Prompt B: {prompt_b}")
    lines.append(f"Device: {args.device}")
    lines.append(f"Dtype: {args.dtype}")
    lines.append(f"CLIP hidden state: {args.clip_hidden_state}")
    lines.append(f"T5 max sequence length: {args.t5_max_sequence_length}")
    lines.append(f"T5 device_map auto: {not args.no_t5_device_map}")
    lines.append(f"Sort by: {args.sort_by}")
    lines.append(f"Std epsilon: {args.std_eps}")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("  mean_norm_abs_delta = mean(abs(value_b - value_a) / std(channel)) across encoder channels")
    lines.append("  mean_abs_delta      = mean(abs(value_b - value_a)) across encoder channels")
    lines.append("  signed_mean_delta   = mean(value_b - value_a) across encoder channels")
    lines.append("  rms_delta           = sqrt(mean((value_b - value_a)^2)) across encoder channels")
    lines.append("  max_abs_delta       = largest raw absolute channel difference inside that token")
    lines.append("")
    lines.append("Encoders:")
    for result in results.values():
        lines.append(
            f"  {result['name']}: {result['label']}, "
            f"{result['channels']} channels, {result['sequence_length']} tokens"
        )
    averaged_rows = build_averaged_rows(results, args.sort_by)
    lines.append("  averaged: arithmetic mean across available encoder scores for each token index")
    lines.append("")

    for result in results.values():
        lines.append("=" * 120)
        lines.append(f"{result['label']} token index legend")
        lines.append("=" * 120)
        lines.append("  token | prompt A -> prompt B")
        lines.append("  ----- | --------------------")
        for token_index, (token_a, token_b) in enumerate(
            zip(result["labels_a"], result["labels_b"])
        ):
            lines.append(f"  {token_index:03d}   | {token_a!r} -> {token_b!r}")
        lines.append("")
        lines.append("=" * 120)
        lines.append(f"{result['label']}: tokens sorted by {args.sort_by} change")
        lines.append("=" * 120)
        lines.extend(format_table(result["rows"], args.max_rows))
        lines.append("")

    lines.append("=" * 120)
    lines.append(f"Averaged encoders: tokens sorted by {args.sort_by} change")
    lines.append("=" * 120)
    lines.extend(format_table(averaged_rows, args.max_rows))
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    require_runtime()
    if args.std_eps <= 0:
        raise ValueError("--std-eps must be > 0.")
    if args.max_rows is not None and args.max_rows <= 0:
        raise ValueError("--max-rows must be greater than 0 when provided.")
    if args.t5_max_sequence_length <= 0:
        raise ValueError("--t5-max-sequence-length must be > 0.")

    dtype = dtype_from_name(args.dtype)
    output_path = Path(args.output_txt)
    plot_path = Path(args.output_plot) if args.output_plot else output_path.with_suffix(".png")

    results = {}
    for encoder_name in args.encoders:
        results[encoder_name] = compare_encoder(
            encoder_name=encoder_name,
            prompt_a=args.prompt_a,
            prompt_b=args.prompt_b,
            dtype=dtype,
            device=args.device,
            clip_hidden_state=args.clip_hidden_state,
            t5_max_sequence_length=args.t5_max_sequence_length,
            std_eps=args.std_eps,
            sort_by=args.sort_by,
            use_t5_device_map=not args.no_t5_device_map,
        )

    write_report(results, output_path, args.prompt_a, args.prompt_b, args)
    print(f"Saved prompt token comparison report: {output_path}")
    if not args.no_plot:
        metric = {
            "normalized": "mean_norm_abs_delta",
            "raw": "mean_abs_delta",
            "rms": "rms_delta",
        }[args.sort_by]
        save_barplot(
            results,
            plot_path,
            args.prompt_a,
            args.prompt_b,
            metric,
            args.token_label_width,
        )
        print(f"Saved prompt token barplot: {plot_path}")


if __name__ == "__main__":
    main()
