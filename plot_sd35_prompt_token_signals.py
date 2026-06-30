#!/usr/bin/env python3
"""Plot SD3.5 text-encoder signal strength for each token in one prompt.

The output plot has two panels:

    1. CLIP-L, CLIP-G, and their average
    2. T5-XXL

Each panel uses x-axis labels from its own tokenizer.

Usage:
  CUDA_VISIBLE_DEVICES=0 python3 plot_sd35_prompt_token_signals.py \
  --device cuda:0 \
  --prompt "A bathroom mat that says hello in bold capital letters, photorealistic" \
  --output-plot text_encoder_inspection/hello_token_signals.png
"""

from __future__ import annotations

import argparse
import gc
import os
import textwrap
from pathlib import Path


MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"

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
    },
    "clip_g": {
        "label": "CLIP-G",
        "tokenizer_subfolder": "tokenizer_2",
        "encoder_subfolder": "text_encoder_2",
        "kind": "clip",
    },
    "t5": {
        "label": "T5-XXL",
        "tokenizer_subfolder": "tokenizer_3",
        "encoder_subfolder": "text_encoder_3",
        "kind": "t5",
    },
}

ENCODER_ORDER = ["clip_l", "clip_g", "t5"]
SCORE_KEYS = ["rms_signal", "mean_abs_signal", "l2_signal", "max_abs_signal"]


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
            "`python -m pip install matplotlib`."
        ) from exc
    plt = pyplot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot SD3.5 prompt token signal strength by text encoder."
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument(
        "--output-plot",
        default="text_encoder_inspection/sd35_prompt_token_signal.png",
    )
    parser.add_argument(
        "--clip-hidden-state",
        choices=["penultimate", "last"],
        default="penultimate",
        help="Which CLIP hidden state to inspect. penultimate matches Diffusers SD3 conditioning.",
    )
    parser.add_argument(
        "--t5-max-sequence-length",
        type=int,
        default=256,
        help="Maximum token length for the T5 text encoder.",
    )
    parser.add_argument(
        "--plot-max-tokens",
        type=int,
        default=77,
        help="Maximum token positions shown on the bar plot.",
    )
    parser.add_argument(
        "--score",
        choices=["rms", "mean_abs", "l2", "max_abs"],
        default="rms",
        help="Scalar signal score to plot.",
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


def score_key_from_name(name: str) -> str:
    return {
        "rms": "rms_signal",
        "mean_abs": "mean_abs_signal",
        "l2": "l2_signal",
        "max_abs": "max_abs_signal",
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


def encode_t5_prompt(tokenizer, text_encoder, prompt: str, max_sequence_length: int):
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


def collect_signal_rows(encoder_name: str, encoder_label: str, embeddings, labels: list[str]) -> list[dict]:
    rows = []
    for token_index in range(embeddings.shape[0]):
        token_embedding = embeddings[token_index]
        abs_embedding = token_embedding.abs()
        rows.append({
            "encoder": encoder_name,
            "description": encoder_label,
            "token_index": token_index,
            "token": labels[token_index],
            "rms_signal": float(torch.sqrt((token_embedding ** 2).mean()).item()),
            "mean_abs_signal": float(abs_embedding.mean().item()),
            "l2_signal": float(torch.linalg.vector_norm(token_embedding).item()),
            "max_abs_signal": float(abs_embedding.max().item()),
        })
    return rows


def inspect_encoder(
    encoder_name: str,
    prompt: str,
    dtype,
    device: str,
    clip_hidden_state: str,
    t5_max_sequence_length: int,
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
        encoded, embeddings = encode_clip_prompt(
            tokenizer, text_encoder, prompt, device, clip_hidden_state
        )
    else:
        encoded, embeddings = encode_t5_prompt(
            tokenizer, text_encoder, prompt, t5_max_sequence_length
        )

    labels = decode_labels(tokenizer, encoded.input_ids, encoded.attention_mask)
    rows = collect_signal_rows(encoder_name, spec["label"], embeddings, labels)

    del text_encoder
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "name": encoder_name,
        "label": spec["label"],
        "channels": embeddings.shape[-1],
        "sequence_length": embeddings.shape[0],
        "rows": rows,
    }


def rows_by_token_index(rows: list[dict]) -> dict[int, dict]:
    return {row["token_index"]: row for row in rows}


def short_token_label(label: str, max_length: int) -> str:
    label = label.replace("\n", "\\n")
    return textwrap.shorten(label, width=max_length, placeholder="...")


def barplot_tick_label(index: int, row: dict | None, max_length: int) -> str:
    if row is None:
        return str(index)
    token = short_token_label(row["token"], max_length=max_length)
    return f"{index}:{token}"


def preferred_label_row(
    index: int,
    rows_by_encoder: dict[str, dict[int, dict]],
    encoder_names: tuple[str, ...],
) -> dict | None:
    for encoder_name in encoder_names:
        if index in rows_by_encoder.get(encoder_name, {}):
            return rows_by_encoder[encoder_name][index]
    return None


def average_rows_for_encoders(
    rows_by_encoder: dict[str, dict[int, dict]],
    encoder_names: list[str],
    metric: str,
    token_index: int,
) -> float:
    values = [
        rows_by_encoder[name][token_index][metric]
        for name in encoder_names
        if name in rows_by_encoder and token_index in rows_by_encoder[name]
    ]
    if not values:
        return float("nan")
    return sum(values) / len(values)


def plot_grouped_series(axis, token_indices: list[int], rows_by_encoder, series, metric: str) -> None:
    width = 0.78 / len(series)
    center_offset = (len(series) - 1) / 2
    for series_index, (series_name, series_label, color, encoder_names) in enumerate(series):
        offset = (series_index - center_offset) * width
        values = []
        for index in token_indices:
            if encoder_names is not None:
                value = average_rows_for_encoders(rows_by_encoder, encoder_names, metric, index)
            else:
                row = rows_by_encoder.get(series_name, {}).get(index)
                value = row[metric] if row is not None else float("nan")
            values.append(value)
        axis.bar(
            [index + offset for index in token_indices],
            values,
            width=width,
            label=series_label,
            color=color,
        )


def save_barplot(
    results: dict[str, dict],
    output_path: Path,
    prompt: str,
    metric: str,
    token_label_width: int,
    plot_max_tokens: int,
) -> None:
    require_matplotlib()

    rows_by_encoder = {
        name: rows_by_token_index(result["rows"])
        for name, result in results.items()
    }
    max_sequence_length = max(result["sequence_length"] for result in results.values())
    plotted_sequence_length = min(max_sequence_length, plot_max_tokens)
    token_indices = list(range(plotted_sequence_length))

    figure_width = max(18, plotted_sequence_length * 0.18)
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(figure_width, 10.0),
        sharex=False,
        constrained_layout=False,
    )

    clip_series = [
        ("clip_l", "CLIP-L", "#4C78A8", None),
        ("clip_g", "CLIP-G", "#F58518", None),
        ("clip_average", "CLIP average", "#B279A2", ["clip_l", "clip_g"]),
    ]
    t5_series = [
        ("t5", "T5-XXL", "#54A24B", None),
    ]
    plot_grouped_series(axes[0], token_indices, rows_by_encoder, clip_series, metric)
    plot_grouped_series(axes[1], token_indices, rows_by_encoder, t5_series, metric)

    clip_tick_labels = [
        barplot_tick_label(
            index,
            preferred_label_row(index, rows_by_encoder, ("clip_l", "clip_g")),
            token_label_width,
        )
        for index in token_indices
    ]
    t5_tick_labels = [
        barplot_tick_label(
            index,
            preferred_label_row(index, rows_by_encoder, ("t5",)),
            token_label_width,
        )
        for index in token_indices
    ]

    axes[0].set_title("CLIP token signal", loc="left", fontsize=12, fontweight="bold")
    axes[1].set_title("T5 token signal", loc="left", fontsize=12, fontweight="bold")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
        axis.margins(x=0.005)
        axis.legend(loc="upper right")
        axis.set_xticks(token_indices)
    axes[0].set_xticklabels(clip_tick_labels, rotation=90, fontsize=5)
    axes[1].set_xticklabels(t5_tick_labels, rotation=90, fontsize=5)

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
        "SD3.5 prompt token signal strength\n"
        f"Prompt: {textwrap.shorten(prompt, width=170, placeholder='...')}"
    )
    figure.suptitle(title, fontsize=11, y=0.985)
    figure.supxlabel(
        "Token position. Top axis uses CLIP tokens and bottom axis uses T5 tokens.",
        fontsize=10,
    )
    figure.tight_layout(rect=(0.06, 0.05, 1, 0.92))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    require_runtime()
    if args.t5_max_sequence_length <= 0:
        raise ValueError("--t5-max-sequence-length must be > 0.")
    if args.plot_max_tokens <= 0:
        raise ValueError("--plot-max-tokens must be > 0.")

    dtype = dtype_from_name(args.dtype)
    metric = score_key_from_name(args.score)

    results = {}
    for encoder_name in ENCODER_ORDER:
        results[encoder_name] = inspect_encoder(
            encoder_name=encoder_name,
            prompt=args.prompt,
            dtype=dtype,
            device=args.device,
            clip_hidden_state=args.clip_hidden_state,
            t5_max_sequence_length=args.t5_max_sequence_length,
            use_t5_device_map=not args.no_t5_device_map,
        )

    save_barplot(
        results,
        Path(args.output_plot),
        args.prompt,
        metric,
        args.token_label_width,
        args.plot_max_tokens,
    )
    print(f"Saved prompt token signal barplot: {args.output_plot}")


if __name__ == "__main__":
    main()
