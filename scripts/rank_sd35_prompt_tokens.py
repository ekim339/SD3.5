#!/usr/bin/env python3
"""Rank SD3.5 prompt tokens by text-encoder signal strength.

For a single prompt, this writes a text report containing four ranked lists:

    1. average across available encoders
    2. CLIP-L
    3. CLIP-G
    4. T5-XXL

The default signal score is RMS across hidden channels for each token. Encoders
are loaded and released one at a time to keep SD3.5 Large memory use lower.

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 rank_sd35_prompt_tokens.py \
    --device cuda:0 \
    --prompt "A cinematic photo of a glass greenhouse on a rainy evening" \
    --output-txt text_encoder_inspection/greenhouse_signal_rank.txt
"""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path


MODEL_ID = "stabilityai/stable-diffusion-3.5-large"

torch = None
CLIPTextModelWithProjection = None
CLIPTokenizer = None
T5EncoderModel = None
T5TokenizerFast = None


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank SD3.5 prompt tokens by text-encoder signal strength."
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument(
        "--output-txt",
        default="text_encoder_inspection/sd35_prompt_token_signal_rank.txt",
    )
    parser.add_argument(
        "--encoders",
        nargs="+",
        choices=ENCODER_ORDER,
        default=ENCODER_ORDER,
        help="Text encoders to inspect.",
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
        "--max-tokens",
        type=int,
        default=77,
        help="Only rank the first N token positions. Default is 77 for the CLIP text encoders.",
    )
    parser.add_argument(
        "--score",
        choices=["rms", "mean_abs", "l2", "max_abs"],
        default="rms",
        help="Scalar signal score used for sorting tokens.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on rows printed in each list.",
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
    max_tokens: int,
    score_key: str,
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
    rows = [row for row in rows if row["token_index"] < max_tokens]
    rows.sort(key=lambda row: row[score_key], reverse=True)

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
        "ranked_rows": rows,
    }


def rows_by_token_index(rows: list[dict]) -> dict[int, dict]:
    return {row["token_index"]: row for row in rows}


def build_averaged_rows(results: dict[str, dict], score_key: str, max_tokens: int) -> list[dict]:
    by_encoder = {
        name: rows_by_token_index(result["ranked_rows"])
        for name, result in results.items()
    }
    rows = []
    for token_index in range(max_tokens):
        available_rows = [
            by_encoder[name][token_index]
            for name in ENCODER_ORDER
            if name in by_encoder and token_index in by_encoder[name]
        ]
        if not available_rows:
            continue

        averaged = {
            "encoder": "averaged",
            "description": "Averaged encoders",
            "token_index": token_index,
            "token": "; ".join(
                f"{ENCODER_SPECS[row['encoder']]['label']}={row['token']!r}"
                for row in available_rows
            ),
        }
        for metric in SCORE_KEYS:
            averaged[metric] = sum(row[metric] for row in available_rows) / len(available_rows)
        rows.append(averaged)

    rows.sort(key=lambda row: row[score_key], reverse=True)
    return rows


def format_table(rows: list[dict], score_key: str, max_rows: int | None) -> list[str]:
    rows_to_print = rows[:max_rows] if max_rows is not None else rows
    headers = [
        "rank",
        "token_index",
        "token",
        score_key,
        "rms_signal",
        "mean_abs_signal",
        "l2_signal",
        "max_abs_signal",
    ]
    table_rows = []
    for rank, row in enumerate(rows_to_print, start=1):
        table_rows.append([
            str(rank),
            f"{row['token_index']:03d}",
            repr(row["token"]),
            f"{row[score_key]:.8f}",
            f"{row['rms_signal']:.8f}",
            f"{row['mean_abs_signal']:.8f}",
            f"{row['l2_signal']:.8f}",
            f"{row['max_abs_signal']:.8f}",
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


def write_report(
    results: dict[str, dict],
    averaged_rows: list[dict],
    output_path: Path,
    args: argparse.Namespace,
    score_key: str,
) -> None:
    lines = []
    lines.append("SD3.5 prompt tokens ranked by text-encoder signal strength")
    lines.append(f"Model: {MODEL_ID}")
    lines.append(f"Prompt: {args.prompt}")
    lines.append(f"Device: {args.device}")
    lines.append(f"Dtype: {args.dtype}")
    lines.append(f"CLIP hidden state: {args.clip_hidden_state}")
    lines.append(f"T5 max sequence length: {args.t5_max_sequence_length}")
    lines.append(f"Max ranked token positions: {args.max_tokens}")
    lines.append(f"T5 device_map auto: {not args.no_t5_device_map}")
    lines.append(f"Sort score: {args.score} ({score_key})")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("  rms_signal      = sqrt(mean(value^2)) across channels for that token")
    lines.append("  mean_abs_signal = mean(abs(value)) across channels for that token")
    lines.append("  l2_signal       = vector L2 norm across channels for that token")
    lines.append("  max_abs_signal  = largest absolute channel value for that token")
    lines.append("  averaged        = arithmetic mean of encoder scores at the same token index")
    lines.append("")
    lines.append("Encoders:")
    for result in results.values():
        lines.append(
            f"  {result['name']}: {result['label']}, "
            f"{result['channels']} channels, {result['sequence_length']} encoded tokens"
        )
    lines.append("")

    sections = [("Averaged encoders", averaged_rows)]
    for encoder_name in ENCODER_ORDER:
        if encoder_name in results:
            result = results[encoder_name]
            sections.append((result["label"], result["ranked_rows"]))

    for title, rows in sections:
        lines.append("=" * 120)
        lines.append(f"{title}: tokens sorted by {score_key}")
        lines.append("=" * 120)
        lines.extend(format_table(rows, score_key, args.max_rows))
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    require_runtime()
    if args.t5_max_sequence_length <= 0:
        raise ValueError("--t5-max-sequence-length must be > 0.")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be > 0.")
    if args.max_rows is not None and args.max_rows <= 0:
        raise ValueError("--max-rows must be greater than 0 when provided.")

    dtype = dtype_from_name(args.dtype)
    score_key = score_key_from_name(args.score)

    results = {}
    for encoder_name in args.encoders:
        results[encoder_name] = inspect_encoder(
            encoder_name=encoder_name,
            prompt=args.prompt,
            dtype=dtype,
            device=args.device,
            clip_hidden_state=args.clip_hidden_state,
            t5_max_sequence_length=args.t5_max_sequence_length,
            max_tokens=args.max_tokens,
            score_key=score_key,
            use_t5_device_map=not args.no_t5_device_map,
        )

    averaged_rows = build_averaged_rows(results, score_key, args.max_tokens)
    output_path = Path(args.output_txt)
    write_report(results, averaged_rows, output_path, args, score_key)
    print(f"Saved prompt token signal ranking report: {output_path}")


if __name__ == "__main__":
    main()
