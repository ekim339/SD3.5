"""Text-conditioning helpers for Self-Prompting SD3.5."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def encode_t5_target_conditioning(
    pipeline: Any,
    target_text: str | Sequence[str],
    *,
    device: Any,
    max_sequence_length: int,
) -> tuple[Any, Any]:
    """Encode target content only with T5 while preserving SD3 conditioning.

    SD3 expects a sequence containing its CLIP and T5 token blocks plus pooled
    conditioning produced by the two CLIP towers. Both CLIP towers therefore
    receive the same content-free empty prompt; only ``prompt_3`` receives the
    target string. This keeps the pipeline's native tensor layout without
    leaking target content into CLIP.
    """

    targets = [target_text] if isinstance(target_text, str) else list(target_text)
    if not targets:
        raise ValueError("At least one target string is required")
    if any(not isinstance(value, str) or not value for value in targets):
        raise ValueError("Target strings must be non-empty strings")

    null_clip_prompts = [""] * len(targets)
    sequence, _, pooled, _ = pipeline.encode_prompt(
        prompt=null_clip_prompts,
        prompt_2=null_clip_prompts,
        prompt_3=targets,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
        max_sequence_length=max_sequence_length,
    )
    if sequence is None or pooled is None:
        raise RuntimeError("SD3 prompt encoding did not return sequence and pooled tensors")
    return sequence, pooled
