"""Prompt rendering for the text-image-editing task."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class PromptTemplateError(ValueError):
    """Raised when a configured prompt template cannot be rendered."""


def render_edit_prompt(
    prompt_config: Mapping[str, Any],
    *,
    source_text: str,
    target_text: str,
    sample_id: str = "",
) -> str:
    template = str(prompt_config.get("template", "")).strip()
    if not template:
        raise PromptTemplateError("The selected prompt set has an empty template.")
    fields = {
        "source_text": source_text,
        "target_text": target_text,
        "sample_id": sample_id,
    }
    try:
        rendered = template.format_map(fields)
    except KeyError as exc:
        raise PromptTemplateError(
            f"Unknown prompt placeholder {exc.args[0]!r}. "
            "Available placeholders are source_text, target_text, and sample_id."
        ) from exc
    return " ".join(rendered.split())


def select_negative_prompt(
    prompt_config: Mapping[str, Any],
    generation_config: Mapping[str, Any],
) -> str | None:
    value = prompt_config.get("negative_prompt")
    if value is None:
        value = generation_config.get("negative_prompt")
    if value is None:
        return None
    rendered = " ".join(str(value).split())
    return rendered or None

