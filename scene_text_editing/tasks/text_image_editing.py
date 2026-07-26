"""Task-level behavior shared by the supported editing networks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from scene_text_editing.prompts import render_edit_prompt, select_negative_prompt


class TextImageEditingTask:
    """Render model inputs for a source-text to target-text replacement."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config
        self.prompt_config = config["prompts"]
        self.generation_config = config["generation"]

    def prompt_for(self, sample: Any) -> str:
        return render_edit_prompt(
            self.prompt_config,
            source_text=str(sample.source_text),
            target_text=str(sample.target_text),
            sample_id=str(sample.sample_id),
        )

    @property
    def negative_prompt(self) -> str | None:
        return select_negative_prompt(
            self.prompt_config,
            self.generation_config,
        )

