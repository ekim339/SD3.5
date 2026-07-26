from __future__ import annotations

import unittest

from scene_text_editing.prompts import (
    PromptTemplateError,
    render_edit_prompt,
    select_negative_prompt,
)


class PromptTests(unittest.TestCase):
    def test_render_edit_prompt(self) -> None:
        rendered = render_edit_prompt(
            {
                "template": 'Replace "{source_text}" with "{target_text}" '
                "for {sample_id}."
            },
            source_text="hello world",
            target_text="good night",
            sample_id="sample-1",
        )
        self.assertEqual(
            rendered,
            'Replace "hello world" with "good night" for sample-1.',
        )

    def test_unknown_placeholder_is_rejected(self) -> None:
        with self.assertRaisesRegex(PromptTemplateError, "Unknown"):
            render_edit_prompt(
                {"template": "{unsupported}"},
                source_text="A",
                target_text="B",
            )

    def test_prompt_set_negative_prompt_takes_precedence(self) -> None:
        self.assertEqual(
            select_negative_prompt(
                {"negative_prompt": "wrong text"},
                {"negative_prompt": "fallback"},
            ),
            "wrong text",
        )


if __name__ == "__main__":
    unittest.main()
