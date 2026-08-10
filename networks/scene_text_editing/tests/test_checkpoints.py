from __future__ import annotations

import unittest

from networks.scene_text_editing.checkpoints import describe_download


class CheckpointPlanTests(unittest.TestCase):
    def test_sd3_plan_lists_base_and_controlnet(self) -> None:
        actions = describe_download(
            {
                "download_base_model": True,
                "download_controlnet": True,
                "network": {
                    "backend": "sd3_controlnet_inpaint",
                    "base_model_id": "base/model",
                    "controlnet_model_id": "control/model",
                },
            }
        )
        self.assertEqual(len(actions), 2)
        self.assertIn("base/model", actions[0])
        self.assertIn("control/model", actions[1])


if __name__ == "__main__":
    unittest.main()
