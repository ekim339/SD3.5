from __future__ import annotations

import unittest
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from scene_text_editing.configuration import ConfigurationError, validate_config


CONFIG_DIR = Path(__file__).parents[1] / "configs"


def compose_mapping(
    config_name: str = "inference",
    overrides: list[str] | None = None,
) -> dict:
    with initialize_config_dir(
        version_base="1.3",
        config_dir=str(CONFIG_DIR.resolve()),
    ):
        config = compose(config_name=config_name, overrides=overrides or [])
    resolved = OmegaConf.to_container(config, resolve=True)
    assert isinstance(resolved, dict)
    return resolved


class ConfigurationTests(unittest.TestCase):
    def test_default_inference_is_textctrl_sd15_pndm(self) -> None:
        config = compose_mapping()
        validate_config(config)
        self.assertEqual(config["network"]["name"], "textctrl_sd15")
        self.assertEqual(config["diffusion"]["name"], "pndm")
        self.assertEqual(config["task"]["dataset"]["name"], "dataset_a")
        self.assertEqual(config["task"]["prompts"]["name"], "prompt_set_a")

    def test_textctrl_pndm_composes(self) -> None:
        config = compose_mapping(
            overrides=["network=textctrl_sd15", "diffusion=pndm"]
        )
        validate_config(config)

    def test_textctrl_rejects_flow_matching(self) -> None:
        config = compose_mapping(
            overrides=["network=textctrl_sd15", "diffusion=flow_matching"]
        )
        with self.assertRaisesRegex(ConfigurationError, "does not support"):
            validate_config(config)

    def test_textctrl_rejects_scheduler_drift(self) -> None:
        config = compose_mapping(
            overrides=["network=textctrl_sd15", "diffusion=pndm"]
        )
        config["diffusion"]["skip_prk_steps"] = False
        with self.assertRaisesRegex(ConfigurationError, "skip_prk_steps=true"):
            validate_config(config)

    def test_sd3_rejects_pndm(self) -> None:
        config = compose_mapping(
            overrides=["network=sd3_inpainting", "diffusion=pndm"]
        )
        with self.assertRaisesRegex(ConfigurationError, "does not support"):
            validate_config(config)

    def test_sd35_requires_explicit_experimental_opt_in(self) -> None:
        config = compose_mapping(
            overrides=["network=sd35_medium", "diffusion=flow_matching"]
        )
        with self.assertRaisesRegex(ConfigurationError, "trained for SD3 Medium"):
            validate_config(config)
        config["network"]["allow_experimental_base_model"] = True
        validate_config(config)

    def test_training_config_selects_synthetic_textctrl_data(self) -> None:
        config = compose_mapping(config_name="train")
        validate_config(config)
        self.assertEqual(config["mode"], "train")
        self.assertEqual(config["task"]["dataset"]["format"], "textctrl_shards")


if __name__ == "__main__":
    unittest.main()
