import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from PIL import Image

from scene_text_editing.checkpoints import validate_textctrl_installation
from scene_text_editing.datasets import DatasetError, load_dataset
from scene_text_editing.networks.factory import (
    BackendError,
    SD3ControlNetInpaintBackend,
    TextCtrlSubprocessBackend,
    _unique_output_stems,
)
from scene_text_editing.textctrl_inference_worker import parse_args
from scene_text_editing.training import (
    TrainingError,
    validate_textctrl_training_data,
)


class RegressionTests(unittest.TestCase):
    def test_download_cache_defaults_to_hf_home_hub(self):
        with patch.dict(os.environ, {"HF_HOME": "/tmp/test-hf-home"}, clear=False):
            os.environ.pop("HF_HUB_CACHE", None)
            with initialize_config_dir(
                version_base="1.3",
                config_dir=str(
                    (Path(__file__).parents[1] / "configs").resolve()
                ),
            ):
                config = compose(config_name="download")
            resolved = OmegaConf.to_container(config, resolve=True)
        self.assertEqual(resolved["cache_dir"], "/tmp/test-hf-home/hub")

    def test_empty_jsonl_source_path_has_a_dataset_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "manifest.jsonl").write_text(
                json.dumps(
                    {
                        "source_image": None,
                        "source_text": "OLD",
                        "target_text": "NEW",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(DatasetError, "source_image must be"):
                load_dataset(
                    {
                        "root_dir": str(root),
                        "format": "jsonl",
                        "manifest": "manifest.jsonl",
                        "strict": False,
                    }
                )

    def test_sanitized_output_collisions_are_rejected(self):
        samples = [
            SimpleNamespace(sample_id="folder/item"),
            SimpleNamespace(sample_id="folder item"),
        ]
        with self.assertRaisesRegex(BackendError, "both map to output stem"):
            _unique_output_stems(samples)

    def test_textctrl_worker_keeps_steps_as_an_integer(self):
        args = parse_args(
            [
                "--repository",
                "/tmp/repository",
                "--checkpoint",
                "/tmp/model.pth",
                "--dataset-dir",
                "/tmp/data",
                "--output-dir",
                "/tmp/output",
                "--seed",
                "7",
                "--starting-layer",
                "10",
                "--num-inference-steps",
                "37",
                "--guidance-scale",
                "2.5",
            ]
        )
        self.assertEqual(args.num_inference_steps, 37)
        self.assertIsInstance(args.num_inference_steps, int)

    def test_textctrl_backend_forwards_step_override(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = root / "TextCtrl"
            weights = repository / "weights"
            repository.mkdir()
            weights.mkdir()
            (repository / "inference.py").write_text("# fixture\n", encoding="utf-8")
            (weights / "model.pth").write_bytes(b"checkpoint")
            source = root / "source.png"
            Image.new("RGB", (8, 8), "white").save(source)
            output = root / "output"
            sample = SimpleNamespace(
                sample_id="sample",
                source_image=source,
                source_text="OLD",
                target_text="NEW",
            )
            config = {
                "seed": 9,
                "network": {
                    "backend": "textctrl_subprocess",
                    "repository_dir": str(repository),
                    "weights_dir": str(weights),
                    "checkpoint_path": str(weights / "model.pth"),
                    "required_checkpoints": ["model.pth"],
                    "required_sd_components": [],
                    "python_executable": "legacy-python",
                    "starting_layer": 10,
                },
                "diffusion": {
                    "num_inference_steps": 37,
                    "guidance_scale": 2.5,
                },
            }
            captured = []

            def fake_run(command, **_kwargs):
                captured.extend(command)
                raw_output = Path(command[command.index("--output-dir") + 1])
                raw_output.mkdir(parents=True)
                Image.new("RGB", (8, 8), "black").save(
                    raw_output / "000000.png"
                )
                return subprocess.CompletedProcess(command, 0)

            backend = TextCtrlSubprocessBackend(config)
            with patch(
                "scene_text_editing.networks.factory.subprocess.run",
                side_effect=fake_run,
            ):
                records = backend.run([sample], output, overwrite=False)
            self.assertEqual(
                captured[captured.index("--num-inference-steps") + 1],
                "37",
            )
            self.assertEqual(records[0]["status"], "generated")

    def test_multi_image_mode_preserves_existing_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.png"
            Image.new("RGB", (16, 16), "white").save(source)
            output = root / "output"
            output.mkdir()
            existing = output / "sample_00.png"
            existing.write_bytes(b"keep-me")
            sample = SimpleNamespace(
                sample_id="sample",
                source_image=source,
                source_text="OLD",
                target_text="NEW",
                mask_image=None,
            )
            config = {
                "seed": 1,
                "device": "cpu",
                "dtype": "float32",
                "network": {"backend": "sd3_controlnet_inpaint"},
                "diffusion": {
                    "num_inference_steps": 2,
                    "guidance_scale": 1.0,
                },
                "task": {
                    "prompts": {
                        "template": "Replace {source_text} with {target_text}."
                    },
                    "generation": {
                        "width": 16,
                        "height": 16,
                        "preserve_source_size": True,
                        "full_crop_mask_when_missing": True,
                        "controlnet_conditioning_scale": 1.0,
                        "num_images_per_prompt": 2,
                        "max_sequence_length": 32,
                    },
                },
            }
            backend = SD3ControlNetInpaintBackend(config)

            class Generator:
                def __init__(self, **_kwargs):
                    pass

                def manual_seed(self, _seed):
                    return self

            backend.torch = SimpleNamespace(Generator=Generator)
            backend.pipe = lambda **_kwargs: SimpleNamespace(
                images=[
                    Image.new("RGB", (16, 16), "red"),
                    Image.new("RGB", (16, 16), "blue"),
                ]
            )
            records = backend.run([sample], output, overwrite=False)
            self.assertEqual(existing.read_bytes(), b"keep-me")
            self.assertTrue((output / "sample_01.png").is_file())
            self.assertEqual(
                [record["status"] for record in records],
                ["skipped_existing", "generated"],
            )

    def test_training_preflight_checks_released_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fonts = root / "fonts"
            fonts.mkdir()
            (fonts / "arial.ttf").write_bytes(b"font")
            for relative in ("train/train-1", "eval/eval-1"):
                shard = root / relative
                (shard / "i_s").mkdir(parents=True)
                (shard / "t_f").mkdir()
                (shard / "i_s.txt").write_text(
                    "000.png OLD\n",
                    encoding="utf-8",
                )
                (shard / "i_t.txt").write_text(
                    "000.png NEW\n",
                    encoding="utf-8",
                )
                (shard / "i_s" / "000.png").write_bytes(b"source")
                (shard / "t_f" / "000.png").write_bytes(b"target")
            dataset = {
                "root_dir": str(root),
                "train_glob": "train/train-*",
                "validation_glob": "eval/eval-*",
                "source_dir": "i_s",
                "target_dir": "t_f",
                "source_labels": "i_s.txt",
                "target_labels": "i_t.txt",
            }
            self.assertEqual(validate_textctrl_training_data(dataset), root)
            (fonts / "arial.ttf").write_bytes(b"")
            with self.assertRaisesRegex(TrainingError, "non-empty Arial"):
                validate_textctrl_training_data(dataset)

    def test_checkpoint_validation_requires_weight_binaries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = root / "TextCtrl"
            weights = repository / "weights"
            (repository / "configs").mkdir(parents=True)
            (weights / "sd" / "unet").mkdir(parents=True)
            (repository / "inference.py").write_text("# fixture\n", encoding="utf-8")
            (repository / "configs" / "inference.yaml").write_text(
                "model: {}\n",
                encoding="utf-8",
            )
            (weights / "model.pth").write_bytes(b"checkpoint")
            network = {
                "repository_dir": str(repository),
                "weights_dir": str(weights),
                "required_source_files": [
                    "inference.py",
                    "configs/inference.yaml",
                ],
                "required_checkpoints": ["model.pth"],
                "required_sd_components": [
                    "sd/unet/diffusion_pytorch_model.bin"
                ],
            }
            missing = validate_textctrl_installation(network)
            self.assertEqual(
                missing,
                [weights / "sd" / "unet" / "diffusion_pytorch_model.bin"],
            )


if __name__ == "__main__":
    unittest.main()
