import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from networks.scene_text_editing.networks.factory import TextCtrlSubprocessBackend
from networks.scene_text_editing.textctrl_residual_inference_worker import parse_args


class ResidualStyleBackendTests(unittest.TestCase):
    def test_worker_argument_types(self):
        args = parse_args([
            "--repository", "/tmp/repository", "--checkpoint", "/tmp/model.pth",
            "--dataset-dir", "/tmp/data", "--output-dir", "/tmp/output",
            "--seed", "7", "--starting-layer", "10", "--num-inference-steps", "37",
            "--guidance-scale", "2.5", "--residual-checkpoint", "/tmp/residual.pt",
            "--adapter-checkpoint", "/tmp/adapter.pt", "--canonical-font", "/tmp/font.ttf",
            "--residual-resolution", "256",
        ])
        self.assertIsInstance(args.num_inference_steps, int)
        self.assertEqual(args.residual_resolution, 256)

    def test_backend_selects_residual_worker_and_forwards_checkpoints(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository, weights = root / "TextCtrl", root / "TextCtrl" / "weights"
            repository.mkdir()
            weights.mkdir()
            (repository / "inference.py").write_text("# fixture\n", encoding="utf-8")
            (weights / "model.pth").write_bytes(b"checkpoint")
            residual, adapter, font = root / "residual.pt", root / "adapter.pt", root / "font.ttf"
            for path in (residual, adapter, font):
                path.write_bytes(b"fixture")
            source = root / "source.png"
            Image.new("RGB", (8, 8), "white").save(source)
            sample = SimpleNamespace(sample_id="sample", source_image=source,
                                     source_text="OLD", target_text="NEW")
            config = {
                "seed": 9,
                "network": {"backend": "textctrl_subprocess",
                            "repository_dir": str(repository), "weights_dir": str(weights),
                            "checkpoint_path": str(weights / "model.pth"),
                            "required_checkpoints": ["model.pth"],
                            "required_sd_components": [], "python_executable": "legacy-python",
                            "starting_layer": 10},
                "diffusion": {"num_inference_steps": 37, "guidance_scale": 2.5},
                "textctrl_style": {"encoder": "residual",
                                   "residual_checkpoint": str(residual),
                                   "adapter_checkpoint": str(adapter),
                                   "canonical_font": str(font), "resolution": 256},
            }
            captured = []

            def fake_run(command, **_kwargs):
                captured.extend(command)
                output = Path(command[command.index("--output-dir") + 1])
                output.mkdir(parents=True)
                Image.new("RGB", (8, 8), "black").save(output / "000000.png")
                return subprocess.CompletedProcess(command, 0)

            with patch("networks.scene_text_editing.networks.factory.subprocess.run",
                       side_effect=fake_run):
                records = TextCtrlSubprocessBackend(config).run(
                    [sample], root / "output", overwrite=False)
            self.assertTrue(captured[1].endswith("textctrl_residual_inference_worker.py"))
            self.assertEqual(captured[captured.index("--residual-checkpoint") + 1], str(residual))
            self.assertEqual(captured[captured.index("--adapter-checkpoint") + 1], str(adapter))
            self.assertEqual(records[0]["style_encoder"], "residual")


if __name__ == "__main__":
    unittest.main()
