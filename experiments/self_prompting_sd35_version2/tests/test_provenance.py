from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from experiments.self_prompting_sd35_version2 import ocr_worker, sd35_worker
from experiments.self_prompting_sd35_version2.data import MODEL_KEY
from experiments.self_prompting_sd35_version2.generation_provenance import (
    GenerationProvenanceError,
    begin,
    finish,
)
from experiments.self_prompting_sd35_version2.provenance import sha256_file


def test_generation_resume_rejects_changed_settings(tmp_path: Path) -> None:
    image = tmp_path / "input.png"
    mask = tmp_path / "mask.png"
    output = tmp_path / "output.png"
    Image.new("RGB", (20, 10), "white").save(image)
    Image.new("L", (20, 10), 255).save(mask)
    manifest = tmp_path / "jobs.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "pytorch_lora_weights.safetensors").write_bytes(b"lora")
    (checkpoint / "input_projection.safetensors").write_bytes(b"projection")
    job = SimpleNamespace(input_path=image, mask_path=mask, output_path=output)
    args = argparse.Namespace(
        manifest=manifest,
        checkpoint=checkpoint,
        base_model="base-model",
        resolution=512,
        steps=28,
        max_sequence_length=256,
        dtype="bfloat16",
        font_path=None,
        overwrite=False,
    )

    state_path, signature = begin(args, [job], Path(sd35_worker.__file__))
    Image.new("RGB", (20, 10), "black").save(output)
    finish(state_path, signature)
    changed = argparse.Namespace(**vars(args))
    changed.steps = 29
    with pytest.raises(GenerationProvenanceError, match="different inputs"):
        begin(changed, [job], Path(sd35_worker.__file__))


def _ocr_fixture(tmp_path: Path) -> tuple[list[str], Path]:
    repository = tmp_path / "TextCtrl"
    config_path = repository / "configs" / "inference.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("config", encoding="utf-8")
    package = repository / "src" / "module" / "abinet"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("# package", encoding="utf-8")
    checkpoint = repository / "ocr_model.pth"
    checkpoint.write_bytes(b"checkpoint")
    image = tmp_path / "generated.png"
    Image.new("RGB", (20, 10), "white").save(image)
    manifest = tmp_path / "jobs.jsonl"
    manifest.write_text(
        json.dumps({"index": 0, "model": MODEL_KEY, "output_path": str(image)})
        + "\n",
        encoding="utf-8",
    )
    predictions = tmp_path / "ocr_predictions.jsonl"
    predictions.write_text(
        json.dumps(
            {
                "index": 0,
                "model": MODEL_KEY,
                "ocr_predicted_text": "abcde",
                "output_path": str(image),
                "image_sha256": sha256_file(image),
                "ocr_checkpoint_sha256": sha256_file(checkpoint),
                "ocr_config_sha256": sha256_file(config_path),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    argv = [
        "--repository",
        str(repository),
        "--checkpoint",
        str(checkpoint),
        "--manifest",
        str(manifest),
        "--predictions",
        str(predictions),
    ]
    return argv, image


def test_ocr_resume_accepts_matching_digests(tmp_path: Path) -> None:
    argv, _ = _ocr_fixture(tmp_path)
    assert ocr_worker.main(argv) == 0


def test_ocr_resume_rejects_changed_generated_pixels(tmp_path: Path) -> None:
    argv, image = _ocr_fixture(tmp_path)
    Image.new("RGB", (20, 10), "black").save(image)
    assert ocr_worker.main(argv) == 2
