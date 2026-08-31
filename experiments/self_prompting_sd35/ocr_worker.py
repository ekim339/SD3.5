"""Run TextCtrl's released ABINet over all generated experiment images."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


VALID_MODELS = {"textctrl", "self_prompting_sd35"}
REQUIRED_JOB_FIELDS = {"index", "model", "output_path"}


def arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Recognize every generated job image with TextCtrl ABINet."
    )
    parser.add_argument("--repository", required=True, help="TextCtrl repository root")
    parser.add_argument("--checkpoint", required=True, help="ABINet ocr_model.pth")
    parser.add_argument("--manifest", required=True, help="JSONL job manifest")
    parser.add_argument("--predictions", required=True, help="Output JSONL predictions")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _resolve(value, base):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _require_file(path, description):
    if not path.is_file():
        raise FileNotFoundError("{} does not exist or is not a file: {}".format(description, path))
    if path.stat().st_size == 0:
        raise ValueError("{} is empty: {}".format(description, path))


def _read_manifest(path, path_base):
    jobs = []
    seen = set()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            job = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Invalid JSON in manifest {} at line {}: {}".format(
                    path, line_number, exc
                )
            ) from exc
        if not isinstance(job, dict):
            raise TypeError(
                "Manifest {} line {} must contain a JSON object".format(
                    path, line_number
                )
            )
        missing = sorted(REQUIRED_JOB_FIELDS.difference(job))
        if missing:
            raise ValueError(
                "Manifest {} line {} is missing fields: {}".format(
                    path, line_number, ", ".join(missing)
                )
            )
        model_name = str(job["model"])
        if model_name not in VALID_MODELS:
            raise ValueError(
                "Manifest {} line {} has unsupported model {!r}; expected one of {}".format(
                    path, line_number, model_name, sorted(VALID_MODELS)
                )
            )
        try:
            index = int(job["index"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Manifest {} line {} has a non-integer index".format(path, line_number)
            ) from exc
        if index in seen:
            raise ValueError("Duplicate job index {} in {}".format(index, path))
        seen.add(index)
        job = dict(job)
        job["index"] = index
        job["_output_path"] = _resolve(job["output_path"], path_base)
        jobs.append(job)
    if not jobs:
        raise ValueError("Job manifest contains no jobs: {}".format(path))
    return jobs


def _read_predictions(path, path_base):
    completed = {}
    if not path.is_file():
        return completed
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Invalid JSON in predictions {} at line {}: {}".format(
                    path, line_number, exc
                )
            ) from exc
        if not isinstance(row, dict):
            raise TypeError(
                "Predictions {} line {} must contain a JSON object".format(
                    path, line_number
                )
            )
        missing = {"index", "ocr_predicted_text", "output_path"}.difference(row)
        if missing:
            raise ValueError(
                "Predictions {} line {} is missing fields: {}".format(
                    path, line_number, ", ".join(sorted(missing))
                )
            )
        try:
            index = int(row["index"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Predictions {} line {} has a non-integer index".format(
                    path, line_number
                )
            ) from exc
        if index in completed:
            raise ValueError("Duplicate prediction index {} in {}".format(index, path))
        row = dict(row)
        row["index"] = index
        row["_output_path"] = _resolve(row["output_path"], path_base)
        completed[index] = row
    return completed


def main(argv=None):
    args = arguments(argv)
    invocation_dir = Path.cwd()
    repository = _resolve(args.repository, invocation_dir)
    checkpoint = _resolve(args.checkpoint, invocation_dir)
    manifest = _resolve(args.manifest, invocation_dir)
    predictions = _resolve(args.predictions, invocation_dir)

    _require_file(repository / "configs" / "inference.yaml", "TextCtrl inference config")
    _require_file(
        repository / "src" / "module" / "abinet" / "__init__.py",
        "TextCtrl ABINet package",
    )
    _require_file(checkpoint, "ABINet checkpoint")
    _require_file(manifest, "job manifest")
    if predictions.exists() and not predictions.is_file():
        raise ValueError("Predictions path exists but is not a file: {}".format(predictions))

    jobs = _read_manifest(manifest, manifest.parent)
    by_index = {job["index"]: job for job in jobs}
    completed = {} if args.overwrite else _read_predictions(predictions, predictions.parent)
    unexpected = sorted(set(completed).difference(by_index))
    if unexpected:
        raise ValueError(
            "Predictions contain indices absent from the manifest: {}".format(unexpected)
        )
    for index, row in completed.items():
        expected_path = by_index[index]["_output_path"]
        if row["_output_path"] != expected_path:
            raise ValueError(
                "Prediction {} refers to {}, but the manifest refers to {}".format(
                    index, row["_output_path"], expected_path
                )
            )

    missing_outputs = [job["_output_path"] for job in jobs if not job["_output_path"].is_file()]
    if missing_outputs:
        preview = "\n  - ".join(str(path) for path in missing_outputs[:20])
        suffix = "\n  ... and {} more".format(len(missing_outputs) - 20) if len(missing_outputs) > 20 else ""
        raise FileNotFoundError(
            "OCR cannot start because {} generated images are missing:\n  - {}{}".format(
                len(missing_outputs), preview, suffix
            )
        )

    from PIL import Image

    for job in jobs:
        try:
            with Image.open(job["_output_path"]) as image:
                image.verify()
        except Exception as exc:
            raise ValueError(
                "Generated image for job {} is unreadable: {}".format(
                    job["index"], job["_output_path"]
                )
            ) from exc

    pending = [job for job in jobs if job["index"] not in completed]
    if not pending:
        print("No pending OCR jobs ({} already complete).".format(len(jobs)))
        return

    predictions.parent.mkdir(parents=True, exist_ok=True)
    os.chdir(str(repository))
    sys.path.insert(0, str(repository))

    import torch
    import torchvision.transforms as transforms
    from omegaconf import OmegaConf
    from tqdm import tqdm
    from src.module.abinet import ABINetIterModel, CharsetMapper, postprocess

    if not torch.cuda.is_available():
        raise RuntimeError(
            "ABINet OCR requires a CUDA GPU and a CUDA-enabled PyTorch "
            "installation; run this worker with the pinned TextCtrl environment."
        )

    config = OmegaConf.load(
        "configs/inference.yaml"
    ).model.params.base_config.ocr_model
    model = ABINetIterModel(config).cuda()
    state_dict = torch.load(str(checkpoint), map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    charset = CharsetMapper(
        filename=str(config.charset_path), max_length=int(config.max_length) + 1
    )
    resize = transforms.Resize([int(config.height), int(config.width)])
    to_tensor = transforms.ToTensor()

    mode = "w" if args.overwrite or not predictions.is_file() else "a"
    with predictions.open(mode, encoding="utf-8", buffering=1) as output:
        for job in tqdm(pending, desc="ABINet OCR"):
            image_path = job["_output_path"]
            try:
                with Image.open(image_path) as image:
                    value = resize(to_tensor(image.convert("RGB"))).unsqueeze(0).cuda()
                with torch.no_grad():
                    prediction = postprocess(
                        model(value, mode="test"), charset, "alignment"
                    )[0][0]
            except Exception as exc:
                raise RuntimeError(
                    "OCR failed for job {} image {}".format(job["index"], image_path)
                ) from exc
            row = {
                "index": job["index"],
                "ocr_predicted_text": prediction,
                "output_path": str(image_path),
            }
            output.write(json.dumps(row, sort_keys=True) + "\n")
            completed[job["index"]] = row


if __name__ == "__main__":
    main()
