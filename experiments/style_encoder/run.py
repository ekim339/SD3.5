from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path

from omegaconf import OmegaConf

from .collage import render_collages
from .data import expand_jobs, prepare_samples, read_jsonl, write_jsonl
from .report import evaluate, write_reports

ROOT = Path(__file__).resolve().parents[2]


def resolve(value):
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def main(argv=None):
    import sys
    config_path = Path(__file__).with_name("configs") / "experiment.yaml"
    cfg = OmegaConf.load(config_path)
    overrides = list(sys.argv[1:] if argv is None else argv)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    config = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(config, Mapping):
        raise TypeError("configuration must be a mapping")
    output_dir = resolve(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "samples.jsonl"
    samples = (prepare_samples(config, ROOT, output_dir)
               if config["overwrite"] or not samples_path.is_file() else read_jsonl(samples_path))
    jobs = expand_jobs(samples, config, output_dir)
    write_jsonl(output_dir / "jobs.jsonl", jobs)
    expected = int(config["sample_count"]) * (len(config["masking_proportions"]) + 17)
    if len(jobs) != expected:
        raise RuntimeError(f"Expected {expected} jobs, got {len(jobs)}")
    if config["stage"] == "prepare":
        print(json.dumps({"samples": len(samples), "jobs": len(jobs),
                          "output_dir": str(output_dir)}, indent=2))
        return
    if config["stage"] != "all":
        raise ValueError("stage must be 'prepare' or 'all'")

    repository = resolve(config["textctrl"]["repository_dir"])
    predictions = output_dir / "predictions.jsonl"
    command = [str(config["textctrl"]["python_executable"]),
               str(Path(__file__).with_name("worker.py").resolve()),
               "--repository", str(repository),
               "--checkpoint", str(resolve(config["textctrl"]["checkpoint_path"])),
               "--ocr-checkpoint", str(resolve(config["ocr"]["checkpoint_path"])),
               "--manifest", str(output_dir / "jobs.jsonl"),
               "--predictions", str(predictions),
               "--starting-layer", str(config["textctrl"]["starting_layer"]),
               "--num-inference-steps", str(config["textctrl"]["num_inference_steps"]),
               "--guidance-scale", str(config["textctrl"]["guidance_scale"])]
    if config["overwrite"]:
        command.append("--overwrite")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, (str(ROOT), str(repository),
                                                               environment.get("PYTHONPATH", ""))))
    subprocess.run(command, cwd=repository, env=environment, check=True)
    predicted = {int(row["index"]): row for row in read_jsonl(predictions)}
    if len(predicted) != len(jobs):
        raise RuntimeError(f"Incomplete predictions: {len(predicted)}/{len(jobs)}")
    rows = evaluate(jobs, predicted, bool(config["metrics"]["case_sensitive"]))
    write_reports(rows, output_dir)
    render_collages(samples, rows, output_dir / "collages", config["collage"])
    (output_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")


if __name__ == "__main__":
    main()
