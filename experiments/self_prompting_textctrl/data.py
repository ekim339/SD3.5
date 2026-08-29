"""Deterministic SRNet sampling, corruption, and target construction."""
from __future__ import annotations

import json
import random
import string
from pathlib import Path

import numpy as np
from PIL import Image


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def read_labels(path: Path) -> dict[str, str]:
    labels = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = raw.strip().split(maxsplit=1)
        if not parts:
            continue
        if len(parts) != 2:
            raise ValueError(f"Expected `filename text` at {path}:{number}")
        labels[parts[0]] = parts[1]
    return labels


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def discover_samples(config, project_root: Path):
    records = []
    dataset = config["dataset"]
    for raw_root in dataset["roots"]:
        root = _resolve(project_root, str(raw_root))
        labels = read_labels(root / str(dataset["source_labels"]))
        for filename, text in labels.items():
            image = root / str(dataset["source_dir"]) / filename
            mask = root / str(dataset["mask_dir"]) / filename
            if len(text) == 5 and image.is_file() and mask.is_file():
                records.append((root, filename, text, image, mask))
    return sorted(records, key=lambda row: (str(row[0]), row[1]))


def _noisy_copy(source: Path, destination: Path, sigma: float, seed: int) -> None:
    with Image.open(source) as opened:
        pixels = np.asarray(opened.convert("RGB"), dtype=np.float32)
    noise = np.random.default_rng(seed).normal(0.0, sigma, pixels.shape)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(pixels + noise, 0, 255).astype(np.uint8)).save(destination)


def _mixed_target(rng: random.Random, letters: int, specials: int,
                  letter_alphabet: str, special_alphabet: str) -> str:
    values = [rng.choice(letter_alphabet) for _ in range(letters)]
    values += [rng.choice(special_alphabet) for _ in range(specials)]
    rng.shuffle(values)
    return "".join(values)


def prepare_samples(config, project_root: Path, output_dir: Path):
    eligible = discover_samples(config, project_root)
    count = int(config["sample_count"])
    if len(eligible) < count:
        raise ValueError(f"Only {len(eligible)} eligible five-character samples; need {count}")
    sample_rng = random.Random(int(config["sample_seed"]))
    sample_rng.shuffle(eligible)
    target_rng = random.Random(int(config["target_seed"]))
    noise_rng = random.Random(int(config["input_noise"]["seed"]))
    letter_alphabet = str(config["targets"].get("letters", string.ascii_letters))
    special_alphabet = str(config["targets"].get("special_characters", string.punctuation))
    patterns = [(5, 0), (4, 1), (3, 2), (2, 1), (1, 4), (0, 5)]
    samples = []
    for index, (root, filename, source_text, image, mask) in enumerate(eligible[:count]):
        sigma = noise_rng.uniform(float(config["input_noise"]["minimum_standard_deviation"]),
                                  float(config["input_noise"]["maximum_standard_deviation"]))
        noise_seed = int(config["input_noise"]["seed"]) + index
        noisy = output_dir / "inputs" / f"{index:04d}_{Path(filename).stem}.png"
        _noisy_copy(image, noisy, sigma, noise_seed)
        case_base = "".join(target_rng.choice(string.ascii_lowercase) for _ in range(5))
        samples.append({
            "sample_index": index, "dataset_root": str(root), "filename": filename,
            "source_text": source_text, "source_path": str(image), "input_path": str(noisy),
            "mask_path": str(mask), "noise_sigma": sigma, "noise_seed": noise_seed,
            "case_lower": case_base, "case_upper": case_base.upper(),
            **{f"special_{position}": _mixed_target(target_rng, *pattern, letter_alphabet,
                                                     special_alphabet)
               for position, pattern in enumerate(patterns, 1)},
        })
    write_jsonl(output_dir / "samples.jsonl", samples)
    return samples


def expand_jobs(samples, config, output_dir: Path, models=("regular", "self_prompting")):
    jobs = []
    target_keys = ("case_upper", "case_lower", *(f"special_{index}" for index in range(1, 7)))
    for model in models:
        for sample in samples:
            for target_key in target_keys:
                jobs.append({
                    **sample, "index": len(jobs), "model": model, "target_key": target_key,
                    "target_text": sample[target_key],
                    "generation_seed": int(config["generation_seed"]) + int(sample["sample_index"]),
                    "output_path": str(output_dir / "generated" / model / target_key /
                                       f"{int(sample['sample_index']):04d}.png"),
                })
    return jobs
