"""Deterministic SRNet sampling, corruption, and experiment job manifests."""

from __future__ import annotations

import json
import random
import string
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


MODEL_KEYS = ("textctrl", "self_prompting_sd35")
CASE_TARGET_KEYS = ("case_upper", "case_lower")
SPECIAL_TARGET_KEYS = tuple(f"special_{index}" for index in range(1, 7))
TARGET_KEYS = CASE_TARGET_KEYS + SPECIAL_TARGET_KEYS

# The fourth tuple intentionally has length three. This follows item b.1-4 in
# the supplied experiment specification: two letters and one special character.
SPECIAL_PATTERNS = ((5, 0), (4, 1), (3, 2), (2, 1), (1, 4), (0, 5))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSON-lines file, ignoring blank lines."""

    source = Path(path)
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"Expected an object at {source}:{line_number}")
        rows.append(value)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write a deterministic JSON-lines manifest."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def read_labels(path: str | Path) -> dict[str, str]:
    """Read SRNet's ``filename text`` label format."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    labels: dict[str, str] = {}
    for line_number, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        parts = raw.strip().split(maxsplit=1)
        if not parts:
            continue
        if len(parts) != 2:
            raise ValueError(f"Expected `filename text` at {source}:{line_number}")
        labels[parts[0]] = parts[1]
    return labels


def resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def discover_samples(config: Mapping[str, Any], project_root: Path) -> list[dict[str, Any]]:
    """Return every aligned SRNet sample whose source label has five characters."""

    records: list[dict[str, Any]] = []
    dataset = config["dataset"]
    for raw_root in dataset["roots"]:
        root = resolve_path(project_root, str(raw_root))
        labels = read_labels(root / str(dataset["source_labels"]))
        for filename, source_text in labels.items():
            source_path = root / str(dataset["source_dir"]) / filename
            mask_path = root / str(dataset["mask_dir"]) / filename
            if len(source_text) == 5 and source_path.is_file() and mask_path.is_file():
                records.append(
                    {
                        "dataset_root": str(root),
                        "filename": filename,
                        "source_text": source_text,
                        "source_path": str(source_path),
                        "mask_path": str(mask_path),
                    }
                )
    return sorted(records, key=lambda row: (row["dataset_root"], row["filename"]))


def add_gaussian_noise(
    source: str | Path,
    destination: str | Path,
    *,
    standard_deviation: float,
    seed: int,
) -> None:
    """Save one clipped RGB Gaussian-noise perturbation of ``source``."""

    source_path, destination_path = Path(source), Path(destination)
    with Image.open(source_path) as opened:
        pixels = np.asarray(opened.convert("RGB"), dtype=np.float32)
    noise = np.random.default_rng(int(seed)).normal(
        0.0, float(standard_deviation), pixels.shape
    )
    perturbed = np.clip(pixels + noise, 0.0, 255.0).astype(np.uint8)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(perturbed, mode="RGB").save(destination_path)


def mixed_target(
    rng: random.Random,
    letter_count: int,
    special_count: int,
    letter_alphabet: str,
    special_alphabet: str,
) -> str:
    """Draw and shuffle a target with the requested character composition."""

    characters = [rng.choice(letter_alphabet) for _ in range(letter_count)]
    characters.extend(rng.choice(special_alphabet) for _ in range(special_count))
    rng.shuffle(characters)
    return "".join(characters)


def validate_target_alphabets(letter_alphabet: str, special_alphabet: str) -> None:
    if not letter_alphabet:
        raise ValueError("targets.letters must not be empty")
    if not special_alphabet:
        raise ValueError("targets.special_characters must not be empty")
    overlap = set(letter_alphabet) & set(special_alphabet)
    if overlap:
        shown = "".join(sorted(overlap))
        raise ValueError(f"Letter and special-character alphabets overlap: {shown!r}")


def validate_samples(samples: Sequence[Mapping[str, Any]], expected_count: int) -> None:
    if len(samples) != expected_count:
        raise ValueError(f"Expected {expected_count} samples, found {len(samples)}")
    required = {
        "sample_index",
        "source_text",
        "input_path",
        "mask_path",
        *TARGET_KEYS,
    }
    seen: set[int] = set()
    for row_number, sample in enumerate(samples, 1):
        missing = sorted(required - sample.keys())
        if missing:
            raise ValueError(f"Sample row {row_number} is missing: {', '.join(missing)}")
        index = int(sample["sample_index"])
        if index in seen:
            raise ValueError(f"Duplicate sample_index: {index}")
        seen.add(index)
        if len(str(sample["source_text"])) != 5:
            raise ValueError(f"Sample {index} source text is not five characters")
        for path_key in ("input_path", "mask_path"):
            if not Path(str(sample[path_key])).is_file():
                raise FileNotFoundError(sample[path_key])


def prepare_samples(
    config: Mapping[str, Any], project_root: Path, output_dir: Path
) -> list[dict[str, Any]]:
    """Select sources, add fixed noise, and construct all eight target strings."""

    eligible = discover_samples(config, project_root)
    count = int(config["sample_count"])
    if len(eligible) < count:
        raise ValueError(
            f"Only {len(eligible)} aligned five-character SRNet samples were found; need {count}"
        )

    sample_rng = random.Random(int(config["sample_seed"]))
    sample_rng.shuffle(eligible)
    target_rng = random.Random(int(config["target_seed"]))
    sigma_rng = random.Random(int(config["input_noise"]["seed"]))
    letter_alphabet = str(config["targets"].get("letters", string.ascii_letters))
    special_alphabet = str(
        config["targets"].get("special_characters", string.punctuation)
    )
    validate_target_alphabets(letter_alphabet, special_alphabet)

    minimum_sigma = float(config["input_noise"]["minimum_standard_deviation"])
    maximum_sigma = float(config["input_noise"]["maximum_standard_deviation"])
    if minimum_sigma < 0 or maximum_sigma < minimum_sigma:
        raise ValueError("input_noise requires 0 <= minimum_standard_deviation <= maximum")

    samples: list[dict[str, Any]] = []
    for index, record in enumerate(eligible[:count]):
        sigma = sigma_rng.uniform(minimum_sigma, maximum_sigma)
        noise_seed = int(config["input_noise"]["seed"]) + index
        input_path = (
            output_dir
            / "inputs"
            / f"{index:04d}_{Path(record['filename']).stem}.png"
        ).resolve()
        add_gaussian_noise(
            record["source_path"],
            input_path,
            standard_deviation=sigma,
            seed=noise_seed,
        )
        case_lower = "".join(target_rng.choice(string.ascii_lowercase) for _ in range(5))
        targets = {
            f"special_{position}": mixed_target(
                target_rng,
                letter_count,
                special_count,
                letter_alphabet,
                special_alphabet,
            )
            for position, (letter_count, special_count) in enumerate(
                SPECIAL_PATTERNS, 1
            )
        }
        samples.append(
            {
                **record,
                "sample_index": index,
                "input_path": str(input_path),
                "noise_standard_deviation": sigma,
                "noise_seed": noise_seed,
                "case_lower": case_lower,
                "case_upper": case_lower.upper(),
                **targets,
            }
        )

    validate_samples(samples, count)
    write_jsonl(output_dir / "samples.jsonl", samples)
    return samples


def expand_jobs(
    samples: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    output_dir: Path,
    models: Sequence[str] = MODEL_KEYS,
) -> list[dict[str, Any]]:
    """Pair every sample/target with each selected model."""

    invalid = [model for model in models if model not in MODEL_KEYS]
    if invalid:
        raise ValueError(f"Unknown model key: {invalid[0]}")
    jobs: list[dict[str, Any]] = []
    base_seed = int(config["generation_seed"])
    for model in models:
        for sample in samples:
            sample_index = int(sample["sample_index"])
            for target_position, target_key in enumerate(TARGET_KEYS):
                jobs.append(
                    {
                        **dict(sample),
                        "index": len(jobs),
                        "model": model,
                        "target_key": target_key,
                        "target_text": str(sample[target_key]),
                        # The same sample/target seed is paired across models.
                        "generation_seed": base_seed + sample_index * len(TARGET_KEYS) + target_position,
                        "output_path": str(
                            (
                                output_dir
                                / "generated"
                                / model
                                / target_key
                                / f"{sample_index:04d}.png"
                            ).resolve()
                        ),
                    }
                )
    return jobs

