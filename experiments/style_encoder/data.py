from __future__ import annotations

import json
import random
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                    encoding="utf-8")


def _resolve(root, value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _labels(path):
    result = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.strip().split(maxsplit=1)
        if parts:
            if len(parts) != 2:
                raise ValueError(f"Expected filename and text at {path}:{number}")
            result[parts[0]] = parts[1]
    return result


def prepare_samples(config, project_root, output_dir):
    import numpy as np
    from PIL import Image

    eligible = []
    dataset = config["dataset"]
    for value in dataset["roots"]:
        root = _resolve(project_root, str(value))
        for filename, text in _labels(root / dataset["source_labels"]).items():
            source = root / dataset["source_dir"] / filename
            if len(text) == 5 and source.is_file():
                eligible.append((root, filename, text, source))
    eligible.sort(key=lambda item: (str(item[0]), item[1]))
    random.Random(int(config["sample_seed"])).shuffle(eligible)
    count = int(config["sample_count"])
    if len(eligible) < count:
        raise ValueError(f"Only {len(eligible)} eligible five-character samples; need {count}")

    target_rng = random.Random(int(config["target_seed"]))
    noise_rng = random.Random(int(config["input_noise"]["seed"]))
    rows = []
    for index, (root, filename, source_text, source) in enumerate(eligible[:count]):
        sigma = noise_rng.uniform(float(config["input_noise"]["minimum_standard_deviation"]),
                                  float(config["input_noise"]["maximum_standard_deviation"]))
        noise_seed = int(config["input_noise"]["seed"]) + index
        destination = output_dir / "inputs" / f"{index:04d}_{Path(filename).stem}.png"
        with Image.open(source) as opened:
            pixels = np.asarray(opened.convert("RGB"), dtype=np.float32)
        noise = np.random.default_rng(noise_seed).normal(0, sigma, pixels.shape)
        noisy = np.clip(pixels + noise, 0, 255).astype(np.uint8)
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(noisy).save(destination)
        rows.append({"sample_index": index, "dataset_root": str(root), "filename": filename,
                     "source_path": str(source), "input_path": str(destination),
                     "source_text": source_text,
                     "target_text": "".join(target_rng.choice(str(config["target_alphabet"]))
                                            for _ in range(5)),
                     "input_noise_standard_deviation": sigma, "input_noise_seed": noise_seed})
    write_jsonl(output_dir / "samples.jsonl", rows)
    return rows


def patch_tokens(square_row: int, square_col: int):
    """Return row-major indices for a 4x4 square in the 16x16 token grid."""
    return [row * 16 + col
            for row in range(square_row * 4, square_row * 4 + 4)
            for col in range(square_col * 4, square_col * 4 + 4)]


def expand_jobs(samples, config, output_dir):
    jobs = []
    seed = int(config["seed"])
    for sample in samples:
        sample_index = int(sample["sample_index"])
        common = {**sample, "generation_seed": seed + 200000 + sample_index}
        for proportion in config["masking_proportions"]:
            proportion = float(proportion)
            name = str(proportion).replace(".", "p")
            jobs.append({**common, "index": len(jobs), "experiment": "proportion",
                         "masking_proportion": proportion, "masked_square": None,
                         "masked_token_indices": [],
                         "style_mask_seed": seed + 100000 + sample_index * 10 + round(proportion * 10),
                         "output_path": str(output_dir / "generated" / "proportions" /
                                            f"mask-{name}" / f"{sample_index:04d}.png")})
        # A separate no-mask baseline makes patch comparisons explicit.
        patch_specs = [(None, None, [])] + [
            (row, col, patch_tokens(row, col)) for row in range(4) for col in range(4)]
        for row, col, tokens in patch_specs:
            label = "none" if row is None else f"r{row}c{col}"
            jobs.append({**common, "index": len(jobs), "experiment": "patch",
                         "masking_proportion": 0.0, "masked_square": None if row is None else [row, col],
                         "masked_token_indices": tokens, "style_mask_seed": None,
                         "output_path": str(output_dir / "generated" / "patches" / label /
                                            f"{sample_index:04d}.png")})
    return jobs
