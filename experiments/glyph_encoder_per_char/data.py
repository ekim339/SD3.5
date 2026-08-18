from __future__ import annotations
import json, random
from pathlib import Path

def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]

def write_jsonl(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in rows), encoding="utf-8")

def resolve(root, value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()

def labels(path):
    result = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.strip().split(maxsplit=1)
        if parts:
            if len(parts) != 2: raise ValueError(f"Expected filename and text at {path}:{number}")
            result[parts[0]] = parts[1]
    return result

def prepare_samples(config, project_root, output_dir):
    import numpy as np
    from PIL import Image
    eligible = []
    dataset = config["dataset"]
    for value in dataset["roots"]:
        root = resolve(project_root, str(value))
        for filename, text in labels(root / dataset["source_labels"]).items():
            image = root / dataset["source_dir"] / filename
            if len(text) == 5 and image.is_file(): eligible.append((root, filename, text, image))
    eligible.sort(key=lambda x: (str(x[0]), x[1]))
    random.Random(int(config["sample_seed"])).shuffle(eligible)
    count = int(config["sample_count"])
    if len(eligible) < count: raise ValueError(f"Only {len(eligible)} eligible samples; need {count}")
    target_rng = random.Random(int(config["target_seed"])); noise_rng = random.Random(int(config["input_noise"]["seed"]))
    rows = []
    for index, (root, filename, source_text, source) in enumerate(eligible[:count]):
        sigma = noise_rng.uniform(float(config["input_noise"]["minimum_standard_deviation"]), float(config["input_noise"]["maximum_standard_deviation"]))
        seed = int(config["input_noise"]["seed"]) + index
        destination = output_dir / "inputs" / f"{index:04d}_{Path(filename).stem}.png"
        with Image.open(source) as opened: pixels = np.asarray(opened.convert("RGB"), dtype=np.float32)
        noisy = np.clip(pixels + np.random.default_rng(seed).normal(0, sigma, pixels.shape), 0, 255).astype(np.uint8)
        destination.parent.mkdir(parents=True, exist_ok=True); Image.fromarray(noisy).save(destination)
        rows.append({"sample_index": index, "dataset_root": str(root), "filename": filename,
            "source_path": str(source), "input_path": str(destination), "source_text": source_text,
            "target_text": "".join(target_rng.choice(str(config["target_alphabet"])) for _ in range(5)),
            "input_noise_standard_deviation": sigma, "input_noise_seed": seed})
    write_jsonl(output_dir / "samples.jsonl", rows); return rows

def expand_jobs(samples, config, output_dir):
    jobs = []
    for sample in samples:
        for proportion in config["masking_proportions"]:
            for position in range(5):
                p = float(proportion); stem = str(p).replace(".", "p")
                jobs.append({**sample, "index": len(jobs), "masking_proportion": p,
                    "masked_character_index": position, "masked_character": sample["target_text"][position],
                    "glyph_mask_seed": int(config["seed"]) + 100000 + int(sample["sample_index"]) * 5 + position,
                    "generation_seed": int(config["seed"]) + 200000 + int(sample["sample_index"]),
                    "output_path": str(output_dir / "generated" / f"mask-{stem}" / f"char-{position+1}" / f"{int(sample['sample_index']):04d}.png")})
    return jobs
