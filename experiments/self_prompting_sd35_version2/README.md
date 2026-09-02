# Self-Prompting SD3.5 version-2 evaluation

This package implements the experiment in `self_promptsing_sd35_v2.md`. All
new manifests, generated images, OCR predictions, reports, and collages stay
inside `experiments/self_prompting_sd35_version2/results`.

## What is held fixed

- The exact 100 version-1 noisy input images are copied byte-for-byte from
  `experiments/self_prompting_sd35_version1/results/inputs`.
- Sample filenames, source strings, masks, noise seeds, and noise standard
  deviations are validated against the version-1 manifest.
- The exact `(filename, target_key, target_text)` combinations and generation
  seeds are deduplicated from the version-1 detailed CSV. The resulting grid is
  exactly 100 samples by 8 target keys, or 800 jobs.
- Metrics are case-sensitive exact accuracy (ACC), normalized edit-distance
  similarity (NED), and reference-normalized character error rate (CER).
  Reported standard deviations are population standard deviations, matching
  version 1.

## Version-2 model path

The generation worker constructs `SelfPromptingSD35` before restoring the
checkpoint, then calls `model.load_lora_weights(...)`. This restores both:

- `pytorch_lora_weights.safetensors` (attention LoRA)
- `input_projection.safetensors` (the learned 65-channel input projection)

The base pipeline is pinned to revision
`b940f670f0eda2d07fbb75229e779da1ad11eb80`, the sole local SD3.5 Medium
snapshot available to the version-2 trainer.

The target string is rendered as the glyph condition and is encoded only by
T5. Both CLIP towers receive an empty, content-free prompt; their native pooled
conditioning tensor is retained so the SD3.5 transformer receives the tensor
layout it expects. The visual style condition remains the VAE encoding of the
source text crop.

## Run

From the project root:

```bash
python -m experiments.self_prompting_sd35_version2.run stage=prepare
python -m experiments.self_prompting_sd35_version2.run stage=generate
python -m experiments.self_prompting_sd35_version2.run stage=ocr
python -m experiments.self_prompting_sd35_version2.run stage=report
```

Or run the complete sequence:

```bash
python -m experiments.self_prompting_sd35_version2.run stage=all
```

Configuration values can be overridden as OmegaConf dot-list arguments. For
example:

```bash
python -m experiments.self_prompting_sd35_version2.run \
  stage=generate self_prompting_sd35.num_inference_steps=40 overwrite=true
```

The default interpreters can also be overridden with
`SELF_PROMPTING_SD35_PYTHON` and `OCR_PYTHON`.

Generation and OCR are resumable by default. Existing generated images and OCR
rows are skipped only when their provenance still matches. Generation records
digests of the manifest, input/mask pixels, checkpoint artifacts, settings, and
runtime source. Every OCR row records digests of its generated image, OCR
checkpoint, and OCR config. A mismatched resume fails with an instruction to
use `overwrite=true`; this prevents mixed or mislabeled result sets.

## Outputs

- `results/config.yaml`: resolved run configuration snapshot
- `results/samples.jsonl`: normalized 100-sample manifest
- `results/jobs.jsonl`: the 800 version-2 generation jobs
- `results/generation_provenance.json`: content signature and completion state
  used to validate generation resumes
- `results/inputs/`: exact byte copies of the v1 noisy inputs
- `results/generated/self_prompting_sd35_version2/<target_key>/`: edited images
- `results/ocr_predictions.jsonl`: per-image ABINet predictions
- `results/detailed_results.csv`: all v2 predictions and metrics
- `results/capital_lowercase_summary.csv`: copied v1 two-table report plus one
  version-2 row in each table
- `results/special_character_summary.csv`: copied v1 report plus one version-2
  row containing all 18 mean/std cells
- `results/capital_lowercase_collage.png`: literal requested 3x5 layout—noisy
  source, v1 uppercase, and v2 lowercase
- `results/special_character_collage.png`: first noisy source plus six v2 targets
  in a 7x1 layout

## Tests

```bash
python3 -m pytest -q experiments/self_prompting_sd35_version2/tests
```

The suite uses the real v1 manifests for its 100-sample/800-target provenance
check but does not load SD3.5 or run GPU generation.

## OCR limitation

The released TextCtrl ABINet charset is primarily lowercase alphanumeric. It
therefore cannot faithfully emit uppercase letters or most punctuation even
when those glyphs are visually correct. The evaluator intentionally preserves
the requested case-sensitive metrics, so uppercase and special-character
scores measure the released OCR system as well as image quality.

## Training/inference caveat

The evaluator enforces T5-only target conditioning at inference. A checkpoint
matches that contract only when its trainer loaded the same code. For the
configured checkpoint-030000, the conditioning and training files predate the
active trainer's launch, and the checkpoint was saved afterward; this strongly
supports matching T5-only provenance.

Training is still self-reconstruction: source and target text are identical,
and the visual style crop contains that same word. At evaluation time the new
target differs while the style crop retains the source content. The model has
not seen disentangled content/style pairs and may therefore copy source glyphs.
