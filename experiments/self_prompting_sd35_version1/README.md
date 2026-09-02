# TextCtrl + SD1.5 versus Self-Prompting SD3.5

This directory implements the experiment in `self_prompting_sd35.md`. A full
run deterministically selects 100 aligned five-character SRNet source crops,
adds one fixed Gaussian-noise perturbation to each crop, and reuses those exact
inputs and targets for both models. It generates:

- uppercase and lowercase versions of one random five-letter target;
- six letter/punctuation target families from the specification;
- OCR-based ACC, NED, and CER for all 1,600 generated images;
- the two requested summary CSVs plus a detailed per-image CSV; and
- the requested 5x5 capitalization and 2x7 special-character collages.

All manifests, noisy inputs, generated images, reports, and collages are kept
under this experiment directory. The run is resumable: generated images and OCR
rows are reused unless `overwrite=true`.

## Paths used by this implementation

The TextCtrl path written in the brief contains `projects` (plural), which does
not exist on this machine. The installed checkout and weights are actually:

```text
/home/ekim339/project/SD3.5/networks/TextCtrl
/home/ekim339/project/SD3.5/networks/TextCtrl/weights
```

The configured Self-Prompting checkpoint is the requested LoRA-only checkpoint:

```text
/home/ekim339/project/SD3.5/CODEX/self_prompting_sd35/checkpoints/checkpoint-050000
```

The SD3.5 worker first constructs the matching 65-channel SD3.5 Medium model and
then loads this LoRA; the checkpoint is not treated as a standalone pipeline.

Target text is encoded only by T5. Both CLIP towers receive content-free empty
prompts so Diffusers still produces SD3.5's native neutral CLIP token block and
required pooled conditioning tensor. Visual style guidance comes from the
VAE-encoded source crop.

Version 1 was trained with nonempty CLIP text, so using it with this T5-only

Set the checkpoint directory in `finetuned_sd35.yaml`:

```yaml
self_prompting_sd35:
  checkpoint_path: /absolute/path/to/your/checkpoint
```

The directory must contain `pytorch_lora_weights.safetensors`. A command-line
`self_prompting_sd35.checkpoint_path=...` override still takes precedence.

## Environments

Run the orchestrator and Self-Prompting SD3.5 worker from the project environment
that satisfies `CODEX/self_prompting_sd35/requirements.txt`. SD3.5 Medium is a
gated Hugging Face model, so accept its license and authenticate before the full
run.

Released TextCtrl vendors an old Diffusers tree and requires its pinned Python
3.8/PyTorch environment. Create it once if it is not already installed:

```bash
cd /home/ekim339/project/SD3.5
conda env create -f networks/TextCtrl/environment.yaml
export TEXTCTRL_PYTHON=/home/ekim339/miniconda3/envs/textctrl/bin/python
```

`config.yaml` also accepts `SELF_PROMPTING_SD35_PYTHON` when the SD3.5 runtime
uses a different interpreter.

## Run

First validate sampling, target construction, and manifests without loading a
model:

```bash
cd /home/ekim339/project/SD3.5
python -m experiments.self_prompting_sd35.run stage=prepare
```

Run the complete experiment:

```bash
python -m experiments.self_prompting_sd35.run
```

The complete workflow is intentionally ordered as follows:

1. Generate all fine-tuned Self-Prompting SD3.5 images.
2. Run OCR on those images and save `self_prompting_sd35_ocr_results.csv`.
3. Generate TextCtrl images only after that CSV has been closed on disk.
4. Run TextCtrl OCR and write the combined reports and collages.

Stages can be resumed independently:

```bash
python -m experiments.self_prompting_sd35.run stage=textctrl
python -m experiments.self_prompting_sd35.run stage=self_prompting_sd35
python -m experiments.self_prompting_sd35.run stage=ocr
python -m experiments.self_prompting_sd35.run stage=report
```

Use `mode=textctrl` or `mode=self_prompting_sd35` for a model-only generation
run. To evaluate another checkpoint on exactly the same noisy sources and target
strings, point `samples_path` at the prior manifest and choose a new output
directory beneath this experiment:

```bash
python -m experiments.self_prompting_sd35.run \
  mode=self_prompting_sd35 stage=self_prompting_sd35 \
  samples_path=experiments/self_prompting_sd35/results/samples.jsonl \
  output_dir=experiments/self_prompting_sd35/results/checkpoint_variant \
  self_prompting_sd35.checkpoint_path=/path/to/checkpoint
```

## Outputs

The default output directory is `experiments/self_prompting_sd35/results`:

```text
results/
├── config.yaml
├── samples.jsonl
├── jobs.jsonl
├── inputs/
├── generated/
│   ├── textctrl/
│   └── self_prompting_sd35/
├── ocr_predictions.jsonl
├── self_prompting_sd35_ocr_results.csv
├── detailed_results.csv
├── capital_lowercase_summary.csv
├── special_character_summary.csv
├── capital_lowercase_collage.png
└── special_character_collage.png
```

`capital_lowercase_summary.csv` contains two tables. Each model row has separate
mean and population-standard-deviation columns for ACC, NED, and CER.
`special_character_summary.csv` has 18 metric columns; each cell is formatted as
`mean/std`.

## OCR interpretation caveat

The bundled and project-standard TextCtrl ABINet checkpoint uses a 36-character
lowercase alphanumeric charset. It cannot emit punctuation or distinguish upper
from lowercase. The default metrics remain case-sensitive because that is what
the experiment requests; consequently, uppercase and punctuation results expose
this detector limitation as well as model performance. Do not interpret those
two slices as a detector-independent measure. Set `metrics.case_sensitive=false`
only for an explicitly case-insensitive auxiliary analysis.

The SD3.5 worker crops the model's centered 512x512 canvas back to the fitted
source region and restores the original source dimensions before OCR. This
prevents the small SRNet word crop from being evaluated as tiny text inside a
large black square and makes the detector input geometry comparable to TextCtrl.

