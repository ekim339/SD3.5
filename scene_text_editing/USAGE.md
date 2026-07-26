# Scene text editing

This sub-repository turns the design brief in `README.md` into a Hydra-driven
text-image-editing workflow. It supports:

- a modern SD3 ControlNet inpainting path with flow matching;
- the released TextCtrl SD1.5 model through an isolated legacy environment;
- JSONL, TextCtrl example, ScenePair, and synthetic TextCtrl dataset layouts;
- composable dataset and prompt sets under each task;
- explicit checkpoint acquisition; and
- a TextCtrl fine-tuning handoff generated from the Hydra training config.

## Compatibility

| Network config | Architecture | Diffusion config | Status |
| --- | --- | --- | --- |
| `sd3_inpainting` | SD3 transformer + inpainting ControlNet | `flow_matching` | Default, native Diffusers inference |
| `textctrl_sd15` | TextCtrl SD1.5 UNet + style/glyph control | `pndm` | Exact released model, isolated environment |
| `sd35_medium` | SD3.5 transformer + SD3 inpainting ControlNet | `flow_matching` | Experimental opt-in only |

TextCtrl's checkpoint cannot be loaded into SD3/SD3.5: its UNet channels,
cross-attention dimensions, text encoder, and epsilon objective differ from
the SD3 transformer and rectified-flow objective. The two supported paths are
therefore intentionally separate.

Although the design note calls the local reference "SDXL", the executable
reference files in this workspace use `StableDiffusion3Pipeline` and SD3.5
checkpoints. The native path follows that inspected implementation; it does
not silently substitute an unrelated SDXL inpainting architecture.

## Repository layout

```text
scene_text_editing/
├── configs/
│   ├── inference.yaml
│   ├── train.yaml
│   ├── download.yaml
│   ├── network/
│   ├── diffusion/
│   │   └── flow_matching.yaml
│   └── tasks/
│       └── text_image_editing/
│           ├── datasets/
│           │   ├── dataset_a.yaml
│           │   ├── dataset_b.yaml
│           │   ├── scenepair.yaml
│           │   ├── synthetic_textctrl.yaml
│           │   └── textctrl_example.yaml
│           └── prompts/
│               ├── prompt_set_a.yaml
│               └── prompt_set_b.yaml
├── data/text_image_editing/
├── datasets/
├── networks/
├── tasks/
├── checkpoints.py
├── download_models.py
├── inference.py
└── train.py
```

Config files describe data; image binaries live under `data/` or at an
external path selected through a Hydra override.

Repository-relative paths are anchored to the workspace root, so embedding the
entrypoints or launching them from another directory does not change datasets,
checkpoints, or output locations.

## Install

From the workspace root:

```bash
python3 -m pip install -r scene_text_editing/requirements.txt
```

Hugging Face downloads use `HF_HUB_CACHE`, or `$HF_HOME/hub` when only
`HF_HOME` is set, so the downloader and Diffusers inference share one cache.

The default SD3 base model is gated. Accept its Hugging Face terms and
authenticate before downloading:

```bash
hf auth login
```

`HF_TOKEN` and `HUGGINGFACE_HUB_TOKEN` are also supported.

## Validate the composed config

This loads neither data nor model weights:

```bash
python3 -m scene_text_editing.inference validate_only=true
```

Hydra can print the composed job directly:

```bash
python3 -m scene_text_editing.inference --cfg job
```

Invalid architecture/scheduler pairs fail early. For example, TextCtrl cannot
be combined with `flow_matching`.

## Prepare a dataset

### JSONL (`dataset_a`)

Copy the example manifest and add the referenced images:

```bash
cp scene_text_editing/data/text_image_editing/dataset_a/manifest.example.jsonl \
  scene_text_editing/data/text_image_editing/dataset_a/manifest.jsonl
```

Each JSON object needs `source_image`, `source_text`, and `target_text`.
`id`, `mask_image`, `target_image`, and `full_image` are optional:

```json
{"id":"sign-001","source_image":"images/sign.png","source_text":"OPEN","target_text":"CLOSED","mask_image":"masks/sign.png"}
```

All relative paths are resolved from the dataset root. With no mask, the
entire source crop is editable by default.

### TextCtrl pairs (`dataset_b`)

```text
dataset_b/
├── i_s/
│   └── example.png
├── i_s.txt
└── i_t.txt
```

Label records use `filename text`. This loader uses `split(maxsplit=1)` and
joins the source/target records by filename, so phrases and differing line
order are handled safely.

Select another nested task dataset or prompt group with an explicit Hydra
package override:

```bash
python3 -m scene_text_editing.inference \
  'tasks/text_image_editing/datasets@task.dataset=dataset_b' \
  'tasks/text_image_editing/prompts@task.prompts=prompt_set_b'
```

## Download pretrained models

Preview any acquisition without downloading:

```bash
python3 -m scene_text_editing.download_models dry_run=true
```

Download the default SD3 Medium base model and its public inpainting
ControlNet to the Hugging Face cache:

```bash
python3 -m scene_text_editing.download_models
```

Download the TextCtrl checkout, released Google Drive checkpoints, and the
SD1.5 VAE/UNet/scheduler:

```bash
python3 -m scene_text_editing.download_models \
  network=textctrl_sd15
```

The released Google Drive bundle includes a `Usage Restriction Statement`.
Review and retain that file with the checkpoints; it requires use to comply
with applicable laws, regulations, and ethical guidelines.

Large model files are intentionally ignored by Git and never downloaded on
module import.

## Run native SD3 editing

```bash
python3 -m scene_text_editing.inference \
  seed=42 \
  limit=1 \
  task.generation.width=1024 \
  task.generation.height=1024
```

Results contain edited PNG files, the fully composed `config.yaml`,
`results.jsonl`, and `summary.json`.

The existing SD3.5 model can be selected only with explicit acknowledgement
that the available inpainting ControlNet was trained for SD3 Medium:

```bash
python3 -m scene_text_editing.inference \
  network=sd35_medium \
  network.allow_experimental_base_model=true
```

## Run released TextCtrl

TextCtrl uses old PyTorch/Diffusers internals and calls CUDA directly. Keep it
in its own Python 3.8 environment. The upstream requirements do not include
Torch, so reproduce the released CUDA 11.6 stack explicitly:

```bash
conda create --name textctrl python=3.8
conda activate textctrl
python -m pip install \
  torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 \
  --extra-index-url https://download.pytorch.org/whl/cu116
python -m pip install -r external/TextCtrl/requirement.txt
python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__)"
```

Then point this adapter at that interpreter:

```bash
export TEXTCTRL_PYTHON=/path/to/textctrl-env/bin/python

python3 -m scene_text_editing.inference \
  network=textctrl_sd15 \
  diffusion=pndm \
  'tasks/text_image_editing/datasets@task.dataset=textctrl_example'
```

The released TextCtrl parser only accepts one token for source and target
labels. Its runtime consumes those labels directly rather than the Hydra
natural-language prompt template. The native SD3 backend renders the selected
prompt set and supports multi-word phrases.

## Fine-tune TextCtrl

The training config captures TextCtrl's SRNet-generated shard structure:

```text
Syn_data/
├── fonts/
├── train/train-*/
└── eval/eval-*/
```

Validation requires a non-empty `fonts/arial.ttf`, at least one train and
validation shard, aligned single-token `i_s.txt`/`i_t.txt` records, and the
corresponding `i_s/` and `t_f/` images. It fails before creating a training
handoff when any prerequisite is missing.

Set `TEXTCTRL_SYNTHETIC_ROOT`, then generate and validate the translated
upstream training config:

```bash
export TEXTCTRL_SYNTHETIC_ROOT=/path/to/Syn_data
python3 -m scene_text_editing.train
```

After checking the generated YAML and isolated environment, launch training:

```bash
python3 -m scene_text_editing.train \
  validate_only=false \
  training.execute=true
```

The worker retains TextCtrl's style encoder, glyph encoder, OCR loss, VGG
loss, and prior-guided training path. The native SD3 ControlNet config is
inference-only; a flow-matching fine-tune needs a flow target and cannot be
created by swapping schedulers on the TextCtrl checkpoint.

## Tests

The fast test suite composes Hydra configs and validates datasets, prompt
rendering, compatibility rules, checkpoint plans, and image/mask preparation
without downloading model weights:

```bash
python3 -m unittest discover -s scene_text_editing/tests -v
```

An end-to-end GPU smoke test remains environment-dependent because the base
models are large/gated and TextCtrl requires its legacy CUDA environment.
