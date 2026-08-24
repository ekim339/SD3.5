# Unified SD3.5 and TextCtrl scene-text editing

The repository now has one entry point and one controlling config:

```bash
python run.py
```

Edit `configs/base_config.yaml` to select:

- `defaults.network: sd3.5` for the glyph/style-conditioned SD3.5 LoRA implementation;
- `defaults.network: textctrl` for released TextCtrl;
- `mode: inference` to generate samples;
- `mode: train` to fine-tune; and
- `model.checkpoint` for the selected network checkpoint.

`diffusion: auto` selects flow matching for SD3.5 and PNDM for TextCtrl. It can
be replaced with `flow_matching` or `pndm` explicitly; incompatible combinations
fail before a model loads.

Install the shared environment with:

```bash
python -m pip install -r requirements.txt
```

TextCtrl still requires its pinned legacy Python/CUDA environment. Set
`TEXTCTRL_PYTHON` to that interpreter. Its source and released weights are in
`networks/external/TextCtrl/`.

## Common commands

Validate configuration and data without loading a model:

```bash
python run.py validate_only=true limit=1
```

Select TextCtrl (the checkpoint must match TextCtrl):

```bash
python run.py network=textctrl \
  model.checkpoint=networks/external/TextCtrl/weights/model.pth
```

TextCtrl uses its released ViT style encoder by default. To replace only that
style encoder during inference with the frozen residual extractor and trained
channel adapter configured in `base_config.yaml`, add:

```bash
python run.py network=textctrl \
  model.checkpoint=networks/external/TextCtrl/weights/model.pth \
  textctrl_style.encoder=residual
```

Set `textctrl_style.encoder=textctrl` to use the original released pipeline.
Residual conditioning renders the source string with the same Pillow renderer,
canonical font, resolution, and normalization used to train the residual
extractor. It is currently an inference-only option.

Fine-tune SD3.5 on the configured SRNet partitions:

```bash
python run.py mode=train dataset=srnet
```

Fine-tune TextCtrl using its isolated environment and an SRNet layout with both
`train/` and `eval/` shards:

```bash
python run.py mode=train network=textctrl dataset=srnet \
  model.checkpoint=networks/external/TextCtrl/weights/model.pth
```

Network, dataset, detector, and diffusion groups live under `configs/`. The
implementation packages are `networks/sd35_implementation/` and `networks/scene_text_editing/`;
large checkpoints, generated results, and utility scripts live in
`checkpoints/`, `results/`, and `tools/` respectively.
