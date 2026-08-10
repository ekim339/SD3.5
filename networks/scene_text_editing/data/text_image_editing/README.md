# Text-image-editing data

Hydra selects dataset descriptions from
`networks/scene_text_editing/configs/tasks/text_image_editing/datasets/`. Keep image
binaries here (or point a config at another absolute path); do not place data
inside the config tree.

## `dataset_a`: JSONL

Each non-empty line in `dataset_a/manifest.jsonl` is an object with:

```json
{"id":"sign-001","source_image":"images/sign-001.png","source_text":"OPEN","target_text":"CLOSED","mask_image":"masks/sign-001.png"}
```

`mask_image` and `target_image` are optional. Paths are relative to the
dataset root. A missing mask means the entire source crop is editable when
`full_crop_mask_when_missing=true`.

## `dataset_b`: TextCtrl pairs

The TextCtrl-compatible format is:

```text
dataset_b/
├── i_s/
├── i_s.txt
└── i_t.txt
```

Both label files use `filename text`, one record per line. This repository's
loader accepts spaces in labels and aligns records by filename. The isolated
upstream TextCtrl inference backend itself only accepts single-token labels;
use the native SD3 backend for multi-word replacements.

Additional ready-made configs describe the upstream TextCtrl examples,
ScenePair evaluation data, and SRNet-generated synthetic training shards.
