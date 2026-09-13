# TextCtrl glyph/style SupCon fine-tuning

This trainer implements `supercon_style_encoder.md` against the vendored TextCtrl
code. It freezes `StyleEncoder.vit` and `spatial_attn_block`, runs neither the
spatial representation nor the downstream removal/segmentation head, and updates
only `glyph_attn_block` using minibatch-wide supervised contrastive loss.

## Batch contract

The normalized font face in SRNet's `font.txt` is the style identity, matching
the specification's Arial-versus-Times examples. A selected record's `i_s` and
`t_f` images share that font and the generated color/effects, placement, and
background while using the source and target strings from `i_s.txt` and
`i_t.txt`. A batch with `styles_per_batch: K` therefore contains `2K` images:
two positive views for each of K distinct font classes. Records with identical
source/target strings are excluded by default.

The batch sampler preferentially puts records sharing an exact text string in
the same batch when their fonts differ. These become the requested
same-glyph/different-style hard negatives. Other font classes in the batch remain
ordinary negatives. It does not construct pairwise losses.

The glyph token sequence `[B, N, 768]` is mean-pooled to `[B, 768]`, L2-normalized,
and used in the standard all-anchor SupCon objective. Set `model.pooling: flatten`
only if token positions are known to be aligned and the larger representation is
desired.

## Checkpoint detail

The upstream pretraining architecture has both `glyph_attn_block` and
`spatial_attn_block`. TextCtrl's released `weights/style_encoder.pth`, however,
contains only the shared `VisionTransformerEncoder` (`patch_embed`, `blocks`, and
`norm`). With the supplied default config, the pretrained backbone is loaded and
the TextCtrl glyph branch begins from its upstream initialization; the trainer
prints a warning making this explicit.

For a genuinely pretrained branch, point `checkpoint.pretrained` to a full
prestyle `StyleNet`/Lightning checkpoint and set
`checkpoint.require_glyph_head_weights: true`. The loader accepts bare
`StyleEncoder`, `encoder.*`, `state_dict`/Lightning, and released backbone-only
layouts. It requires every backbone tensor and rejects partially present glyph
heads.

## Run

From the repository root, use the existing TextCtrl environment:

```bash
/home/ekim339/miniconda3/envs/textctrl/bin/python3.8 \
  -m CODEX.style_encoder_supercon.train \
  --config CODEX/style_encoder_supercon/config.yaml
```

Edit `config.yaml` to change data, precision, batch size, or checkpoint paths.
`styles_per_batch: 16` means 32 input images. The code reports training and
validation loss plus positive/negative cosine similarity.

Outputs are written below `training.output_dir`. Each checkpoint stores only the
fine-tuned `glyph_attn_block`, optimizer/scaler state, resolved configuration,
and the immutable source-checkpoint path. Set `training.resume_from` to a saved
file to continue training. Keeping the head delta separate prevents each save
from duplicating the frozen 340 MB backbone.

## Test

```bash
/home/ekim339/miniconda3/envs/textctrl/bin/python3.8 \
  -m unittest discover -s CODEX/style_encoder_supercon/tests -v
```
