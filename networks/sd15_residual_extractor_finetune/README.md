# SD1.5 residual-extractor fine-tuning

This package continues from the released TextCtrl checkpoint while replacing
its ViT style encoder with a frozen pretrained residual extractor.

For every same-style SRNet source/target pair:

1. The source text is rendered with the exact Pillow renderer, canonical font,
   sizing loop, centering, LANCZOS resize, and `[-1,1]` normalization used during
   residual-extractor training.
2. The frozen extractor computes `R(E(I_s), E(G_s))` with shape
   `[B,256,16,16]`.
3. The trainable residual adapter maps it to `[B,768,16,16]`, and the original
   pretrained TextCtrl style pyramid creates 13 U-Net controls.
4. A separate CNN consumes the standard-font target glyph and creates another
   13 controls. Every output projection is zero-initialized.
5. The two branches are combined with trainable per-level scales. TextCtrl's
   frozen glyph encoder still supplies the target-text cross-attention features.

The full SD1.5 U-Net, residual adapter, original TextCtrl style pyramid, glyph
spatial adapter, and control scales train jointly. The VAE, TextCtrl glyph
encoder, OCR/VGG loss networks, and residual extractor remain frozen.

Validate paths and configuration without loading models:

```bash
/home/ekim339/miniconda3/envs/ste/bin/python \
  -m networks.sd15_residual_extractor_finetune.run validate_only=true
```

Start training on GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 \
TEXTCTRL_PYTHON=/home/ekim339/miniconda3/envs/textctrl/bin/python \
/home/ekim339/miniconda3/envs/ste/bin/python \
  -m networks.sd15_residual_extractor_finetune.run
```

Checkpoints, the resolved runtime config, and the generated TextCtrl config are
stored under `networks/sd15_residual_extractor_finetune/checkpoints/`. Use
`training.resume_from=/path/to/last.ckpt` to resume. Full-U-Net optimizer
checkpoints require several gigabytes; free sufficient disk space before starting
a production run.

The default loss continues TextCtrl's diffusion, OCR, and VGG reconstruction
objectives. For a lower-memory diagnostic run, OCR and reconstruction can be
disabled with `loss.ocr_supervised=false loss.reconstruction=false`.
