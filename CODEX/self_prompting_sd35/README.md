# Self-prompting SD3.5 scene-text editing

This package implements the requested SD3.5 adaptation over the four local
SRNet-Datagen 50k shards (200k examples). For each example it constructs the
masked source, a tight style crop padded back to 512x512, and a single-line
Pillow glyph map. The frozen SD3.5 VAE encodes each image independently.

The MM-DiT input is `[z_t, z_masked, z_glyph, z_style, mask]`: 65 channels for
the standard 16-channel SD3.5 latent. The original 16-channel weights are copied
into the expanded patch projection and all added weights start at zero. All
MM-DiT parameters are optimized; the VAE and all three SD3 text encoders remain
frozen. `StableDiffusion3Pipeline.encode_prompt` supplies the native CLIP-L,
OpenCLIP bigG, and T5 token/pooled conditioning for the replacement string.

From the repository root:

```bash
python -m pip install -r CODEX/self_prompting_sd35/requirements.txt
accelerate launch -m CODEX.self_prompting_sd35.train \
  --config CODEX/self_prompting_sd35/config.yaml
```

Accept the gated Stability AI model license and authenticate with Hugging Face
before training. Full SD3.5 training is memory intensive; use an Accelerate
DeepSpeed/FSDP configuration for multi-GPU sharding. Checkpoints and final
Diffusers transformer weights are written below this package by default.

Resume an Accelerate state checkpoint with:

```bash
accelerate launch -m CODEX.self_prompting_sd35.train \
  --resume CODEX/self_prompting_sd35/checkpoints/checkpoint-002500
```
