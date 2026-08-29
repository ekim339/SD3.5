# Self-prompting SD3.5 scene-text editing

This package trains a parameter-efficient SD3.5 adaptation over the four local
SRNet-Datagen 50k shards (200k examples). For each example it constructs the
masked source, a tight style crop padded back to 512x512, and a single-line
Pillow glyph map. The frozen SD3.5 VAE encodes each image independently.

The MM-DiT input is `[z_t, z_masked, z_glyph, z_style, mask]`: 65 channels for
the standard 16-channel SD3.5 latent. The pretrained 16-channel patch projection
is copied into the expanded projection and the 49 added channels start at zero.

The SD3.5 transformer, VAE, and all three text encoders remain frozen. PEFT LoRA
adapters are trained on the expanded input projection and SD3 joint-attention
projections. Including `pos_embed.proj` is essential: its convolutional LoRA
provides the only trainable path from the masked-image, glyph, style, and mask
channels into the frozen backbone. Rank, alpha, dropout, and target modules are
configured under `model.lora` in `config.yaml`.

Install and launch from the repository root:

```bash
python -m pip install -r CODEX/self_prompting_sd35/requirements.txt

NCCL_P2P_DISABLE=1 accelerate launch \
  --multi_gpu --num_processes 2 --mixed_precision bf16 \
  -m CODEX.self_prompting_sd35.train \
  --config CODEX/self_prompting_sd35/config.yaml
```

`NCCL_P2P_DISABLE=1` is required on `mmplab2`, where direct NCCL P2P between
the two RTX A4500 GPUs hangs; it makes NCCL use shared-memory transport. It may
be omitted on hosts whose P2P collective test succeeds.

Accept the gated Stability AI model license and authenticate with Hugging Face
before training. LoRA removes full-transformer gradients and optimizer states,
but each worker still loads the frozen SD3.5 pipeline, so BF16 and gradient
checkpointing remain enabled.

## Checkpoints

Periodic `checkpoint-NNNNNN` directories contain:

- `pytorch_lora_weights.safetensors`: only the transformer LoRA tensors
- Accelerate optimizer, scaler, and RNG state needed to resume

The output directory itself receives the final
`pytorch_lora_weights.safetensors`. Frozen SD3.5 transformer or encoder
weights are never duplicated in these checkpoints.

Resume with:

```bash
NCCL_P2P_DISABLE=1 accelerate launch \
  --multi_gpu --num_processes 2 --mixed_precision bf16 \
  -m CODEX.self_prompting_sd35.train \
  --config CODEX/self_prompting_sd35/config.yaml \
  --resume CODEX/self_prompting_sd35/checkpoints/checkpoint-002500
```

Loading this adapter requires constructing `SelfPromptingSD35` first so its
patch projection is expanded from 16 to 65 channels, then calling
`model.load_lora_weights(path)`. A stock 16-channel SD3.5 pipeline cannot load
the input-projection LoRA directly because its tensor shape is intentionally
different.
