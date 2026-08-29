# Self-prompting SD3.5

This directory implements the supplied specification as a self-reconstruction
training task. Every SRNet source image `i_s` is masked with `mask_s`; its own
text from `i_s.txt` is rendered as the glyph prompt and its masked-region crop
is the style prompt. The reconstruction target is the unchanged source image.
At inference, only the rendered/text-encoder string changes.

The frozen SD3.5 VAE separately encodes the masked image, glyph, and style
images. These three 16-channel latents are concatenated with the noisy
16-channel target and one mask channel. The resulting 65-channel tensor enters
the MM-DiT. The base transformer, VAE, CLIP-L, OpenCLIP bigG, and T5 remain
frozen. PEFT LoRA parameters are trained on the expanded input convolution and
joint-attention projections.

Install and train from the repository root:

```bash
python -m pip install -r CODEX/self_prompting_sd35/requirements.txt
accelerate launch -m CODEX.self_prompting_sd35.train \
  --config CODEX/self_prompting_sd35/config.yaml
```

Edit a marked region after training:

```bash
python -m CODEX.self_prompting_sd35.inference \
  --image source.png --mask mask.png --text "replacement" \
  --lora CODEX/self_prompting_sd35/checkpoints/lora-final \
  --output edited.png
```

The Markdown mentions `/home/ekim339/project/...`, but this checkout's dataset
is under `/home/ekim339/projects/SD3.5/datasets/SRNet_Datagen`; the portable
configuration therefore uses repository-relative paths to its four 50k shards.
