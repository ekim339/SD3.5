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

Before the first run, accept the gated SD3.5 Medium license on Hugging Face and
authenticate with `hf auth login`.

The checked-in configuration disables NCCL peer-to-peer transport before
Accelerate initializes distributed training because P2P fails on mmplab2's RTX
A4500 pair. Set `distributed.nccl_p2p_disable: false` on a host with working
GPU peer-to-peer access.

Periodic `checkpoint-*` directories contain LoRA weights plus Accelerate
training state (optimizer, sampler/scaler when applicable, and RNG state).
Resume one with:

```bash
accelerate launch -m CODEX.self_prompting_sd35.train \
  --config CODEX/self_prompting_sd35/config.yaml \
  --resume CODEX/self_prompting_sd35/checkpoints/checkpoint-002500
```

Full-finetuning checkpoints are not compatible with this LoRA optimizer state.
Resume with the same base model, LoRA rank, and LoRA target modules used to
create the checkpoint.

Edit a marked region after training:

```bash
python -m CODEX.self_prompting_sd35.inference \
  --image source.png --mask mask.png --text "replacement" \
  --lora CODEX/self_prompting_sd35/checkpoints/lora-final \
  --output edited.png
```

If training changed the base model or resolution, pass the matching `--model`
and `--resolution` values during inference.

Dataset paths are repository-relative; launch from the repository root.
