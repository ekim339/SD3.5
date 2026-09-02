# Self-prompting SD3.5

This directory implements the supplied specification as a self-reconstruction
training task. Every SRNet source image `i_s` is masked with `mask_s`; its own
text from `i_s.txt` is rendered as the glyph prompt and its masked-region crop
is the style prompt. The reconstruction target is the unchanged source image.
At inference, the rendered glyph and T5 target string change. Style guidance
comes only from the visual style crop. CLIP-L and OpenCLIP receive empty prompts
to retain SD3.5's native sequence layout and pooled conditioning without
encoding target content.

The frozen SD3.5 VAE separately encodes the masked image, glyph, and style
images. These three 16-channel latents are concatenated with the noisy
16-channel target and one mask channel. The resulting 65-channel tensor enters
the MM-DiT. The base transformer, VAE, CLIP-L, OpenCLIP bigG, and T5 remain
frozen. The expanded input convolution is trained in full, while PEFT LoRA
adapters train only the MM-DiT joint-attention projections. Target text is
encoded only by T5. The two CLIP towers encode content-free empty prompts and
provide the neutral CLIP token block and pooled tensor required by SD3.5.

Checkpoints trained with nonempty CLIP text require retraining for this contract.

Install and train from the repository root:

```bash
python -m pip install -r CODEX/self_prompting_sd35/requirements.txt
accelerate launch -m CODEX.self_prompting_sd35.train
```

Before the first run, accept the gated SD3.5 Medium license on Hugging Face and
authenticate with `hf auth login`.

The checked-in configuration disables NCCL peer-to-peer transport before
Accelerate initializes distributed training because P2P fails on mmplab2's RTX
A4500 pair. Set `distributed.nccl_p2p_disable: false` on a host with working
GPU peer-to-peer access.

Periodic `checkpoint-*` directories contain the attention LoRA, the full
65-channel input projection, and Accelerate state (optimizer, scaler when
applicable, and RNG state). Set the checkpoint root and resume policy directly
in `config.yaml`:

```yaml
training:
  output_dir: CODEX/self_prompting_sd35/checkpoints
  resume_from_checkpoint: latest  # null, latest, or a checkpoint-<step> directory
```

Hydra values can also be overridden without editing the file:

```bash
accelerate launch -m CODEX.self_prompting_sd35.train \
  training.output_dir=/path/to/training-output \
  training.resume_from_checkpoint=latest
```

Full-finetuning checkpoints are not compatible with this LoRA optimizer state.
Resume with the same base model, LoRA rank, and LoRA target modules used to
create the checkpoint.

Edit a marked region after training:

```bash
python -m CODEX.self_prompting_sd35.inference \
  --image source.png --mask mask.png \
  --text "replacement" \
  --lora CODEX/self_prompting_sd35/checkpoints/lora-final \
  --output edited.png
```

If training changed the base model or resolution, pass the matching `--model`
and `--resolution` values during inference.

Dataset paths are repository-relative; launch from the repository root.
