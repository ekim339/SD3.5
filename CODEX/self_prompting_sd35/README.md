# Self-prompting SD3.5

This directory supports two training modes selected by the top-level `mode`
configuration:

- `self_reconstruction` masks each SRNet `i_s` image and reconstructs that
  same image using its text from `i_s.txt`. It does not construct or encode a
  visual style crop; the 16 style-conditioning channels are exact zeros.
- `cooldown` uses each native same-filename SRNet edit pair: `i_s` is the
  source and `t_f` is the ground-truth edited output. The generator renders
  both with the same font, geometric transform, color, effects, and background
  while `i_s.txt` and `i_t.txt` provide different source and target strings.
  Only this stage constructs and VAE-encodes the original source-text crop as
  the visual style prompt.

In both modes, `mask_s` constructs the masked source. The target string is
rendered as the glyph prompt and encoded by T5.
Cooldown filters the two equal-text rows in the 200k dataset, leaving 199,998
different-glyph pairs. Its target-only `mask_t` is never given to MMDiT; the
union of `mask_s` and `mask_t` is used only to weight the training loss.
At inference, style guidance comes only from the visual style crop. CLIP-L and
OpenCLIP receive empty prompts to retain SD3.5's native sequence layout and
pooled conditioning without encoding target content.

The frozen SD3.5 VAE separately encodes the masked image and glyph. It also
encodes the style image during cooldown; self-reconstruction substitutes an
exact zero latent for that condition. These 16-channel blocks are concatenated
with the noisy 16-channel target and one mask channel in both stages, so the
MM-DiT input remains 65 channels and checkpoints are shape-compatible. The base
transformer, VAE, CLIP-L, OpenCLIP bigG, and T5 remain frozen. The expanded
input convolution is trained in full, while PEFT LoRA adapters train only the
MM-DiT joint-attention projections. Target text is encoded only by T5. The two
CLIP towers encode content-free empty prompts and provide the neutral CLIP
token block and pooled tensor required by SD3.5.

Self-reconstruction checkpoints trained with the old readable style crop, or
with nonempty CLIP text, require retraining for this contract.

Install dependencies from the repository root:

```bash
python -m pip install -r CODEX/self_prompting_sd35/requirements.txt
```

Run the stages separately and keep their checkpoint directories separate.
First train style-free self-reconstruction for 50,000 optimizer steps:

```bash
accelerate launch -m CODEX.self_prompting_sd35.train \
  mode=self_reconstruction \
  training.output_dir=CODEX/self_prompting_sd35/checkpoints/self_reconstruction \
  training.resume_from_checkpoint=null \
  training.max_steps=50000
```

Then initialize cooldown from that completed training-state checkpoint:

```bash
accelerate launch -m CODEX.self_prompting_sd35.train \
  mode=cooldown \
  training.output_dir=CODEX/self_prompting_sd35/checkpoints/cooldown \
  training.resume_from_checkpoint=CODEX/self_prompting_sd35/checkpoints/self_reconstruction/checkpoint-050000 \
  training.max_steps=70000
```

`max_steps` is an absolute global-step limit, so this example performs 20,000
cooldown steps after resuming step 50,000. To resume an interrupted cooldown
run, keep `mode=cooldown` and its output directory but use
`training.resume_from_checkpoint=latest`. `latest` searches only the current
stage output directory; the first cooldown run must name the self-reconstruction
checkpoint explicitly.

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
create the checkpoint. For a cooldown continuation, point
`training.resume_from_checkpoint` at the self-reconstruction
`checkpoint-<step>` directory and set `training.max_steps` above that
checkpoint's step; `max_steps` is an absolute global-step limit.

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
