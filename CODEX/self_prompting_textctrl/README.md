# Self-Prompting TextCtrl (SD1.5)

This directory implements `self_prompting_textctrl.md` without modifying the original TextCtrl checkout.

The full SD1.5 UNet is trainable; the VAE and CLIP text encoder are frozen. Each sample supplies a noisy target latent (4 channels), masked-source latent (4), rendered-glyph latent (4), source-style latent (4), and latent edit mask (1). The resulting UNet input has 17 channels. Original four-channel weights are copied into the expanded input convolution, while new channels are zero-initialized.

## Data

The default configuration reads up to 200,000 complete samples from `datasets/SRNet_Datagen/train/train-50k-{1,2,3,4}`. Each split needs `i_s`, `t_f`, `mask_s`, `i_s.txt`, and `i_t.txt`. If generation is running, incomplete rows are ignored when the dataset is created.

## Environment

The existing `sd3` environment contains the runtime packages. Initial model construction downloads the SD1.5 tokenizer and CLIP text encoder named by `model.text_model_path`, unless it points to a local SD1.5 directory.

## Train

From this directory:

```bash
conda run -n sd3 accelerate launch train.py --config config.yaml
```

Checkpoints are written under `outputs/`. Resume with:

```bash
conda run -n sd3 accelerate launch train.py \
  --config config.yaml --resume outputs/checkpoint-00005000
```

## Inference

The mask uses white pixels for the editable region:

```bash
conda run -n sd3 python infer.py \
  --checkpoint outputs/checkpoint-00005000 \
  --source source.png --mask mask.png \
  --text "replacement" --output edited.png
```

The result preserves source pixels outside the mask.
