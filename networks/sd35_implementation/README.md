# Implement Glyph Structure- and Style-Conditioned Scene Text Editing with SD3.5 MMDiT

Under this folder sd35_implementation, implement a research prototype that adapts Stable Diffusion 3.5 Medium for scene-text editing.

The model must edit the text in a source image while preserving the original background and text style. It should use:

- SD3.5 Medium as the pretrained MMDiT backbone.
- A frozen pretrained glyph structure vision encoder to extract target-glyph features.
- The pretrained TextCtrl style encoder to extract style features from the source image.
- SRNet-Datagen paired samples for supervised training.
- The source image and source mask as spatial conditioning.
- The final edited image t_f as the ground-truth training target.

Extracted features from pretrained glyph structure encoder and TextCtrl style encoder will be fed into SD3.5 as conditions.

Do not replace the pretrained SD3.5 text-conditioning pathway. Add Glyph structure and style conditioning through trainable adapters initialized so that the initial model remains close to the pretrained SD3.5 model.

Keep a hydra config file to control the configurations.

## Directories

Dataset: datasets/SRNet_Datagen/train/train-50k-1

TextCtrl Glyph structure encoder: networks/external/TextCtrl/weights/text_encoder.pth

TextCtrl Style encoder: networks/external/TextCtrl/weights/style_encoder.pth

check under /home/ekim339/projects/SD3.5/ to search for anything else. Do not go above this directory.

## Primary objective

For each training sample, use:

- source image: i_s
  - Use TextCtrl style encoder to extract information from this image
- source text label: i_s.txt
- target text label: i_t.txt
- target glyph rendering: i_t
  - Use glyph structure encoder to extract information from this image
- source text mask: mask_s
- target final image: t_f

Source, target, and mask images are resized with
`s = min(resolution / width, resolution / height)` and centered on a square
canvas. RGB padding is black and mask padding is zero. The style input uses the
same transform at its configured resolution.

## Conditions

We concatenate the native editing prompt, a T5-only target-text prompt, glyph
structure features, and style features as the SD3.5 textual condition.

Output of TextCtrl glyph structure encoder has dimension [B, 24, 768]

Output of TextCtrl style encoder has dimension [B, 256, 768]

These have to match the native SD3.5 prompt dimension [B, 333, 4096]

Use separate projectors for Glyph structure and style encoders to project the last dimension into 4096. Then concatenate along the token axis.

With the default padding lengths, the final condition is:
[B, 333 + 256 + 24 + 256, 4096] = [B, 869, 4096].

## Native Prompt Construction

There will be two text prompts which will be concatenated along the token axis.

First, construct a natural-language editing prompt that includes both the scene-editing instruction and target text.

For example:
Replace "OPEN" with "CLOSED" in the marked region while preserving the original font, color, geometry, lighting, and background.

Implement prompt templates in configuration:

prompt:
  template: >
    Replace "{source_text}" with "{target_text}" in the masked region while preserving the original font, color, geometry, lighting, texture, and background.

Encode this prompt through the standard SD3.5 text-encoding pipeline. Do not bypass or replace the native CLIP/T5 embeddings.

For the second prompt, encode the target text (i_t.txt) only with T5 and concatenate with the first prompt along token axis. 

Final condition: [B, 333 + 256 + 24 + 256, 4096]

## Source Image and Mask Conditioning

The glyph structure and style sequences do not replace spatial conditioning.

Use the source image i_s and source mask mask_s to construct a masked source image: $(1-M_s)\odot I_s+c_{fill}M_s.$

Encode the masked source image using the frozen SD3.5 VAE and resize the source mask to the latent resolution.

Use an SD3-compatible inpainting input with 33 channels:
- 16 noisy target-latent channels
- 16 masked-source-latent channels
- 1 source-mask channel

The pretrained 16-channel SD3 patch projection remains frozen. A parallel
zero-initialized projection learns the 17 new conditioning channels.

Do not concatenate the source image or mask with the text condition tokens.

The complete conditioning should therefore be:

Textual sequence conditions:
    native SD3.5 prompt embeddings
    + T5-only target-text embeddings
    + TextCtrl glyph structure tokens
    + TextCtrl global style tokens

Spatial conditions:
    masked source image latent
    + source mask

<!-- ## LoRA fine-tuning

The default `model.training_mode` is `lora`. The SD3.5 MMDiT base weights, VAE,
Glyph Structure encoder, and TextCtrl style encoder remain frozen. Rank-16 LoRA adapters train
the `to_q`, `to_k`, `to_v`, and `to_out.0` attention projections together with the
Glyph structure/style projectors and inpainting projection. Use `frozen` for adapter-only training or
`full` for full MMDiT fine-tuning. -->

Run with:

```bash
conda activate ste
python -m networks.sd35_implementation.train
```

## COCO-Text Evaluation

`evaluate_coco_text.py` applies the same 1,000-image protocol as the released
TextCtrl baseline. It selects one five-character annotation per COCO image,
adds deterministic Gaussian noise, assigns the same random five-character
target produced by seed 42, and rasterizes the selected annotation as the
SD3.5 inpainting mask. Generated crops are recognized with pretrained ABINet.

The configured checkpoint is
`outputs/sd35_scene_text_sd3_inpaint_dual_prompt_style_pad_200k/adapters-015000.pt`.
Run or resume the full evaluation with:

```bash
conda activate ste
CUDA_VISIBLE_DEVICES=0 \
  python -m networks.sd35_implementation.evaluate_coco_text
```

Hydra settings are in `evaluate_coco_text.yaml`. Existing generations and OCR
predictions are reused unless `overwrite=true` is supplied. Outputs include
source/noisy/generated crops, per-word masks, the exact evaluation manifest,
`results.csv`, and `summary.json`. The CSV reports `image_title`, target text as
`ground_truth_text`, `ocr_predicted_text`, case-insensitive exact `ACC`, NED
similarity, and CER.

## TextCtrl and SD3.5 Comparison Collage

`create_coco_text_collage.py` prepares ten shared noisy COCO-Text word crops,
runs TextCtrl SD1.5 and the configured SD3.5 adapter on identical targets, OCRs
both result sets, and renders a captioned 3-by-10 comparison image.

```bash
conda activate ste
CUDA_VISIBLE_DEVICES=1 \
  python -m networks.sd35_implementation.create_coco_text_collage
```

Settings are in `create_coco_text_collage.yaml`. Source crops, masks, both
generated image sets, manifests, OCR predictions, and the final
`comparison_collage.png` are retained under the configured output directory.

## Report

### Training Levels
1. Projection heads only
2. Projection heads + MMDiT LoRA
3. Partial or full MMDiT backbone

### Spatial Conditioning Pathways
1. SD3 ControlNet

- Advantages
  - Gives spatial information at multiple network depths -> useful for perserving text location, background, etc
  - keeps spatial conditioning separate from your token conditions
- Disadvantages
  - expensive because it duplicates part of the transformer computation

2. Zero-initialized source-image adapter
A source adapter is a smaller custom module that maps the source image and mask directly into the MMDiT image-token space.

Possible injection levels:
- Input-only adapter
- Multi-block adapter
- Cross-attention adapter

- Advantages
  - Easiest implementation, few additional params
- Disadvantages
  - Input-only injection may weaken as information passes through many transformer blocks.
  - It is also a custom architecture, so you must implement and test its token alignment carefully.

**3. SD3-compatible inpainting pathway**

Options:
- modify SD3.5’s latent input projection
- separate patch embeddings, then concatenate/add tokens
- ControlNet-based inpainting

- Advantages
  - Inpainting provides the strongest task-specific inductive bias
  - The model sees the preserved source context at every denoising step. This is ideal because scene-text editing is fundamentally localized reconstruction.
- Disadvantages
  - Requires careful & detailed implementation. You may need to:
    - change the patch embedding
    - initialize new input channels
    - Ensure source information does not leak the old word too strongly


| Pathway                         | Spatial condition enters through                              | Backbone modification |    Training cost | Background preservation | Best use                                         |
| ------------------------------- | ------------------------------------------------------------- | --------------------: | ---------------: | ----------------------: | ------------------------------------------------ |
| SD3 ControlNet                  | Residuals added across transformer blocks                     |       Low to moderate |             High |                  Strong | Robust, established spatial conditioning         |
| Zero-initialized source adapter | Custom residual added to image tokens or selected blocks      |              Moderate |  Low to moderate |      Moderate to strong | Fast research prototype                          |
| SD3-compatible inpainting       | Masked source latent and mask integrated into denoising input |      Moderate to high | Moderate to high |   Potentially strongest | Scene-text replacement as true localized editing |


### Conditioning Routes for MMDiT

1. Joint attention token conditioning
2. Input-token or latent concatenation
  - closest to inpainting input pathway
3. Transformer-block residual injection
4. SD3 ControlNet-style residual conditioning
5. Adaptive LayerNorm modulation
6. Attention K,V injection or cross-branch attention


FLUX-Text explicitly studies multiple ways to inject glyph information into FLUX-Fill:
- convolutional glyph/position feature extraction → 2. Input-token or latent concatenation
- Canny/VAE glyph features → 2. Input-token or latent concatenation
- direct VAE glyph latent injection → 2. Input-token or latent concatenation
- OCR-derived text-token injection → 1. Joint attention token conditioning
- Glyph-ByT5 token injection → 1. Joint attention token conditioning

FLUX-Text explored learned convolutional preprocessing, but did not choose per-block residual injection. They found that direct VAE glyph embedding converged better and required fewer parameters.

### 


tidy the repository

weights and biases

100 evaluation with noise

code (*config), experiments, insights

config: model, dataset, detectors