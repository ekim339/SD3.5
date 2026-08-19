# [TextCtrl: Diffusion-based Scene Text Editing with Prior Guidance Control](https://arxiv.org/html/2410.10133v1)

GAN based methods divide task into text removal, new-text generation, image fusion. This simplifies training, but errors from any module can damage the final output. <br/>
→ bucket effect: overall performance is restricted by the weakest submodule.

Diffusion models generate better looking images but often have spelling errors and style deviation. TextCtrl therefore separates which text to generate from how those characters should look.

Components: SD1.5 VAE, SD1.5UNet, glyph structure encoder, text style encoder, a pretrained text-recognition vision encoder

Workflow
- Target string → char level text encoder → glyph feature structures: $c_{glyph}$
- Source text crop → style encoder → style features: $c_{style}$
Both conditions are fed into UNet
TextCtrl does not require rendering a standard-font glyph image every time at inference 

Instead, it trains a character-level text encoder so that the text embedding becomes aligned with visual character structure

## Glyph Guidance

The authors describe the desired representation as being related to the cluster centroid of visual features from different fonts. This lets the glyph feature answer What character sequence must be drawn? without forcing a particular font.
- Paper uses 730 fonts of text images to generate vOPEN(k)
- Glyph encoder with font variance performed better than wo font variance or CLIP

$g_{OPEN} \approx v_{OPEN}^{(1)} \approx v_{OPEN}^{(2)} \approx \dots$
- $g_{OPEN}$: produced by character level text encoder
- $v_{OPEN}^{(1)}$: produced by frozen scene-text-recognition vision encoder, font 1
- $v_{OPEN}^{(2)}$: produced by frozen scene-text-recognition vision encoder, font 2

### Glyph Encoder

TextCtrl trains the glyph encoder with a CLIP-style contrastive objective that aligns character-level text embeddings with visual text features extracted by a frozen OCR/scene-text recognizer.

For text 'OPEN, the training pipeline is roughly: "OPEN" → glyph encoder → $C_struct$

and separately: "OPEN" → render with random font → ViTSTR OCR encoder → $C_visual$

Then the two are projected into a common space and trained contrastively.

1. Generate synthetic text images
  - create text strings and renders them using fonts from a font directory
  - The glyph pretraining config uses strings up to length 24 and generates 100,000 samples
2. Encode the string with the glyph encoder
  - The input string is first converted character-by-character to integer IDs.
  - The code pads all strings to L=24
  - Those IDs are passed through a character embedding: $X_0 = E_{\mathrm{char}}(s) \in \mathbb{R}^{24 \times 768}.$
  - Then positional encoding is added: $X_1 = X_0 + P$
  - Then a 12-layer Transformer encoder processes the sequence: $C_{\mathrm{struct}} = \mathcal{T}(X_1)$
  - released config is $L=24, d=768$, 12 Transformation layers, 8 attention heads
3. Encode the rendered word with a frozen OCR vision encoder
$I_i \xrightarrow{\text{ViTSTR}} V_i.$
  - The rendered image is fed into a pretrained ViTSTR scene-text recognition model
  - Because it is a ViT with 224×224 input and 16×16 patches, the visual output contains 197 tokens: $V_i \in \mathbb{R}^{197 \times 768}$
    - 196 patch tokens+1 CLS token.
  - ViSTR is frozen; the OCR model acts as the teacher visual space.
4. Project both sides into a common 1024-D space
  - Glyph representation: $[B,24,768] \rightarrow [B,1024]$
  - Visual side: $[B,197,768] \rightarrow [B,1024]$
5. L2 normalize the representations
  - normalize both projected vectors <br/>
  $\hat{t}_i = \frac{t_i}{\|t_i\|_2}, \qquad \hat{v}_i = \frac{v_i}{\|v_i\|_2}.$
  - therefore their dot product is cosine similarity $\hat{v}_i^\top \hat{t}_j = \cos(v_i, t_j)$
6. Compute CLIP-style similarity across the batch
  - For a batch of B examples, they compute all pairwise similarities: $S_{ij} = \tau \hat{v}_i^\top \hat{t}_j$ <br/>
  where $\tau = \exp(\text{logit\_scale})$ is a learnable temperature parameter 
7. Symmetric contrastive loss
  - The correct text for image i is the text at the same batch index: $y_i=i$
  - They compute cross-entropy in both directions.
    - image to text: $\mathcal{L}_{v \rightarrow t} = \mathrm{CE}(S, y)$
    - text to image: $\mathcal{L}_{t \rightarrow v} = \mathrm{CE}(S^\top, y)$
    - then: $\mathcal{L}_{\mathrm{glyph}} = \frac{1}{2} \left( \mathcal{L}_{v \rightarrow t} + \mathcal{L}_{t \rightarrow v} \right)$
    
The frozen OCR model is the teacher:
The trainable components include the glyph encoder, character embeddings, projection heads, and the contrastive temperature.


**Glyph encoder output**:

$C_{\mathrm{struct}} \in \mathbb{R}^{L \times d}$
- L: character-sequence length; $L=24$
- d: embedding dimension; $d=768$

## Style Guidance

### Text style encoder: extracts style from source text image
- Original input image is 128\*128 with 16\*16 patches
  - Therefore 8*8=64 patches are produced
  - The $x_{\mathrm{feature}} \in \mathbb{R}^{B \times 64 \times 768}$ representation splits to $x{spatial}$ and $x_{glyph}$ through separate Transformer attention blocks
- The style encoder uses a ViT backbone and produces features that are projected into approximately two categories: glyph and spatial features
  - Glyph features are used for text color transfer and font transfer
  - Spatial features are used for text removal, text segmentation
- This is intended to make the encoder explicitly learn different aspects of text appearance rather than relying on an unconstrained latent feature.

- [Glyph/style branch] Text color transfer
  - The network receives a black-and-white text glyph and style information from the source image. - It must reconstruct the source color appearance: $x_{glyph}, c_t \rightarrow c_r$
  - Code uses a pretrained-style ResNet feature extractor plus AdaIN (Adaptive Instance Normalization module) to inject color/style statistics.
  - Training loss is MSE: <br/>
  $\mathcal{L}_{\mathrm{color}} = \|\hat{c}_r - c_r\|_2^2$
- [Glyph/style branch] Text font transfer
  - Dataset renders the target text with 1. fixed standard font ($f_t$) 2. target font ($f_r$)
  - The network receives text rendered in a template font and transforms its boundary into the source font: $x_{glyph}, f_t \rightarrow f_r$
  - This task forces the style representation to preserve font shape characteristics.
  - Font output supervised by Dice Loss
  - A lightweight encoder-decoder with pyramid pooling performs this boundary transformation.
- [Spatial branch] Text removal: The model must remove text from the source crop and restore the hidden background.
  - "Can $x_{spatial}$ tell us enough about the source image to remove the existing text?"
  - Spatial head predicts $\hat{I_{bg}}$; background image
  - Weighted L1 loss: <br/>
  $\mathcal{L}_{\mathrm{removal}} = 2 \left\| \hat{I}_{\mathrm{bg}} \odot M - I_{\mathrm{bg}} \odot M \right\|_1 + 0.5 \left\| \hat{I}_{\mathrm{bg}} \odot (1 - M) - I_{\mathrm{bg}} \odot (1 - M) \right\|_1$
- [Spatial branch] Text segmentation: The model predicts a binary text mask. This teaches explicit spatial separation between text and background.
  - "Where is the text"
  - Dice Loss: <br/>
  $\mathcal{L}_{\mathrm{seg}} = 1 - \frac{2 \sum \hat{M} M + \epsilon}{\sum \hat{M} + \sum M + \epsilon}$

During training of style encoder, there are two branches
$x_{spatial}, x_{glyph}$ with both having shape [B, 64, 768]
- The source image is resized into 128×128 and split into 16×16 patches

The style encoder is trained using a combination of losses:
- MSE for color transfer,
- MAE for font transfer,
- Dice-based losses for text removal and segmentation
Synthetic data provides ground truth for all four subtasks. 

**Glyph encoder output**: <br/>

The source image is $I_s \in \mathbb{R}^{B \times 3 \times 256 \times 256}$

There are 256/16=16 patches per spatial dimension therefore 16*16=256 tokens

After ViT, [B,3,256,256] → [B,768,16,16] → [B,256,768]

$H_{style} \in [B,256,768]$ is the raw style ViT feature sequence in the final TextCtrl architecture.

Then the code immediately reshapes to [B,256,768] → [B,768,16,16]

StylePyramidNet then applies adaptive pooling to create 32*32, 16*16, 8*8, 4*4 style/control feature maps, which are projected with zero-initialized convolutions and supplied to different U-Net stages.

## How style and glyph guidance enter UNet
- Glyph guidance through cross attention: glyph feature provides keys and values in cross-attention. The latent U-Net features query the glyph representation to determine which characters should appear.
- Style guidance through feature injection: injected into the U-Net middle block and UNet decoder skip connections

Overall training loss
- denoising loss
- construction loss: includes pixel-level MSE, perceptual loss using VGG-19 features, style loss using Gram matrices
- linguistic loss: A pretrained text recognizer reads the generated image. The OCR prediction is compared with the target character sequence to penalize spelling mistakes.

## SD3.5 Application

Concatenate $C=[T; G; S]$ along the channel axis and C enters MMDiT as a condition. Conditions are interpreted though joint attention.

### Flux-Text
1. Replace each editable text line in the caption with a placeholder token S∗
2. Encode the modified caption using T5.
3. Render the target word as a glyph image.
4. Pass the rendered glyph image through a pretrained OCR encoder.
5. Project the OCR feature to the T5 embedding dimension.
6. Replace the placeholder embedding with the projected OCR feature.

### Self Prompting DiT

- T5: target text → glyph-text condition
- CLIP: style description → style-text condition
- VAE: rendered glyph text → visual glyph prompt
- VAE: source text crop → visual style prompt

| Condition                    | FLUX-Text                             | Self-Prompting DiT                                              |
| ---------------------------- | ------------------------------------- | --------------------------------------------------------------- |
| Image/scene description      | T5 + native CLIP pathway              | T5 for full semantic prompt; CLIP for concise style description |
| Target text string           | Included in T5 prompt                 | Explicitly encoded by T5                                        |
| Target glyph shape           | Rendered image encoded by VAE         | Rendered image encoded by VAE                                   |
| Text location                | Position map / mask                   | Mask and composite visual arrangement                           |
| Original target-region style | Not explicitly retained after masking | Original text-region crop used as visual style prompt           |
| External OCR/glyph encoder   | No, in final model                    | No                                                              |

1. Joint Attention Token Conditioning
- Use TextCtrl glyph encoder and style encoder and project onto SD3.5 space
- Concatenate with text prompt
- Feed the combined sequence into SD3.5
- Finetune backbone with LoRA
- Dataset: datasets/SRNet_Datagen/train/train-50k-1
- Ckpt: checkpoints/sd35_scene_text_glyph_weighted/adapters-010000.pt


## Failures in TextCtrl

- Use SRNet test set for the evaluation

- Dig in to why it is failing
  - Architecture breakdown for TextCtrl
    - Abalation by removing individual components; which one has greatest effect on accuracy
    - stick with SD1.5+TextCtrl and change configs

  - Make another report

- Control glyph guidance vectors
  - way to destroy the vector: add Gussian noise of varying magnitude, masking (randomly; certain %)
  - How dependent is text guidance in glyph guidance?
  - 5*5*5 (5 guidance scale, 5 gaussian noise, 5 masking %)

- Create one single report : what is the most important

- 