Implement Self Prompting TextCtrl based on the paper Self Prompting DiT https://arxiv.org/pdf/2605.15523

repository for TextCtrl: /home/ekim339/project/SD3.5/networks/TextCtrl

TextCtrl uses explicit glyph and style encoders. However, you will omit these pretrained glyph and style encoders, replicating the visual self-prompting concept of Self Prompting DiT.

Instead of using glyph and style encoders, render the target glyph with Pillow and use it as a visual glyph prompt. Use the original source text crop as a visual style prompt.

Retain the frozen SD1.5 VAE, UNet backbone, and diffusion objective used by TextCtrl. Adapt the UNet input projection to receive the additional VAE-encoded visual prompts and mask.

Use 200k of the SRNet_Datagen dataset and fine-tune the full SD1.5 UNet while keeping the VAE frozen.

Dataset directory: /home/ekim339/project/SD3.5/datasets/SRNet_Datagen

### Masked Image Construction

Given a source text crop $I \in \mathbb{R}^{H \times W \times 3}$ and its binary mask $M \in \{0,1\}^{H \times W}$ indicating the text region, construct a masked image:

$I_m = I \odot (1 - M)$

\* $\odot$ denotes element-wise multiplication.

The masked image $I_m$, visual glyph prompt, and visual style prompt are encoded separately using the frozen SD1.5 VAE. Their latents are concatenated with the noisy target latent and the resized binary mask as input to the adapted SD1.5 UNet.

### Text Prompt Encoding
Use the SD1.5 CLIP text encoder to encode the target replacement string. Inject the resulting embeddings into the UNet through its existing cross-attention layers. Style guidance is provided by the visual style prompt.

### SD1.5 Diffusion Objective

$\mathcal{L}_{\text{diff}} = \mathbb{E}_{z_0, \epsilon, t} \left[ \|\epsilon_\theta(z_t, t, c) - \epsilon\|_2^2 \right],$

- $z_t$: noisy latent
- $z_0$: clean latent
- $\epsilon$: sampled Gaussian noise
- c: visual prompt latents, CLIP text embeddings, and spatial mask information

### Style Prompt Construction
To preserve the visual appearance of the original text, construct a visual style prompt:
- Visual prompt:
  - Input image: $I \in \mathbb{R}^{H \times W \times 3}$
  - Binary mask indicating target text region: $M \in \{0,1\}^{H \times W}$
  - Compute the maximal enclosing bounding rectangle of the
masked area and crop the corresponding region from $I$ to obtain the visual style prompt $I_s$
  - This cropped patch encodes region-specific appearance information, such as color, texture, font characteristics, and local illumination, and serves as a visual reference for style preservation.

### Glyph Prompt Construction
represents the desired textual structure and the semantic meanings of the text prompt

- Target text is rendered into a single-line glyph image using the Pillow library, producing a white-on-black glyph map $I_g$
- Encode the target text string using the SD1.5 **CLIP** text encoder
- Inject the resulting embeddings into the UNet through cross-attention

### Summary

1. Mask the text region
  - given source image $I$, create masked image $I_m$ <br/> $I_m = I \odot (1 - M)$
  - binary mask is provided by SRNet_Datagen
  - model receives both masked image and binary mask
2. Visual style prompt
  - The paper computes the smallest rectangle enclosing the masked text region
  - It then crops this region from the unmasked original image
3. Visual glyph prompt
  - Render target string using Pillow into a white-on-black glyph image <br/>
  $I_g = R(y_{\text{tgt}})$
4. Composite visual input
  - Encode the masked image, glyph prompt, and style prompt separately with the frozen VAE <br/>
  $z_m=E_{\text{VAE}}(I_m),\quad z_g=E_{\text{VAE}}(I_g),\quad z_s=E_{\text{VAE}}(I_s)$
5. VAE encoding
  - Concatenate the noisy target latent, visual prompt latents, and latent-resolution mask <br/>
  $x_t=\text{Concat}_{\text{channel}}(z_t,z_m,z_g,z_s,m)$
  - The VAE remains frozen; adapt the UNet input projection for the expanded channel count
6. Textual glyph condition
  - Target text string is encoded with CLIP
7. Construct noisy target latent
  - $z_t$ is produced using the existing SD1.5 forward diffusion process
8. The expanded latent input and CLIP embeddings enter the adapted SD1.5 UNet