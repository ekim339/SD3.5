Implement the Self Prompting scene text editing with SD3.5 based on the paper Self Prompting DiT https://arxiv.org/pdf/2605.15523

Use 200k of the SRNet_Datagen dataset and fine-tune the full SD3.5 backbone while keeping the VAE frozen.

Dataset directory: /home/ekim339/project/SD3.5/datasets/SRNet_Datagen

### Masked Image Construction

Given a source text crop $I \in \mathbb{R}^{H \times W \times 3}$ and its binary mask $M \in \{0,1\}^{H \times W}$ indicating the text region, construct a masked image:

$I_m = I \odot (1 - M)$

\* $\odot$ denotes element-wise multiplication.

The masked image $I_m$, visual glyph prompt, and visual style prompt are encoded separately using the frozen SD1.5 VAE. Their latents are concatenated with the noisy target latent and the resized binary mask as input to the adapted SD1.5 UNet.

### Text Prompt Encoding
Use the SD1.5 CLIP text encoder to encode the target replacement string. Inject the resulting embeddings into the UNet through its existing cross-attention layers. Style guidance is provided by the visual style prompt.

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
  - Computes the smallest rectangle enclosing the masked text region
  - Then crop this region from the unmasked original image
  - The cropped style prompt $I_s$ must either be padded/resized back to full image dimensions $(H, W)$ before VAE encoding, 
3. Visual glyph prompt
  - Render target string using Pillow into a white-on-black glyph image <br/>
  $I_g = R(y_{\text{tgt}})$
4. Composite visual input
  - Encode the masked image, glyph prompt, and style prompt separately with the frozen VAE <br/>
  $z_m=E_{\text{VAE}}(I_m),\quad z_g=E_{\text{VAE}}(I_g),\quad z_s=E_{\text{VAE}}(I_s)$
5. VAE encoding
  - Concatenate the noisy target latent, visual prompt latents, and latent-resolution mask <br/>
  $x_t=\text{Concat}_{\text{channel}}(z_t,z_m,z_g,z_s,m)$
  - The VAE remains frozen
6. Textual glyph condition
  - Target text string is encoded with frozen T5 and textual visual/style prompt is encoded using a frozen CLIP ViT-L & OpenCLIP BigG.
  - The resulting text token embeddings from CLIP and T5 are concatenated together to form a single continuous sequence of text tokens.
7. Construct noisy target latent
  - $z_t$ is produced using the existing SD3.5 forward diffusion process
8. DiT Forward Pass:
  - Feed expanded $x_t$ along with text embeddings $(c_{\text{seq}}, c_{\text{pooled}})$ into the adapted SD3.5 MM-DiT block. Fine-tune the backbone using full-parameter training.