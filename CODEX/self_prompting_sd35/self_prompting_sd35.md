Implement Self Prompting scene-text editing with SD3.5 based on the paper
Self Prompting DiT: https://arxiv.org/pdf/2605.15523

Use 200k examples from SRNet_Datagen. Train PEFT LoRA adapters on the SD3.5
MM-DiT while keeping the base transformer, VAE, and text encoders frozen. The
expanded 65-channel patch projection is itself a convolutional LoRA target so
all self-prompt inputs retain a trainable path into the frozen backbone.

Dataset directory: `/home/ekim339/project/SD3.5/datasets/SRNet_Datagen`

### Masked image construction

Given a source text crop $I \in \mathbb{R}^{H \times W \times 3}$ and its
binary mask $M \in \{0,1\}^{H \times W}$ indicating the text region,
construct a masked image:

$I_m = I \odot (1 - M)$

The masked image $I_m$, visual glyph prompt, and visual style prompt are encoded
separately with the frozen SD3.5 VAE. Their latents are concatenated with the
noisy target latent and resized binary mask as input to the adapted MM-DiT.

### Text prompt encoding

Encode the target replacement string with SD3.5's frozen CLIP-L, OpenCLIP bigG,
and T5 encoders. Inject the resulting sequence and pooled embeddings through
MM-DiT's existing joint-attention layers. Style guidance is supplied by the
visual style prompt.

### Style prompt construction

To preserve the visual appearance of the original text:

- Start with source image $I$ and binary text mask $M$.
- Compute the smallest rectangle enclosing the masked region.
- Crop that region from $I$ to obtain visual style prompt $I_s$.
- Pad the crop back to the model canvas before VAE encoding.

The crop carries local color, texture, font, and illumination information.

### Glyph prompt construction

- Render the target string as a centered, single-line white-on-black image
  $I_g$ with Pillow.
- Encode $I_g$ with the frozen SD3.5 VAE.
- Also encode the target string with the three frozen SD3.5 text encoders.

### Summary

1. Mask the text region:

   $I_m = I \odot (1 - M)$

2. Build the visual style crop $I_s$ and pad it to $(H, W)$.

3. Render the target glyph image:

   $I_g = R(y_{\text{tgt}})$

4. Encode the masked image, glyph prompt, and style prompt:

   $z_m=E_{\text{VAE}}(I_m),\quad z_g=E_{\text{VAE}}(I_g),\quad
   z_s=E_{\text{VAE}}(I_s)$

5. Concatenate the noisy target latent, prompt latents, and latent-resolution
   mask:

   $x_t=\text{Concat}_{\text{channel}}(z_t,z_m,z_g,z_s,m)$

6. Encode the replacement string with frozen CLIP-L, OpenCLIP bigG, and T5.

7. Produce $z_t$ with SD3.5's flow-matching noise schedule.

8. Feed expanded $x_t$ and text embeddings
   $(c_{\text{seq}}, c_{\text{pooled}})$ into MM-DiT. Optimize only LoRA
   tensors on the input projection and joint-attention projections; keep every
   pretrained base parameter frozen.
