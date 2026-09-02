Implement Self Prompting scene-text editing with SD3.5 based on the paper
Self Prompting DiT: https://arxiv.org/pdf/2605.15523

Use 200k examples from SRNet_Datagen. Train PEFT LoRA adapters on the SD3.5
MM-DiT while keeping the base transformer, VAE, and text encoders frozen. The
expanded 65-channel patch projection is itself a convolutional LoRA target so
all self-prompt inputs retain a trainable path into the frozen backbone.

Dataset directory: `/home/ekim339/project/SD3.5/datasets/SRNet_Datagen`

check this md file and generate code accordingly. keep all files under CODEX/self_prompting_sd35

First mask the source text region of the input image. Then pass in the rendered
target glyph as the visual glyph prompt and the cropped source text as the visual
style prompt. Both visual prompts are VAE-encoded; only the target string is
text-encoded, using T5. CLIP receives no target or style text.

### Masked image construction

Given a source text crop $I \in \mathbb{R}^{H \times W \times 3}$ and its
binary mask $M \in \{0,1\}^{H \times W}$ indicating the text region,
construct a masked image:

$I_m = I \odot (1 - M)$

The masked image $I_m$, visual glyph prompt, and visual style prompt are encoded
separately with the frozen SD3.5 VAE. Their latents are concatenated with the
noisy target latent and resized binary mask as input to the adapted MM-DiT.

### Text prompt encoding

Encode the target replacement string only with SD3.5's frozen T5 encoder. Send
batch-matched empty prompts to frozen CLIP-L and OpenCLIP bigG; these provide
SD3.5's neutral 77-token CLIP block and required pooled tensor without receiving
target content. Use SD3.5's native prompt assembly to concatenate the neutral
CLIP block with the T5 target sequence and inject the resulting sequence and
pooled tensors through MM-DiT's existing joint-attention layers. Style guidance
is supplied only by the visual style prompt.

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
- Also encode the target string only with the frozen T5 encoder.

### Summary

1. Mask the text region:
   $I_m = I \odot (1 - M)$
   - M is the mask

2. Crop the original text region using the maximal bounding rectangle of M. This is the visual prompt $I_s$. Resize/pad $I_s$ into the visual-prompt canvas.

3. Render the target glyph image:
   $I_g = R(y_{\text{tgt}})$

4. Encode three visual conditions using frozen SD3.5 VAE: masked image, glyph prompt, and style prompt

   $z_m=E_{\text{VAE}}(I_m),\quad z_g=E_{\text{VAE}}(I_g),\quad
   z_s=E_{\text{VAE}}(I_s)$

5. Encode the ground truth target image (this is the original source image because model will be trained to recover the original image) and construct the noisy flow matching stat using the exact SD3.5 scheduler convention in your training code.

$z_0 = E_{VAE}(I_{tgt})$

$z_t = (1 - \sigma_t)z_0 + \sigma_t\epsilon$
$\epsilon \sim \mathcal{N}(0,I)$

6. Resize the binary mask to latent resolution

$m = Resize(M, H_z, W_z)$

Then concatenate spatial conditions
$x_t=\text{Concat}_{\text{channel}}(z_t,z_m,z_g,z_s,m)$

7. Encode the target string only with T5 as glyph-text guidance. Give CLIP-L and
   OpenCLIP bigG empty prompts, then use SD3.5's native prompt assembly to create
   the sequence and pooled conditioning tensors. The target string must never be
   passed to either CLIP encoder.

8. Expand SD3.5's patch/input projection

$P_{old}: 16 \rightarrow d$

becomes

$P_{new}: 65 \rightarrow d$

Initialize: $Wnew​=[W_{old}​∣0_{zm​}​∣0_{zg}​​∣0_{zs}​​∣0_m​].$

Initially: $P_{new}(x_t) = P_{old}(z_t)$

and training gradually learns how to use the conditioning channels. The SD3.5 implementation indeed patchifies the latent through a learned convolutional projection

9. Feed $x_t, c_{seq}, c_{pooled}, t$ into MMDiT. Train expanded input projection full + LoRA on MMDiT attention while freezing VAE, T5, CLIP, and base MMDiT weights.

Do not change the output projection to 65 channels. the transformer should still predict only the target latent flow/velocity. The extra 49 channels are conditioning inputs only.

input channels = 65

prediction channels = 16