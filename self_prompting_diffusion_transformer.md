# Scene Text Editing (STE)

### Task: Edit text in an image while keeping all other things (background, font, lighting etc) the same. 

##  [Self-Prompting Diffusion Transformer for Open-Vocabulary Scene Text Editing via In-Context Learning (ICML 2026)](https://arxiv.org/pdf/2605.15523)

Problems of earlier approaches:
1. Masking discards the original text style
2. Pretraind OCR glyph encoders may limit vocabulary. <br/> Previous workflow: target string → renderer → OCR glyph encoder (feature extractor) → glyph features


What this paper proposes:
- Extracts style and glyph prompts directly from the source image instead of relying on separate external style and glyph encoder
- Sufficiently powerful diffusion transformers may be able to derive glyph and style conditions internally, reducing dependence on separately pretrained OCR or style encoders.
  - Feed the rendered glyph image directly into the multimodal diffusion transformer rather than first compressing it through an OCR encoder
- Uses a latent rectified-flow transformer, not a traditional Stable Diffusion UNet predicting ϵ

Previous works FluxText (Lan et al., 2025) and TextFlux (Xie et al., 2025) also remove glyph encoders and instead render glyph directly but spatial extent of the target region can hinder the accurate capture fine-grained glyph details. Loss of pre-existing style information still exists.

TextCtrl (Zeng et al., 2024) employs a text style encoder but the representation mismatch between the glyph-structure encoder and the VAE image latent space restricts the method’s scalability beyond isolated text regions.

## Main approach

1. The paper's scene text editing model separates text content and text appearance
- The target text tells the model what characters to generate <br/>
→ generate a high-fidelity glyph map by rendering the target text
- The original text region tells it how those characters should look <br/>
→ extract the original pixel information from
the target text region in the input image to form a style prompt

These two prompts are then concatenated with the full input image to form a multi-modal input tensor for the MMDiT backbone.

**Earlier glyph-encoder approaches** <br/>
target text → rendered glyph → OCR/glyph encoder → glyph feature embeddings

**This paper** <br/>
target text → rendered glyph map → directly supplied to MMDiT

2. Freeze the model’s encoder and decoder components and exclusively train the backbone using large-scale self-supervised image-text datasets. This step equips the model with fundamental text inpainting capabilities without overfitting to specific text styles.

3. Utilize a mask-free image editing tool to collect and filter high-quality paired image datasets, where each pair consists of an original image and its corresponding style-consistent edited version. These are used for the cooldown training, during which the model learns to align the generated text with the original style of the target region.

## Preliminaries

### Latent Rectified-flow Transformer
 - generates latent space
 - uses rectified flow as generative process
 - transformer predicts a flow direction
 - the paper uses [Flux-Fill](https://github.com/black-forest-labs/flux), an inpainting variant of Multi Modal Diffusion Transformer (MMDiT)

Network that predicts velocity $v_0$ for rectified flow is implemented as a transformer.

The latent tensor $z_t$ is divided into spatial patches:

$z_t \to \{x_t^{(1)}, x_t^{(2)}, \dots, x_t^{(N)}\}.$

These patches become Transformer tokens. The model then uses:
- self-attention between latent image tokens,
- time-step conditioning,
- cross-attention to text embeddings,
- feed-forward layers,
- sometimes adaptive normalization such as AdaLN

Its output has the same spatial shape as the latent and represents the predicted velocity field.

### Rectified Flow 

$z_t = (1 - t)z_0 + tz_1, \quad t \in [0, 1].$
- straight line interpolation between noise $z_0$ and data $z_1$

true velocity: $\frac{dz_t}{dt} = z_1 - z_0$

Neural net (transformer in this case) is trained to predict this velocity. $v_\theta(z_t, t, c) \approx z_1 - z_0$
\* c: text prompt / condition

Loss function: $\mathcal{L}_{\text{RF}} = \mathbb{E}_{z_0, z_1, t} \left[ \| v_\theta(z_t, t, c) - (z_1 - z_0) \|_2^2 \right].$

At inference, generation solves the ODE $\frac{dz_t}{dt} = v_\theta(z_t, t, c),$

VAE decoder decodes the final latent $z_1$ to image.

### Diffusion Transformer (DiT)
A Diffusion Transformer (DiT) is a diffusion model whose denoising network is a Transformer instead of a U-Net.

A Transformer expects a sequence of tokens, but the image latent is a spatial tensor $z_t \in \mathbb{R}^{H \times W \times C}.$ 

The latent is divided into patches and then each patch is flattened and projected into a vector. 

$p_i \in \mathbb{R}^{2 \cdot 2 \cdot 4} \longrightarrow x_i \in \mathbb{R}^D.$

This produces a sequence $X = [x_1, x_2, \dots, x_N]$

A positional embedding is added so the Transformer knows where every patch came from

$x_i \leftarrow x_i + p_i^{\text{pos}}$

The noisy latent image has now become a sequence of image tokens. The tokens pass through multiple Transformer blocks.

X → self attention → feed forward network <br/>
$X' = X + \text{Attention}(\text{Norm}(X)),$ <br/>
$X'' = X' + \text{MLP}(\text{Norm}(X')).$ 

For every token, the Transformer computes query, key, and value vectors:

$Q = XW_Q, \quad K = XW_K, \quad V = XW_V.$ 

$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V.$

**Timestep conditioning**: scalar timestep is converted into sinusoidal embedding. The timestep embedding can then modify each Transformer block through adaptive normalization therefore controls the behavior of the entire transformer.

**Text conditioning**: prompt is processsed by text encoder and injected as a text conditioning in several ways

1. Cross attention: image tokens are queries, while text tokens are keys and values. This allows an image patch to retrieve information from relevant prompt tokens.

2. Joint attention: Newer multimodal diffusion Transformers may place image and text tokens into a shared attention mechanis.

After the Transformer blocks, each token is projected back into a latent patch:

$x_i^{\text{out}} \rightarrow \hat{p}_i.$

The patches are rearranged into a spatial tensor:

$\{\hat{p}_i\}_{i=1}^N \rightarrow \hat{\epsilon} \in \mathbb{R}^{H \times W \times C}.$ <br/>
The output has the same spatial dimensions as the noisy latent.

At inference, the prediction is used by the sampler to update the latent $z_t \rightarrow z_{t-1}$

After multiple updates we obtain $z_0$ which is processed by VAE decoder to produce the final image.

<!-- **DiT vs UNet**

UNet
- Hierarchical spatial scales <br/>
$64 \times 64 \rightarrow 32 \times 32 \rightarrow 16 \times 16 \rightarrow 32 \times 32 \rightarrow 64 \times 64.$
- Skip connections across scales

DiT
- Sequence of latent patches <br/>
$N \text{ patch tokens} \rightarrow N \text{ patch tokens} \rightarrow N \text{ patch tokens}.$
- Residual connections across blocks -->

### Multi Modal Diffusion Transformer (MMDiT)

A Multi-Modal Diffusion Transformer (MMDiT) is a diffusion or flow-model backbone that processes image tokens and text tokens together inside Transformer attention layers. DiT primarily processes image tokens. MMDiT jointly processes image tokens and text tokens. It was designed so that image and language representations can interact more deeply than in a conventional U-Net or simple cross-attention architecture.

Image tokens: noisy latent is divided into patches.

$z_t \in \mathbb{R}^{H \times W \times C}$ <br/>
$z_t \rightarrow X_{\text{img}} = [x_1, x_2, \dots, x_N].$ <br/>
Each $x_i$ represents one spatial latent patch.

Text tokens: a text encoder converts the prompt into embeddings.

So MMDiT has two token streams 

$X_{\text{img}} \in \mathbb{R}^{N \times d}, \quad X_{\text{text}} \in \mathbb{R}^{M \times d}.$

**Multi-modal**

Two modalities are image / latent image tokens and text tokens. Instead of treating the text as a small side condition, MMDiT lets both modalities participate in the main Transformer computation. This allows image and text tokens to attend to each other.

$[\text{image tokens}, \text{text tokens}] \rightarrow \text{joint attention}.$

A central property of MMDiT is that image and text tokens can use different projection matrices since image patches and language tokens have very different statistical structures.

For image tokens: <br/>
$Q_I = X_I W_Q^I, \quad K_I = X_I W_K^I, \quad V_I = X_I W_V^I.$

For text tokens: <br/>
$Q_T = X_T W_Q^T, \quad K_T = X_T W_K^T, \quad V_T = X_T W_V^T.$

The projections are modality specific <br/>
$W_Q^I \neq W_Q^T, \quad W_K^I \neq W_K^T, \quad W_V^I \neq W_V^T.$

**Joint Attention**

After computing modality-specific queries, keys, and values, MMDiT combines them and computes one attention matrix.

$Q = \begin{bmatrix} Q_I \\ Q_T \end{bmatrix}, \quad K = \begin{bmatrix} K_I \\ K_T \end{bmatrix}, \quad V = \begin{bmatrix} V_I \\ V_T \end{bmatrix}.$

$A = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right).$

Because both modalities are included, the attention matrix contains four kinds of interactions: <br/>
$A = \begin{bmatrix} A_{I \rightarrow I} & A_{I \rightarrow T} \\ A_{T \rightarrow I} & A_{T \rightarrow T} \end{bmatrix}.$

Output: $Y=AV$

After joint attention, the resulting sequence is split back into image and text streams:
$Y \rightarrow (Y_I, Y_T).$ Each stream can then pass through its own output projection, normalization, and feed-forward network:

Therefore, the modalities
- keep separate parameters,
- interact in joint attention,
- return to separate processing streams

**Why not just concatenate the tokens and use one Transformer?**

Applying the same projection matrices after concatenating allows joint attention, but the same parameters process both modalities. MMDiT gives modality-specific representation learning. 

The final image tokens are projected back into latent patches and the patches are rearranged into a latent tensor.

$X_I^{\text{final}} \rightarrow \hat{p}_1, \dots, \hat{p}_N.$

$\hat{v}_t \in \mathbb{R}^{H \times W \times C}.$

Depending on the model, this output may be:
- predicted noise
- velocity
- rectified-flow vector field (our case; transformer predicts $v_\theta$)
- another parameterization of the denoising target

### Flux-Fill

FLUX.1 Fill is a version of FLUX specialized for inpainting and outpainting. It takes source image + binary mask + text prompt and generates new content inside the masked area while trying to preserve and continue the unmasked image.

Basic idea: <br/>
$(I, M, p) \rightarrow I_{\text{edited}}.$
- I: image
- M: mask (1: region to generate, 0: region to preserve)
- p: prompt

Flux-Fill generate the requested object inside the mask and make it geometrically and visually consistent with the visible context. The official implementation expects the mask to have the same size as the conditioning image and to contain black and white pixels.

**Inpainting** changes or reconstructs an area inside an existing image.

**Outpainting** expands an image beyond its original boundaries. The unknown surrounding regions are masked, and FLUX Fill generates a continuation consistent with the original image. [Flux-Fill by Black Forest Labs](https://github.com/black-forest-labs/flux/blob/main/docs/fill.md) supports both inpainting and outpainting.

**Main Components**
- Text encoders: uses CLIP and T5. T5 provides token-level semantic information, while the pooled CLIP representation provides a more global prompt condition.
- VAE: compresses image to latent space which the FLUX operates on.
- Rectified flow Transformer: The Transformer predicts the direction in which the current noisy latent should move $v_\theta(z_t, t, c, \text{image condition}, M).$
  - Then the sample updates $z_t \rightarrow z_{t-1}$

**FLUX Guidance**
- FLUX.1 [dev] was trained using guidance distillation. Its goal is to imitate a teacher model performing CFG, but using only one forward pass. The official model card identifies FLUX.1 [dev] as guidance-distilled, and the Transformer implementation explicitly supports a guidance embedding.
- Suppose a teacher produces <br/>
$v_{\text{teacher}}^{(s)} = v_{\text{uncond}} + s (v_{\text{cond}} - v_{\text{uncond}})$
- The distilled FLUX model is trained to approximate that guided output directly <br/>
$v_\theta(z_t, t, c, s) \approx v_{\text{teacher}}^{(s)}.$

How the guidance scale enters FLUX:
- Guidance scale $s \in \mathbb{R}$ and timestep $t$ are converted into an embedding <br/>
$s \rightarrow e_s$ <br/>
$t \rightarrow e_t$
- FLUX combines these with pooled text conditioning to form a global modulation vector <br/>
$h_{\text{global}} = f(e_t, e_s, c_{\text{CLIP}})$
- This global vector modulates Transformer blocks, for example through adaptive normalization or learned gates <br/>
$\text{AdaLN}(X; h_{\text{global}}) = \gamma(h_{\text{global}}) \odot \text{LN}(X) + \beta(h_{\text{global}})$

### OCR Encoder

OCR encoder is the feature extraction part of an Optical Character Recognition model. It takes a text image and converts the visible character strokes into a sequence of visual features.

#### OCR model vs OCR Encoder

OCR model = visual encoder + recognition decoder/classifier

The encoder processes the image: <br/>
$F = E_{OCR}(I)$

The recognition head then converts those features into character probabilities: <br/>
$p(y \mid I) = D_{\text{OCR}}(F).$

OCR encoder focuses on how the word visually appears

#### In older glyph control methods
target string → glyph renderer → glyph image → OCR encoder → glyph features.

- OCR systems are normally trained as classifiers over a fixed character vocabulary
- OCR encoder approach may generalize poorly on the characters that were rarely seen

## Method

### Masked Image Construction

Given an input image $I \in \mathbb{R}^{H \times W \times 3}$ and a binary mask $M \in \{0,1\}^{H \times W}$ indicating the text region, FLUX-Fill constructs a masked image:

$I_m = I \odot (1 - M)$

\* $\odot$ denotes element-wise multiplication.

Both the original image $I$ and the masked image $I_m$
are encoded into latent representations using a frozen VAE
encoder, producing visual tokens that serve as the input
to the diffusion transformer. The binary mask M is also
embedded and provided to the model as an explicit spatial
prior, enabling region-aware generation during denoising.

### Text Prompt Encoding
- CLIP: encode concise textual descriptors that emphasize visual alignment, such as object names or short phrases.
- T5: process the full natural-language prompt, capturing high-level semantic intent and contextual information.

The resulting embeddings are injected into the diffusion transformer
through cross-attention layers, guiding global content generation and scene-level consistency. 

**For this paper, T5 encodes target replacement text (glyph prompt) only and CLIP encodes image description only (style prompt)**

### Dual Stream and Single Stream Transformer

FLUX-Fill is a hybrid transformer design that alternates between dual-stream and single-stream blocks. FLUX-style double stream blocks commonly perform multimodal attention jointly while retaining separate image and text streams and separate modality specific projections. The single-stream blocks then concatenate the streams and process them with ordinary unified self-attention.

**Dual Stream Block**
Image ($X$) and text ($T$) tokens remain represented as separate tokens. X and T have separate projection matrices. 

$\begin{aligned}
Q_x &= X W_Q^x, & K_x &= X W_K^x, & V_x &= X W_V^x, \\
Q_t &= T W_Q^t, & K_t &= T W_K^t, & V_t &= T W_V^t.
\end{aligned}$

The projections are modality specific

$W_Q^x \neq W_Q^t, \quad W_K^x \neq W_K^t, \quad W_V^x \neq W_V^t.$

Project keys and values are concatenated.

$K = \begin{bmatrix} K_x \\ K_t \end{bmatrix}, \quad V = \begin{bmatrix} V_x \\ V_t \end{bmatrix}.$

Image queries attend to both image and text keys: <br/>
$Y_x = \text{softmax}\left(\frac{Q_x K^\top}{\sqrt{d_h}}\right) V.$

Image queries contain two types of attention scores; $Q_x K_x^\top$ (image to image interaction) and $Q_x K_t^\top$ (image to text interaction) <br/>
$Y_x = \text{softmax}\left(\frac{Q_x \begin{bmatrix} K_x^\top & K_t^\top \end{bmatrix}}{\sqrt{d_h}}\right) \begin{bmatrix} V_x \\ V_t \end{bmatrix}.$

Text queries also attend to both image and text keys: <br/>
$Y_t = \text{softmax}\left(\frac{Q_t K^\top}{\sqrt{d_h}}\right) V.$

Text queries contain two types of attention scores; $Q_t K_t^\top$ (text to text interaction) and $Q_t K_t^\top$ (text to image interaction)

Complete attention-score matrix:

$\begin{bmatrix} Q_x K_x^\top & Q_x K_t^\top \\ Q_t K_x^\top & Q_t K_t^\top \end{bmatrix}.$

**Single Stream Block**
After the dual-stream stage, the image and text tokens are concatenated and one shared set of projection matrices is applied.

$H = \begin{bmatrix} X \\ T \end{bmatrix} \in \mathbb{R}^{(N_x + N_t) \times d}.$

$Q = HW_Q, \quad K = HW_K, \quad V = HW_V.$

Unified self attention operation:

$Y = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}\right) V.$

Since H contains both modalities, the attention matrix has size $(N_x + N_t) \times (N_x + N_t).$

Matrix in blocks: $QK^\top = \begin{bmatrix} Q_x K_x^\top & Q_x K_t^\top \\ Q_t K_x^\top & Q_t K_t^\top \end{bmatrix}.$

### Rectified Flow Objective

$\mathcal{L}_{\text{RF}} = \mathbb{E}_{t, z_0, z_1} \left[ \|\hat{v}_\theta(z_t, t, c) - (z_1 - z_0)\|_2^2 \right],$

- $z_t$: noisy latent
- $z_0$: clean latent
- c: multimodal conditioning inputs (visual tokens, text embeddings, and spatial mask information)

### Style Prompt Construction
To preserve the visual appearance of the original text, we construct a style prompt from
both visual and textual perspectives
- Visual: 
  - Input image: $I \in \mathbb{R}^{H \times W \times 3}$
  - Binary mask indicating target text region: $M \in \{0,1\}^{H \times W}$
  - Compute the maximal enclosing bounding rectangle of the
masked area and crop the corresponding region from $I$ to obtain the visual style prompt $I_s$
  - This cropped patch encodes region-specific appearance information, such as color, texture, font characteristics, and local illumination, and serves as a visual reference for style preservation.
- Textual: encode the input text description using the **CLIP** text encoder

### Glyph Prompt Construction
represents the desired textual structure and the semantic meanings of the text prompt

- Target text is rendered into a single-line glyph image using the Pillow library, producing a whiteon-black glyph map $I_g$
- Encode the target text string using the **T5** text encoder
- Resulting embeddings capture high-level semantic and syntactic information of the target text and are injected into the MMDiT backbone through cross attention

### Denoising Process

$I_{\text{input}} = \text{Concat}(I_g, I_s, I_m)$
- $I_g$: visual glyph prompt
- $I_s$: visual style prompt
- $I_m$: masked image

These three are concatenated along channel axis to form composite visual input. The composite input is encoded by a frozen VAE encoder to obtain the latent representation $z_0$, which is processed by the MMDiT backbone together with the textual glyph and style embeddings. After denoising, final latent is decoded back to image by VAE decoder. Since the masked image is used only for conditioning, the final output is obtained by cropping the decoded result to the spatial region corresponding to the target text area. 

### Training

#### Dataset + Self Supervised Pretraining

**Workflow: official pretrained FLUX-Fill → self supervised pretraining on Any-Word 3M → cooldown training**

1. Cool down training dataset: Use instruction based image editing model (Nano Banana Pro) to generate edited images conditioned on explicit editing instructions + Manually filter correct pairs
  - Each data pair consists of an original image and a corresponding edited image (only target text is modified) <br/>
  **Annotators manually mark the edited text regions with bounding boxes, and those boxes are converted into training masks**
2. Pretrain the model on the AnyWord-3M dataset, which provides large-scale self-supervised data for multilingual scene text rendering. 
  - Optimization objective: <br/>
  $\mathcal{L}_{\text{RF}} = \mathbb{E}_{z_0, z_1, t} \left[ \| v_\theta(z_t, t, c) - (z_1 - z_0) \|_2^2 \right].$
  - AnyWord-3M contains 1.6M Chinese+1.39M English+10K multilingual images
  - train on this dataset for one epoch
  - freeze the encoder and decoder and train the diffusion Transformer backbone

#### Cooldown Training

1. Treat the original text region in the input image as metainformation during training. 
2. Continue training from pretrained checkpoint using 4000 manually filtered images.
3. Target-region-oriented training objective for the
cooldown stage to mitigate the degenerate optimization where original text region features unduly dominate the target region.
  - This objective restricts the learning signal
to localized text transformations

Interpolated latent variable: 
$z_t = (1 - \sigma_t) z_0^{\text{src}} + \sigma_t z_0^{\text{tgt}},$
- $z_0^{\text{src}}$: latent source image
- $z_0^{\text{tgt}}$: latent edited image
- $\sigma_t \in [0,1]$: time dependent interpolation constant

Model predicts the velocity field with objective

$\mathcal{L}_{\text{CD}} = \mathbb{E}_t \left[ \|\hat{v}_\theta(z_t, t, c) - (z_0^{\text{tgt}} - z_0^{\text{src}})\|_2^2 \right]$
- c includes the original text region as meta-information through the style prompt

### Summary

1. Mask the text region
  - given source image $I$, create masked image $I_m$ <br/> $I_m = I \odot (1 - M)$
  - binary mask is manually specified
  - model receives both masked image and binary mask
2. Visual style prompt
  - The paper computes the smallest rectangle enclosing the masked text region
  - It then crops this region from the unmasked original image
3. Visual glyph prompt
  - Render target string using Pillow into a white-on-black glyph image <br/>
  $I_g = R(y_{\text{tgt}})$
4. Composite visual input
  - Concatenates three visual inputs along channel axis <br/>
  $I_c = \text{Concat}_{\text{channel}}(I_{\text{mask}}, I_g, I_s)$
5. VAE encoding
  - Composite visual input is encoded by VAE encoder, creating a conditioning latents <br/>
  $z_c = E_{\text{VAE}}(I_c)$
  - VAE is frozen, so the authors do not train a new glyph encoder or style encoder. They rely on the pretrained image encoder to convert all three visual prompts into latent visual representations.
  - mask is also resized or embedded into a latent-compatible representation <br/>
  $m = E_M(M)$
6. Textual glyph condition
  - target text string is encoded with T5
7. Textual style condition
  - source image or source-text description is encoded with CLIP
8. Construct interpolated target latent
  - $z_t = (1 - t) z_0 + t z_1.$
    - $z_1$: VAE encoded clean target image
    - $z_0$: sampled gaussian noise
    - $t$: timestep (between 0-1)
9. Information enters MMDiT
  - All glyph and style conditions are jointly fed into MMDiT, and the model uses a single global FLUX guidance value of 30.
    - scalar guidance value is itself supplied to the Transformer as a conditioning signal
  - Ground truth velocity <br/>
  $u_t = \frac{d z_t}{d t} = z_1 - z_0$
  - MMDiT predicts <br/>
  $\hat{u}_t = v_\theta (z_t, t, z_c, m, C_g, c_s)$
    - $z_t$: current noisy latent being generated
    - $t$: rectified flow timestep
    - $z_c$: conditioning latent (latent encoding of visual prompts) <br/> $$z_c = E_{\text{VAE}}(\text{Concat}(I_{\text{mask}}, I_g, I_s))$$
      - $I_{\text{mask}}$: masked source image
      - $I_g$: rendered target glyph image
      - $I_s$: cropepd visual style prompt
    - m: embedded binary edit mask
    - $C_g$: T5 encoding of target replacement text
    - $c_s$: CLIP encoding of image description
  - Loss <br/>
  $\mathcal{L}_{\text{RF}} = \mathbb{E} \left[ \|v_\theta (z_t, t; z_c, m, C_g, c_s) - (z_1 - z_0)\|_2^2 \right]$

### Limitations

1. Text-length changes are poorly handled
  - few to many or many to few
  - The model receives a fixed target mask, but it does not explicitly optimize typography variables such as font size, character spacing, line breaking etc
  - A stronger system could first predict a target layout $L_{\text{target}} = f(\boldsymbol{y}_{\text{target}}, M, I_{\text{source}})$ and then render the glyph prompt according to that predicted layout
2. It does not properly support multiline text
  - The visual glyph prompt is rendered as a single line using Pillow
3. Performance degrades for long text
  - The appendix reports that quality starts degrading when text exceeds approximately 16 characters, producing stroke distortion and reduced structural consistency
4. The “style prompt” is not actually style-disentangled
  - The visual style prompt is simply a crop of the original text region which includes undesired content
  - The paper itself observes that adding style prompts during self-supervised pretraining can degrade results because the source and target text are identical. The model learns copying behavior rather than true style transfer.
  - How does the model extract source style without copying the original glyph?


##  [GlyphMastero: A Glyph Encoder for High-Fidelity Scene Text Editing (CVPR 2025)](https://arxiv.org/pdf/2505.04915)

Problems of earlier approaches:
1. Earlier methods often pass OCR features directly into the diffusion model, but those features do not explicitly model the hierarchy strokes → characters → complete text line
- GAN based methods: encountered challenges when dealing with complex text structures and diverse style variations, frequently resulting in generated text that exhibited unrealistic characteristics and compromised visual
quality
2. Diffusion based methods: 


What this paper proposes: 
- processes the target text at both the individual-character level and the whole-line level, then fuses them before conditioning the diffusion UNet
- proposes GlyphMaestro, a specialized glyph encoder that gives the diffusion model stronger character-structure guidance. Built on a Stable Diffusion 2.1 inpainting UNet.

Workflow: <br/>
target string → glyph renderer → PaddleOCR-v4 → GlyphMastero encoder → cross-attention in SD2.1 UNet

## Preliminaries

### Diffusion Based Methods Conditioning Strategy

**Cross Attention Guidance**

**Latent Space Guidance**s

### OCR backbone vs neck features



## Method

### Masked Image Construction

$x_m = x \odot (1 - m)$
- m: binary mask indicating region to modify

Concatenation along channel axis; <br/>
$\hat{z}_t = [z_t; m; \mathcal{E}(x_m)]$
- $z_t$: latent representation of masked image at timestpe t
- $m$: mask
- $\mathcal{E}()$: image encoder function

### Dual Stream Glyoh Integration
- PaddleOCR-v4 as feature extractor for input text
- extract local level stream and global level stream and integrate these through cross-level and multi-scale fusion to derive fine-grained glyph guidance c
- target string is rendered twice

**Local Stream Level**

Given a target string y, render a series of single-character glyph images $x_l \in \mathbb{R}^{N \times H_l \times W_l},$
- $N$: # chars
- $H_l, W_l$: height and width of each chars

OCR model’s last-layer backbone output $l_b$, and the neck output $l_n$, form the local-level stream feature representations.

**Global Stream Level**

Input text y is rendered as a unified glyph image $x_g \in \mathbb{R}^{H_g \times W_g}.$ 

Neck output $g_n$ is extracted for subsequent prrocessing. 

Unlike the local stream, we integrate M hierarchical backbone features (m=5 for PaddleOCR-v4) for the global stream through a FPN which fuses higher resolution, fine-grained features in shallow layers with semantic-rich features at lower resolutions in deeper layers, yielding the enhanced backbone features $g_b$.

**Further Processing**

- Employ two **glyph attention modules** $T_n$ and $T_b$ to capture interactions between local and global features for both the backbone and neck features.
- The cross-level interaction-enhanced features $o_n, o_b \in \mathbb{R}^{N \times d_o}$ are then obtained through two glyph attention modules $T_n$ and $T_b$ <br/>
$o_n = T_n(l_n, g_n), \quad o_b = T_b(l_b, g_b)$ 
  - $d_o$: output dimension of glyph attention modules
- Finally, an aggregator A concatenates and projects the two features <br/>
  $c = A(o_b, o_n)$
  - $c \in \mathbb{R}^{N \times D}$: guides the UNet during both training and inference phases of scene text editing through cross-attention
  - $D=d_o$


### Glyph Attention Module

GOAL: to use cross-attention to capture the interaction between character-level local features and line-level global features to get a better representation of text glyphs

1. repeat global feature $g \in \mathbb{R}^{1 \times d_g}$ N times to obtain $g \in \mathbb{R}^{N \times d_g}$ to match local features $l \in \mathbb{R}^{N \times d_l}$ sequence length
2. Features $l$ and $\hat{g}$ are then projected to attention space dimension $\tilde{g}$ through learnable linear transformations $ψ_l$ and $ψ_g$
  - Projected local features: $l^p = \psi_l(l)$
  - Projected global features: $g^p = \psi_g(\hat{g})$
  - both in $\mathbb{R}^{N \times \tilde{d}}$
3. Add positional embeddings to the two streams of features with rotary positional embedding (RoPE) by <br/>
$\bar{l}^p, \bar{g}^p = \text{RoPE}(l^p, g^p)$ <br/>
4. To capture interactions between local and global representations, perform multi head cross attention where positionally encoded $\bar{l^p}$ serves as queries and $\bar{g^p}$ as keys and values followed by layer normalization (LN)
- Produces attention map $z \in \mathbb{R}^{N \times \tilde{d}}.$
5. Linear projection $ψ_o$ is applied to map $z$ from attention dimension $\tilde{d}$ to the output size $d_o$ with $o = \psi_o(z) \in \mathbb{R}^{N \times \tilde{d}}$

### Glyph condition enters diffusion UNet

- CFG guidance

### Style
- Glyph Maestro doesnt have a dedicated style encoder
- Style is inferred from inpainting context (unmasked texts, background)