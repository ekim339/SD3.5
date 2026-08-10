# [TextCtrl: Diffusion-based Scene Text Editing with Prior Guidance Control](https://arxiv.org/html/2410.10133v1)

GAN based methods divide task into text removal, new-text generation, image fusion. This simplifies training, but errors from any module can damage the final output. <br/>
→ bucket effect: overall performance is restricted by the weakest submodule.

Diffusion models generate better looking images but often have spelling errors and style deviation. TextCtrl therefore separates which text to generate from how those characters should look.

Components: SD1.5 VAE, SD1.5UNet, glyph structure encoder, text style encoder, a pretrained text-recognition vision encoder

Workflow
- Target string → char level text encoder → glyph feature structures: cglyph
- Source text crop → style encoder → style features: cstyle
Both conditions are fed into UNet
TextCtrl does not require rendering a standard-font glyph image every time at inference 

Instead, it trains a character-level text encoder so that the text embedding becomes aligned with visual character structure


## Glyph Guidance

The authors describe the desired representation as being related to the cluster centroid of visual features from different fonts. This lets the glyph feature answer What character sequence must be drawn? without forcing a particular font.
- Paper uses 730 fonts of text images to generate vOPEN(k)
- Glyph encoder with font variance performed better than wo font variance or CLIP


## Style Guidance

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

