Create a code that finetunes the entire sd1.5 backbone along with residual adapter, style pyramid and rendered glyph image adapter. Continue from the pretrained TextCtrl weights.

Pretrained TextCtrl weights: /home/ekim339/projects/SD3.5/networks/external/TextCtrl/weights/model.pth

In a TextCtrl pipeline, I want to replace TextCtrl style encoder with pretrained residual extractor.

Use dataset: /home/ekim339/projects/SD3.5/datasets/SRNet_Datagen

Take a pair of images with same style and different texts from the SRNet dataset. Use one as source image and one as ground truth target image. You will perform text editing from the source text to the target text.

When using this residual extractor, 
1. Render the source text using Pillow
   - When rendering the source text, use the exact same way that was used in training residual extractor: check code under /home/ekim339/projects/SD3.5/encoders
2. Extract the residual between the source image and Pillow rendered source glyph image from step 1.
  - Feed the source image $I_s$ and the Pillow-rendered source glyph image $G_s$ into the pretrained residual extractor.
3. Train a residual adapter and style pyramid so we can feed this residual into SD1.5
  - The residual adapter consumes $R_s$. Residual adapter converts the residual of dimension [B,256,16,16] to [B,768,16,16].
  - The style pyramid converts it into the multi-scale residual features required by the SD1.5 U-Net.
  - Train these jointly with SD1.5 finetuning.
  - Dimensions: <br/>
  $256 \xrightarrow{\text{residual adapter}} 768 \xrightarrow{\text{original TextCtrl style pyramid}} 768 \xrightarrow{\text{zero convs}} 320/640/1280/1280$
4. Input the Pillow rendered target glyph image to the model. The model will use residual and Pillow rendered target glyph image to learn to generate the target image.
  - When feeding in this Pillow rendered target glyph image, use CNN / glyph spatial adapter, produce multi-scale spatial features, and inject those as residuals into the corresponding SD1.5 U-Net blocks, similar to ControlNet.
  - Make the Pillow-rendering adapter's output layers zero-initialized, ControlNet-style such that pretrained TextCtrl behavior is preserved.

Since both the residual extractor and Pillow target are spatial, maintain two spatial control branches.

$F_{\text{style}} = E_R(I_s, G_s)$

$F_{\text{glyph-spatial}} = E_G(G_t)$

At each UNet block, $h_l' = h_l + \alpha_l F_{\text{style},l} + \beta_l F_{\text{glyph-spatial},l}$

Basically the residual serves as style information. Since the model receives Pillow rendered target glyph image + residual (style information of source image), it will generate the target image, which is of the same style but different glyph with the source image. Keep the glyph features as same as the TextCtrl. Use TextCtrl glyph encoder and use the same cross attention pathway.




