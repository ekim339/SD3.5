# Adapter Training

I want to replace the style encoder of TextCtrl with my custom pretrained residual extractor. This residual extractor is trained to extract the residual between original source image and rendered glyph image in standard style with the same source text. 

Output dimension of the residual extractor is [B,256,16,16] however TextCtrl style encoder has the output dimension of [B,768,16,16].

TextCtrl's pretrained style pyramid expects 768 input chennls so I want to train a channel adapter that projects residal extractor output [B,256,16,16] to [B,768,16,16].

Residual extractor directory: /home/ekim339/projects/SD3.5/encoders/checkpoints/residual_style/residual-style-100000.pt

Your task is creating a code to train this channel adapter. Keep all files under 'SD3.5/adapter/' and store the adapter checkpoint under this directory as well.