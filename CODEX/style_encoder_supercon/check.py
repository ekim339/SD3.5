import torch

ckpt = torch.load("/home/ekim339/projects/SD3.5/networks/TextCtrl/weights/style_encoder.pth", map_location="cpu")

if "state_dict" in ckpt:
    ckpt = ckpt["state_dict"]

for k in ckpt.keys():
    if "glyph_attn_block" in k:
        print(k)