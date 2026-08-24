# Residual-to-TextCtrl channel adapter

This trains a pointwise `Conv2d(256, 768, kernel_size=1)` adapter that maps the
frozen residual extractor output `[B,256,16,16]` to TextCtrl's style-grid shape
`[B,768,16,16]`.

Training uses frozen-feature distillation. For each SRNet image `I`, its matching
canonical glyph `G` is rendered and the input feature is:

```text
R(E(I), E(G)) -> [B,256,16,16]
```

The frozen TextCtrl style ViT supplies the target:

```text
TextCtrlStyleEncoder(I) -> [B,256,768] -> [B,768,16,16]
```

Only the 1x1 adapter is optimized, using MSE plus flattened cosine distance. The
residual extractor and TextCtrl style encoder remain frozen. Both members of each
same-style SRNet pair are used as separate training examples.

Run on GPU from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 /home/ekim339/miniconda3/envs/ste/bin/python \
  -m adapter.train
```

Checkpoints and the resolved configuration are stored under
`adapter/checkpoints/`. To resume:

```bash
CUDA_VISIBLE_DEVICES=0 /home/ekim339/miniconda3/envs/ste/bin/python \
  -m adapter.train \
  training.resume_from=adapter/checkpoints/channel-adapter-latest.pt
```

Load the trained projection with:

```python
from adapter import load_adapter

adapter = load_adapter("adapter/checkpoints/channel-adapter-latest.pt", "cuda")
textctrl_grid = adapter(residual_features)  # [B,768,16,16]
```

This checkpoint supplies the channel-compatible grid; replacing TextCtrl's ViT
at inference still requires routing source image plus canonical source glyph
through the residual extractor and feeding the adapted grid into StylePyramidNet.
