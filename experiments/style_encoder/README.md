# TextCtrl style-token masking

This experiment selects 50 five-character SRNet-Datagen source images, adds
deterministic Gaussian pixel noise, and records both its standard deviation and
seed. It generates one deterministic random five-character target for each
source and evaluates two interventions on TextCtrl + SD1.5:

- random masking of 0%, 10%, 30%, 50%, and 70% of the 256 complete style tokens;
- no masking plus each of the sixteen 4x4 squares in the 16x16 style-token grid.

The style ViT output is validated as `[B, 256, 768]` and complete token vectors
are zeroed before TextCtrl constructs its style feature pyramid. Source inversion
uses intact features; only editing is masked. Identical generation seeds make
conditions paired. The specification's final token for square `[0,0]` is treated
as a typo: row-major indexing gives token `51`, not `21`.

Prepare noisy inputs and the 1,100-job manifest without using a GPU:

```bash
python -m experiments.style_encoder.run stage=prepare
```

Run TextCtrl generation and ABINet OCR in the pinned CUDA environment:

```bash
TEXTCTRL_PYTHON=/path/to/textctrl/python python -m experiments.style_encoder.run
```

Results are stored under `experiments/style_encoder/results`: all generated
images, `samples.jsonl` (including noise metadata), `jobs.jsonl`, OCR predictions,
`detailed_results.csv`, and `summary.csv`. The summary has five proportion rows,
then the patch baseline and sixteen spatial-mask rows, with mean/population-std
ACC, NED, CER, and all five position accuracies. Collages include one 5x7
proportion image and one patch grid for each of three samples. Runs resume by
default; use `overwrite=true` to regenerate existing outputs.
