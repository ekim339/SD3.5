# Paired per-character glyph masking

This implementation selects 50 five-character SRNet-Datagen images, adds and
records small deterministic pixel noise, and creates one random five-character
target per image. Every image is evaluated under all 25 combinations of five
target-character positions and five masking proportions. Thus the position
comparisons—including the zero-mask baseline—use identical samples and seeds.

Prepare the 1,250-job manifest without inference:

```bash
python -m experiments.glyph_encoder_per_char.run stage=prepare
```

Run TextCtrl SD1.5 generation and ABINet OCR on CUDA:

```bash
TEXTCTRL_PYTHON=/path/to/textctrl/python python -m experiments.glyph_encoder_per_char.run
```

Outputs under `results/` include noisy inputs, all generated images,
`detailed_results.csv` (1,250 rows), `summary.csv` (25 rows, each aggregating
the same 50 samples), and three 5×5 collages. The summary reports ACC/NED/CER,
masked-character accuracy, and accuracy for each of the four remaining
characters. `masked_character_index` is zero-based. Runs resume by default;
set `overwrite=true` to regenerate everything.
