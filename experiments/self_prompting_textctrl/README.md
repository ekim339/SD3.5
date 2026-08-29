# Self-Prompting TextCtrl comparison

This experiment implements `self_prompting_textctrl.md` and keeps its code and
outputs within this directory. It deterministically selects 100 five-character
SRNet source crops, adds paired pixel noise, generates the capitalization and
six letter/punctuation target families with regular and self-prompting TextCtrl,
runs TextCtrl's ABINet OCR, and writes the requested reports and collages.

The full run produces 1,600 images and is resumable. Existing generated images
and OCR rows are reused unless `overwrite=true`.
Regular TextCtrl and OCR run in the legacy `textctrl` environment; the
self-prompting checkpoint runs in the modern `ste` environment.

```bash
cd /home/ekim339/projects/SD3.5
/home/ekim339/miniconda3/envs/ste/bin/python \
  -m experiments.self_prompting_textctrl.run
```

Run or resume individual stages with `stage=prepare`, `stage=regular`,
`stage=self_prompting`, `stage=ocr`, or `stage=report`. For example:

```bash
/home/ekim339/miniconda3/envs/ste/bin/python \
  -m experiments.self_prompting_textctrl.run stage=prepare sample_count=2 \
  output_dir=experiments/self_prompting_textctrl/results_smoke
```

## Evaluate new self-prompting weights on the same samples

Edit `config.yaml` (or provide the same values as command-line overrides):

```yaml
mode: self_prompting
samples_path: experiments/self_prompting_textctrl/results/samples.jsonl
output_dir: experiments/self_prompting_textctrl/results/new_self_prompting
self_prompting:
  checkpoint_path: checkpoints/my-new-self-prompting-checkpoint
```

Then run the normal command. This loads the prior manifest verbatim, including
the same noisy input paths, target strings, noise seeds, and generation seeds.
It creates 800 self-prompting jobs and does not run regular TextCtrl. Set
`regular.checkpoint_path` and `self_prompting.checkpoint_path` independently for
`mode: both` runs.

Outputs include `detailed_results.csv`, `capital_lowercase_summary.csv` (two
tables), `special_character_summary.csv` (18 metric columns),
`capital_lowercase_collage.png` (5x5), and `special_character_collage.png`
(2x7).

The released ABINet checkpoint uses a 36-character alphanumeric charset and
cannot emit punctuation or preserve letter case. Metrics default to
case-sensitive as requested, so this detector limitation is visible rather than
silently normalized. Set `metrics.case_sensitive=false` for the conventional
TextCtrl case-insensitive evaluation.
