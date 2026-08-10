"""Typed bridge into TextCtrl's released inference module.

The upstream CLI omits ``type=int`` for ``--num_inference_steps``. Calling its
parser directly with an override therefore passes a string to Diffusers. This
worker runs inside the pinned TextCtrl environment and constructs the namespace
with correctly typed values.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--starting-layer", type=int, required=True)
    parser.add_argument("--num-inference-steps", type=int, required=True)
    parser.add_argument("--guidance-scale", type=float, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    repository = Path(args.repository).expanduser().resolve()
    if not (repository / "inference.py").is_file():
        raise FileNotFoundError(
            f"TextCtrl inference.py does not exist under {repository}"
        )

    sys.path.insert(0, str(repository))
    os.chdir(repository)
    from inference import main as run_upstream_inference

    options = argparse.Namespace(
        seed=args.seed,
        target_height=256,
        teaget_width=256,
        style_height=256,
        style_width=256,
        ckpt_path=args.checkpoint,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        starting_layer=args.starting_layer,
        num_inference_steps=args.num_inference_steps,
        num_sample_per_image=1,
        guidance_scale=args.guidance_scale,
        benchmark=True,
    )
    run_upstream_inference(options)


if __name__ == "__main__":
    main()
