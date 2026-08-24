#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONFAULTHANDLER=1

exec conda run -n sd3 accelerate launch \
  --num_processes 2 \
  --num_machines 1 \
  --mixed_precision fp16 \
  --dynamo_backend no \
  "${script_dir}/train.py" \
  --config "${script_dir}/config.yaml" \
  "$@"
