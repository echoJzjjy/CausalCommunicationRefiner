#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ljz/llm-mas/CausalCommunicationRefiner"
PYTHON="/home/ljz/miniconda3/envs/qwen3_vllm_clean/bin/python"

cd "$ROOT"

exec "$PYTHON" -u adapters/gdesigner_posthoc_runner.py \
  --config "$ROOT/configs/omni_topologies.json" \
  --output-root "$ROOT/results/gdesigner_posthoc" \
  --api-key EMPTY \
  --seed 888
