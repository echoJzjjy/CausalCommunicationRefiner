#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ljz/llm-mas/CausalCommunicationRefiner"
PYTHON="/home/ljz/miniconda3/envs/qwen3_vllm_clean/bin/python"
VLLM="/home/ljz/miniconda3/envs/qwen3_vllm_clean/bin/vllm"
MODEL="/home/ljz/models/Qwen3-8B"
STAMP="${1:-$(date -u +%Y%m%d_%H%M%S)}"
GPUS_CSV="${CCR_GPUS:-2,3,4,5}"
PORT_BASE="${CCR_PORT_BASE:-8040}"
BUDGET_GRID="${CCR_BUDGET_GRID:-0.25:0.0,0.50:0.0,0.75:0.0,0.25:0.2,0.50:0.2,0.25:0.4}"
LOG_DIR="$ROOT/logs/humaneval_shards_$STAMP"
OUTPUT_ROOT="$ROOT/results/gdesigner_explainer_pareto/humaneval_shards_$STAMP"
mkdir -p "$LOG_DIR" "$OUTPUT_ROOT"

IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
NUM_SHARDS="${#GPUS[@]}"
if [ "$NUM_SHARDS" -lt 1 ]; then
  echo "[error] no GPUs configured" >&2
  exit 1
fi

VLLM_SESSIONS=()

cleanup() {
  for session in "${VLLM_SESSIONS[@]:-}"; do
    tmux kill-session -t "$session" 2>/dev/null || true
  done
}
trap cleanup EXIT

start_vllm() {
  local gpu="$1"
  local port="$2"
  local session="ccr_humaneval_vllm_gpu${gpu}_${port}_${STAMP}"
  local log="$LOG_DIR/vllm_gpu${gpu}_${port}.log"
  tmux new-session -d -s "$session" \
    "CUDA_VISIBLE_DEVICES=$gpu $VLLM serve $MODEL --host 0.0.0.0 --port $port --dtype float16 --max-model-len 12288 --gpu-memory-utilization 0.8 --served-model-name qwen3-8b --max-num-seqs 1 > '$log' 2>&1"
  VLLM_SESSIONS+=("$session")
}

wait_port() {
  local port="$1"
  local log="$2"
  for _ in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    if grep -qiE "out of memory|traceback|error|insufficient" "$log" 2>/dev/null; then
      return 1
    fi
    sleep 5
  done
  return 1
}

PIDS=()
READY=()
for i in "${!GPUS[@]}"; do
  gpu="${GPUS[$i]}"
  port="$((PORT_BASE + i))"
  start_vllm "$gpu" "$port"
done

for i in "${!GPUS[@]}"; do
  gpu="${GPUS[$i]}"
  port="$((PORT_BASE + i))"
  log="$LOG_DIR/vllm_gpu${gpu}_${port}.log"
  if wait_port "$port" "$log"; then
    READY+=("${i}:${gpu}:${port}")
    echo "[ready] shard=$i gpu=$gpu port=$port" | tee -a "$LOG_DIR/ready_workers.txt"
  else
    echo "[skip] shard=$i gpu=$gpu port=$port vllm not ready" | tee -a "$LOG_DIR/failed_vllm.txt"
    tmux kill-session -t "ccr_humaneval_vllm_gpu${gpu}_${port}_${STAMP}" 2>/dev/null || true
  fi
done

if [ "${#READY[@]}" -eq 0 ]; then
  echo "[error] no ready workers" >&2
  exit 1
fi

for worker in "${READY[@]}"; do
  shard="${worker%%:*}"
  rest="${worker#*:}"
  gpu="${rest%%:*}"
  port="${rest##*:}"
  log="$LOG_DIR/humaneval_shard${shard}_gpu${gpu}.log"
  (
    cd "$ROOT"
    "$PYTHON" -u adapters/evaluate_gdesigner_explainer_pareto.py \
      --dataset humaneval \
      --output-root "$OUTPUT_ROOT" \
      --run-name "shard${shard}_of${NUM_SHARDS}" \
      --num-shards "$NUM_SHARDS" \
      --shard-index "$shard" \
      --base-urls "http://127.0.0.1:${port}/v1" \
      --api-key EMPTY \
      --budget-grid "$BUDGET_GRID" \
      --intervention-types both \
      --batch-size 2 \
      --max-tokens 4096 \
      --explainer-label-source causal_local \
      --explainer-correctness-weight 0.8 \
      --explainer-entropy-weight 0.2 \
      --explainer-cost-penalty 0.0 \
      --seed 888 > "$log" 2>&1
  ) &
  PIDS+=("$!")
done

FAILED=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || FAILED=1
done

"$PYTHON" - <<PY
import csv, json
from pathlib import Path
root = Path("$OUTPUT_ROOT")
rows = []
for summary_path in sorted(root.glob("humaneval/*/summary.json")):
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    original = data["original"]
    original_tokens = original["usage"].get("total_tokens", 0)
    for posthoc in data.get("posthoc", []):
        posthoc_tokens = posthoc["usage"].get("total_tokens", 0)
        rows.append({
            "dataset": data["dataset"],
            "shard_index": data.get("shard", {}).get("shard_index"),
            "num_shards": data.get("shard", {}).get("num_shards"),
            "budget": posthoc["name"],
            "original_correct": original["correct"],
            "original_total": original["total"],
            "posthoc_correct": posthoc["correct"],
            "posthoc_total": posthoc["total"],
            "original_total_tokens": original_tokens,
            "posthoc_total_tokens": posthoc_tokens,
            "summary": str(summary_path),
        })
out_json = root / "humaneval_shards_raw.json"
out_csv = root / "humaneval_shards_raw.csv"
out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
if rows:
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
print(f"[aggregate] {len(rows)} rows -> {out_csv}")
PY

cleanup
exit "$FAILED"
