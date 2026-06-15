#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ljz/llm-mas/CausalCommunicationRefiner"
PYTHON="/home/ljz/miniconda3/envs/qwen3_vllm_clean/bin/python"
VLLM="/home/ljz/miniconda3/envs/qwen3_vllm_clean/bin/vllm"
MODEL="/home/ljz/models/Qwen3-8B"
STAMP="${1:-$(date -u +%Y%m%d_%H%M%S)}"
LOG_DIR="$ROOT/logs/pareto_$STAMP"
OUTPUT_ROOT="$ROOT/results/gdesigner_explainer_pareto/$STAMP"
mkdir -p "$LOG_DIR" "$OUTPUT_ROOT"

declare -A GPU_BY_DATASET=(
  [humaneval]=2
  [gsm8k]=3
  [svamp]=4
  [mmlu]=5
  [multiarith]=6
  [aqua]=7
)

declare -A PORT_BY_GPU=(
  [2]=8032
  [3]=8033
  [4]=8034
  [5]=8035
  [6]=8036
  [7]=8037
)

declare -A MAX_TOKENS_BY_DATASET=(
  [humaneval]=4096
  [gsm8k]=1024
  [svamp]=1024
  [mmlu]=1024
  [multiarith]=1024
  [aqua]=1024
)

declare -A BATCH_BY_DATASET=(
  [humaneval]=2
  [gsm8k]=4
  [svamp]=4
  [mmlu]=4
  [multiarith]=4
  [aqua]=4
)

DATASETS=(multiarith aqua mmlu svamp gsm8k humaneval)
BUDGET_GRID="${CCR_BUDGET_GRID:-0.25:0.0,0.50:0.0,0.75:0.0,0.25:0.2,0.50:0.2,0.25:0.4}"
VLLM_SESSIONS=()

start_vllm() {
  local gpu="$1"
  local port="$2"
  local session="ccr_pareto_vllm_gpu${gpu}_${port}_${STAMP}"
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
    if grep -qiE "out of memory|traceback|error" "$log" 2>/dev/null; then
      return 1
    fi
    sleep 5
  done
  return 1
}

run_dataset_on_worker() {
  local dataset="$1"
  local gpu="$2"
  local port="$3"
  local log="$LOG_DIR/${dataset}_gpu${gpu}.log"
  local base_url="http://127.0.0.1:${port}/v1"
  local max_tokens="${MAX_TOKENS_BY_DATASET[$dataset]}"
  local batch="${BATCH_BY_DATASET[$dataset]}"
  echo "[worker gpu=$gpu] start dataset=$dataset" | tee -a "$LOG_DIR/worker_gpu${gpu}.log"
  cd "$ROOT"
  "$PYTHON" -u adapters/evaluate_gdesigner_explainer_pareto.py \
    --dataset "$dataset" \
    --output-root "$OUTPUT_ROOT" \
    --base-urls "$base_url" \
    --api-key EMPTY \
    --budget-grid "$BUDGET_GRID" \
    --intervention-types both \
    --batch-size "$batch" \
    --max-tokens "$max_tokens" \
    --explainer-label-source causal_local \
    --explainer-correctness-weight 0.8 \
    --explainer-entropy-weight 0.2 \
    --explainer-cost-penalty 0.0 \
    --seed 888 > "$log" 2>&1
  echo "[worker gpu=$gpu] done dataset=$dataset status=$?" | tee -a "$LOG_DIR/worker_gpu${gpu}.log"
}

aggregate_results() {
  "$PYTHON" - <<PY
import csv, json
from pathlib import Path
root = Path("$OUTPUT_ROOT")
rows = []
for summary_path in sorted(root.glob("*/*/summary.json")):
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    original = data["original"]
    original_tokens = original["usage"].get("total_tokens", 0)
    for posthoc in data.get("posthoc", []):
        posthoc_tokens = posthoc["usage"].get("total_tokens", 0)
        rows.append({
            "dataset": data["dataset"],
            "budget": posthoc["name"],
            "edge_pruning_rate": posthoc["edge_pruning_rate"],
            "node_pruning_rate": posthoc["node_pruning_rate"],
            "original_accuracy": round(original["accuracy"] * 100, 4),
            "posthoc_accuracy": round(posthoc["accuracy"] * 100, 4),
            "accuracy_delta": round((posthoc["accuracy"] - original["accuracy"]) * 100, 4),
            "original_total_tokens": original_tokens,
            "posthoc_total_tokens": posthoc_tokens,
            "token_delta_pct": round(100.0 * (posthoc_tokens - original_tokens) / original_tokens, 4) if original_tokens else 0.0,
            "mean_original_spatial": round(posthoc["edge_stats"]["mean_original_spatial"], 4),
            "mean_posthoc_spatial": round(posthoc["edge_stats"]["mean_posthoc_spatial"], 4),
            "mean_original_nodes": round(posthoc["edge_stats"]["mean_original_nodes"], 4),
            "mean_posthoc_nodes": round(posthoc["edge_stats"]["mean_posthoc_nodes"], 4),
            "summary": str(summary_path),
        })
out_json = root / "aggregate_pareto.json"
out_csv = root / "aggregate_pareto.csv"
out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
if rows:
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
print(f"[aggregate] {len(rows)} rows -> {out_csv}")
PY
}

cleanup_all_vllm() {
  for session in "${VLLM_SESSIONS[@]:-}"; do
    tmux kill-session -t "$session" 2>/dev/null || true
  done
}

trap cleanup_all_vllm EXIT

for dataset in "${DATASETS[@]}"; do
  gpu="${GPU_BY_DATASET[$dataset]}"
  port="${PORT_BY_GPU[$gpu]}"
  start_vllm "$gpu" "$port"
done

READY_WORKERS=()
for dataset in "${DATASETS[@]}"; do
  gpu="${GPU_BY_DATASET[$dataset]}"
  port="${PORT_BY_GPU[$gpu]}"
  log="$LOG_DIR/vllm_gpu${gpu}_${port}.log"
  if wait_port "$port" "$log"; then
    READY_WORKERS+=("${gpu}:${port}")
    echo "[ready] gpu=$gpu port=$port" | tee -a "$LOG_DIR/ready_workers.txt"
  else
    echo "[skip] vllm not ready for dataset=$dataset gpu=$gpu port=$port" | tee -a "$LOG_DIR/failed_vllm.txt"
    tmux kill-session -t "ccr_pareto_vllm_gpu${gpu}_${port}_${STAMP}" 2>/dev/null || true
  fi
done

if [ "${#READY_WORKERS[@]}" -eq 0 ]; then
  echo "[error] no vLLM workers are ready" | tee -a "$LOG_DIR/failed_vllm.txt"
  exit 1
fi

QUEUE_FILE="$LOG_DIR/dataset_queue.txt"
LOCK_DIR="$LOG_DIR/queue.lock"
printf "%s\n" "${DATASETS[@]}" > "$QUEUE_FILE"

worker_loop() {
  local worker="$1"
  local gpu="${worker%%:*}"
  local port="${worker##*:}"
  while true; do
    local dataset=""
    while ! mkdir "$LOCK_DIR" 2>/dev/null; do
      sleep 0.2
    done
    if [ -s "$QUEUE_FILE" ]; then
      dataset="$(head -n 1 "$QUEUE_FILE")"
      tail -n +2 "$QUEUE_FILE" > "$QUEUE_FILE.tmp"
      mv "$QUEUE_FILE.tmp" "$QUEUE_FILE"
    fi
    rmdir "$LOCK_DIR"
    if [ -z "$dataset" ]; then
      break
    fi
    if ! run_dataset_on_worker "$dataset" "$gpu" "$port"; then
      echo "[failed] dataset=$dataset gpu=$gpu" | tee -a "$LOG_DIR/failed_jobs.txt"
    fi
  done
  tmux kill-session -t "ccr_pareto_vllm_gpu${gpu}_${port}_${STAMP}" 2>/dev/null || true
}

PIDS=()
for worker in "${READY_WORKERS[@]}"; do
  worker_loop "$worker" &
  PIDS+=("$!")
done

for pid in "${PIDS[@]}"; do
  wait "$pid"
done

aggregate_results
cleanup_all_vllm
