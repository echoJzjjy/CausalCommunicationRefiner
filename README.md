# Causal Communication Refiner

Causal Communication Refiner (CCR) is a post-hoc compression module for LLM-based multi-agent systems. It takes a communication graph produced by an existing topology generator, scores the causal utility of its nodes and edges, and removes low-utility communication while trying to preserve task accuracy.

```text
Task + Graph Generator -> Candidate MAS Graph -> CCR Explainer -> Refined MAS Graph
```

The current implementation focuses on GDesigner-style task-conditioned graphs, but the method is intended to be generator-agnostic: the generator can be GDesigner, AgentPrune, AgentDropout, or a hand-crafted topology as long as its graph can be serialized into nodes, spatial edges, temporal edges, and execution traces.

## Method

CCR separates graph generation from graph refinement.

1. A backbone generator first produces an initial communication graph `G(x)` for a task `x`.
2. CCR builds counterfactual supervision by deleting one graph component at a time and measuring the change in the downstream behavior.
3. A lightweight explainer network is trained offline to predict the utility of each edge from task, agent-role, and graph-structure features.
4. At test time, the generator is frozen. CCR scores the generated graph once and applies an edge/node pruning budget to obtain a smaller graph.
5. The final report compares original vs refined graph accuracy and token cost under multiple pruning budgets.

For an edge `e`, the causal-local training target is:

```text
y(e) = wc * max(0, Q(G, x) - Q(G \ e, x)) + we * Hlocal(e)
```

where `Q` is the final-answer correctness/quality score from leave-one-edge-out rollouts, and `Hlocal(e)` is the local semantic instability of the source/target agents after deleting `e`. In the current experiments:

```text
wc = 0.8, we = 0.2
```

The token-cost term is not used as a training label by default. Token reduction is evaluated directly in the Pareto sweep. This keeps the explainer focused on preserving useful communication, while the pruning budget controls how aggressively the graph is compressed.

## Explainer

The implemented explainer is `LocalEntropyEdgeExplainer`, an MLP edge scorer. For each candidate edge `(u, v)`, it receives:

- source-agent role embedding and target-agent role embedding;
- task embedding;
- edge type, edge index, source/target degree features, mask/logit features, and graph-size features.

It outputs an edge utility score in `[0, 1]`. Node scores are derived by aggregating incident edge scores, so the same explainer can support edge-only pruning or combined edge/node pruning.

The pruning rule is budget-based:

```text
remove lowest-scoring ceil(edge_pruning_rate * active_edges) edges
remove lowest-scoring ceil(node_pruning_rate * active_nodes) nodes
```

Node pruning removes the selected node and all incident edges, with `min_nodes` as a safety floor.

## Datasets

The six benchmark datasets used in the current experiments are:

```text
MMLU, GSM8K, MultiArith, SVAMP, AQuA, HumanEval
```

MMLU follows the GDesigner/AgentPrune-style split: graph learning uses the dev split and evaluation uses the 153-example validation subset used by AgentPrune. For GSM8K and HumanEval, the original GDesigner/AgentPrune adapters are used. MultiArith, SVAMP, and AQuA use added adapters aligned with the AgentDropout/AgentPrune data format because the original GDesigner repository does not provide all of them.

## Cost Accounting

The main accuracy-token comparison reports inference-time cost only:

```text
Original GDesigner inference tokens vs GDesigner + frozen CCR inference tokens
```

One-time costs are excluded from the main inference table:

- GDesigner backbone training;
- counterfactual leave-one-edge-out dataset construction;
- local semantic entropy sampling;
- offline explainer training.

These costs can be reported separately as amortized training/calibration overhead if needed.

## Repository Layout

```text
ccr/                  generator-independent scoring, pruning, and explainer code
adapters/             GDesigner/AgentPrune/AgentDropout dataset and graph adapters
configs/              experiment configs
scripts/              launchers for full benchmark runs
docs/                 method notes
tests/                lightweight unit tests
```

## Quick Test

```bash
cd /home/ljz/llm-mas/CausalCommunicationRefiner
/home/ljz/miniconda3/envs/qwen3_vllm_clean/bin/python -m pytest tests
```

## Train The Explainer Offline

Entropy-only labels:

```bash
cd /home/ljz/llm-mas/CausalCommunicationRefiner
/home/ljz/miniconda3/envs/qwen3_vllm_clean/bin/python -u adapters/train_local_entropy_explainer.py \
  --cache-roots results/gdesigner_mmlu_local_entropy_shards \
  --datasets mmlu \
  --label-source local_entropy
```

Causal-local labels:

```bash
cd /home/ljz/llm-mas/CausalCommunicationRefiner
/home/ljz/miniconda3/envs/qwen3_vllm_clean/bin/python -u adapters/train_local_entropy_explainer.py \
  --cache-roots results/gdesigner_mmlu_local_entropy_shards \
  --rollout-roots results/gdesigner_fixed_graph_counterfactual_mmlu_shards/20260611_121522 \
  --datasets mmlu \
  --label-source causal_local \
  --correctness-weight 0.8 \
  --entropy-weight 0.2
```

## Pareto Evaluation

Run one dataset against one vLLM endpoint:

```bash
cd /home/ljz/llm-mas/CausalCommunicationRefiner
/home/ljz/miniconda3/envs/qwen3_vllm_clean/bin/python -u adapters/evaluate_gdesigner_explainer_pareto.py \
  --dataset mmlu \
  --base-urls http://127.0.0.1:8035/v1 \
  --budget-grid 0.25:0.0,0.50:0.0,0.75:0.0,0.25:0.2,0.50:0.2,0.25:0.4 \
  --intervention-types both \
  --explainer-label-source causal_local
```

Launch the current six-dataset run on GPUs 2-7:

```bash
cd /home/ljz/llm-mas/CausalCommunicationRefiner
tmux new-session -d -s ccr_pareto_6gpu './scripts/launch_pareto_six_gpus.sh'
```

The launcher first tries to start one vLLM worker on each GPU from 2 to 7. If some GPUs are already occupied and vLLM fails to start, those GPUs are skipped and the six datasets are queued over the workers that are actually ready. This makes the run robust to shared-server GPU availability while preserving the same dataset order and output format.

The launcher writes logs to:

```text
logs/pareto_<timestamp>/
```

and results to:

```text
results/gdesigner_explainer_pareto/<timestamp>/
```

Each dataset directory contains `summary.json`, `pareto_metrics.csv`, per-budget predictions, masks, and usage logs. The top-level launcher also writes `aggregate_pareto.csv/json` after all launched jobs finish.

## Fairness Protocol

- The backbone graph generator is frozen during final evaluation.
- CCR is trained offline from calibration/counterfactual data and is frozen during final evaluation.
- Test labels are used only for final metric reporting, not for test-time graph selection.
- The main table should compare inference-time accuracy and token cost; training/calibration cost should be reported separately if included.
