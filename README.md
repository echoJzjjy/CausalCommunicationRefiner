# Causal Communication Refiner

Generator-agnostic post-hoc refinement for LLM-based multi-agent communication graphs.

The method treats a topology generator as a black box:

```text
Generator -> Candidate Graph -> Causal Communication Refiner -> Refined Graph
```

It audits an existing communication graph, re-scores nodes/edges by counterfactual influence,
and compresses low-contribution communication structures. The first implementation includes:

- Non-parametric causal refinement with node/edge ablation labels.
- A lightweight amortized critic interface for edge/node score prediction.
- A GDesigner adapter for running the method on existing LLM-MAS experiments.

## Core Idea

For a generated graph `G0`, the refiner estimates component influence:

```text
S(c) = alpha * DeltaQ + beta * DeltaA + gamma * DeltaD
       + delta * U * (1 - R) - lambda * C
```

where `c` is an edge or node, `DeltaQ` is verifier quality drop, `DeltaA` is final answer
change, `DeltaD` is downstream output change, `U` is message usage, `R` is redundancy, and
`C` is communication cost.

## Project Layout

```text
ccr/                  framework-independent method code
adapters/             backend adapters, currently GDesigner
configs/              experiment configs
scripts/              runnable shell scripts
docs/                 method notes
tests/                lightweight unit tests
```

## Quick Smoke Test

```bash
cd /home/ljz/llm-mas/CausalCommunicationRefiner
/home/ljz/miniconda3/envs/qwen3_vllm_clean/bin/python -m pytest tests
```

## GDesigner Smoke Run

```bash
cd /home/ljz/llm-mas/CausalCommunicationRefiner
/home/ljz/miniconda3/envs/qwen3_vllm_clean/bin/python -u adapters/gdesigner_posthoc_runner.py \
  --datasets humaneval \
  --generators Complete Chain \
  --limit 2 \
  --calibration-examples 1 \
  --mask-samples 2 \
  --explainer-epochs 1 \
  --max-tokens 512
```

## Fairness Protocol

- Full causal refinement may do test-time ablations only with unlabeled signals.
- Ground-truth answers can be used for training/calibration labels, but not for test-time topology selection.
- The amortized critic must be frozen during final test evaluation.
- Test labels are used only for final reported metrics.

