# Method Review Notes

## What Is Clear

- The refiner is a post-hoc module, not a graph generator.
- It is generator-agnostic: any generator only needs to output a candidate graph.
- The method naturally decomposes into `audit -> refine -> compress`.
- The non-parametric refiner can be used as both a method and a label generator for an amortized critic.

## Points To Tighten

- Define edge messages operationally. In many MAS frameworks, `m_ij` is not a separate object;
  it is the source agent output injected into the target prompt.
- Separate two protocols:
  - Full causal refinement can run test-time ablations, but only with unlabeled signals.
  - Amortized critic must be trained on train/calibration labels and frozen at test time.
- If verifier `Q` uses an LLM judge, document the prompt and ensure it does not see ground truth on test.
- `DeltaA` measures answer change, not answer improvement. It should not dominate `DeltaQ`.
- Node pruning can remove important diversity roles. Keep a `min_nodes` constraint and report node removal.
- Cost reduction should report both structural cost and actual token/call usage, because removing a graph edge
  does not always reduce LLM calls in every framework.

## Recommended First Experiments

1. HumanEval, five topology generators, post-hoc refiner only.
2. Add generator-only baseline or reuse existing topology baseline for comparison.
3. Run random-prune and degree-prune at the same edge/node budget.
4. Then expand to MMLU-153 and math datasets.

