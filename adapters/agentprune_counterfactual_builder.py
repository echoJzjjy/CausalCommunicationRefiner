#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import copy
import csv
import json
import math
import os
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LLM_MAS_ROOT = PROJECT_ROOT.parent
AGENTPRUNE_ROOT = LLM_MAS_ROOT / "AgentPrune"

sys.path.insert(0, str(AGENTPRUNE_ROOT))
sys.path.insert(0, str(AGENTPRUNE_ROOT / "experiments"))
sys.stdout.reconfigure(encoding="utf-8")


DEFAULT_CHECKPOINTS = {
    "mmlu": AGENTPRUNE_ROOT / "result" / "agentprune" / "20260514_010835" / "mmlu.checkpoint.pt",
    "gsm8k": AGENTPRUNE_ROOT / "result" / "agentprune_remaining" / "20260514_024935" / "gsm8k.checkpoint.pt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an AgentPrune leave-one-edge-out rollout cache. "
            "The cache stores raw outputs only; rewards/contributions can be recomputed offline."
        )
    )
    parser.add_argument("--datasets", nargs="+", default=["gsm8k"])
    parser.add_argument("--data-root", type=Path, default=LLM_MAS_ROOT / "single_agent" / "data")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results" / "agentprune_counterfactual_cache")
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help="Dataset-specific checkpoint in the form dataset=/path/to/checkpoint.pt. Can be passed multiple times.",
    )
    parser.add_argument("--use-default-checkpoints", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--llm-name", default="qwen3-8b")
    parser.add_argument("--base-urls", default="http://127.0.0.1:8003/v1,http://127.0.0.1:8004/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--top-p", default="0.95")
    parser.add_argument("--disable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--samples-per-graph", type=int, default=3)
    parser.add_argument("--parallel-samples", type=int, default=3)

    parser.add_argument("--mode", default="FullConnected")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--num-rounds", type=int, default=None)
    parser.add_argument("--optimized-spatial", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--optimized-temporal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--topology-mode",
        choices=["mask", "threshold", "stochastic"],
        default="mask",
        help=(
            "mask: freeze the learned active masks as deterministic topology; "
            "threshold: keep active masks whose sigmoid(logit) exceeds --logit-threshold; "
            "stochastic: keep AgentPrune stochastic edge sampling."
        ),
    )
    parser.add_argument("--logit-threshold", type=float, default=0.5)
    parser.add_argument("--edge-types", choices=["spatial", "temporal", "both"], default="both")
    parser.add_argument(
        "--subset",
        choices=["train", "eval"],
        default="train",
        help="train uses the graph-learning subset; eval uses the benchmark evaluation order.",
    )
    parser.add_argument(
        "--train-examples",
        type=int,
        default=40,
        help="Number of graph-learning examples to cache. Default matches 10 iterations x batch size 4.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Override number of examples; 0 uses subset default.")
    parser.add_argument("--edge-limit", type=int, default=0, help="Debug limit per example; 0 means all active edges.")
    parser.add_argument("--node-timeout", type=int, default=900)
    parser.add_argument("--max-tries", type=int, default=3)
    parser.add_argument("--code-timeout", type=int, default=5)
    parser.add_argument("--seed", type=int, default=888)
    return parser.parse_args()


class SimpleExample:
    def __init__(self, *, id: str, dataset: str, task: str, answer: str, kind: str, meta: dict[str, Any]) -> None:
        self.id = id
        self.dataset = dataset
        self.task = task
        self.answer = answer
        self.kind = kind
        self.meta = meta


def mmlu_task(question: str, a: str, b: str, c: str, d: str) -> str:
    return f"{question}\nOption A: {a}\nOption B: {b}\nOption C: {c}\nOption D: {d}"


def load_mmlu_split_examples(data_root: Path, split: str, limit: int) -> list[SimpleExample]:
    import numpy as np

    split_dir = data_root / "mmlu" / "data" / split
    suffix = f"_{split}.csv"
    rows = []
    for csv_path in sorted(split_dir.glob(f"*{suffix}")):
        subject = csv_path.name.removesuffix(suffix)
        with csv_path.open(encoding="utf-8") as f:
            for i, row in enumerate(csv.reader(f)):
                if len(row) < 6:
                    continue
                question, a, b, c, d, answer = row[:6]
                rows.append(
                    SimpleExample(
                        id=f"{split}:{subject}:{i}",
                        dataset="mmlu",
                        task=mmlu_task(question, a, b, c, d),
                        answer=answer.strip(),
                        kind="choice",
                        meta={"subject": subject, "choices": "ABCD", "split": split},
                    )
                )

    rng = np.random.default_rng(888)
    rows = [rows[i] for i in rng.permutation(len(rows))]
    return rows[:limit] if limit else rows


def load_subset_examples(dataset: str, args: argparse.Namespace) -> list[Any]:
    from experiments.benchmark_adapters import load_examples

    limit = args.limit if args.limit else args.train_examples
    if args.subset == "train" and dataset == "mmlu":
        return load_mmlu_split_examples(args.data_root, "dev", limit)

    examples = load_examples(args.data_root, dataset)
    if limit:
        examples = examples[:limit]
    return examples


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def configure_env(args: argparse.Namespace, run_dir: Path) -> None:
    os.environ["BASE_URLS"] = args.base_urls
    os.environ["API_KEY"] = args.api_key
    os.environ["TOP_P"] = str(args.top_p)
    os.environ["QWEN_DISABLE_THINKING"] = "1" if args.disable_thinking else "0"
    os.environ["AGENTPRUNE_USAGE_LOG"] = str(run_dir / "usage.jsonl")


def usage_snapshot(path: Path, scope: str | None = None) -> dict[str, int]:
    totals = {
        "requests": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "prompt_chars": 0,
        "response_chars": 0,
    }
    if not path.exists():
        return totals
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if scope is not None and row.get("scope") != scope:
                continue
            totals["requests"] += 1
            for key in totals:
                if key != "requests":
                    totals[key] += int(row.get(key) or 0)
    return totals


def usage_delta(after: dict[str, int], before: dict[str, int]) -> dict[str, int]:
    return {key: int(after.get(key, 0)) - int(before.get(key, 0)) for key in before}


def parse_checkpoint_args(items: list[str]) -> dict[str, Path]:
    out = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--checkpoint must be dataset=/path form, got {item!r}")
        dataset, path = item.split("=", 1)
        out[dataset.strip()] = Path(path).expanduser().resolve()
    return out


def checkpoint_for(dataset: str, args: argparse.Namespace, explicit: dict[str, Path]) -> Path | None:
    if dataset in explicit:
        return explicit[dataset]
    if args.checkpoint_dir is not None:
        candidate = args.checkpoint_dir / f"{dataset}.checkpoint.pt"
        if candidate.exists():
            return candidate
    if args.use_default_checkpoints:
        candidate = DEFAULT_CHECKPOINTS.get(dataset)
        if candidate is not None and candidate.exists():
            return candidate
    return None


def set_deterministic_topology(graph: Any, mode: str, threshold: float) -> None:
    if mode == "stochastic":
        return
    if mode == "threshold":
        import torch

        spatial_prob = torch.sigmoid(graph.spatial_logits.detach())
        temporal_prob = torch.sigmoid(graph.temporal_logits.detach())
        graph.spatial_masks.data = ((graph.spatial_masks.detach() > 0) & (spatial_prob >= threshold)).float()
        graph.temporal_masks.data = ((graph.temporal_masks.detach() > 0) & (temporal_prob >= threshold)).float()
    graph.optimized_spatial = False
    graph.optimized_temporal = False
    graph.spatial_logits.requires_grad_(False)
    graph.temporal_logits.requires_grad_(False)


def tensor_list(value: Any) -> list[float]:
    return [float(item) for item in value.detach().cpu().view(-1).tolist()]


def edge_rows(graph: Any, edge_types: str, num_rounds: int) -> list[dict[str, Any]]:
    rows = []
    if edge_types in {"spatial", "both"}:
        for idx, (source, target) in enumerate(graph.potential_spatial_edges):
            if float(graph.spatial_masks[idx]) > 0:
                rows.append({"kind": "spatial", "index": idx, "source": source, "target": target})
    if num_rounds > 1 and edge_types in {"temporal", "both"}:
        for idx, (source, target) in enumerate(graph.potential_temporal_edges):
            if float(graph.temporal_masks[idx]) > 0:
                rows.append({"kind": "temporal", "index": idx, "source": source, "target": target})
    return rows


def graph_metadata(graph: Any) -> dict[str, Any]:
    return {
        "node_ids": list(graph.nodes.keys()),
        "nodes": [
            {
                "id": node_id,
                "agent_name": node.agent_name,
                "role": getattr(node, "role", None),
            }
            for node_id, node in graph.nodes.items()
        ],
        "potential_spatial_edges": [
            {"index": idx, "source": source, "target": target}
            for idx, (source, target) in enumerate(graph.potential_spatial_edges)
        ],
        "potential_temporal_edges": [
            {"index": idx, "source": source, "target": target}
            for idx, (source, target) in enumerate(graph.potential_temporal_edges)
        ],
        "spatial_masks": tensor_list(graph.spatial_masks),
        "temporal_masks": tensor_list(graph.temporal_masks),
        "spatial_logits": tensor_list(graph.spatial_logits),
        "temporal_logits": tensor_list(graph.temporal_logits),
    }


def clone_without_edge(graph: Any, edge: dict[str, Any] | None) -> Any:
    g = copy.deepcopy(graph)
    if edge is None:
        return g
    if edge["kind"] == "spatial":
        g.spatial_masks.data[int(edge["index"])] = 0.0
    elif edge["kind"] == "temporal":
        g.temporal_masks.data[int(edge["index"])] = 0.0
    else:
        raise ValueError(f"Unknown edge kind: {edge['kind']}")
    return g


def normalize_cluster_key(prediction: Any, correct: bool, kind: str) -> str:
    if kind == "code":
        return "pass" if correct else f"fail:{str(prediction)[:120]}"
    value = str(prediction).strip().lower()
    return " ".join(value.split()) or "<empty>"


def semantic_entropy(samples: list[dict[str, Any]], kind: str) -> float:
    if not samples:
        return 0.0
    counts = Counter(
        normalize_cluster_key(sample.get("prediction", ""), bool(sample.get("correct")), kind)
        for sample in samples
    )
    total = sum(counts.values())
    return -sum((count / total) * math.log(max(count / total, 1e-12)) for count in counts.values())


def aggregate_samples(samples: list[dict[str, Any]], kind: str, usage: dict[str, int]) -> dict[str, Any]:
    correct_rate = statistics.mean(float(sample["correct"]) for sample in samples) if samples else 0.0
    entropy = semantic_entropy(samples, kind)
    return {
        "samples": len(samples),
        "correct_score": correct_rate,
        "semantic_entropy": entropy,
        "confidence": math.exp(-entropy),
        "prediction_clusters": dict(Counter(
            normalize_cluster_key(sample.get("prediction", ""), bool(sample.get("correct")), kind)
            for sample in samples
        )),
        "usage": usage,
        "mean_total_tokens": usage["total_tokens"] / len(samples) if samples else 0.0,
    }


async def run_one_sample(graph: Any, example: Any, args: argparse.Namespace, sample_index: int) -> dict[str, Any]:
    from experiments.benchmark_adapters import evaluate_raw_answer

    realized_graph = copy.deepcopy(graph)
    raw_answer, _log_prob = await realized_graph.arun(
        {"task": example.task},
        args.num_rounds_for_dataset,
        max_tries=args.max_tries,
        max_time=args.node_timeout,
    )
    judged = evaluate_raw_answer(raw_answer, example, args.code_timeout)
    row = {
        "sample_index": sample_index,
        "raw_answer": raw_answer,
        "prediction": judged.get("prediction"),
        "correct": bool(judged.get("correct")),
        "trace": getattr(realized_graph, "last_trace", []),
    }
    for key in ("code", "error"):
        if key in judged:
            row[key] = judged[key]
    return row


async def run_variant(graph: Any, example: Any, variant: dict[str, Any], args: argparse.Namespace, usage_log: Path) -> dict[str, Any]:
    scope = f"{example.dataset}:{example.id}:{variant['id']}"
    os.environ["AGENTPRUNE_USAGE_SCOPE"] = scope
    before = usage_snapshot(usage_log, scope)
    semaphore = asyncio.Semaphore(max(1, args.parallel_samples))

    async def guarded(sample_index: int) -> dict[str, Any]:
        async with semaphore:
            return await run_one_sample(graph, example, args, sample_index)

    samples = await asyncio.gather(*(guarded(i) for i in range(args.samples_per_graph)))
    after = usage_snapshot(usage_log, scope)
    usage = usage_delta(after, before)
    return {
        "dataset": example.dataset,
        "example_id": example.id,
        "variant": variant,
        "answer": example.answer,
        "kind": example.kind,
        "meta": example.meta,
        "task": example.task,
        "samples": samples,
        "aggregate": aggregate_samples(samples, example.kind, usage),
    }


async def build_dataset_cache(dataset: str, args: argparse.Namespace, run_dir: Path, explicit_checkpoints: dict[str, Path]) -> dict[str, Any]:
    import torch
    from AgentPrune.graph.graph import Graph
    from AgentPrune.llm.llm import LLM
    from experiments.benchmark_adapters import get_dataset_config
    from experiments.run_agentprune import get_kwargs, load_checkpoint

    LLM.DEFAULT_MAX_TOKENS = args.max_tokens
    LLM.DEFAULT_TEMPERATURE = args.temperature

    started = time.time()
    config = get_dataset_config(dataset)
    graph_domain = config.graph_domain
    agent_names = [name for name, num in zip(config.agent_names, config.agent_nums) for _ in range(num)]
    num_rounds = args.num_rounds if args.num_rounds is not None else config.num_rounds
    args.num_rounds_for_dataset = num_rounds

    graph = Graph(
        domain=graph_domain,
        llm_name=args.llm_name,
        agent_names=agent_names,
        decision_method=config.decision_method,
        optimized_spatial=args.optimized_spatial,
        optimized_temporal=args.optimized_temporal,
        **get_kwargs(args.mode, len(agent_names)),
    )
    optimizer = torch.optim.Adam([graph.spatial_logits, graph.temporal_logits], lr=args.lr)
    checkpoint = checkpoint_for(dataset, args, explicit_checkpoints)
    if checkpoint is not None:
        load_checkpoint(checkpoint, graph, optimizer)
    set_deterministic_topology(graph, args.topology_mode, args.logit_threshold)

    examples = load_subset_examples(dataset, args)

    dataset_dir = run_dir / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    rollout_path = dataset_dir / "rollouts.jsonl"
    manifest_path = dataset_dir / "manifest.json"
    graph_path = dataset_dir / "graph.json"
    graph_path.write_text(json.dumps(graph_metadata(graph), ensure_ascii=False, indent=2), encoding="utf-8")

    edges = edge_rows(graph, args.edge_types, num_rounds)
    if args.edge_limit:
        edges = edges[:args.edge_limit]
    variants = [{"id": "base", "type": "base", "edge": None}] + [
        {
            "id": f"drop_{edge['kind']}_{edge['index']}_{edge['source']}_to_{edge['target']}",
            "type": "drop_edge",
            "edge": edge,
        }
        for edge in edges
    ]

    completed = 0
    with rollout_path.open("w", encoding="utf-8") as out:
        for example_idx, example in enumerate(examples, 1):
            print(
                f"[{dataset}] example {example_idx}/{len(examples)} variants={len(variants)} "
                f"samples={args.samples_per_graph}",
                flush=True,
            )
            for variant in variants:
                variant_graph = clone_without_edge(graph, variant["edge"])
                row = await run_variant(variant_graph, example, variant, args, run_dir / "usage.jsonl")
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                completed += 1
                print(
                    f"[{dataset}] {example.id} {variant['id']} "
                    f"correct={row['aggregate']['correct_score']:.3f} "
                    f"H={row['aggregate']['semantic_entropy']:.3f} "
                    f"tokens={row['aggregate']['usage']['total_tokens']}",
                    flush=True,
                )

    summary = {
        "dataset": dataset,
        "graph_domain": graph_domain,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "topology_mode": args.topology_mode,
        "num_rounds": num_rounds,
        "subset": args.subset,
        "train_examples": args.train_examples,
        "limit": args.limit,
        "examples": len(examples),
        "edge_types": args.edge_types,
        "edges": len(edges),
        "variants_per_example": len(variants),
        "samples_per_graph": args.samples_per_graph,
        "rollout_rows": completed,
        "seconds": round(time.time() - started, 2),
        "rollout_file": str(rollout_path),
        "graph_file": str(graph_path),
    }
    manifest_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


async def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = (args.output_root / timestamp).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    configure_env(args, run_dir)
    (run_dir / "args.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
    explicit_checkpoints = parse_checkpoint_args(args.checkpoint)

    rows = []
    for dataset in args.datasets:
        rows.append(await build_dataset_cache(dataset, args, run_dir, explicit_checkpoints))
        (run_dir / "summary.json").write_text(json.dumps({
            "method": "AgentPrune counterfactual rollout cache",
            "run_dir": str(run_dir),
            "updated_at_utc": utc_now(),
            "datasets": rows,
            "usage": usage_snapshot(run_dir / "usage.jsonl"),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {run_dir}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
