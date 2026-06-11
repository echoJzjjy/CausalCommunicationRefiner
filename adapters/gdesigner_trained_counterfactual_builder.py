#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import builtins
import contextlib
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

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LLM_MAS_ROOT = PROJECT_ROOT.parent
GDESIGNER_ROOT = LLM_MAS_ROOT / "GDesigner"
AGENTPRUNE_ROOT = LLM_MAS_ROOT / "AgentPrune"

sys.path.insert(0, str(GDESIGNER_ROOT))
sys.path.insert(0, str(GDESIGNER_ROOT / "experiments"))
sys.path.insert(0, str(AGENTPRUNE_ROOT / "experiments"))
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(GDESIGNER_ROOT)

from run_granger_gdesigner import DatasetAdapter, build_graph, process_batch, summarize_usage, utc_now  # noqa: E402
from GDesigner.graph.graph import min_max_norm  # noqa: E402


DATASETS = ["mmlu", "gsm8k", "multiarith", "svamp", "aqua", "humaneval"]
_ORIGINAL_PRINT = builtins.print


def install_agent_print_filter() -> None:
    if getattr(builtins.print, "_ccr_agent_filter", False):
        return

    def filtered_print(*args: Any, **kwargs: Any) -> None:
        if args:
            first = str(args[0])
            if first.startswith("################") or first.startswith("#################"):
                return
        _ORIGINAL_PRINT(*args, **kwargs)

    setattr(filtered_print, "_ccr_agent_filter", True)
    builtins.print = filtered_print


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a G-Designer generator with the original 10x4 protocol, then build a "
            "task-level leave-one-edge-out counterfactual cache on the 40 training/calibration examples."
        )
    )
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results" / "gdesigner_trained_counterfactual")
    parser.add_argument("--llm-name", default="qwen3-8b")
    parser.add_argument("--base-urls", default="http://127.0.0.1:8003/v1,http://127.0.0.1:8004/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--top-p", default="0.95")
    parser.add_argument("--disable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--mode", default="FullConnected", choices=["FullConnected", "Random", "Chain", "Debate", "Layered", "Star", "Mesh"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-iterations", type=int, default=10)
    parser.add_argument("--train-examples", type=int, default=40)
    parser.add_argument("--agent-nums", nargs="+", type=int, default=None, help="Override DatasetAdapter.agent_nums, e.g. --agent-nums 6 for GDesigner MMLU README parity.")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--num-rounds", type=int, default=None, help="Override original per-dataset defaults.")
    parser.add_argument("--optimized-spatial", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--optimized-temporal", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--train-mlp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-training", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pretrained-cache-dir", type=Path, default=None, help="Dataset cache dir containing graph_architecture.json and gdesigner_trained_generator.pt.")
    parser.add_argument("--record-indices", nargs="+", type=int, default=None, help="Optional explicit indices within the selected train_examples records to cache.")
    parser.add_argument("--shard-index", type=int, default=0, help="0-based shard index over selected training records.")
    parser.add_argument("--num-shards", type=int, default=1, help="Number of shards over selected training records.")
    parser.add_argument("--edge-types", choices=["spatial", "temporal", "both"], default="both")
    parser.add_argument("--edge-limit", type=int, default=0)
    parser.add_argument("--logit-threshold", type=float, default=0.0)
    parser.add_argument("--cache-mode", choices=["full", "local_entropy"], default="full")
    parser.add_argument("--samples-per-graph", type=int, default=1)
    parser.add_argument("--parallel-samples", type=int, default=1)
    parser.add_argument("--local-samples", type=int, default=5)
    parser.add_argument("--parallel-local-samples", type=int, default=5)
    parser.add_argument("--local-entropy-threshold", type=float, default=0.82)
    parser.add_argument("--node-timeout", type=int, default=900)
    parser.add_argument("--max-tries", type=int, default=3)
    parser.add_argument("--mmlu-limit", type=int, default=153)
    parser.add_argument("--trace-code-timeout", type=int, default=2)
    parser.add_argument("--final-code-timeout", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0, help="Debug adapter limit before training split; 0 means full.")
    parser.add_argument("--code-timeout", type=int, default=5)
    parser.add_argument("--suppress-agent-stdout", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=888)
    return parser.parse_args()


def configure_env(args: argparse.Namespace, run_dir: Path) -> None:
    os.environ["BASE_URLS"] = args.base_urls
    os.environ["BASE_URL"] = args.base_urls.split(",")[0]
    os.environ["OPENAI_BASE_URL"] = args.base_urls.split(",")[0]
    os.environ["API_KEY"] = args.api_key
    os.environ["TOP_P"] = str(args.top_p)
    os.environ["QWEN_DISABLE_THINKING"] = "1" if args.disable_thinking else "0"
    os.environ["GDESIGNER_USAGE_LOG"] = str(run_dir / "usage.jsonl")
    os.environ["AGENTPRUNE_USAGE_LOG"] = str(run_dir / "usage.jsonl")


def refresh_gdesigner_llm_routing(args: argparse.Namespace) -> None:
    """GDesigner reads API routing at module import time; keep it aligned with CLI args."""
    try:
        import GDesigner.llm.gpt_chat as gpt_chat
    except Exception:
        return
    gpt_chat.BASE_URLS = [url.strip() for url in args.base_urls.split(",") if url.strip()]
    gpt_chat.API_KEY = args.api_key
    gpt_chat.TOP_P = float(args.top_p)
    gpt_chat.DISABLE_THINKING = args.disable_thinking
    gpt_chat._BASE_URL_INDEX = 0


def set_scope(scope: str, run_dir: Path) -> None:
    os.environ["GDESIGNER_USAGE_LOG"] = str(run_dir / "usage.jsonl")
    os.environ["GDESIGNER_USAGE_SCOPE"] = scope
    os.environ["AGENTPRUNE_USAGE_LOG"] = str(run_dir / "usage.jsonl")
    os.environ["AGENTPRUNE_USAGE_SCOPE"] = scope


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


def dataset_rounds(dataset: str, args: argparse.Namespace) -> int:
    if args.num_rounds is not None:
        return args.num_rounds
    return 2 if dataset == "humaneval" else 1


def dataset_kind(dataset: str) -> str:
    if dataset == "humaneval":
        return "code"
    if dataset in {"mmlu", "aqua"}:
        return "choice"
    return "number"


def record_id(adapter: DatasetAdapter, record: Any, index: int, train: bool) -> str:
    prefix = "train" if train else "eval"
    if adapter.name == "mmlu":
        return f"{prefix}:{index}"
    if isinstance(record, dict):
        for key in ("id", "ID", "iIndex", "name", "task_id"):
            if key in record:
                return f"{prefix}:{record[key]}"
    return f"{prefix}:{index}"


def select_training_records(adapter: DatasetAdapter, args: argparse.Namespace) -> tuple[list[Any], bool]:
    if adapter.name == "mmlu":
        rng = np.random.default_rng(args.seed)
        order = rng.permutation(len(adapter.train_records))
        records = [adapter.train_records[int(i)] for i in order[: args.train_examples]]
        return records, True
    records = adapter.eval_records[: args.train_examples]
    return records, False


def tensor_list(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        return [float(item) for item in value.detach().cpu().view(-1).tolist()]
    return [float(item) for item in value]


def graph_architecture_metadata(adapter: DatasetAdapter, graph: Any, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "method": "gdesigner_trained",
        "graph_source": "trained_gdesigner_task_conditioned_generator",
        "dataset": adapter.name,
        "graph_domain": adapter.graph_domain,
        "mode": args.mode,
        "train_examples": args.train_examples,
        "batch_size": args.batch_size,
        "num_iterations": args.num_iterations,
        "train_mlp": args.train_mlp,
        "node_ids": list(graph.nodes.keys()),
        "nodes": [
            {"id": node_id, "agent_name": getattr(node, "agent_name", None), "role": getattr(node, "role", None)}
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
        "fixed_spatial_masks": tensor_list(graph.spatial_masks),
        "fixed_temporal_masks": tensor_list(graph.temporal_masks),
    }


def load_pretrained_generator(graph: Any, checkpoint_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    graph.gcn.load_state_dict(checkpoint["gcn_state_dict"])
    graph.mlp.load_state_dict(checkpoint["mlp_state_dict"])
    graph.gcn.eval()
    graph.mlp.eval()
    return {
        "training_log_file": None,
        "checkpoint_file": str(checkpoint_path),
        "iterations": len(checkpoint.get("training_log") or []),
        "records": int(checkpoint.get("trained_records") or 0),
        "mean_train_utility": statistics.mean([row["mean_utility"] for row in checkpoint.get("training_log") or []]) if checkpoint.get("training_log") else 0.0,
        "reused_pretrained": True,
    }


def shard_records(records: list[Any], args: argparse.Namespace) -> tuple[list[Any], list[int]]:
    if args.num_shards <= 1:
        return records, list(range(len(records)))
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError(f"shard-index must be in [0, {args.num_shards}); got {args.shard_index}")
    selected = [(idx, record) for idx, record in enumerate(records) if idx % args.num_shards == args.shard_index]
    return [record for _idx, record in selected], [idx for idx, _record in selected]


def task_spatial_logits(graph: Any, task: str) -> torch.Tensor:
    with torch.no_grad():
        new_features = graph.construct_new_features(task)
        logits = graph.gcn(new_features, graph.role_adj_matrix)
        logits = graph.mlp(logits)
        return min_max_norm(torch.flatten(logits @ logits.t())).detach().float()


def task_fixed_graph(graph: Any, task: str, threshold: float) -> tuple[Any, dict[str, Any]]:
    spatial_logits = task_spatial_logits(graph, task)
    fixed_mask = graph.spatial_masks.detach().clone().float()
    active = ((fixed_mask > 0) & (spatial_logits >= threshold)).float()
    if float(active.sum()) <= 0 and float(fixed_mask.sum()) > 0:
        masked_logits = spatial_logits.clone()
        masked_logits[fixed_mask <= 0] = masked_logits.min() - 1.0
        active[int(torch.argmax(masked_logits).item())] = 1.0

    fixed_graph = copy.deepcopy(graph)
    fixed_graph.optimized_spatial = False
    fixed_graph.optimized_temporal = False
    fixed_graph.spatial_logits = spatial_logits.clone()
    fixed_graph.spatial_masks = torch.nn.Parameter(active, requires_grad=False)
    fixed_graph.temporal_masks = torch.nn.Parameter(graph.temporal_masks.detach().clone().float(), requires_grad=False)
    if hasattr(fixed_graph.temporal_logits, "detach"):
        fixed_graph.temporal_logits = torch.nn.Parameter(graph.temporal_logits.detach().clone().float(), requires_grad=False)

    meta = {
        "spatial_logits": tensor_list(spatial_logits),
        "spatial_masks": tensor_list(active),
        "spatial_threshold": threshold,
    }
    return fixed_graph, meta


def active_edges(graph: Any, num_rounds: int, edge_types: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if edge_types in {"spatial", "both"}:
        for idx, (source, target) in enumerate(graph.potential_spatial_edges):
            if float(graph.spatial_masks[idx]) > 0:
                rows.append({"kind": "spatial", "index": idx, "source": source, "target": target})
    if num_rounds > 1 and edge_types in {"temporal", "both"}:
        for idx, (source, target) in enumerate(graph.potential_temporal_edges):
            if float(graph.temporal_masks[idx]) > 0:
                rows.append({"kind": "temporal", "index": idx, "source": source, "target": target})
    return rows


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


def aggregate_samples(samples: list[dict[str, Any]], kind: str, usage: dict[str, int]) -> dict[str, Any]:
    correct_rate = statistics.mean(float(sample["correct"]) for sample in samples) if samples else 0.0
    counts = Counter(
        normalize_cluster_key(sample.get("prediction", ""), bool(sample.get("correct")), kind)
        for sample in samples
    )
    total = sum(counts.values())
    entropy = -sum((count / total) * math.log(max(count / total, 1e-12)) for count in counts.values()) if total else 0.0
    return {
        "samples": len(samples),
        "correct_score": correct_rate,
        "semantic_entropy": entropy,
        "confidence": math.exp(-entropy),
        "prediction_clusters": dict(counts),
        "usage": usage,
        "mean_total_tokens": usage["total_tokens"] / len(samples) if samples else 0.0,
    }


def text_tokens(text: Any) -> Counter:
    import re

    return Counter(re.findall(r"\w+|[^\w\s]", str(text or "").lower()))


def cosine_counter(left: Counter, right: Counter) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    common = set(left) & set(right)
    dot = sum(left[token] * right[token] for token in common)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return dot / max(left_norm * right_norm, 1e-12)


def semantic_cluster_labels(texts: list[Any], threshold: float) -> list[str]:
    labels: list[str] = []
    centroids: list[Counter] = []
    for text in texts:
        tokens = text_tokens(text)
        best_idx = -1
        best_score = -1.0
        for idx, centroid in enumerate(centroids):
            score = cosine_counter(tokens, centroid)
            if score > best_score:
                best_idx = idx
                best_score = score
        if best_idx >= 0 and best_score >= threshold:
            labels.append(f"cluster_{best_idx}")
            centroids[best_idx].update(tokens)
        else:
            labels.append(f"cluster_{len(centroids)}")
            centroids.append(tokens)
    return labels


def entropy_from_clusters(labels: list[str]) -> dict[str, Any]:
    counts = Counter(labels)
    total = sum(counts.values())
    entropy = -sum((count / total) * math.log(max(count / total, 1e-12)) for count in counts.values()) if total else 0.0
    return {
        "entropy": entropy,
        "confidence": math.exp(-entropy),
        "clusters": dict(counts),
        "num_clusters": len(counts),
    }


def aggregate_local_samples(samples: list[dict[str, Any]], usage: dict[str, int], threshold: float) -> dict[str, Any]:
    source_outputs = [sample.get("source_output", "") for sample in samples]
    target_outputs = [sample.get("target_output", "") for sample in samples]
    pair_outputs = [
        f"{sample.get('source_output', '')}\n---TARGET---\n{sample.get('target_output', '')}"
        for sample in samples
    ]
    source_labels = semantic_cluster_labels(source_outputs, threshold)
    target_labels = semantic_cluster_labels(target_outputs, threshold)
    pair_labels = semantic_cluster_labels(pair_outputs, threshold)
    source_stats = entropy_from_clusters(source_labels)
    target_stats = entropy_from_clusters(target_labels)
    pair_stats = entropy_from_clusters(pair_labels)
    return {
        "samples": len(samples),
        "source_semantic_entropy": source_stats["entropy"],
        "target_semantic_entropy": target_stats["entropy"],
        "pair_semantic_entropy": pair_stats["entropy"],
        "source_confidence": source_stats["confidence"],
        "target_confidence": target_stats["confidence"],
        "pair_confidence": pair_stats["confidence"],
        "source_clusters": source_stats["clusters"],
        "target_clusters": target_stats["clusters"],
        "pair_clusters": pair_stats["clusters"],
        "usage": usage,
        "mean_total_tokens": usage["total_tokens"] / len(samples) if samples else 0.0,
    }


def latest_output(graph: Any, node_id: str) -> str:
    node = graph.nodes.get(node_id)
    if node is None:
        return ""
    outputs = getattr(node, "outputs", [])
    if isinstance(outputs, list):
        return str(outputs[-1]) if outputs else ""
    return str(outputs)


def trace_output(trace: list[dict[str, Any]], node_id: str, round_index: int) -> str:
    if not trace:
        return ""
    round_index = max(0, min(round_index, len(trace) - 1))
    outputs = trace[round_index].get("nodes", {}).get(node_id, {}).get("outputs") or []
    if isinstance(outputs, list):
        return str(outputs[-1]) if outputs else ""
    return str(outputs)


async def run_agent_graph_without_decision(
    graph: Any,
    input_dict: dict[str, str],
    num_rounds: int,
    max_tries: int,
    max_time: int,
) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    if graph.optimized_spatial:
        new_features = graph.construct_new_features(input_dict["task"])
        logits = graph.gcn(new_features, graph.role_adj_matrix)
        logits = graph.mlp(logits)
        graph.spatial_logits = min_max_norm(torch.flatten(logits @ logits.t()))
    else:
        graph.spatial_logits = torch.zeros(len(graph.potential_spatial_edges), dtype=torch.float32)

    for round_idx in range(num_rounds):
        graph.construct_spatial_connection()
        graph.construct_temporal_connection(round_idx)
        in_degree = {node_id: len(node.spatial_predecessors) for node_id, node in graph.nodes.items()}
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]

        while queue:
            current_node_id = queue.pop(0)
            tries = 0
            while tries < max_tries:
                try:
                    await asyncio.wait_for(graph.nodes[current_node_id].async_execute(input_dict), timeout=max_time)
                    break
                except Exception as err:
                    print(f"Error during local execution of node {current_node_id}: {err}")
                tries += 1
            for successor in graph.nodes[current_node_id].spatial_successors:
                if successor.id not in graph.nodes:
                    continue
                in_degree[successor.id] -= 1
                if in_degree[successor.id] == 0:
                    queue.append(successor.id)

        trace.append(graph._snapshot_round_trace())
        graph.update_memory()

    graph.last_trace = trace
    return trace


async def run_local_edge_sample(
    graph: Any,
    adapter: DatasetAdapter,
    record: Any,
    train: bool,
    edge: dict[str, Any],
    args: argparse.Namespace,
    sample_index: int,
) -> dict[str, Any]:
    input_dict = adapter.train_input_for(record) if train else adapter.input_for(record)
    trace = await run_agent_graph_without_decision(
        graph,
        input_dict,
        args.num_rounds_for_dataset,
        args.max_tries,
        args.node_timeout,
    )

    target_round = len(trace) - 1
    source_round = max(0, target_round - 1) if edge["kind"] == "temporal" else target_round
    return {
        "sample_index": sample_index,
        "source": edge["source"],
        "target": edge["target"],
        "source_role": getattr(graph.nodes.get(edge["source"]), "role", ""),
        "target_role": getattr(graph.nodes.get(edge["target"]), "role", ""),
        "source_round": source_round,
        "target_round": target_round,
        "source_output": trace_output(trace, edge["source"], source_round),
        "target_output": trace_output(trace, edge["target"], target_round),
        "trace_edges": {
            "spatial": trace[-1].get("spatial_edges", []) if trace else [],
            "temporal": trace[-1].get("temporal_edges", []) if trace else [],
        },
    }


async def run_local_entropy_variant(
    graph: Any,
    adapter: DatasetAdapter,
    record: Any,
    record_key: str,
    train: bool,
    variant: dict[str, Any],
    args: argparse.Namespace,
    run_dir: Path,
) -> dict[str, Any]:
    edge = variant.get("edge")
    if edge is None:
        raise ValueError("local_entropy mode expects drop-edge variants only.")
    scope = f"gdesigner_local_entropy:{adapter.name}:{record_key}:{variant['id']}"
    set_scope(scope, run_dir)
    usage_log = run_dir / "usage.jsonl"
    before = usage_snapshot(usage_log, scope)
    semaphore = asyncio.Semaphore(max(1, args.parallel_local_samples))

    async def guarded(sample_index: int) -> dict[str, Any]:
        async with semaphore:
            sample_graph = copy.deepcopy(graph)
            return await run_local_edge_sample(sample_graph, adapter, record, train, edge, args, sample_index)

    samples = await asyncio.gather(*(guarded(i) for i in range(args.local_samples)))
    usage = usage_delta(usage_snapshot(usage_log, scope), before)
    input_dict = adapter.train_input_for(record) if train else adapter.input_for(record)
    return {
        "method": "gdesigner_trained_local_entropy",
        "dataset": adapter.name,
        "example_id": record_key,
        "variant": variant,
        "kind": dataset_kind(adapter.name),
        "meta": {"train_phase": train, "local_endpoint": [edge["source"], edge["target"]]},
        "task": input_dict["task"],
        "samples": samples,
        "aggregate": aggregate_local_samples(samples, usage, args.local_entropy_threshold),
    }


async def run_one_sample(
    graph: Any,
    adapter: DatasetAdapter,
    record: Any,
    train: bool,
    args: argparse.Namespace,
    sample_index: int,
) -> dict[str, Any]:
    input_dict = adapter.train_input_for(record) if train else adapter.input_for(record)
    if args.suppress_agent_stdout:
        with open(os.devnull, "w", encoding="utf-8") as devnull, contextlib.redirect_stdout(devnull):
            raw_answer, _log_prob = await graph.arun(
                input_dict,
                args.num_rounds_for_dataset,
                max_tries=args.max_tries,
                max_time=args.node_timeout,
            )
    else:
        raw_answer, _log_prob = await graph.arun(
            input_dict,
            args.num_rounds_for_dataset,
            max_tries=args.max_tries,
            max_time=args.node_timeout,
        )

    target = adapter.target_for(record, train=train)
    prediction = adapter.postprocess_train(raw_answer, record) if train else adapter.postprocess_final(raw_answer, record)
    correct = adapter.is_correct(prediction, target)
    row = {
        "sample_index": sample_index,
        "raw_answer": raw_answer,
        "prediction": prediction,
        "correct": bool(correct),
        "trace": getattr(graph, "last_trace", []),
    }
    return row


async def run_variant(
    graph: Any,
    adapter: DatasetAdapter,
    record: Any,
    record_key: str,
    train: bool,
    variant: dict[str, Any],
    args: argparse.Namespace,
    run_dir: Path,
) -> dict[str, Any]:
    scope = f"gdesigner_trained:{adapter.name}:{record_key}:{variant['id']}"
    set_scope(scope, run_dir)
    usage_log = run_dir / "usage.jsonl"
    before = usage_snapshot(usage_log, scope)
    semaphore = asyncio.Semaphore(max(1, args.parallel_samples))

    async def guarded(sample_index: int) -> dict[str, Any]:
        async with semaphore:
            sample_graph = copy.deepcopy(graph)
            return await run_one_sample(sample_graph, adapter, record, train, args, sample_index)

    samples = await asyncio.gather(*(guarded(i) for i in range(args.samples_per_graph)))
    usage = usage_delta(usage_snapshot(usage_log, scope), before)
    input_dict = adapter.train_input_for(record) if train else adapter.input_for(record)
    return {
        "method": "gdesigner_trained",
        "dataset": adapter.name,
        "example_id": record_key,
        "variant": variant,
        "answer": adapter.target_for(record, train=train),
        "kind": dataset_kind(adapter.name),
        "meta": {"train_phase": train},
        "task": input_dict["task"],
        "samples": samples,
        "aggregate": aggregate_samples(samples, dataset_kind(adapter.name), usage),
    }


async def train_gdesigner(adapter: DatasetAdapter, graph: Any, records: list[Any], train: bool, args: argparse.Namespace, dataset_dir: Path, run_dir: Path) -> dict[str, Any]:
    params = list(graph.gcn.parameters())
    if args.train_mlp:
        params.extend(list(graph.mlp.parameters()))
    optimizer = torch.optim.Adam(params, lr=args.lr)
    graph.gcn.train()
    graph.mlp.train(args.train_mlp)
    train_log = dataset_dir / "training_log.jsonl"
    rows: list[dict[str, Any]] = []

    set_scope(f"gdesigner_trained:{adapter.name}:train", run_dir)
    for i_iter in range(args.num_iterations):
        batch = records[i_iter * args.batch_size : (i_iter + 1) * args.batch_size]
        if not batch:
            break
        started = time.time()
        if args.suppress_agent_stdout:
            with open(os.devnull, "w", encoding="utf-8") as devnull, contextlib.redirect_stdout(devnull):
                batch_rows, loss_list, utilities, _traces = await process_batch(adapter, graph, batch, args, train=train)
        else:
            batch_rows, loss_list, utilities, _traces = await process_batch(adapter, graph, batch, args, train=train)
        if loss_list:
            total_loss = torch.mean(torch.stack(loss_list))
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            loss_value = float(total_loss.detach().cpu())
        else:
            loss_value = 0.0
        row = {
            "iteration": i_iter,
            "records": len(batch),
            "utilities": utilities,
            "mean_utility": statistics.mean(utilities) if utilities else 0.0,
            "loss": loss_value,
            "seconds": round(time.time() - started, 2),
        }
        rows.append(row)
        with train_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(
            f"[gdesigner][{adapter.name}] train iter {i_iter + 1}/{args.num_iterations} "
            f"mean_utility={row['mean_utility']:.3f} loss={loss_value:.4f}",
            flush=True,
        )

    graph.gcn.eval()
    graph.mlp.eval()
    checkpoint = dataset_dir / "gdesigner_trained_generator.pt"
    torch.save(
        {
            "dataset": adapter.name,
            "args": vars(args),
            "gcn_state_dict": graph.gcn.state_dict(),
            "mlp_state_dict": graph.mlp.state_dict(),
            "trained_records": len(records),
            "training_log": rows,
        },
        checkpoint,
    )
    return {
        "training_log_file": str(train_log),
        "checkpoint_file": str(checkpoint),
        "iterations": len(rows),
        "records": len(records),
        "mean_train_utility": statistics.mean([row["mean_utility"] for row in rows]) if rows else 0.0,
    }


async def build_dataset_cache(dataset: str, args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    dataset_dir = run_dir / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_args = copy.copy(args)
    dataset_args.num_rounds = dataset_rounds(dataset, args)
    dataset_args.num_rounds_for_dataset = dataset_args.num_rounds

    adapter = DatasetAdapter(dataset, dataset_args)
    if dataset_args.agent_nums is not None:
        if len(dataset_args.agent_nums) != len(adapter.agent_names):
            raise ValueError("--agent-nums length must match the dataset adapter's agent_names length.")
        adapter.agent_nums = list(dataset_args.agent_nums)
    graph = build_graph(adapter, dataset_args)
    all_train_records, train_flag = select_training_records(adapter, dataset_args)
    all_train_records = all_train_records[: dataset_args.train_examples]
    if dataset_args.record_indices is not None:
        invalid = [idx for idx in dataset_args.record_indices if idx < 0 or idx >= len(all_train_records)]
        if invalid:
            raise ValueError(f"--record-indices out of range for {len(all_train_records)} records: {invalid}")
        train_records = [all_train_records[idx] for idx in dataset_args.record_indices]
        original_indices = list(dataset_args.record_indices)
    else:
        train_records, original_indices = shard_records(all_train_records, dataset_args)
    graph_path = dataset_dir / "graph_architecture.json"
    graph_metadata = graph_architecture_metadata(adapter, graph, dataset_args)
    graph_metadata["shard_index"] = dataset_args.shard_index
    graph_metadata["num_shards"] = dataset_args.num_shards
    graph_metadata["original_record_indices"] = original_indices
    graph_path.write_text(json.dumps(graph_metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    started = time.time()
    if dataset_args.skip_training:
        if dataset_args.pretrained_cache_dir is None:
            raise ValueError("--skip-training requires --pretrained-cache-dir")
        checkpoint_path = dataset_args.pretrained_cache_dir / "gdesigner_trained_generator.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing pretrained checkpoint: {checkpoint_path}")
        train_info = load_pretrained_generator(graph, checkpoint_path)
    else:
        train_info = await train_gdesigner(adapter, graph, all_train_records, train_flag, dataset_args, dataset_dir, run_dir)
    rollout_path = dataset_dir / ("local_entropy.jsonl" if dataset_args.cache_mode == "local_entropy" else "rollouts.jsonl")
    example_graph_path = dataset_dir / "example_graphs.jsonl"

    completed = 0
    with rollout_path.open("w", encoding="utf-8") as rollout_out, example_graph_path.open("w", encoding="utf-8") as graph_out:
        for local_idx, record in enumerate(train_records):
            idx = original_indices[local_idx]
            record_key = record_id(adapter, record, idx, train_flag)
            input_dict = adapter.train_input_for(record) if train_flag else adapter.input_for(record)
            fixed_graph, example_graph_meta = task_fixed_graph(graph, input_dict["task"], dataset_args.logit_threshold)
            edges = active_edges(fixed_graph, dataset_args.num_rounds, dataset_args.edge_types)
            if dataset_args.edge_limit:
                edges = edges[: dataset_args.edge_limit]
            graph_out.write(json.dumps({
                "dataset": adapter.name,
                "example_id": record_key,
                "task": input_dict["task"],
                "num_rounds": dataset_args.num_rounds,
                "active_spatial_edges": [
                    edge for edge in active_edges(fixed_graph, 1, "spatial")
                ],
                "active_temporal_edges": [
                    edge for edge in active_edges(fixed_graph, dataset_args.num_rounds, "temporal")
                ],
                **example_graph_meta,
            }, ensure_ascii=False) + "\n")
            graph_out.flush()

            drop_variants = [
                {
                    "id": f"drop_{edge['kind']}_{edge['index']}_{edge['source']}_to_{edge['target']}",
                    "type": "drop_edge",
                    "edge": edge,
                }
                for edge in edges
            ]
            variants = drop_variants if dataset_args.cache_mode == "local_entropy" else [{"id": "base", "type": "base", "edge": None}] + drop_variants
            print(
                f"[gdesigner][{adapter.name}] cache example {idx + 1}/{len(all_train_records)} "
                f"mode={dataset_args.cache_mode} shard={dataset_args.shard_index}/{dataset_args.num_shards} "
                f"variants={len(variants)}",
                flush=True,
            )
            for variant in variants:
                variant_graph = clone_without_edge(fixed_graph, variant["edge"])
                if dataset_args.cache_mode == "local_entropy":
                    row = await run_local_entropy_variant(variant_graph, adapter, record, record_key, train_flag, variant, dataset_args, run_dir)
                else:
                    row = await run_variant(variant_graph, adapter, record, record_key, train_flag, variant, dataset_args, run_dir)
                rollout_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                rollout_out.flush()
                completed += 1
                if dataset_args.cache_mode == "local_entropy":
                    print(
                        f"[gdesigner][{adapter.name}] {record_key} {variant['id']} "
                        f"pair_entropy={row['aggregate']['pair_semantic_entropy']:.3f} "
                        f"tokens={row['aggregate']['usage']['total_tokens']}",
                        flush=True,
                    )
                else:
                    print(
                        f"[gdesigner][{adapter.name}] {record_key} {variant['id']} "
                        f"correct={row['aggregate']['correct_score']:.3f} "
                        f"tokens={row['aggregate']['usage']['total_tokens']}",
                        flush=True,
                    )

    summary = {
        "method": "gdesigner_trained_counterfactual",
        "dataset": dataset,
        "protocol": "mmlu_dev_40_train_then_cache" if dataset == "mmlu" else "online_first_40_train_then_cache",
        "num_rounds": dataset_args.num_rounds,
        "train_examples": len(train_records),
        "total_train_examples": len(all_train_records),
        "shard_index": dataset_args.shard_index,
        "num_shards": dataset_args.num_shards,
        "original_record_indices": original_indices,
        "batch_size": dataset_args.batch_size,
        "num_iterations": dataset_args.num_iterations,
        "edge_types": dataset_args.edge_types,
        "edge_limit": dataset_args.edge_limit,
        "cache_mode": dataset_args.cache_mode,
        "skip_training": dataset_args.skip_training,
        "pretrained_cache_dir": str(dataset_args.pretrained_cache_dir) if dataset_args.pretrained_cache_dir else None,
        "local_samples": dataset_args.local_samples,
        "parallel_local_samples": dataset_args.parallel_local_samples,
        "local_entropy_threshold": dataset_args.local_entropy_threshold,
        "rollout_rows": completed,
        "seconds": round(time.time() - started, 2),
        "training": train_info,
        "graph_architecture_file": str(graph_path),
        "example_graphs_file": str(example_graph_path),
        "rollout_file": str(rollout_path),
    }
    (dataset_dir / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def write_summary(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    total_usage = usage_snapshot(run_dir / "usage.jsonl")
    summary = {
        "method": "GDesigner trained counterfactual cache",
        "run_dir": str(run_dir),
        "updated_at_utc": utc_now(),
        "datasets": rows,
        "usage": total_usage,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (run_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "protocol", "train_examples", "num_rounds", "rollout_rows", "seconds", "checkpoint", "rollout_file"])
        for row in rows:
            writer.writerow([
                row["dataset"],
                row["protocol"],
                row["train_examples"],
                row["num_rounds"],
                row["rollout_rows"],
                row["seconds"],
                row["training"]["checkpoint_file"],
                row["rollout_file"],
            ])


async def main() -> None:
    args = parse_args()
    if args.suppress_agent_stdout:
        install_agent_print_filter()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    run_dir = (args.output_root / timestamp).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    configure_env(args, run_dir)
    (run_dir / "args.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")

    import GDesigner.agents  # noqa: F401
    import GDesigner.llm  # noqa: F401
    import GDesigner.prompt  # noqa: F401
    from GDesigner.llm.llm import LLM

    refresh_gdesigner_llm_routing(args)
    LLM.DEFAULT_MAX_TOKENS = args.max_tokens
    LLM.DEFAULT_TEMPERATURE = args.temperature

    rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        row = await build_dataset_cache(dataset, args, run_dir)
        rows.append(row)
        write_summary(run_dir, rows)
    print(f"[done] {run_dir}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
