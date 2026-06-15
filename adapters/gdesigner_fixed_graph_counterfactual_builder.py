#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import builtins
import contextlib
import copy
import csv
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LLM_MAS_ROOT = PROJECT_ROOT.parent
GDESIGNER_ROOT = LLM_MAS_ROOT / "GDesigner"
AGENTPRUNE_ROOT = LLM_MAS_ROOT / "AgentPrune"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(GDESIGNER_ROOT))
sys.path.insert(0, str(GDESIGNER_ROOT / "experiments"))
sys.path.insert(0, str(AGENTPRUNE_ROOT / "experiments"))
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(GDESIGNER_ROOT)

from adapters.gdesigner_trained_counterfactual_builder import (  # noqa: E402
    DATASETS,
    active_edges,
    aggregate_samples,
    clone_without_edge,
    configure_env,
    dataset_kind,
    graph_architecture_metadata,
    install_agent_print_filter,
    load_pretrained_generator,
    record_id,
    refresh_gdesigner_llm_routing,
    run_one_sample,
    select_training_records,
    set_scope,
    shard_records,
    tensor_list,
    usage_delta,
    usage_snapshot,
)
from run_granger_gdesigner import DatasetAdapter, build_graph, utc_now  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build final-correctness leave-one-edge-out rollouts from previously saved "
            "GDesigner per-task graphs. This does not regenerate task-conditioned graphs."
        )
    )
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=["mmlu"])
    parser.add_argument("--saved-graph-root", type=Path, required=True, help="Directory containing DATASET/example_graphs.jsonl, or the dataset directory itself.")
    parser.add_argument("--pretrained-cache-dir", type=Path, required=True, help="Dataset cache dir containing gdesigner_trained_generator.pt.")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results" / "gdesigner_fixed_graph_counterfactual")
    parser.add_argument("--llm-name", default="qwen3-8b")
    parser.add_argument("--base-urls", default="http://127.0.0.1:8012/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--top-p", default="0.95")
    parser.add_argument("--disable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--mode", default="FullConnected", choices=["FullConnected", "Random", "Chain", "Debate", "Layered", "Star", "Mesh"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-iterations", type=int, default=10)
    parser.add_argument("--train-examples", type=int, default=40)
    parser.add_argument("--agent-nums", nargs="+", type=int, default=None)
    parser.add_argument("--num-rounds", type=int, default=None)
    parser.add_argument("--optimized-spatial", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--optimized-temporal", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--train-mlp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--record-indices", nargs="+", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--edge-types", choices=["spatial", "temporal", "both"], default="both")
    parser.add_argument("--edge-limit", type=int, default=0)
    parser.add_argument("--samples-per-graph", type=int, default=1)
    parser.add_argument("--parallel-samples", type=int, default=1)
    parser.add_argument("--mmlu-limit", type=int, default=153)
    parser.add_argument("--trace-code-timeout", type=int, default=2)
    parser.add_argument("--final-code-timeout", type=int, default=100)
    parser.add_argument("--node-timeout", type=int, default=900)
    parser.add_argument("--max-tries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--code-timeout", type=int, default=5)
    parser.add_argument("--suppress-agent-stdout", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=888)
    return parser.parse_args()


def dataset_rounds(dataset: str, args: argparse.Namespace) -> int:
    if args.num_rounds is not None:
        return args.num_rounds
    return 2 if dataset == "humaneval" else 1


def resolve_saved_dataset_dir(root: Path, dataset: str) -> Path:
    root = root.expanduser().resolve()
    if (root / "example_graphs.jsonl").exists():
        return root
    if (root / dataset / "example_graphs.jsonl").exists():
        return root / dataset
    raise FileNotFoundError(f"Missing saved example_graphs.jsonl for {dataset} under {root}")


def load_saved_graphs(dataset_dir: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(dataset_dir / "example_graphs.jsonl")
    return {str(row.get("example_id")): row for row in rows}


def load_saved_architecture(dataset_dir: Path) -> dict[str, Any]:
    path = dataset_dir / "graph_architecture.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def align_graph_node_ids(graph: Any, saved_architecture: dict[str, Any]) -> None:
    saved_nodes = saved_architecture.get("nodes") or []
    saved_node_ids = [str(node.get("id")) for node in saved_nodes if node.get("id") is not None]
    current_node_ids = list(graph.nodes.keys())
    if not saved_node_ids or len(saved_node_ids) != len(current_node_ids):
        return

    remapped_nodes = {}
    for old_id, new_id in zip(current_node_ids, saved_node_ids):
        node = graph.nodes[old_id]
        node.id = new_id
        remapped_nodes[new_id] = node
    graph.nodes = remapped_nodes
    graph.potential_spatial_edges = [
        [str(edge.get("source")), str(edge.get("target"))]
        for edge in saved_architecture.get("potential_spatial_edges", [])
    ]
    graph.potential_temporal_edges = [
        [str(edge.get("source")), str(edge.get("target"))]
        for edge in saved_architecture.get("potential_temporal_edges", [])
    ]


def restore_saved_graph(base_graph: Any, graph_row: dict[str, Any]) -> Any:
    graph = copy.deepcopy(base_graph)
    spatial_masks = torch.tensor(graph_row.get("spatial_masks") or [], dtype=torch.float32)
    if spatial_masks.numel() != len(graph.potential_spatial_edges):
        raise ValueError(
            f"Saved spatial mask length {spatial_masks.numel()} does not match graph edges {len(graph.potential_spatial_edges)}"
        )
    graph.optimized_spatial = False
    graph.optimized_temporal = False
    graph.spatial_masks = torch.nn.Parameter(spatial_masks, requires_grad=False)
    graph.spatial_logits = torch.tensor(graph_row.get("spatial_logits") or [0.0] * len(graph.potential_spatial_edges), dtype=torch.float32)

    temporal_masks = graph_row.get("temporal_masks")
    if temporal_masks is not None:
        temporal_tensor = torch.tensor(temporal_masks, dtype=torch.float32)
    else:
        temporal_tensor = base_graph.temporal_masks.detach().clone().float()
    graph.temporal_masks = torch.nn.Parameter(temporal_tensor, requires_grad=False)
    if hasattr(base_graph.temporal_logits, "detach"):
        graph.temporal_logits = torch.nn.Parameter(base_graph.temporal_logits.detach().clone().float(), requires_grad=False)
    return graph


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
    scope = f"gdesigner_fixed_graph:{adapter.name}:{record_key}:{variant['id']}"
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
        "method": "gdesigner_fixed_graph_counterfactual",
        "dataset": adapter.name,
        "example_id": record_key,
        "variant": variant,
        "answer": adapter.target_for(record, train=train),
        "kind": dataset_kind(adapter.name),
        "meta": {"train_phase": train, "saved_graph": True},
        "task": input_dict["task"],
        "samples": samples,
        "aggregate": aggregate_samples(samples, dataset_kind(adapter.name), usage),
    }


async def build_dataset_cache(dataset: str, args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    dataset_dir = run_dir / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    saved_dataset_dir = resolve_saved_dataset_dir(args.saved_graph_root, dataset)
    saved_graphs = load_saved_graphs(saved_dataset_dir)
    saved_architecture = load_saved_architecture(saved_dataset_dir)

    dataset_args = copy.copy(args)
    dataset_args.num_rounds = dataset_rounds(dataset, args)
    dataset_args.num_rounds_for_dataset = dataset_args.num_rounds
    adapter = DatasetAdapter(dataset, dataset_args)
    if dataset_args.agent_nums is not None:
        if len(dataset_args.agent_nums) != len(adapter.agent_names):
            raise ValueError("--agent-nums length must match the dataset adapter's agent_names length.")
        adapter.agent_nums = list(dataset_args.agent_nums)
    graph = build_graph(adapter, dataset_args)
    align_graph_node_ids(graph, saved_architecture)
    train_info = load_pretrained_generator(graph, args.pretrained_cache_dir / "gdesigner_trained_generator.pt")

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
    graph_metadata["graph_source"] = "saved_gdesigner_task_graphs"
    graph_metadata["saved_graph_dir"] = str(saved_dataset_dir)
    graph_metadata["shard_index"] = dataset_args.shard_index
    graph_metadata["num_shards"] = dataset_args.num_shards
    graph_metadata["original_record_indices"] = original_indices
    graph_path.write_text(json.dumps(graph_metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    rollout_path = dataset_dir / "rollouts.jsonl"
    example_graph_path = dataset_dir / "example_graphs.jsonl"
    started = time.time()
    completed = 0
    missing_graphs: list[str] = []
    with rollout_path.open("w", encoding="utf-8") as rollout_out, example_graph_path.open("w", encoding="utf-8") as graph_out:
        for local_idx, record in enumerate(train_records):
            idx = original_indices[local_idx]
            record_key = record_id(adapter, record, idx, train_flag)
            graph_row = saved_graphs.get(record_key)
            if graph_row is None:
                missing_graphs.append(record_key)
                continue
            fixed_graph = restore_saved_graph(graph, graph_row)
            edges = active_edges(fixed_graph, dataset_args.num_rounds, dataset_args.edge_types)
            if dataset_args.edge_limit:
                edges = edges[: dataset_args.edge_limit]
            graph_out.write(json.dumps({
                **graph_row,
                "source_example_graph_file": str(saved_dataset_dir / "example_graphs.jsonl"),
                "active_spatial_edges": active_edges(fixed_graph, 1, "spatial"),
                "active_temporal_edges": active_edges(fixed_graph, dataset_args.num_rounds, "temporal"),
                "spatial_masks": tensor_list(fixed_graph.spatial_masks),
                "spatial_logits": tensor_list(fixed_graph.spatial_logits),
            }, ensure_ascii=False) + "\n")
            graph_out.flush()

            variants = [{"id": "base", "type": "base", "edge": None}] + [
                {
                    "id": f"drop_{edge['kind']}_{edge['index']}_{edge['source']}_to_{edge['target']}",
                    "type": "drop_edge",
                    "edge": edge,
                }
                for edge in edges
            ]
            print(
                f"[gdesigner-fixed][{adapter.name}] cache {record_key} "
                f"from saved graph variants={len(variants)}",
                flush=True,
            )
            for variant in variants:
                variant_graph = clone_without_edge(fixed_graph, variant["edge"])
                row = await run_variant(variant_graph, adapter, record, record_key, train_flag, variant, dataset_args, run_dir)
                rollout_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                rollout_out.flush()
                completed += 1
                print(
                    f"[gdesigner-fixed][{adapter.name}] {record_key} {variant['id']} "
                    f"correct={row['aggregate']['correct_score']:.3f} "
                    f"tokens={row['aggregate']['usage']['total_tokens']}",
                    flush=True,
                )

    summary = {
        "method": "gdesigner_fixed_graph_counterfactual",
        "dataset": dataset,
        "protocol": "saved_task_graph_leave_one_edge_out",
        "num_rounds": dataset_args.num_rounds,
        "train_examples": len(train_records),
        "total_train_examples": len(all_train_records),
        "shard_index": dataset_args.shard_index,
        "num_shards": dataset_args.num_shards,
        "original_record_indices": original_indices,
        "edge_types": dataset_args.edge_types,
        "edge_limit": dataset_args.edge_limit,
        "samples_per_graph": dataset_args.samples_per_graph,
        "parallel_samples": dataset_args.parallel_samples,
        "saved_graph_dir": str(saved_dataset_dir),
        "missing_graphs": missing_graphs,
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
        "method": "GDesigner fixed saved-graph counterfactual cache",
        "run_dir": str(run_dir),
        "updated_at_utc": utc_now(),
        "datasets": rows,
        "usage": total_usage,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (run_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "train_examples", "rollout_rows", "seconds", "saved_graph_dir", "rollout_file"])
        for row in rows:
            writer.writerow([
                row["dataset"],
                row["train_examples"],
                row["rollout_rows"],
                row["seconds"],
                row["saved_graph_dir"],
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
