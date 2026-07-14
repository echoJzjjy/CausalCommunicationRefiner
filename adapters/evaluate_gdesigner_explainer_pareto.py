#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import copy
import csv
import hashlib
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

from adapters.evaluate_mmlu_gdesigner_explainer import (  # noqa: E402
    DEFAULT_CACHE_ROOTS,
    DEFAULT_PRETRAINED,
    DEFAULT_ROLLOUT_ROOTS,
    active_graph_stats,
    install_agent_print_filter,
    run_fixed_graph,
)
from adapters.gdesigner_trained_counterfactual_builder import (  # noqa: E402
    load_pretrained_generator,
    refresh_gdesigner_llm_routing,
    set_scope,
    task_fixed_graph,
)
from ccr.local_entropy_explainer import (  # noqa: E402
    train_causal_local_explainer_from_cache,
    train_local_entropy_explainer_from_cache,
)
from run_causal_explainer_gdesigner import apply_predicted_pruning, predict_scores  # noqa: E402
from run_granger_gdesigner import DatasetAdapter, batch_records, build_graph, summarize_usage, utc_now  # noqa: E402


DATASETS = ["mmlu", "gsm8k", "multiarith", "svamp", "aqua", "humaneval"]
DEFAULT_BUDGETS = "0.25:0.0,0.50:0.0,0.75:0.0,0.25:0.2,0.50:0.2,0.25:0.4"


def parse_budget_grid(value: str) -> list[tuple[float, float]]:
    budgets: list[tuple[float, float]] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise argparse.ArgumentTypeError("Each budget must be EDGE:NODE, e.g. 0.25:0.2.")
        edge_text, node_text = item.split(":", 1)
        edge_rate = float(edge_text)
        node_rate = float(node_text)
        if not (0.0 <= edge_rate <= 1.0 and 0.0 <= node_rate <= 1.0):
            raise argparse.ArgumentTypeError(f"Budget rates must be in [0,1], got {item!r}.")
        budgets.append((edge_rate, node_rate))
    if not budgets:
        raise argparse.ArgumentTypeError("At least one pruning budget is required.")
    return budgets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pareto sweep for GDesigner + CCR explainer.")
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results" / "gdesigner_explainer_pareto")
    parser.add_argument("--pretrained-cache-dir", type=Path, default=None)
    parser.add_argument("--explainer-cache-roots", nargs="+", type=Path, default=None)
    parser.add_argument("--explainer-rollout-roots", nargs="+", type=Path, default=None)
    parser.add_argument("--llm-name", default="qwen3-8b")
    parser.add_argument("--base-urls", default="http://127.0.0.1:8032/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--top-p", default="0.95")
    parser.add_argument("--disable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--mode", default="FullConnected", choices=["FullConnected", "Random", "Chain", "Debate", "Layered", "Star", "Mesh"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-rounds", type=int, default=None)
    parser.add_argument("--agent-nums", nargs="+", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--mmlu-limit", type=int, default=153)
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1, help="Split eval records into this many deterministic shards.")
    parser.add_argument("--shard-index", type=int, default=0, help="0-based shard index to evaluate.")
    parser.add_argument("--drop-remainder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--logit-threshold", type=float, default=0.0)
    parser.add_argument("--budget-grid", type=parse_budget_grid, default=parse_budget_grid(DEFAULT_BUDGETS))
    parser.add_argument("--intervention-types", choices=["edges", "nodes", "both"], default="both")
    parser.add_argument("--min-nodes", type=int, default=2)
    parser.add_argument("--optimized-spatial", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--optimized-temporal", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--trace-code-timeout", type=int, default=2)
    parser.add_argument("--final-code-timeout", type=int, default=100)
    parser.add_argument("--node-timeout", type=int, default=900)
    parser.add_argument("--max-tries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--explainer-hidden-dim", type=int, default=128)
    parser.add_argument("--explainer-epochs", type=int, default=100)
    parser.add_argument("--explainer-lr", type=float, default=1e-3)
    parser.add_argument("--explainer-weight-decay", type=float, default=1e-4)
    parser.add_argument("--explainer-dropout", type=float, default=0.15)
    parser.add_argument("--explainer-batch-size", type=int, default=64)
    parser.add_argument("--explainer-val-ratio", type=float, default=0.2)
    parser.add_argument("--explainer-ranking-weight", type=float, default=0.25)
    parser.add_argument("--explainer-label-source", choices=["local_entropy", "causal_local"], default="causal_local")
    parser.add_argument("--explainer-label-mode", choices=["pair", "source", "target", "combined"], default="combined")
    parser.add_argument("--explainer-correctness-weight", type=float, default=0.8)
    parser.add_argument("--explainer-entropy-weight", type=float, default=0.2)
    parser.add_argument("--explainer-cost-penalty", type=float, default=0.0)
    parser.add_argument("--explainer-positive-weight", type=float, default=4.0)
    parser.add_argument("--score-mode", choices=["explainer", "random"], default="explainer")
    parser.add_argument("--random-score-seed", type=int, default=None)
    parser.add_argument(
        "--baseline-run-dir",
        type=Path,
        default=None,
        help="Reuse a previously saved GDesigner baseline run_dir that contains {dataset}_fixed_graphs.jsonl and summary.json.",
    )
    parser.add_argument("--seed", type=int, default=888)
    parser.add_argument("--suppress-agent-stdout", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.num_rounds is None:
        args.num_rounds = 2 if args.dataset == "humaneval" else 1
    if args.pretrained_cache_dir is None:
        args.pretrained_cache_dir = DEFAULT_PRETRAINED[args.dataset]
    if args.explainer_cache_roots is None:
        args.explainer_cache_roots = DEFAULT_CACHE_ROOTS[args.dataset]
    if args.explainer_rollout_roots is None:
        args.explainer_rollout_roots = DEFAULT_ROLLOUT_ROOTS[args.dataset]
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1.")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard-index must satisfy 0 <= shard_index < num_shards.")
    if args.random_score_seed is None:
        args.random_score_seed = args.seed
    return args


def configure_env(args: argparse.Namespace, run_dir: Path) -> None:
    os.environ["BASE_URLS"] = args.base_urls
    os.environ["BASE_URL"] = args.base_urls.split(",")[0]
    os.environ["OPENAI_BASE_URL"] = args.base_urls.split(",")[0]
    os.environ["API_KEY"] = args.api_key
    os.environ["TOP_P"] = str(args.top_p)
    os.environ["QWEN_DISABLE_THINKING"] = "1" if args.disable_thinking else "0"
    os.environ["GDESIGNER_USAGE_LOG"] = str(run_dir / "usage.jsonl")
    os.environ["AGENTPRUNE_USAGE_LOG"] = str(run_dir / "usage.jsonl")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_baseline_run_dir(baseline_run_dir: Path, dataset: str) -> Path:
    baseline_run_dir = baseline_run_dir.expanduser().resolve()
    if (baseline_run_dir / f"{dataset}_fixed_graphs.jsonl").exists():
        return baseline_run_dir
    candidates = sorted(baseline_run_dir.glob(f"{dataset}/*/"))
    for candidate in candidates:
        if (candidate / f"{dataset}_fixed_graphs.jsonl").exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find {dataset}_fixed_graphs.jsonl under baseline run dir {baseline_run_dir}"
    )


def load_baseline_cache(
    baseline_run_dir: Path,
    dataset: str,
) -> tuple[Path, list[dict[str, Any]], dict[str, Any] | None]:
    resolved = resolve_baseline_run_dir(baseline_run_dir, dataset)
    graph_rows = load_jsonl(resolved / f"{dataset}_fixed_graphs.jsonl")
    summary_path = resolved / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
    return resolved, graph_rows, summary


def budget_name(edge_rate: float, node_rate: float) -> str:
    return f"e{edge_rate:.2f}_n{node_rate:.2f}".replace(".", "p")


def make_prune_args(args: argparse.Namespace, edge_rate: float, node_rate: float) -> argparse.Namespace:
    out = copy.copy(args)
    out.pruning_rate = edge_rate
    out.node_pruning_rate = node_rate
    return out


def random_scores_for_graph(graph: Any, task: str, seed: int) -> dict[str, dict[str, float]]:
    node_ids = list(graph.nodes.keys())
    spatial_edges = [f"{edge[0]}->{edge[1]}" for edge in graph.potential_spatial_edges]
    temporal_edges = [f"{edge[0]}->{edge[1]}" for edge in graph.potential_temporal_edges]
    payload = "||".join([
        str(seed),
        task,
        "|".join(node_ids),
        "|".join(spatial_edges),
        "|".join(temporal_edges),
    ])
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    rng = random.Random(int(digest[:16], 16))
    return {
        "node": {node_id: rng.random() for node_id in node_ids},
        "spatial": {edge: rng.random() for edge in spatial_edges},
        "temporal": {edge: rng.random() for edge in temporal_edges},
    }


async def evaluate_original(
    adapter: DatasetAdapter,
    dataset: str,
    generator_graph: Any,
    records: list[Any],
    args: argparse.Namespace,
    run_dir: Path,
    fixed_cache: list[dict[str, Any]],
) -> dict[str, Any]:
    result_path = run_dir / f"{dataset}_original.jsonl"
    graph_path = run_dir / f"{dataset}_fixed_graphs.jsonl"
    correct = 0
    total = 0
    graph_stats_rows: list[dict[str, Any]] = []
    with result_path.open("w", encoding="utf-8") as result_out, graph_path.open("w", encoding="utf-8") as graph_out:
        for i_batch, batch in enumerate(batch_records(records, args.batch_size, drop_remainder=args.drop_remainder and adapter.drop_remainder)):
            print(f"[{dataset}][original] batch {i_batch + 1} size={len(batch)}", flush=True)
            tasks = []
            metas = []
            for record in batch:
                input_dict = adapter.input_for(record)
                fixed_graph, gdesigner_meta = task_fixed_graph(generator_graph, input_dict["task"], args.logit_threshold)
                stats = active_graph_stats(fixed_graph)
                metas.append((record, input_dict, fixed_graph, gdesigner_meta, stats))
                tasks.append(run_fixed_graph(fixed_graph, adapter, record, args))
            set_scope(f"{dataset}:original", run_dir)
            rows = await asyncio.gather(*tasks)
            for (record, input_dict, fixed_graph, gdesigner_meta, stats), row in zip(metas, rows):
                row["index"] = total
                correct += int(row["correct"])
                total += 1
                row["running_accuracy"] = correct / total
                result_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                fixed_cache.append({
                    "record": record,
                    "input": input_dict,
                    "graph": fixed_graph,
                    "gdesigner_meta": gdesigner_meta,
                    "graph_stats": stats,
                })
                graph_out.write(json.dumps({
                    "index": total - 1,
                    "input": input_dict,
                    "gdesigner_meta": gdesigner_meta,
                    "graph_stats": stats,
                }, ensure_ascii=False) + "\n")
                graph_stats_rows.append(stats)
            result_out.flush()
            graph_out.flush()
            print(f"[{dataset}][original] running={correct}/{total}={correct / total:.4f}", flush=True)

    usage = summarize_usage(run_dir / "usage.jsonl", f"{dataset}:original")
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "usage": usage,
        "result_file": str(result_path),
        "fixed_graph_file": str(graph_path),
        "edge_stats": {
            "mean_spatial_edges": statistics.mean([row["spatial_edges"] for row in graph_stats_rows]) if graph_stats_rows else 0.0,
            "mean_temporal_edges": statistics.mean([row["temporal_edges"] for row in graph_stats_rows]) if graph_stats_rows else 0.0,
            "mean_nodes": statistics.mean([row["nodes"] for row in graph_stats_rows]) if graph_stats_rows else 0.0,
        },
    }


def build_fixed_cache_from_baseline(
    adapter: DatasetAdapter,
    graph: Any,
    records: list[Any],
    baseline_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if len(baseline_rows) != len(records):
        raise ValueError(
            f"Baseline graph rows ({len(baseline_rows)}) do not match selected eval records ({len(records)}). "
            "Make sure the baseline run dir and the current eval split/shard are aligned."
        )

    fixed_cache: list[dict[str, Any]] = []
    for idx, (record, baseline_row) in enumerate(zip(records, baseline_rows)):
        input_dict = adapter.input_for(record)
        baseline_input = baseline_row.get("input") or {}
        if baseline_input.get("task") != input_dict.get("task"):
            raise ValueError(
                f"Baseline task mismatch at index {idx}: "
                f"{baseline_input.get('task')!r} != {input_dict.get('task')!r}. "
                "The baseline run dir likely does not belong to this dataset split."
            )
        fixed_graph, gdesigner_meta = task_fixed_graph(graph, input_dict["task"], args.logit_threshold)
        stats = active_graph_stats(fixed_graph)
        baseline_stats = baseline_row.get("graph_stats") or {}
        if baseline_stats and stats != baseline_stats:
            raise ValueError(
                f"Baseline graph stats mismatch at index {idx}: {stats!r} != {baseline_stats!r}. "
                "This usually means the current pretrained checkpoint or threshold differs from the baseline run."
            )
        fixed_cache.append({
            "record": record,
            "input": input_dict,
            "graph": fixed_graph,
            "gdesigner_meta": gdesigner_meta,
            "graph_stats": stats,
            "baseline_row": baseline_row,
        })
    return fixed_cache


async def evaluate_budget(
    adapter: DatasetAdapter,
    dataset: str,
    explainer: Any,
    fixed_cache: list[dict[str, Any]],
    args: argparse.Namespace,
    run_dir: Path,
    edge_rate: float,
    node_rate: float,
) -> dict[str, Any]:
    name = budget_name(edge_rate, node_rate)
    result_path = run_dir / f"{dataset}_posthoc_{name}.jsonl"
    mask_path = run_dir / f"{dataset}_masks_{name}.jsonl"
    comparison_path = run_dir / f"{dataset}_comparison_{name}.jsonl"
    prune_args = make_prune_args(args, edge_rate, node_rate)
    scope = f"{dataset}:posthoc:{name}"
    correct = 0
    total = 0
    edge_stats_rows: list[dict[str, Any]] = []
    with (
        result_path.open("w", encoding="utf-8") as result_out,
        mask_path.open("w", encoding="utf-8") as mask_out,
        comparison_path.open("w", encoding="utf-8") as comparison_out,
    ):
        for i_batch, batch in enumerate(batch_records(fixed_cache, args.batch_size, drop_remainder=False)):
            print(f"[{dataset}][{name}] batch {i_batch + 1} size={len(batch)}", flush=True)
            tasks = []
            metas = []
            for item in batch:
                fixed_graph = item["graph"]
                input_dict = item["input"]
                if args.score_mode == "random":
                    scores = random_scores_for_graph(fixed_graph, input_dict["task"], args.random_score_seed)
                else:
                    scores = predict_scores(explainer, fixed_graph, input_dict["task"])
                pruned_graph, prune_info = apply_predicted_pruning(fixed_graph, scores, prune_args)
                metas.append((item, prune_info))
                tasks.append(run_fixed_graph(pruned_graph, adapter, item["record"], args))
            set_scope(scope, run_dir)
            rows = await asyncio.gather(*tasks)
            for (item, prune_info), row in zip(metas, rows):
                edge_stats_rows.append({
                    "original_spatial": item["graph_stats"]["spatial_edges"],
                    "posthoc_spatial": prune_info["spatial_active"],
                    "removed_spatial": len(prune_info.get("removed_spatial_edges") or []),
                    "original_temporal": item["graph_stats"]["temporal_edges"],
                    "posthoc_temporal": prune_info["temporal_active"],
                    "removed_temporal": len(prune_info.get("removed_temporal_edges") or []),
                    "original_nodes": item["graph_stats"]["nodes"],
                    "posthoc_nodes": prune_info["nodes_active"],
                    "removed_nodes": len(prune_info.get("removed_nodes") or []),
                })
            for (item, prune_info), row in zip(metas, rows):
                row["index"] = total
                row["prune_info"] = prune_info
                correct += int(row["correct"])
                total += 1
                row["running_accuracy"] = correct / total
                result_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                mask_out.write(json.dumps({
                    "index": total - 1,
                    "input": item["input"],
                    "original_graph": item["graph_stats"],
                    "posthoc_prune_info": prune_info,
                    "edge_rate": edge_rate,
                    "node_rate": node_rate,
                }, ensure_ascii=False) + "\n")
                comparison_out.write(json.dumps({
                    "index": total - 1,
                    "answer": row["answer"],
                    "posthoc_prediction": row["prediction"],
                    "posthoc_correct": row["correct"],
                    "original_graph": item["graph_stats"],
                    "posthoc_graph": {
                        "nodes": prune_info["nodes_active"],
                        "spatial_edges": prune_info["spatial_active"],
                        "temporal_edges": prune_info["temporal_active"],
                    },
                }, ensure_ascii=False) + "\n")
            result_out.flush()
            mask_out.flush()
            comparison_out.flush()
            print(f"[{dataset}][{name}] running={correct}/{total}={correct / total:.4f}", flush=True)

    usage = summarize_usage(run_dir / "usage.jsonl", scope)
    return {
        "name": name,
        "edge_pruning_rate": edge_rate,
        "node_pruning_rate": node_rate,
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "usage": usage,
        "result_file": str(result_path),
        "mask_file": str(mask_path),
        "comparison_file": str(comparison_path),
        "edge_stats": {
            "mean_original_spatial": statistics.mean([row["original_spatial"] for row in edge_stats_rows]) if edge_stats_rows else 0.0,
            "mean_posthoc_spatial": statistics.mean([row["posthoc_spatial"] for row in edge_stats_rows]) if edge_stats_rows else 0.0,
            "mean_removed_spatial": statistics.mean([row["removed_spatial"] for row in edge_stats_rows]) if edge_stats_rows else 0.0,
            "mean_original_temporal": statistics.mean([row["original_temporal"] for row in edge_stats_rows]) if edge_stats_rows else 0.0,
            "mean_posthoc_temporal": statistics.mean([row["posthoc_temporal"] for row in edge_stats_rows]) if edge_stats_rows else 0.0,
            "mean_removed_temporal": statistics.mean([row["removed_temporal"] for row in edge_stats_rows]) if edge_stats_rows else 0.0,
            "mean_original_nodes": statistics.mean([row["original_nodes"] for row in edge_stats_rows]) if edge_stats_rows else 0.0,
            "mean_posthoc_nodes": statistics.mean([row["posthoc_nodes"] for row in edge_stats_rows]) if edge_stats_rows else 0.0,
            "mean_removed_nodes": statistics.mean([row["removed_nodes"] for row in edge_stats_rows]) if edge_stats_rows else 0.0,
        },
    }


def write_metrics(run_dir: Path, summary: dict[str, Any]) -> None:
    original = summary["original"]
    rows = []
    for posthoc in summary["posthoc"]:
        original_tokens = original["usage"]["total_tokens"]
        posthoc_tokens = posthoc["usage"]["total_tokens"]
        rows.append({
            "dataset": summary["dataset"],
            "budget": posthoc["name"],
            "edge_pruning_rate": posthoc["edge_pruning_rate"],
            "node_pruning_rate": posthoc["node_pruning_rate"],
            "original_accuracy": original["accuracy"],
            "posthoc_accuracy": posthoc["accuracy"],
            "accuracy_delta": posthoc["accuracy"] - original["accuracy"],
            "original_total_tokens": original_tokens,
            "posthoc_total_tokens": posthoc_tokens,
            "token_delta": posthoc_tokens - original_tokens,
            "token_delta_pct": 100.0 * (posthoc_tokens - original_tokens) / original_tokens if original_tokens else 0.0,
            "mean_original_spatial": posthoc["edge_stats"]["mean_original_spatial"],
            "mean_posthoc_spatial": posthoc["edge_stats"]["mean_posthoc_spatial"],
            "mean_original_nodes": posthoc["edge_stats"]["mean_original_nodes"],
            "mean_posthoc_nodes": posthoc["edge_stats"]["mean_posthoc_nodes"],
        })
    with (run_dir / "pareto_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    (run_dir / "pareto_metrics.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


async def main() -> None:
    args = parse_args()
    if args.suppress_agent_stdout:
        install_agent_print_filter()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    run_name = args.run_name or timestamp
    run_dir = (args.output_root / args.dataset / run_name).resolve()
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

    adapter = DatasetAdapter(args.dataset, args)
    if args.agent_nums is not None:
        if len(args.agent_nums) != len(adapter.agent_names):
            raise ValueError("--agent-nums length must match the dataset adapter's agent_names length.")
        adapter.agent_nums = list(args.agent_nums)
    graph = build_graph(adapter, args)
    checkpoint = args.pretrained_cache_dir / "gdesigner_trained_generator.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing GDesigner checkpoint: {checkpoint}")
    pretrained_info = load_pretrained_generator(graph, checkpoint)

    explainer_dir = run_dir / "explainer"
    if args.score_mode == "random":
        explainer = None
        explainer_info = {
            "mode": "random",
            "seed": args.random_score_seed,
            "note": "Random scores reused the same pruning path; no explainer was trained.",
        }
    else:
        train_kwargs = {
            "hidden_dim": args.explainer_hidden_dim,
            "epochs": args.explainer_epochs,
            "lr": args.explainer_lr,
            "weight_decay": args.explainer_weight_decay,
            "dropout": args.explainer_dropout,
            "batch_size": args.explainer_batch_size,
            "val_ratio": args.explainer_val_ratio,
            "ranking_weight": args.explainer_ranking_weight,
            "seed": args.seed,
            "checkpoint_name": f"{args.dataset}_{args.explainer_label_source}_explainer.pt",
        }
        if args.explainer_label_source == "causal_local":
            explainer, explainer_info = train_causal_local_explainer_from_cache(
                output_dir=explainer_dir,
                local_cache_roots=args.explainer_cache_roots,
                rollout_roots=args.explainer_rollout_roots,
                datasets=[args.dataset],
                label_mode=args.explainer_label_mode,
                correctness_weight=args.explainer_correctness_weight,
                entropy_weight=args.explainer_entropy_weight,
                cost_penalty=args.explainer_cost_penalty,
                positive_weight=args.explainer_positive_weight,
                **train_kwargs,
            )
        else:
            explainer, explainer_info = train_local_entropy_explainer_from_cache(
                output_dir=explainer_dir,
                cache_roots=args.explainer_cache_roots,
                datasets=[args.dataset],
                label_mode=args.explainer_label_mode,
                cost_penalty=args.explainer_cost_penalty,
                positive_weight=args.explainer_positive_weight,
                **train_kwargs,
            )

    records = adapter.eval_records[: min(len(adapter.eval_records), args.eval_limit)] if args.eval_limit else list(adapter.eval_records)
    if args.num_shards > 1:
        records = [record for idx, record in enumerate(records) if idx % args.num_shards == args.shard_index]
    started = time.time()
    fixed_cache: list[dict[str, Any]]
    original: dict[str, Any]
    baseline_summary: dict[str, Any] | None = None
    baseline_resolved_dir: Path | None = None
    if args.baseline_run_dir is not None:
        baseline_resolved_dir, baseline_rows, baseline_summary = load_baseline_cache(args.baseline_run_dir, args.dataset)
        if baseline_summary is None or "original" not in baseline_summary:
            raise FileNotFoundError(
                f"Baseline run dir {baseline_resolved_dir} is missing summary.json with an original result block."
            )
        fixed_cache = build_fixed_cache_from_baseline(adapter, graph, records, baseline_rows, args)
        original = baseline_summary["original"]
    else:
        fixed_cache = []
        original = await evaluate_original(adapter, args.dataset, graph, records, args, run_dir, fixed_cache)
    posthoc_rows = []
    for edge_rate, node_rate in args.budget_grid:
        posthoc_rows.append(await evaluate_budget(adapter, args.dataset, explainer, fixed_cache, args, run_dir, edge_rate, node_rate))

    summary = {
        "method": "GDesigner + CausalCommunicationRefiner Pareto sweep",
        "dataset": args.dataset,
        "run_dir": str(run_dir),
        "started_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "seconds": round(time.time() - started, 2),
        "seed": args.seed,
        "random_score_seed": args.random_score_seed,
        "baseline_run_dir": str(baseline_resolved_dir) if baseline_resolved_dir is not None else None,
        "protocol": (
            "mmlu_dev40_trained_gdesigner_val153_eval_fixed_graph_multi_budget_posthoc"
            if args.dataset == "mmlu"
            else f"{args.dataset}_first40_trained_gdesigner_eval_fixed_graph_multi_budget_posthoc"
        ),
        "score_mode": args.score_mode,
        "shard": {
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "records": len(records),
        },
        "cost_accounting": {
            "main_comparison": "evaluation-time LLM usage only: original scope vs each posthoc budget scope in usage.jsonl",
            "not_included_in_inference_cost": "one-time GDesigner training, counterfactual cache construction, and offline explainer training",
        },
        "args": vars(args),
        "baseline": {
            "enabled": args.baseline_run_dir is not None,
            "resolved_run_dir": str(baseline_resolved_dir) if baseline_resolved_dir is not None else None,
            "summary_file": str(baseline_resolved_dir / "summary.json") if baseline_resolved_dir is not None else None,
            "fixed_graph_file": str(baseline_resolved_dir / f"{args.dataset}_fixed_graphs.jsonl") if baseline_resolved_dir is not None else None,
            "original_result_file": str(baseline_resolved_dir / f"{args.dataset}_original.jsonl") if baseline_resolved_dir is not None else None,
        },
        "pretrained_gdesigner": pretrained_info,
        "explainer_train": explainer_info,
        "original": original,
        "posthoc": posthoc_rows,
        "usage_total": summarize_usage(run_dir / "usage.jsonl"),
        "usage_total_file": str(run_dir / "usage.jsonl"),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    write_metrics(run_dir, summary)
    print(f"[done] {run_dir}", flush=True)
    for row in posthoc_rows:
        original_tokens = original["usage"]["total_tokens"]
        posthoc_tokens = row["usage"]["total_tokens"]
        token_delta_pct = 100.0 * (posthoc_tokens - original_tokens) / original_tokens if original_tokens else 0.0
        print(
            f"[result][{args.dataset}][{row['name']}] original={original['accuracy']:.4f} "
            f"posthoc={row['accuracy']:.4f} acc_delta={row['accuracy'] - original['accuracy']:+.4f} "
            f"token_delta={token_delta_pct:+.2f}%",
            flush=True,
        )


if __name__ == "__main__":
    asyncio.run(main())
