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
    load_pretrained_generator,
    refresh_gdesigner_llm_routing,
    set_scope,
    task_fixed_graph,
    usage_snapshot,
)
from ccr.local_entropy_explainer import (  # noqa: E402
    train_causal_local_explainer_from_cache,
    train_local_entropy_explainer_from_cache,
)
from run_causal_explainer_gdesigner import apply_predicted_pruning, predict_scores  # noqa: E402
from run_granger_gdesigner import DatasetAdapter, build_graph, batch_records, summarize_usage, utc_now  # noqa: E402


DEFAULT_PRETRAINED = {
    "mmlu": PROJECT_ROOT / "results" / "gdesigner_trained_counterfactual_40" / "20260608_033934" / "mmlu",
    "gsm8k": PROJECT_ROOT / "results" / "gdesigner_local_entropy_gsm8k_gpu4" / "20260608_053344" / "gsm8k",
    "multiarith": PROJECT_ROOT / "results" / "gdesigner_local_entropy_multiarith_gpu5" / "20260608_053344" / "multiarith",
    "svamp": PROJECT_ROOT / "results" / "gdesigner_local_entropy_svamp_gpu6" / "20260608_053344" / "svamp",
    "aqua": PROJECT_ROOT / "results" / "gdesigner_local_entropy_aqua_gpu7" / "20260608_053344" / "aqua",
    "humaneval": PROJECT_ROOT / "results" / "gdesigner_local_entropy_humaneval_gpu2" / "20260608_053344" / "humaneval",
}
DEFAULT_CACHE_ROOTS = {
    "mmlu": [PROJECT_ROOT / "results" / "gdesigner_mmlu_local_entropy_shards"],
    "gsm8k": [PROJECT_ROOT / "results" / "gdesigner_local_entropy_gsm8k_gpu4" / "20260608_053344"],
    "multiarith": [PROJECT_ROOT / "results" / "gdesigner_local_entropy_multiarith_gpu5" / "20260608_053344"],
    "svamp": [PROJECT_ROOT / "results" / "gdesigner_local_entropy_svamp_gpu6" / "20260608_053344"],
    "aqua": [PROJECT_ROOT / "results" / "gdesigner_local_entropy_aqua_gpu7" / "20260608_053344"],
    "humaneval": [
        PROJECT_ROOT / "results" / "gdesigner_local_entropy_humaneval_gpu2" / "20260608_053344",
        PROJECT_ROOT / "results" / "gdesigner_local_entropy_humaneval_resume_gpu1" / "20260611_072101",
    ],
}
DEFAULT_ROLLOUT_ROOTS = {
    "mmlu": [PROJECT_ROOT / "results" / "gdesigner_fixed_graph_counterfactual_mmlu_shards" / "20260611_121522"],
    "gsm8k": [PROJECT_ROOT / "results" / "gdesigner_fixed_graph_counterfactual_remaining" / "20260611_163926" / "gsm8k_s0_gpu3"],
    "multiarith": [PROJECT_ROOT / "results" / "gdesigner_fixed_graph_counterfactual_remaining" / "20260611_163926" / "multiarith_s0_gpu6"],
    "svamp": [PROJECT_ROOT / "results" / "gdesigner_fixed_graph_counterfactual_remaining" / "20260611_163926" / "svamp_s0_gpu7"],
    "aqua": [PROJECT_ROOT / "results" / "gdesigner_fixed_graph_counterfactual_remaining" / "20260611_163926" / "aqua_s0_gpu4"],
    "humaneval": [
        PROJECT_ROOT / "results" / "gdesigner_fixed_graph_counterfactual_remaining" / "20260611_163926" / "humaneval_s0_gpu2",
        PROJECT_ROOT / "results" / "gdesigner_fixed_graph_counterfactual_remaining" / "20260611_163926" / "humaneval_s1_gpu5",
    ],
}
_ORIGINAL_PRINT = builtins.print


def install_agent_print_filter() -> None:
    if getattr(builtins.print, "_ccr_mmlu_eval_filter", False):
        return

    def filtered_print(*args: Any, **kwargs: Any) -> None:
        if args:
            first = str(args[0])
            if first.startswith("################") or first.startswith("#################"):
                return
        _ORIGINAL_PRINT(*args, **kwargs)

    setattr(filtered_print, "_ccr_mmlu_eval_filter", True)
    builtins.print = filtered_print


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate trained GDesigner on the MMLU-153 subset, then attach a MMLU-only "
            "local-entropy explainer as a post-hoc edge pruner and compare accuracy/token cost."
        )
    )
    parser.add_argument("--dataset", choices=["mmlu", "gsm8k", "multiarith", "svamp", "aqua", "humaneval"], default="mmlu")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results" / "gdesigner_explainer_eval")
    parser.add_argument("--pretrained-cache-dir", type=Path, default=None)
    parser.add_argument("--explainer-cache-roots", nargs="+", type=Path, default=None)
    parser.add_argument("--explainer-rollout-roots", nargs="+", type=Path, default=None)
    parser.add_argument("--llm-name", default="qwen3-8b")
    parser.add_argument("--base-urls", default="http://127.0.0.1:8003/v1,http://127.0.0.1:8005/v1,http://127.0.0.1:8006/v1,http://127.0.0.1:8007/v1,http://127.0.0.1:8008/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--top-p", default="0.95")
    parser.add_argument("--disable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--mode", default="FullConnected", choices=["FullConnected", "Random", "Chain", "Debate", "Layered", "Star", "Mesh"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-rounds", type=int, default=1)
    parser.add_argument("--agent-nums", nargs="+", type=int, default=None, help="Override DatasetAdapter.agent_nums, e.g. --agent-nums 6 for GDesigner MMLU README parity.")
    parser.add_argument("--run-name", default=None, help="Optional output subdirectory name under output-root.")
    parser.add_argument("--mmlu-limit", type=int, default=153)
    parser.add_argument("--eval-limit", type=int, default=0, help="Number of eval examples; 0 means all examples exposed by the adapter.")
    parser.add_argument("--drop-remainder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--logit-threshold", type=float, default=0.0, help="Deterministic GDesigner graph threshold, matching the local-entropy cache builder.")
    parser.add_argument("--pruning-rate", type=float, default=0.25)
    parser.add_argument("--node-pruning-rate", type=float, default=0.0)
    parser.add_argument("--intervention-types", choices=["edges", "nodes", "both"], default="edges")
    parser.add_argument("--min-nodes", type=int, default=2)
    parser.add_argument("--optimized-spatial", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--optimized-temporal", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--trace-code-timeout", type=int, default=2)
    parser.add_argument("--final-code-timeout", type=int, default=100)
    parser.add_argument("--node-timeout", type=int, default=900)
    parser.add_argument("--max-tries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="Passed to DatasetAdapter; keep 0 for the standard MMLU limit path.")
    parser.add_argument("--explainer-hidden-dim", type=int, default=128)
    parser.add_argument("--explainer-epochs", type=int, default=100)
    parser.add_argument("--explainer-lr", type=float, default=1e-3)
    parser.add_argument("--explainer-weight-decay", type=float, default=1e-4)
    parser.add_argument("--explainer-dropout", type=float, default=0.15)
    parser.add_argument("--explainer-batch-size", type=int, default=64)
    parser.add_argument("--explainer-val-ratio", type=float, default=0.2)
    parser.add_argument("--explainer-ranking-weight", type=float, default=0.25)
    parser.add_argument("--explainer-label-source", choices=["local_entropy", "causal_local"], default="local_entropy")
    parser.add_argument("--explainer-label-mode", choices=["pair", "source", "target", "combined"], default="combined")
    parser.add_argument("--explainer-correctness-weight", type=float, default=0.8)
    parser.add_argument("--explainer-entropy-weight", type=float, default=0.2)
    parser.add_argument("--explainer-cost-penalty", type=float, default=0.0)
    parser.add_argument("--explainer-positive-weight", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=888)
    parser.add_argument("--suppress-agent-stdout", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.pretrained_cache_dir is None:
        args.pretrained_cache_dir = DEFAULT_PRETRAINED[args.dataset]
    if args.explainer_cache_roots is None:
        args.explainer_cache_roots = DEFAULT_CACHE_ROOTS[args.dataset]
    if args.explainer_rollout_roots is None and args.explainer_label_source == "causal_local":
        args.explainer_rollout_roots = DEFAULT_ROLLOUT_ROOTS[args.dataset]
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


def add_usage(*items: dict[str, int]) -> dict[str, int]:
    keys = ["requests", "prompt_tokens", "completion_tokens", "total_tokens", "prompt_chars", "response_chars"]
    return {key: sum(int(item.get(key) or 0) for item in items) for key in keys}


async def run_fixed_graph(
    graph: Any,
    adapter: DatasetAdapter,
    record: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    realized_graph = copy.deepcopy(graph)
    input_dict = adapter.input_for(record)
    raw_answer, _log_prob = await realized_graph.arun(
        input_dict,
        args.num_rounds,
        max_tries=args.max_tries,
        max_time=args.node_timeout,
    )
    target = adapter.target_for(record, train=False)
    prediction = adapter.postprocess_final(raw_answer, record)
    correct = adapter.is_correct(prediction, target)
    return {
        "input": input_dict,
        "answer": target,
        "prediction": prediction,
        "correct": bool(correct),
        "raw_answer": raw_answer,
        "trace": getattr(realized_graph, "last_trace", []),
    }


def active_graph_stats(graph: Any) -> dict[str, int]:
    return {
        "nodes": len(graph.nodes),
        "spatial_edges": int((graph.spatial_masks.detach().float() > 0).sum().item()),
        "temporal_edges": int((graph.temporal_masks.detach().float() > 0).sum().item()),
    }


async def evaluate_pair(
    adapter: DatasetAdapter,
    dataset: str,
    generator_graph: Any,
    explainer: Any,
    records: list[Any],
    args: argparse.Namespace,
    run_dir: Path,
) -> dict[str, Any]:
    original_path = run_dir / f"{dataset}_original.jsonl"
    posthoc_path = run_dir / f"{dataset}_posthoc.jsonl"
    mask_path = run_dir / f"{dataset}_posthoc_masks.jsonl"
    comparison_path = run_dir / f"{dataset}_comparison.jsonl"

    original_correct = 0
    posthoc_correct = 0
    total = 0
    edge_stats: list[dict[str, Any]] = []

    with (
        original_path.open("w", encoding="utf-8") as original_out,
        posthoc_path.open("w", encoding="utf-8") as posthoc_out,
        mask_path.open("w", encoding="utf-8") as mask_out,
        comparison_path.open("w", encoding="utf-8") as comparison_out,
    ):
        for i_batch, batch in enumerate(batch_records(records, args.batch_size, drop_remainder=args.drop_remainder and adapter.drop_remainder)):
            print(f"[{dataset}] eval batch {i_batch + 1} size={len(batch)}", flush=True)
            original_tasks = []
            posthoc_tasks = []
            metas = []
            for record in batch:
                input_dict = adapter.input_for(record)
                fixed_graph, gdesigner_meta = task_fixed_graph(generator_graph, input_dict["task"], args.logit_threshold)
                scores = predict_scores(explainer, fixed_graph, input_dict["task"])
                pruned_graph, prune_info = apply_predicted_pruning(fixed_graph, scores, args)
                metas.append((record, input_dict, gdesigner_meta, active_graph_stats(fixed_graph), prune_info))
                set_scope(f"{dataset}:original", run_dir)
                original_tasks.append(run_fixed_graph(fixed_graph, adapter, record, args))
                set_scope(f"{dataset}:posthoc", run_dir)
                posthoc_tasks.append(run_fixed_graph(pruned_graph, adapter, record, args))

            # Scopes are read at request time inside each coroutine, so launch original and posthoc
            # separately to keep usage accounting unambiguous.
            set_scope(f"{dataset}:original", run_dir)
            original_rows = await asyncio.gather(*original_tasks)
            set_scope(f"{dataset}:posthoc", run_dir)
            posthoc_rows = await asyncio.gather(*posthoc_tasks)

            for (record, input_dict, gdesigner_meta, original_graph_stats, prune_info), original_row, posthoc_row in zip(
                metas,
                original_rows,
                posthoc_rows,
            ):
                total += 1
                original_correct += int(original_row["correct"])
                posthoc_correct += int(posthoc_row["correct"])
                original_row["index"] = total - 1
                original_row["running_accuracy"] = original_correct / total
                posthoc_row["index"] = total - 1
                posthoc_row["running_accuracy"] = posthoc_correct / total
                posthoc_row["prune_info"] = prune_info
                original_out.write(json.dumps(original_row, ensure_ascii=False) + "\n")
                posthoc_out.write(json.dumps(posthoc_row, ensure_ascii=False) + "\n")
                mask_out.write(json.dumps({
                    "index": total - 1,
                    "input": input_dict,
                    "gdesigner_meta": gdesigner_meta,
                    "original_graph": original_graph_stats,
                    "posthoc_prune_info": prune_info,
                }, ensure_ascii=False) + "\n")
                comparison_out.write(json.dumps({
                    "index": total - 1,
                    "answer": original_row["answer"],
                    "original_prediction": original_row["prediction"],
                    "original_correct": original_row["correct"],
                    "posthoc_prediction": posthoc_row["prediction"],
                    "posthoc_correct": posthoc_row["correct"],
                    "original_graph": original_graph_stats,
                    "posthoc_graph": {
                        "nodes": prune_info["nodes_active"],
                        "spatial_edges": prune_info["spatial_active"],
                        "temporal_edges": prune_info["temporal_active"],
                    },
                }, ensure_ascii=False) + "\n")
                edge_stats.append({
                    "original_spatial": original_graph_stats["spatial_edges"],
                    "posthoc_spatial": prune_info["spatial_active"],
                    "removed_spatial": len(prune_info.get("removed_spatial_edges") or []),
                    "original_temporal": original_graph_stats["temporal_edges"],
                    "posthoc_temporal": prune_info["temporal_active"],
                    "removed_temporal": len(prune_info.get("removed_temporal_edges") or []),
                })
            original_out.flush()
            posthoc_out.flush()
            mask_out.flush()
            comparison_out.flush()
            print(
                f"[{dataset}] running original={original_correct}/{total}={original_correct / total:.4f} "
                f"posthoc={posthoc_correct}/{total}={posthoc_correct / total:.4f}",
                flush=True,
            )

    original_usage = summarize_usage(run_dir / "usage.jsonl", f"{dataset}:original")
    posthoc_usage = summarize_usage(run_dir / "usage.jsonl", f"{dataset}:posthoc")
    return {
        "total": total,
        "original": {
            "correct": original_correct,
            "accuracy": original_correct / total if total else 0.0,
            "usage": original_usage,
            "result_file": str(original_path),
        },
        "posthoc": {
            "correct": posthoc_correct,
            "accuracy": posthoc_correct / total if total else 0.0,
            "usage": posthoc_usage,
            "result_file": str(posthoc_path),
            "mask_file": str(mask_path),
        },
        "delta": {
            "accuracy": (posthoc_correct - original_correct) / total if total else 0.0,
            "correct": posthoc_correct - original_correct,
            "requests": posthoc_usage["requests"] - original_usage["requests"],
            "total_tokens": posthoc_usage["total_tokens"] - original_usage["total_tokens"],
            "total_tokens_pct": (
                100.0 * (posthoc_usage["total_tokens"] - original_usage["total_tokens"]) / original_usage["total_tokens"]
                if original_usage["total_tokens"]
                else 0.0
            ),
        },
        "edge_stats": {
            "mean_original_spatial": statistics.mean([row["original_spatial"] for row in edge_stats]) if edge_stats else 0.0,
            "mean_posthoc_spatial": statistics.mean([row["posthoc_spatial"] for row in edge_stats]) if edge_stats else 0.0,
            "mean_removed_spatial": statistics.mean([row["removed_spatial"] for row in edge_stats]) if edge_stats else 0.0,
            "mean_original_temporal": statistics.mean([row["original_temporal"] for row in edge_stats]) if edge_stats else 0.0,
            "mean_posthoc_temporal": statistics.mean([row["posthoc_temporal"] for row in edge_stats]) if edge_stats else 0.0,
            "mean_removed_temporal": statistics.mean([row["removed_temporal"] for row in edge_stats]) if edge_stats else 0.0,
        },
        "comparison_file": str(comparison_path),
    }


def write_metrics(run_dir: Path, summary: dict[str, Any]) -> None:
    with (run_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "method",
            "total",
            "correct",
            "accuracy",
            "requests",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "avg_tokens_per_example",
            "mean_spatial_edges",
        ])
        for key in ["original", "posthoc"]:
            row = summary["evaluation"][key]
            usage = row["usage"]
            edge_key = "mean_original_spatial" if key == "original" else "mean_posthoc_spatial"
            writer.writerow([
                "GDesigner" if key == "original" else f"GDesigner+{summary['dataset'].upper()}Explainer",
                summary["evaluation"]["total"],
                row["correct"],
                round(row["accuracy"] * 100, 4),
                usage["requests"],
                usage["prompt_tokens"],
                usage["completion_tokens"],
                usage["total_tokens"],
                round(usage["total_tokens"] / max(summary["evaluation"]["total"], 1), 2),
                round(summary["evaluation"]["edge_stats"][edge_key], 2),
            ])


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

    explainer_dir = run_dir / "explainer_mmlu_only"
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
        "checkpoint_name": f"{args.dataset}_local_entropy_explainer.pt",
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
    started = time.time()
    evaluation = await evaluate_pair(adapter, args.dataset, graph, explainer, records, args, run_dir)

    summary = {
        "method": f"{args.dataset} trained GDesigner + {args.dataset} local-entropy explainer posthoc evaluation",
        "dataset": args.dataset,
        "run_dir": str(run_dir),
        "started_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "seconds": round(time.time() - started, 2),
        "protocol": (
            "mmlu_dev40_trained_gdesigner_val153_eval_deterministic_threshold0_posthoc"
            if args.dataset == "mmlu"
            else f"{args.dataset}_first40_trained_gdesigner_eval_deterministic_threshold0_posthoc"
        ),
        "cost_accounting": {
            "main_comparison": "evaluation-time LLM usage only: original vs posthoc scopes in usage.jsonl",
            "not_included_in_inference_cost": "one-time GDesigner training, local-entropy cache construction, and offline explainer training",
            "training_cache_usage": "see the original cache-building run summaries if amortized training cost is needed",
        },
        "args": vars(args),
        "pretrained_gdesigner": pretrained_info,
        "explainer_train": explainer_info,
        "evaluation": evaluation,
        "usage_total_file": str(run_dir / "usage.jsonl"),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    write_metrics(run_dir, summary)
    print(f"[done] {run_dir}", flush=True)
    print(
        f"[result][{args.dataset}] original={evaluation['original']['correct']}/{evaluation['total']} "
        f"({evaluation['original']['accuracy']:.4f}) "
        f"posthoc={evaluation['posthoc']['correct']}/{evaluation['total']} "
        f"({evaluation['posthoc']['accuracy']:.4f}) "
        f"token_delta={evaluation['delta']['total_tokens_pct']:.2f}%",
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
