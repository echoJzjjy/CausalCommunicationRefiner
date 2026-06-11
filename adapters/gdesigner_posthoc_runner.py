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

from adapters.omni_math_adapter import OmniMathAdapter  # noqa: E402
from run_causal_explainer_gdesigner import (  # noqa: E402
    collect_labels_loo,
    collect_labels_sampling,
    evaluate_with_explainer,
    normalize_label_examples,
    train_explainer,
)
from run_causal_gdesigner import build_static_graph, evaluate_frozen, split_records  # noqa: E402
from run_granger_gdesigner import DatasetAdapter, configure_env, summarize_usage, utc_now  # noqa: E402


GENERATOR_TO_MODE = {
    "complete": ("Complete", "FullConnected"),
    "fullconnected": ("Complete", "FullConnected"),
    "full_connected": ("Complete", "FullConnected"),
    "random": ("Random", "Random"),
    "layer": ("Layer", "Layered"),
    "layered": ("Layer", "Layered"),
    "chain": ("Chain", "Chain"),
    "star": ("Star", "Star"),
    "mesh": ("Mesh", "Mesh"),
    "debate": ("Debate", "Debate"),
}

SUPPORTED_DATASETS = ["mmlu", "gsm8k", "multiarith", "svamp", "aqua", "humaneval", "omni"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Causal Communication Refiner as a post-hoc processor over GDesigner-style graph generators."
    )
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON config file. CLI arguments override config values.")
    parser.add_argument("--datasets", nargs="+", choices=SUPPORTED_DATASETS, default=["humaneval"])
    parser.add_argument("--generators", nargs="+", default=["Complete", "Random", "Layer", "Chain", "Star"])
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results" / "gdesigner_posthoc")
    parser.add_argument("--llm-name", default="qwen3-8b")
    parser.add_argument("--base-urls", default="http://127.0.0.1:8003/v1,http://127.0.0.1:8004/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--top-p", default="0.95")
    parser.add_argument("--disable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-rounds", type=int, default=2)
    parser.add_argument("--calibration-examples", type=int, default=8)
    parser.add_argument("--estimator", choices=["loo", "mask_sampling"], default="mask_sampling")
    parser.add_argument("--mask-samples", type=int, default=12)
    parser.add_argument("--edge-keep-prob", type=float, default=0.75)
    parser.add_argument("--node-keep-prob", type=float, default=0.8)
    parser.add_argument("--intervention-types", choices=["edges", "nodes", "both"], default="both")
    parser.add_argument("--pruning-rate", type=float, default=0.25)
    parser.add_argument("--node-pruning-rate", type=float, default=0.2)
    parser.add_argument("--min-nodes", type=int, default=2)
    parser.add_argument("--parallel-interventions", type=int, default=1)
    parser.add_argument("--mmlu-limit", type=int, default=153)
    parser.add_argument("--trace-code-timeout", type=int, default=2)
    parser.add_argument("--final-code-timeout", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--omni-local-path", type=Path, default=None, help="Optional local Omni-MATH-512 snapshot for offline runs.")
    parser.add_argument("--eval-includes-calibration", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--evaluate-original", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--explainer-hidden-dim", type=int, default=128)
    parser.add_argument("--explainer-epochs", type=int, default=200)
    parser.add_argument("--explainer-lr", type=float, default=1e-3)
    parser.add_argument("--explainer-weight-decay", type=float, default=1e-4)
    parser.add_argument("--explainer-dropout", type=float, default=0.15)
    parser.add_argument("--explainer-training-source", choices=["local_entropy", "online_labels"], default="local_entropy")
    parser.add_argument("--explainer-cache-roots", nargs="+", type=Path, default=None)
    parser.add_argument("--explainer-cache-auto", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--explainer-label-mode", choices=["pair", "source", "target", "combined"], default="combined")
    parser.add_argument("--explainer-cost-penalty", type=float, default=0.0)
    parser.add_argument("--explainer-positive-weight", type=float, default=4.0)
    parser.add_argument("--explainer-include-smoke", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--explainer-batch-size", type=int, default=64)
    parser.add_argument("--explainer-val-ratio", type=float, default=0.2)
    parser.add_argument("--explainer-ranking-weight", type=float, default=0.25)
    parser.add_argument("--sparsity-weight", type=float, default=0.02)
    parser.add_argument("--entropy-weight", type=float, default=0.001)
    parser.add_argument("--mask-scope", choices=["task", "global"], default="task")
    parser.add_argument("--seed", type=int, default=888)
    defaults = vars(parser.parse_args([]))
    args = parser.parse_args()
    if args.config is None:
        return args

    with args.config.open(encoding="utf-8") as f:
        config = json.load(f)

    cli_args = parser.parse_args()
    merged = argparse.Namespace(**defaults)
    for key, value in config.items():
        if not hasattr(merged, key):
            raise ValueError(f"Unknown config key: {key}")
        setattr(merged, key, value)

    for key, default_value in defaults.items():
        cli_value = getattr(cli_args, key)
        if key == "config":
            setattr(merged, key, cli_value)
        elif cli_value != default_value:
            setattr(merged, key, cli_value)
    return merged


def build_adapter(dataset_name: str, args: argparse.Namespace) -> Any:
    if dataset_name == "omni":
        return OmniMathAdapter(args, local_path=args.omni_local_path)
    return DatasetAdapter(dataset_name, args)


def normalize_generator(name: str) -> tuple[str, str]:
    key = name.strip().replace("-", "_").replace(" ", "_").lower()
    if key not in GENERATOR_TO_MODE:
        choices = ", ".join(sorted({value[0] for value in GENERATOR_TO_MODE.values()}))
        raise ValueError(f"Unknown generator {name!r}. Supported generators: {choices}.")
    return GENERATOR_TO_MODE[key]


def stable_seed(base_seed: int, *parts: str) -> int:
    digest = hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()
    return base_seed + int(digest[:8], 16) % 1_000_000


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)


def set_usage_scope(scope: str) -> None:
    os.environ["GDESIGNER_USAGE_SCOPE"] = scope
    os.environ["AGENTPRUNE_USAGE_SCOPE"] = scope


def add_usages(*items: dict[str, int]) -> dict[str, int]:
    keys = ["requests", "prompt_tokens", "completion_tokens", "total_tokens", "prompt_chars", "response_chars"]
    return {key: sum(int(item.get(key) or 0) for item in items) for key in keys}


def graph_stats(graph: Any) -> dict[str, int]:
    return {
        "nodes": len(graph.nodes),
        "spatial_edges": int((graph.spatial_masks > 0).sum().item()),
        "temporal_edges": int((graph.temporal_masks > 0).sum().item()),
    }


def generator_args(args: argparse.Namespace, mode: str) -> argparse.Namespace:
    out = copy.copy(args)
    out.mode = mode
    return out


async def collect_causal_labels(
    adapter: DatasetAdapter,
    graph: Any,
    calibration_records: list[Any],
    calibration_train: bool,
    args: argparse.Namespace,
    rng: random.Random,
    label_path: Path,
    intervention_path: Path,
) -> list[dict[str, Any]]:
    label_examples: list[dict[str, Any]] = []
    with label_path.open("w", encoding="utf-8") as label_out:
        for idx, record in enumerate(calibration_records):
            print(f"[{adapter.name}] Causal label {idx + 1}/{len(calibration_records)}", flush=True)
            if args.estimator == "loo":
                label = await collect_labels_loo(graph, adapter, record, args, calibration_train, intervention_path)
            else:
                label = await collect_labels_sampling(graph, adapter, record, args, calibration_train, rng, intervention_path)
            label_examples.append(label)
            label_out.write(json.dumps(label, ensure_ascii=False) + "\n")
            label_out.flush()
    return label_examples


async def run_generator_dataset(
    adapter: DatasetAdapter,
    generator_name: str,
    mode: str,
    args: argparse.Namespace,
    run_dir: Path,
) -> dict[str, Any]:
    started_at = utc_now()
    started = time.time()
    seed = stable_seed(args.seed, adapter.name, generator_name)
    seed_everything(seed)
    gen_args = generator_args(args, mode)
    generator_dir = run_dir / generator_name / adapter.name
    generator_dir.mkdir(parents=True, exist_ok=True)
    graph = build_static_graph(adapter, gen_args)
    initial_graph = graph_stats(graph)
    calibration_records, evaluation_records, calibration_train = split_records(adapter, gen_args)
    label_path = generator_dir / f"{adapter.name}_causal_labels.jsonl"
    normalized_path = generator_dir / f"{adapter.name}_causal_labels_normalized.jsonl"
    intervention_path = generator_dir / f"{adapter.name}_interventions.jsonl"
    result_path = generator_dir / f"{adapter.name}.jsonl"
    mask_path = generator_dir / f"{adapter.name}_predicted_masks.jsonl"
    original_path = generator_dir / f"{adapter.name}_original.jsonl"
    usage_path = run_dir / "usage.jsonl"
    scope_prefix = f"{generator_name}:{adapter.name}"

    print(
        f"[{adapter.name}][{generator_name}] start mode={mode} initial={initial_graph} "
        f"calibration={len(calibration_records)} eval={len(evaluation_records)}",
        flush=True,
    )

    original_info: dict[str, Any] | None = None
    if args.evaluate_original:
        set_usage_scope(f"{scope_prefix}:original")
        original_correct, original_total = await evaluate_frozen(adapter, graph, evaluation_records, gen_args, original_path)
        original_info = {
            "total": original_total,
            "correct": original_correct,
            "accuracy": original_correct / original_total if original_total else 0.0,
            "result_file": str(original_path),
            "usage": summarize_usage(usage_path, f"{scope_prefix}:original"),
        }

    if getattr(args, "explainer_training_source", "local_entropy") == "local_entropy":
        label_examples = []
        for record in calibration_records:
            input_dict = adapter.train_input_for(record) if calibration_train else adapter.input_for(record)
            label_examples.append({"task": input_dict["task"], "targets": {}, "weights": {}})
        label_path.write_text("", encoding="utf-8")
        intervention_path.write_text("", encoding="utf-8")
        label_min, label_max = 0.0, 1.0
    else:
        set_usage_scope(f"{scope_prefix}:calibration")
        label_examples = await collect_causal_labels(
            adapter,
            graph,
            calibration_records,
            calibration_train,
            gen_args,
            random.Random(seed),
            label_path,
            intervention_path,
        )
        label_min, label_max = normalize_label_examples(label_examples)
    with normalized_path.open("w", encoding="utf-8") as out:
        for label in label_examples:
            out.write(json.dumps(label, ensure_ascii=False) + "\n")

    explainer, train_info = train_explainer(graph, label_examples, gen_args, generator_dir, adapter.name)

    set_usage_scope(f"{scope_prefix}:posthoc")
    correct, total = await evaluate_with_explainer(
        adapter,
        graph,
        explainer,
        label_examples,
        evaluation_records,
        gen_args,
        result_path,
        mask_path,
    )

    calibration_usage = summarize_usage(usage_path, f"{scope_prefix}:calibration")
    posthoc_usage = summarize_usage(usage_path, f"{scope_prefix}:posthoc")
    original_usage = original_info["usage"] if original_info else {}
    total_usage = add_usages(calibration_usage, posthoc_usage, original_usage)

    protocol = "mmlu_dev_causal_labels_val153_posthoc_eval" if adapter.name == "mmlu" else "prefix_causal_labels_posthoc_eval"
    if adapter.name == "omni":
        protocol = "omni_train_split_prefix_causal_labels_posthoc_eval"

    return {
        "generator": generator_name,
        "generator_mode": mode,
        "dataset": adapter.name,
        "protocol": protocol,
        "note": adapter.note,
        "started_at_utc": started_at,
        "ended_at_utc": utc_now(),
        "seconds": round(time.time() - started, 2),
        "seed": seed,
        "initial_graph": initial_graph,
        "calibration_examples": len(calibration_records),
        "label_min": label_min,
        "label_max": label_max,
        "explainer_train": train_info,
        "original": original_info,
        "posthoc": {
            "total": total,
            "correct": correct,
            "accuracy": correct / total if total else 0.0,
            "result_file": str(result_path),
            "predicted_mask_file": str(mask_path),
        },
        "usage": {
            "calibration": calibration_usage,
            "posthoc": posthoc_usage,
            "original": original_info["usage"] if original_info else None,
            "total": total_usage,
        },
        "files": {
            "label_file": str(label_path),
            "normalized_label_file": str(normalized_path),
            "intervention_log": str(intervention_path),
            "generator_dir": str(generator_dir),
        },
    }


def write_summary(run_dir: Path, args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    dataset_order = ["mmlu", "gsm8k", "multiarith", "svamp", "aqua", "omni", "humaneval"]
    generators: list[str] = []
    for row in rows:
        if row["generator"] not in generators:
            generators.append(row["generator"])
    posthoc_accs = [row["posthoc"]["accuracy"] for row in rows if row["posthoc"]["total"]]
    total_usage = add_usages(*(row["usage"]["total"] for row in rows))
    summary = {
        "method": "CausalCommunicationRefiner",
        "role": "generator-agnostic audit/refine/compress post-processor",
        "run_dir": str(run_dir),
        "updated_at_utc": utc_now(),
        "llm_name": args.llm_name,
        "disable_thinking": args.disable_thinking,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "generators": args.generators,
        "datasets_requested": args.datasets,
        "num_rounds": args.num_rounds,
        "batch_size": args.batch_size,
        "calibration_examples": args.calibration_examples,
        "estimator": args.estimator,
        "mask_samples": args.mask_samples,
        "intervention_types": args.intervention_types,
        "pruning_rate": args.pruning_rate,
        "node_pruning_rate": args.node_pruning_rate,
        "explainer_hidden_dim": args.explainer_hidden_dim,
        "explainer_epochs": args.explainer_epochs,
        "mask_scope": args.mask_scope,
        "evaluate_original": args.evaluate_original,
        "rows": rows,
        "totals": {
            "average_posthoc_accuracy": statistics.mean(posthoc_accs) if posthoc_accs else 0.0,
            "total_seconds": round(sum(float(row["seconds"]) for row in rows), 2),
            "usage": total_usage,
        },
        "caveat": (
            "Ground-truth labels are not used for test-time graph refinement. "
            "The frozen explainer predicts masks from task/role graph features after calibration."
        ),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    by_pair = {(row["generator"], row["dataset"]): row for row in rows}
    with (run_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Generator"] + dataset_order + ["Avg."])
        for generator in generators:
            values = []
            accs = []
            for dataset in dataset_order:
                row = by_pair.get((generator, dataset))
                if row is None:
                    values.append("")
                    continue
                acc = row["posthoc"]["accuracy"] * 100
                values.append(round(acc, 2))
                accs.append(acc)
            writer.writerow([generator] + values + [round(statistics.mean(accs), 2) if accs else ""])
    with (run_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Generator",
            "Dataset",
            "Mode",
            "InitialNodes",
            "InitialSpatialEdges",
            "InitialTemporalEdges",
            "CalibrationExamples",
            "PostHocTotal",
            "PostHocCorrect",
            "PostHocAccuracy",
            "OriginalAccuracy",
            "Seconds",
            "Requests",
            "TotalTokens",
            "ResultFile",
            "MaskFile",
        ])
        for row in rows:
            original = row.get("original")
            writer.writerow([
                row["generator"],
                row["dataset"],
                row["generator_mode"],
                row["initial_graph"]["nodes"],
                row["initial_graph"]["spatial_edges"],
                row["initial_graph"]["temporal_edges"],
                row["calibration_examples"],
                row["posthoc"]["total"],
                row["posthoc"]["correct"],
                round(row["posthoc"]["accuracy"] * 100, 2),
                round(original["accuracy"] * 100, 2) if original else "",
                row["seconds"],
                row["usage"]["total"]["requests"],
                row["usage"]["total"]["total_tokens"],
                row["posthoc"]["result_file"],
                row["posthoc"]["predicted_mask_file"],
            ])


async def main() -> None:
    args = parse_args()
    normalized = [normalize_generator(name) for name in args.generators]
    args.generators = [name for name, _mode in normalized]
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    run_dir = (args.output_root / timestamp).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    configure_env(args, run_dir)
    (run_dir / "args.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")

    import GDesigner.agents  # noqa: F401
    import GDesigner.llm  # noqa: F401
    import GDesigner.prompt  # noqa: F401
    from GDesigner.llm.llm import LLM

    LLM.DEFAULT_MAX_TOKENS = args.max_tokens
    LLM.DEFAULT_TEMPERATURE = args.temperature

    rows: list[dict[str, Any]] = []
    for dataset_name in args.datasets:
        adapter = build_adapter(dataset_name, args)
        for generator_name, mode in normalized:
            row = await run_generator_dataset(adapter, generator_name, mode, args, run_dir)
            rows.append(row)
            write_summary(run_dir, args, rows)
            print(
                f"[{dataset_name}][{generator_name}] done "
                f"acc={row['posthoc']['correct']}/{row['posthoc']['total']} seconds={row['seconds']}",
                flush=True,
            )
    print(f"[done] {run_dir / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
