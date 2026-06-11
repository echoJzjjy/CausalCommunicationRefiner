#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score an AgentPrune counterfactual rollout cache without re-running LLM calls."
    )
    parser.add_argument("--cache-run", type=Path, required=True, help="Run directory created by agentprune_counterfactual_builder.py.")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--output-name", default="scored_contributions")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta-confidence", type=float, default=0.0)
    parser.add_argument("--lambda-entropy", type=float, default=0.05)
    parser.add_argument("--lambda-cost", type=float, default=1e-5)
    parser.add_argument("--cost-field", choices=["total_tokens", "requests", "completion_tokens"], default="total_tokens")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def utility(row: dict[str, Any], args: argparse.Namespace) -> float:
    aggregate = row["aggregate"]
    correct = float(aggregate.get("correct_score") or 0.0)
    entropy = float(aggregate.get("semantic_entropy") or 0.0)
    confidence = float(aggregate.get("confidence") or math.exp(-entropy))
    usage = aggregate.get("usage") or {}
    cost = float(usage.get(args.cost_field) or 0.0)
    return (
        args.alpha * correct
        + args.beta_confidence * confidence
        - args.lambda_entropy * entropy
        - args.lambda_cost * cost
    )


def score_dataset(dataset_dir: Path, args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rollout_path = dataset_dir / "rollouts.jsonl"
    rows = read_jsonl(rollout_path)
    by_example: dict[str, dict[str, Any]] = {}
    for row in rows:
        by_example.setdefault(row["example_id"], {})[row["variant"]["id"]] = row

    scored = []
    for example_id, variants in by_example.items():
        base = variants.get("base")
        if base is None:
            continue
        base_u = utility(base, args)
        for variant_id, row in variants.items():
            if variant_id == "base":
                continue
            u = utility(row, args)
            edge = row["variant"].get("edge") or {}
            scored.append({
                "dataset": row["dataset"],
                "example_id": example_id,
                "edge_kind": edge.get("kind"),
                "edge_index": edge.get("index"),
                "source": edge.get("source"),
                "target": edge.get("target"),
                "base_utility": base_u,
                "drop_utility": u,
                "contribution": base_u - u,
                "base_correct_score": base["aggregate"].get("correct_score"),
                "drop_correct_score": row["aggregate"].get("correct_score"),
                "base_entropy": base["aggregate"].get("semantic_entropy"),
                "drop_entropy": row["aggregate"].get("semantic_entropy"),
                "base_cost": (base["aggregate"].get("usage") or {}).get(args.cost_field),
                "drop_cost": (row["aggregate"].get("usage") or {}).get(args.cost_field),
                "base_clusters": base["aggregate"].get("prediction_clusters"),
                "drop_clusters": row["aggregate"].get("prediction_clusters"),
            })

    mean_contribution = sum(item["contribution"] for item in scored) / len(scored) if scored else 0.0
    summary = {
        "dataset": dataset_dir.name,
        "rollout_file": str(rollout_path),
        "examples": len(by_example),
        "edges_scored": len(scored),
        "mean_contribution": mean_contribution,
    }
    return scored, summary


def write_outputs(run_dir: Path, scored: list[dict[str, Any]], summaries: list[dict[str, Any]], args: argparse.Namespace) -> None:
    out_jsonl = run_dir / f"{args.output_name}.jsonl"
    out_csv = run_dir / f"{args.output_name}.csv"
    out_summary = run_dir / f"{args.output_name}_summary.json"

    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in scored:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    fields = [
        "dataset",
        "example_id",
        "edge_kind",
        "edge_index",
        "source",
        "target",
        "base_utility",
        "drop_utility",
        "contribution",
        "base_correct_score",
        "drop_correct_score",
        "base_entropy",
        "drop_entropy",
        "base_cost",
        "drop_cost",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in scored:
            writer.writerow({key: row.get(key) for key in fields})

    out_summary.write_text(json.dumps({
        "cache_run": str(run_dir),
        "output_jsonl": str(out_jsonl),
        "output_csv": str(out_csv),
        "reward": {
            "formula": "alpha*CorrectScore + beta_confidence*Confidence - lambda_entropy*H_sem - lambda_cost*Cost",
            "alpha": args.alpha,
            "beta_confidence": args.beta_confidence,
            "lambda_entropy": args.lambda_entropy,
            "lambda_cost": args.lambda_cost,
            "cost_field": args.cost_field,
        },
        "datasets": summaries,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = args.cache_run.resolve()
    if args.datasets is None:
        datasets = sorted(path.name for path in run_dir.iterdir() if (path / "rollouts.jsonl").exists())
    else:
        datasets = args.datasets

    all_scored = []
    summaries = []
    for dataset in datasets:
        scored, summary = score_dataset(run_dir / dataset, args)
        all_scored.extend(scored)
        summaries.append(summary)
    write_outputs(run_dir, all_scored, summaries, args)
    print(f"[done] scored {len(all_scored)} counterfactual edges under {run_dir}", flush=True)


if __name__ == "__main__":
    main()
