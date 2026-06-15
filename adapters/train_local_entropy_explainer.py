#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(PROJECT_ROOT))

from ccr.local_entropy_explainer import (  # noqa: E402
    load_causal_local_training_examples,
    load_local_entropy_training_examples,
    train_local_entropy_explainer_from_examples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the CCR explainer offline from GDesigner local-entropy counterfactual caches. "
            "This does not call the LLM."
        )
    )
    parser.add_argument(
        "--cache-roots",
        nargs="+",
        type=Path,
        default=[PROJECT_ROOT / "results"],
        help="Directories or local_entropy.jsonl files to scan.",
    )
    parser.add_argument(
        "--rollout-roots",
        nargs="+",
        type=Path,
        default=None,
        help="Directories or rollouts.jsonl files with base/drop final correctness. Required for --label-source causal_local.",
    )
    parser.add_argument("--datasets", nargs="+", default=None, help="Optional dataset filter.")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results" / "local_entropy_explainer")
    parser.add_argument("--label-source", choices=["local_entropy", "causal_local"], default="local_entropy")
    parser.add_argument("--label-mode", choices=["pair", "source", "target", "combined"], default="combined")
    parser.add_argument("--correctness-weight", type=float, default=0.8)
    parser.add_argument("--entropy-weight", type=float, default=0.2)
    parser.add_argument("--cost-penalty", type=float, default=0.0)
    parser.add_argument("--positive-weight", type=float, default=4.0)
    parser.add_argument("--include-smoke", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--ranking-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=888)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    run_dir = (args.output_root / timestamp).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")

    if args.label_source == "causal_local":
        examples, cache_info = load_causal_local_training_examples(
            local_cache_roots=args.cache_roots,
            rollout_roots=args.rollout_roots,
            datasets=args.datasets,
            label_mode=args.label_mode,
            correctness_weight=args.correctness_weight,
            entropy_weight=args.entropy_weight,
            cost_penalty=args.cost_penalty,
            positive_weight=args.positive_weight,
            include_smoke=args.include_smoke,
        )
    else:
        examples, cache_info = load_local_entropy_training_examples(
            cache_roots=args.cache_roots,
            datasets=args.datasets,
            label_mode=args.label_mode,
            cost_penalty=args.cost_penalty,
            positive_weight=args.positive_weight,
            include_smoke=args.include_smoke,
        )
    if not examples:
        raise SystemExit(f"No explainer examples found under {args.cache_roots}.")

    model, info = train_local_entropy_explainer_from_examples(
        examples,
        output_dir=run_dir,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        ranking_weight=args.ranking_weight,
        seed=args.seed,
    )
    del model
    info["cache"] = cache_info
    info["label_source"] = args.label_source
    info["label_mode"] = args.label_mode
    info["correctness_weight"] = args.correctness_weight
    info["entropy_weight"] = args.entropy_weight
    info["cost_penalty"] = args.cost_penalty
    info["positive_weight"] = args.positive_weight
    (run_dir / "local_entropy_explainer_summary.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[done] trained local entropy explainer on {info['examples']} edges -> {info['model_file']}", flush=True)
    print(f"[metrics] train={info['train_metrics']} val={info['val_metrics']}", flush=True)


if __name__ == "__main__":
    main()
