#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a short evaluation-only descriptor score sweep for a frozen training run.",
    )
    parser.add_argument("--train-run-root", required=True, help="Path to the frozen training run under runs/.")
    parser.add_argument("--checkpoint", default="", help="Optional checkpoint override. Defaults to the run's best checkpoint.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--run-name", default="score_sweep")
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--max-visualizations", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--device-mode", default="")
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    from src.experiments.score_sweep import run_descriptor_score_sweep

    result = run_descriptor_score_sweep(
        train_run_root=args.train_run_root,
        checkpoint_path=args.checkpoint or None,
        data_root=args.data_root,
        output_root=args.output_root,
        run_name=args.run_name,
        max_eval_batches=args.max_eval_batches,
        max_visualizations=args.max_visualizations,
        device=args.device,
        device_mode=args.device_mode,
        seed=args.seed,
    )

    print("score_sweep_run_dir:", result["summary"]["run_dir"])
    print("recommended_variant:", result["summary"]["recommended_variant"])
    print("summary:", result["summary"])


if __name__ == "__main__":
    main()
