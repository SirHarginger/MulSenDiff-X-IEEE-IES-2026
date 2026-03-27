#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_categories_arg(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a detector integrity audit and evaluation-only ablation suite for a frozen shared eval run.",
    )
    parser.add_argument("--eval-run-root", required=True, help="Path to the frozen evaluation run under runs/.")
    parser.add_argument("--checkpoint", default="", help="Optional checkpoint override. Defaults to the checkpoint recorded in the eval run summary.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--run-name", default="detector_audit")
    parser.add_argument("--priority-categories", default="screw,screen,capsule,cotton,zipper,light")
    parser.add_argument("--skip-ablations", action="store_true")
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--max-visualizations", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--device-mode", default="")
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    from src.experiments.detector_audit import run_detector_integrity_audit

    result = run_detector_integrity_audit(
        eval_run_root=args.eval_run_root,
        checkpoint_path=args.checkpoint or None,
        data_root=args.data_root,
        output_root=args.output_root,
        run_name=args.run_name,
        priority_categories=parse_categories_arg(args.priority_categories),
        run_ablations=not args.skip_ablations,
        max_eval_batches=args.max_eval_batches,
        max_visualizations=args.max_visualizations,
        device=args.device,
        device_mode=args.device_mode,
        seed=args.seed,
    )

    print("audit_run_dir:", result["summary"]["audit_run_dir"])
    print("summary:", result["summary"])


if __name__ == "__main__":
    main()
