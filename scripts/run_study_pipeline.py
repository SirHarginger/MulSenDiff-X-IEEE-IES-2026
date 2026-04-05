#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.project_layout import processed_root_ready


def _run_command(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the final study workflow: auto-preprocess if needed, then train and evaluate CCDD, CADD, and CSDD.",
    )
    parser.add_argument("--config", default="config/diffusion.yaml")
    parser.add_argument("--raw-root", default="data/raw/MulSen_AD")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--processed-root", default="data/processed")
    parser.add_argument("--categories", default="")
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--overwrite-preprocess", action="store_true")
    parser.add_argument("--device", default="")
    parser.add_argument("--device-mode", default="")
    parser.add_argument("--object-score-strategy", default="legacy_raw")
    parser.add_argument("--enable-internal-defect-gate", action="store_true")
    parser.add_argument("--run-name", default="main")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--max-visualizations", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--export-models", action="store_true")
    parser.add_argument("--force-model-export", action="store_true")
    parser.add_argument("--regimes", default="ccdd,cadd,csdd")
    return parser


def _should_run_preprocess(args: argparse.Namespace) -> bool:
    if args.skip_preprocess:
        print(f"preprocess skipped by flag for {args.processed_root}", flush=True)
        return False
    if args.overwrite_preprocess:
        print(f"preprocess forced for {args.processed_root}", flush=True)
        return True
    if processed_root_ready(args.processed_root):
        print(
            f"processed dataset already present at {args.processed_root}; skipping preprocess "
            "(use --overwrite-preprocess to rebuild)",
            flush=True,
        )
        return False
    print(f"processed dataset not found or incomplete at {args.processed_root}; running preprocess", flush=True)
    return True


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    regimes = [item.strip().lower() for item in args.regimes.split(",") if item.strip()]
    if not regimes:
        raise SystemExit("no regimes specified")

    if _should_run_preprocess(args):
        preprocess_command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_data_pipeline.py"),
            "--raw-root",
            args.raw_root,
            "--data-root",
            args.data_root,
            "--processed-root",
            args.processed_root,
        ]
        if args.categories:
            preprocess_command.extend(["--categories", args.categories])
        if args.overwrite_preprocess:
            preprocess_command.append("--overwrite")
        _run_command(preprocess_command)

    for regime in regimes:
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_regime_pipeline.py"),
            "--regime",
            regime,
            "--config",
            args.config,
            "--raw-root",
            args.raw_root,
            "--data-root",
            args.data_root,
            "--processed-root",
            args.processed_root,
            "--skip-preprocess",
            "--run-name",
            args.run_name,
            "--object-score-strategy",
            args.object_score_strategy,
        ]
        if args.categories:
            command.extend(["--categories", args.categories])
        if args.device:
            command.extend(["--device", args.device])
        if args.device_mode:
            command.extend(["--device-mode", args.device_mode])
        if args.enable_internal_defect_gate:
            command.append("--enable-internal-defect-gate")
        if args.max_train_batches > 0:
            command.extend(["--max-train-batches", str(args.max_train_batches)])
        if args.max_eval_batches > 0:
            command.extend(["--max-eval-batches", str(args.max_eval_batches)])
        if args.max_visualizations > 0:
            command.extend(["--max-visualizations", str(args.max_visualizations)])
        if args.seed is not None:
            command.extend(["--seed", str(args.seed)])
        if args.export_models:
            command.append("--export-model")
        if args.force_model_export:
            command.append("--force-model-export")
        _run_command(command)


if __name__ == "__main__":
    main()
