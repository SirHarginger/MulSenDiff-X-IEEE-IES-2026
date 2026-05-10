#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.generalization import resolve_held_out_folds, summarize_held_out_generalization
from src.project_layout import default_output_root
from src.utils.logger import create_run_dir, write_history_csv, write_json


def _run_command(command: Sequence[str]) -> None:
    print("$", " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def _snapshot_dir_names(root: Path) -> set[str]:
    if not root.exists():
        return set()
    return {path.name for path in root.iterdir() if path.is_dir()}


def _detect_new_run_dir(output_root: Path, before: set[str]) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    after = sorted((path for path in output_root.iterdir() if path.is_dir()), key=lambda path: path.name)
    new_dirs = [path for path in after if path.name not in before]
    if new_dirs:
        return new_dirs[-1]
    if after:
        return after[-1]
    raise RuntimeError(f"no run directory was created under {output_root}")


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _common_args(args: argparse.Namespace) -> list[str]:
    common = [
        "--config",
        args.config,
        "--data-root",
        args.data_root,
        "--processed-root",
        args.processed_root,
    ]
    if args.device:
        common.extend(["--device", args.device])
    if args.device_mode:
        common.extend(["--device-mode", args.device_mode])
    if args.object_score_strategy:
        common.extend(["--object-score-strategy", args.object_score_strategy])
    if args.enable_internal_defect_gate:
        common.append("--enable-internal-defect-gate")
    if args.seed is not None:
        common.extend(["--seed", str(args.seed)])
    return common


def _run_fold_training_and_eval(
    *,
    args: argparse.Namespace,
    fold_name: str,
    training_categories: Sequence[str],
    held_out_categories: Sequence[str],
) -> dict[str, str]:
    train_label = f"{args.run_name}_{fold_name}_train"
    eval_label = f"{args.run_name}_{fold_name}_unknown"
    train_output_root = default_output_root(
        regime_paper="CCDD",
        run_type="train",
        scope="all",
        label=train_label,
        repo_root=REPO_ROOT,
    )
    eval_output_root = default_output_root(
        regime_paper="CCDD",
        run_type="eval",
        scope="all",
        label=eval_label,
        repo_root=REPO_ROOT,
    )

    before_train = _snapshot_dir_names(train_output_root)
    train_command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_training.py"),
        *_common_args(args),
        "--categories",
        ",".join(training_categories),
        "--run-name",
        train_label,
    ]
    if args.max_train_batches > 0:
        train_command.extend(["--max-train-batches", str(args.max_train_batches)])
    if args.max_eval_batches > 0:
        train_command.extend(["--max-eval-batches", str(args.max_eval_batches)])
    if args.max_visualizations > 0:
        train_command.extend(["--max-visualizations", str(args.max_visualizations)])
    _run_command(train_command)
    train_run_dir = _detect_new_run_dir(train_output_root, before_train)
    checkpoint_path = train_run_dir / "checkpoints" / "best.pt"

    before_eval = _snapshot_dir_names(eval_output_root)
    eval_command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_evaluation.py"),
        *_common_args(args),
        "--checkpoint",
        str(checkpoint_path.resolve().relative_to(REPO_ROOT.resolve())),
        "--categories",
        ",".join(held_out_categories),
        "--unknown-category-inference",
        "--run-name",
        eval_label,
    ]
    if args.max_eval_batches > 0:
        eval_command.extend(["--max-eval-batches", str(args.max_eval_batches)])
    if args.max_visualizations > 0:
        eval_command.extend(["--max-visualizations", str(args.max_visualizations)])
    _run_command(eval_command)
    eval_run_dir = _detect_new_run_dir(eval_output_root, before_eval)

    return {
        "fold": fold_name,
        "training_categories": ",".join(training_categories),
        "held_out_categories": ",".join(held_out_categories),
        "train_run_dir": str(train_run_dir),
        "eval_run_dir": str(eval_run_dir),
        "checkpoint_path": str(checkpoint_path),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run and summarize the 3-fold held-out-category CCDD generalization study.",
    )
    parser.add_argument("--phase", choices=["run", "summary", "full"], default="full")
    parser.add_argument("--folds", default="fold_a,fold_b,fold_c")
    parser.add_argument("--config", default="config/diffusion.yaml")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--processed-root", default="data/processed")
    parser.add_argument("--device", default="")
    parser.add_argument("--device-mode", default="")
    parser.add_argument("--object-score-strategy", default="legacy_raw")
    parser.add_argument("--enable-internal-defect-gate", action="store_true")
    parser.add_argument("--run-name", default="heldout_generalization")
    parser.add_argument("--output-root", default="runs/ccdd/generalization")
    parser.add_argument("--closed-set-eval-run", default="")
    parser.add_argument("--fold-runs-manifest", default="")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--max-visualizations", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    return parser


def _load_fold_manifest(path: Path) -> dict[str, Path]:
    payload: dict[str, Path] = {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    for row in raw:
        payload[str(row["fold"]).strip().lower()] = Path(str(row["eval_run_dir"]))
    return payload


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    folds = resolve_held_out_folds(_parse_csv(args.folds))
    requested_output_root = Path(args.output_root)
    reuse_existing_run = args.phase == "summary" and (requested_output_root / "metrics").exists()
    if reuse_existing_run:
        run_root = requested_output_root
        (run_root / "metrics").mkdir(parents=True, exist_ok=True)
    else:
        run_root = create_run_dir(
            output_root=args.output_root,
            run_name=args.run_name,
            category="heldout_generalization",
        ).root

    fold_rows: list[dict[str, str]] = []
    manifest_path = Path(args.fold_runs_manifest) if args.fold_runs_manifest else run_root / "metrics" / "fold_runs.json"
    if args.phase in {"run", "full"}:
        for fold in folds:
            fold_rows.append(
                _run_fold_training_and_eval(
                    args=args,
                    fold_name=fold.name,
                    training_categories=fold.training_categories,
                    held_out_categories=fold.held_out_categories,
                )
            )
        write_json(manifest_path, fold_rows)
        write_history_csv(run_root / "metrics" / "fold_runs.csv", fold_rows)
    elif manifest_path.exists():
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        fold_rows = [dict(row) for row in raw]

    if args.phase in {"summary", "full"}:
        if not args.closed_set_eval_run:
            raise SystemExit("--closed-set-eval-run is required for the summary phase")
        if not manifest_path.exists():
            raise SystemExit(f"fold manifest not found: {manifest_path}")
        fold_eval_runs = _load_fold_manifest(manifest_path)
        summary = summarize_held_out_generalization(
            closed_set_eval_run=args.closed_set_eval_run,
            fold_eval_runs=fold_eval_runs,
            output_dir=run_root / "metrics" / "summary",
        )
        write_json(
            run_root / "summary.json",
            {
                "phase": args.phase,
                "fold_runs_manifest": str(manifest_path),
                "closed_set_eval_run": args.closed_set_eval_run,
                "folds": [fold.to_dict() for fold in folds],
                "overall_summary": summary["overall_summary"],
            },
        )
        print("generalization_summary_dir:", run_root / "metrics" / "summary")
    else:
        write_json(
            run_root / "summary.json",
            {
                "phase": args.phase,
                "fold_runs_manifest": str(manifest_path),
                "folds": [fold.to_dict() for fold in folds],
            },
        )

    print("generalization_run_dir:", run_root)


if __name__ == "__main__":
    main()
