#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.category_policies import ALL_CATEGORIES
from src.project_layout import default_output_root, processed_root_ready, resolve_scope

REGIME_PAPERS = {
    "ccdd": "CCDD",
    "cadd": "CADD",
    "csdd": "CSDD",
}


def parse_categories_arg(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


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


def _common_command_args(args: argparse.Namespace) -> list[str]:
    common = [
        "--config",
        args.config,
        "--data-root",
        args.data_root,
        "--processed-root",
        args.processed_root,
        "--run-name",
        args.run_name,
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


def _build_preprocess_command(args: argparse.Namespace) -> list[str]:
    command = [
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
        command.extend(["--categories", args.categories])
    if args.overwrite_preprocess:
        command.append("--overwrite")
    return command


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


def _build_train_command(
    *,
    args: argparse.Namespace,
    regime: str,
    output_root: Path,
    category: str = "",
    categories: Sequence[str] | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_training.py"),
        * _common_command_args(args),
        "--output-root",
        str(output_root),
    ]
    if args.max_train_batches > 0:
        command.extend(["--max-train-batches", str(args.max_train_batches)])
    if args.max_eval_batches > 0:
        command.extend(["--max-eval-batches", str(args.max_eval_batches)])
    if args.max_visualizations > 0:
        command.extend(["--max-visualizations", str(args.max_visualizations)])
    if regime == "cadd":
        command.append("--disable-category-embedding")
    if regime in {"ccdd", "cadd"}:
        requested_categories = list(categories or ["all"])
        command.extend(["--categories", ",".join(requested_categories)])
    else:
        command.extend(["--category", category])
    return command


def _build_eval_command(
    *,
    args: argparse.Namespace,
    regime: str,
    output_root: Path,
    checkpoint_path: Path,
    category: str = "",
    categories: Sequence[str] | None = None,
) -> list[str]:
    relative_checkpoint = checkpoint_path.resolve().relative_to(REPO_ROOT.resolve())
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_evaluation.py"),
        * _common_command_args(args),
        "--checkpoint",
        str(relative_checkpoint),
        "--output-root",
        str(output_root),
    ]
    if args.max_eval_batches > 0:
        command.extend(["--max-eval-batches", str(args.max_eval_batches)])
    if args.max_visualizations > 0:
        command.extend(["--max-visualizations", str(args.max_visualizations)])
    if regime == "cadd":
        command.append("--disable-category-embedding")
    if regime in {"ccdd", "cadd"}:
        requested_categories = list(categories or ["all"])
        command.extend(["--categories", ",".join(requested_categories)])
    else:
        command.extend(["--category", category])
    return command


def _export_bundle(eval_run_dir: Path, regime: str, *, force: bool) -> None:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "export_model_bundle.py"),
        "--eval-run",
        str(eval_run_dir),
        "--regime",
        regime,
    ]
    if force:
        command.append("--force")
    _run_command(command)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run train -> eval for one final study regime, with preprocess only when needed.",
    )
    parser.add_argument("--regime", required=True, choices=["ccdd", "cadd", "csdd"])
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
    parser.add_argument("--export-model", action="store_true")
    parser.add_argument("--force-model-export", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if _should_run_preprocess(args):
        _run_command(_build_preprocess_command(args))

    requested_categories = parse_categories_arg(args.categories) if args.categories else []
    if any(category.lower() == "all" for category in requested_categories):
        selected_categories = list(ALL_CATEGORIES)
    else:
        selected_categories = requested_categories or list(ALL_CATEGORIES)
    regime_paper = REGIME_PAPERS[args.regime]

    if args.regime in {"ccdd", "cadd"}:
        scope = resolve_scope(selected_categories, category="shared")
        train_output_root = default_output_root(
            regime_paper=regime_paper,
            run_type="train",
            scope=scope,
            label=args.run_name,
            repo_root=REPO_ROOT,
        )
        eval_output_root = default_output_root(
            regime_paper=regime_paper,
            run_type="eval",
            scope=scope,
            label=args.run_name,
            repo_root=REPO_ROOT,
        )
        before_train = _snapshot_dir_names(train_output_root)
        _run_command(
            _build_train_command(
                args=args,
                regime=args.regime,
                output_root=train_output_root,
                categories=selected_categories,
            )
        )
        train_run_dir = _detect_new_run_dir(train_output_root, before_train)
        checkpoint_path = train_run_dir / "checkpoints" / "best.pt"

        before_eval = _snapshot_dir_names(eval_output_root)
        _run_command(
            _build_eval_command(
                args=args,
                regime=args.regime,
                output_root=eval_output_root,
                checkpoint_path=checkpoint_path,
                categories=selected_categories,
            )
        )
        eval_run_dir = _detect_new_run_dir(eval_output_root, before_eval)
        print("train_run_dir:", train_run_dir)
        print("eval_run_dir:", eval_run_dir)
        if args.export_model:
            _export_bundle(eval_run_dir, args.regime, force=args.force_model_export)
        return

    for category_name in selected_categories:
        scope = resolve_scope([category_name], category=category_name)
        train_output_root = default_output_root(
            regime_paper=regime_paper,
            run_type="train",
            scope=scope,
            label=args.run_name,
            repo_root=REPO_ROOT,
        )
        eval_output_root = default_output_root(
            regime_paper=regime_paper,
            run_type="eval",
            scope=scope,
            label=args.run_name,
            repo_root=REPO_ROOT,
        )

        before_train = _snapshot_dir_names(train_output_root)
        _run_command(
            _build_train_command(
                args=args,
                regime=args.regime,
                output_root=train_output_root,
                category=category_name,
            )
        )
        train_run_dir = _detect_new_run_dir(train_output_root, before_train)
        checkpoint_path = train_run_dir / "checkpoints" / "best.pt"

        before_eval = _snapshot_dir_names(eval_output_root)
        _run_command(
            _build_eval_command(
                args=args,
                regime=args.regime,
                output_root=eval_output_root,
                checkpoint_path=checkpoint_path,
                category=category_name,
            )
        )
        eval_run_dir = _detect_new_run_dir(eval_output_root, before_eval)
        print(f"{category_name}_train_run_dir:", train_run_dir)
        print(f"{category_name}_eval_run_dir:", eval_run_dir)
        if args.export_model:
            _export_bundle(eval_run_dir, args.regime, force=args.force_model_export)


if __name__ == "__main__":
    main()
