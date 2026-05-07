#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.explainer.audit import (
    build_explanation_audit_manifest,
    run_explanation_ablation,
    summarize_explanation_ratings,
)
from src.utils.logger import create_run_dir, write_json


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and run the explanation-branch ablation audit for CCDD evaluation artifacts.",
    )
    parser.add_argument(
        "--phase",
        choices=["build_manifest", "run_modes", "summarize_ratings", "full"],
        default="full",
    )
    parser.add_argument("--eval-run", default="")
    parser.add_argument("--audit-manifest-csv", default="")
    parser.add_argument("--output-root", default="runs/ccdd/explanation_ablation")
    parser.add_argument("--knowledge-base-root", default="data/retrieval")
    parser.add_argument("--retrieval-top-k", type=int, default=3)
    parser.add_argument("--modes", default="retrieval_only,generator_only,full")
    parser.add_argument("--run-name", default="explanation_audit")
    parser.add_argument("--rating-csvs", default="")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    requested_output_root = Path(args.output_root)
    reuse_existing_run = args.phase in {"run_modes", "summarize_ratings"} and (
        (requested_output_root / "metrics").exists() or (requested_output_root / "ratings").exists()
    )
    if reuse_existing_run:
        run_root = requested_output_root
        (run_root / "metrics").mkdir(parents=True, exist_ok=True)
        (run_root / "ratings").mkdir(parents=True, exist_ok=True)
    else:
        run_root = create_run_dir(
            output_root=args.output_root,
            run_name=args.run_name,
            category="explanation_ablation",
        ).root
    manifest_path = (
        Path(args.audit_manifest_csv)
        if args.audit_manifest_csv
        else run_root / "metrics" / "audit_manifest.csv"
    )

    if args.phase in {"build_manifest", "full"}:
        if not args.eval_run:
            raise SystemExit("--eval-run is required to build the explanation audit manifest")
        cases = build_explanation_audit_manifest(
            eval_run_root=args.eval_run,
            output_path=manifest_path,
        )
        write_json(
            run_root / "metrics" / "audit_manifest_summary.json",
            {
                "eval_run": args.eval_run,
                "audit_manifest_csv": str(manifest_path),
                "cases": len(cases),
            },
        )

    if args.phase in {"run_modes", "full"}:
        if not manifest_path.exists():
            raise SystemExit(f"audit manifest not found: {manifest_path}")
        summary = run_explanation_ablation(
            audit_manifest_csv=manifest_path,
            output_dir=run_root,
            knowledge_base_root=args.knowledge_base_root or None,
            retrieval_top_k=args.retrieval_top_k,
            modes=_parse_csv(args.modes),
        )
        write_json(run_root / "metrics" / "run_modes_summary.json", summary)

    if args.phase in {"summarize_ratings", "full"} and args.rating_csvs:
        rating_paths = _parse_csv(args.rating_csvs)
        if not rating_paths:
            raise SystemExit("no rating csv paths were provided")
        summary = summarize_explanation_ratings(
            output_dir=run_root,
            rating_csv_paths=rating_paths,
        )
        write_json(run_root / "metrics" / "rating_summary.json", summary)

    write_json(
        run_root / "summary.json",
        {
            "phase": args.phase,
            "audit_manifest_csv": str(manifest_path),
            "output_dir": str(run_root),
            "knowledge_base_root": args.knowledge_base_root,
            "modes": _parse_csv(args.modes),
            "rating_csvs": _parse_csv(args.rating_csvs),
        },
    )
    print("explanation_ablation_run_dir:", run_root)


if __name__ == "__main__":
    main()
