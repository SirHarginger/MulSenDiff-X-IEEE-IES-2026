#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_loader import build_training_manifests
from src.preprocessing.dataset import build_dataset_index
from src.preprocessing.descriptor_pipeline import run_descriptor_pipeline


def _parse_csv_set(value: str) -> set[str] | None:
    items = {item.strip() for item in value.split(",") if item.strip()}
    return items or None


def _parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the MulSenDiff-X data pipeline: dataset index, descriptors, audits, and model manifests."
    )
    parser.add_argument("--raw-root", default="data/raw/MulSen_AD")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--categories", default="")
    parser.add_argument("--splits", default="")
    parser.add_argument("--defects", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-anomalous", action="store_true")
    parser.add_argument("--no-normal", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--skip-manifests", action="store_true")
    args = parser.parse_args()

    dataset_index = build_dataset_index(args.raw_root, args.data_root)
    descriptor_result = run_descriptor_pipeline(
        data_root=args.data_root,
        index_csv=dataset_index["manifest_csv"],
        categories=_parse_csv_set(args.categories),
        splits=_parse_csv_set(args.splits),
        defect_labels=_parse_csv_set(args.defects),
        include_anomalous=not args.no_anomalous,
        include_normal=not args.no_normal,
        limit=args.limit,
        skip_existing=not args.overwrite,
        continue_on_error=not args.fail_fast,
    )

    manifests = None
    if not args.skip_manifests:
        manifests = build_training_manifests(
            descriptor_index_csv=descriptor_result["descriptor_index"]["csv_path"],
            data_root=args.data_root,
            categories=_parse_csv_list(args.categories),
        )

    print("dataset_records:", len(dataset_index["records"]))
    print("dataset_issues:", len(dataset_index["issues"]))
    print("dataset_manifest:", dataset_index["manifest_csv"])
    print("descriptor_records:", len(descriptor_result["records"]))
    print("generated:", descriptor_result["generated"])
    print("descriptor_failures:", len(descriptor_result["failures"]))
    print("descriptor_summary:", descriptor_result["pipeline_summary_path"])
    print("descriptor_index:", descriptor_result["descriptor_index"]["csv_path"])
    if manifests is not None:
        print("train_manifest:", manifests["train_csv"])
        print("eval_manifest:", manifests["eval_csv"])
        print("global_stats:", manifests["global_stats_json"])


if __name__ == "__main__":
    main()

