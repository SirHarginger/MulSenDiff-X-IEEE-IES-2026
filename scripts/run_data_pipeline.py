#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from datetime import datetime
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


def _log(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the MulSenDiff-X data pipeline: dataset index, category stats, sample folders, audits, and manifests."
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
    parser.add_argument("--quiet", action="store_true", help="Disable progress logging.")
    args = parser.parse_args()

    logger = None if args.quiet else _log

    _log("starting dataset indexing") if logger else None
    dataset_index = build_dataset_index(args.raw_root, args.data_root)
    _log(
        f"dataset indexing complete ({len(dataset_index['records'])} records, {len(dataset_index['issues'])} issues)"
    ) if logger else None

    _log("starting descriptor pipeline") if logger else None
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
        log_progress=logger,
    )
    _log(
        "descriptor pipeline complete "
        f"({descriptor_result['generated']['samples']} samples generated, {len(descriptor_result['failures'])} failures)"
    ) if logger else None

    manifests = None
    if not args.skip_manifests:
        _log("building training/eval manifests") if logger else None
        manifests = build_training_manifests(data_root=args.data_root, categories=_parse_csv_list(args.categories))
        _log("manifest generation complete") if logger else None

    print("dataset_records:", len(dataset_index["records"]))
    print("dataset_issues:", len(dataset_index["issues"]))
    print("dataset_manifest:", dataset_index["manifest_csv"])
    print("descriptor_records:", len(descriptor_result["records"]))
    print("generated:", descriptor_result["generated"])
    print("descriptor_failures:", len(descriptor_result["failures"]))
    print("descriptor_summary:", descriptor_result["pipeline_summary_path"])
    print("samples_root:", descriptor_result["samples_root"])
    if manifests is not None:
        print("train_manifest:", manifests["train_csv"])
        print("eval_manifest:", manifests["eval_csv"])
        print("global_stats:", manifests["global_stats_json"])


if __name__ == "__main__":
    main()
