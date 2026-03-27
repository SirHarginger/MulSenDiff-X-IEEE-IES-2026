#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export publication-grade tables and figures from explicit MulSenDiff-X evaluation runs.",
    )
    parser.add_argument("--shared-eval-run", required=True)
    parser.add_argument("--shared-nocat-eval-run", required=True)
    parser.add_argument("--percat-eval-manifest", required=True)
    parser.add_argument("--qualitative-manifest", required=True)
    parser.add_argument("--out-dir", default="publication")
    parser.add_argument("--visibility-mapping", default="")
    parser.add_argument("--robustness-manifest", default="")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--device-mode", default="")
    parser.add_argument("--max-eval-batches", type=int, default=0)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    from src.publication.export import export_publication_package

    outputs = export_publication_package(
        shared_eval_run=args.shared_eval_run,
        shared_nocat_eval_run=args.shared_nocat_eval_run,
        percat_eval_manifest=args.percat_eval_manifest,
        qualitative_manifest=args.qualitative_manifest,
        out_dir=args.out_dir,
        visibility_mapping=args.visibility_mapping or None,
        robustness_manifest=args.robustness_manifest or None,
        data_root=args.data_root,
        device=args.device,
        device_mode=args.device_mode,
        max_eval_batches=args.max_eval_batches,
    )

    print("publication_out_dir:", args.out_dir)
    print("outputs:", outputs)


if __name__ == "__main__":
    main()
