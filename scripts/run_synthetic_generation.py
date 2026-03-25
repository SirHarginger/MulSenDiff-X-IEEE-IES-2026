#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.synthetic.generator import RECIPES, generate_synthetic_anomaly_bundles


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate offline synthetic anomaly bundles from processed normal samples.")
    parser.add_argument("--samples-root", default="data/processed/samples")
    parser.add_argument("--output-root", default="data/synthetic")
    parser.add_argument("--recipes", default=",".join(RECIPES))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    result = generate_synthetic_anomaly_bundles(
        samples_root=args.samples_root,
        output_root=args.output_root,
        recipes=[item for item in args.recipes.split(",") if item],
        limit=args.limit,
    )
    print("manifest_csv:", result["manifest_csv"])
    print("index_json:", result["index_json"])


if __name__ == "__main__":
    main()
