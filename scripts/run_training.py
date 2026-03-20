#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.train_diffusion import train_model


def load_config(path: str) -> dict:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MulSenDiff-X and save a complete run folder.")
    parser.add_argument("--config", default="config/diffusion.yaml")
    parser.add_argument("--descriptor-index", default="")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--category", default="")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=0.0)
    parser.add_argument("--device", default="")
    parser.add_argument("--score-mode", default="")
    parser.add_argument("--anomaly-timestep", type=int, default=0)
    parser.add_argument("--descriptor-weight", type=float, default=0.0)
    parser.add_argument("--pixel-threshold-percentile", type=float, default=0.0)
    parser.add_argument("--object-mask-threshold", type=float, default=0.0)
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--max-visualizations", type=int, default=6)
    args = parser.parse_args()

    config = load_config(args.config)
    smoke_cfg = config.get("smoke", {})
    training_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    inference_cfg = config.get("inference", {})

    result = train_model(
        data_root=args.data_root,
        descriptor_index_csv=args.descriptor_index or data_cfg.get("descriptor_index", "data/processed/manifests/descriptor_index.csv"),
        category=args.category or smoke_cfg.get("category", "capsule"),
        epochs=args.epochs or int(training_cfg.get("epochs", 5)),
        batch_size=args.batch_size or int(training_cfg.get("batch_size", smoke_cfg.get("batch_size", 2))),
        target_size=(args.image_size or int(data_cfg.get("image_size", 256)),) * 2,
        device=args.device or training_cfg.get("device", "cpu"),
        learning_rate=args.learning_rate or float(training_cfg.get("learning_rate", 1e-4)),
        score_mode=args.score_mode or str(inference_cfg.get("score_mode", "noise_error")),
        anomaly_timestep=args.anomaly_timestep or int(inference_cfg.get("anomaly_timestep", 200)),
        descriptor_weight=args.descriptor_weight or float(inference_cfg.get("descriptor_weight", 0.25)),
        pixel_threshold_percentile=args.pixel_threshold_percentile or float(inference_cfg.get("pixel_threshold_percentile", 0.9)),
        object_mask_threshold=args.object_mask_threshold or float(inference_cfg.get("object_mask_threshold", 0.02)),
        output_root=args.output_root or training_cfg.get("output_root", "runs"),
        run_name=args.run_name or None,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        max_visualizations=args.max_visualizations,
    )

    print("run_dir:", result["summary"]["run_dir"])
    print("summary:", result["summary"])


if __name__ == "__main__":
    main()
