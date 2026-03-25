#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_config(path: str) -> dict:
    import yaml

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def parse_csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_categories_arg(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def coerce_bool_arg(value: str, fallback: bool) -> bool:
    if not value:
        return fallback
    return value.strip().lower() in {"1", "true", "yes", "on"}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train MulSenDiff-X in baseline per-category mode (`--category`) "
            "or shared joint mode (`--categories`)."
        ),
        epilog=(
            "Examples:\n"
            "  python scripts/run_training.py --category capsule --device-mode cpu\n"
            "  python scripts/run_training.py --categories all --device-mode cuda"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="config/diffusion.yaml")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--category", default="", help="Baseline mode: train one category such as 'capsule'.")
    parser.add_argument(
        "--categories",
        default="",
        help="Shared mode: comma-separated categories or 'all' via config selection.",
    )
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--epochs-long", action="store_true")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=0.0)
    parser.add_argument("--device", default="")
    parser.add_argument("--device-mode", default="")
    parser.add_argument("--score-mode", default="")
    parser.add_argument("--anomaly-timestep", type=int, default=0)
    parser.add_argument("--descriptor-weight", type=float, default=0.0)
    parser.add_argument("--global-descriptor-weight", type=float, default=0.0)
    parser.add_argument("--multi-timestep-scoring-enabled", default="")
    parser.add_argument("--multi-timestep-fractions", default="")
    parser.add_argument("--anomaly-map-gaussian-sigma", type=float, default=0.0)
    parser.add_argument("--lambda-residual", type=float, default=0.0)
    parser.add_argument("--lambda-noise", type=float, default=0.0)
    parser.add_argument("--lambda-descriptor-spatial", type=float, default=0.0)
    parser.add_argument("--lambda-descriptor-global", type=float, default=0.0)
    parser.add_argument("--pixel-threshold-percentile", type=float, default=0.0)
    parser.add_argument("--object-mask-threshold", type=float, default=0.0)
    parser.add_argument("--object-crop-enabled", default="")
    parser.add_argument("--object-crop-margin-ratio", type=float, default=0.0)
    parser.add_argument("--object-crop-mask-source", default="")
    parser.add_argument("--rgb-normalization-mode", default="")
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--max-visualizations", type=int, default=6)
    parser.add_argument("--base-channels", type=int, default=0)
    parser.add_argument("--global-embedding-dim", type=int, default=0)
    parser.add_argument("--time-embedding-dim", type=int, default=0)
    parser.add_argument("--category-embedding-dim", type=int, default=0)
    parser.add_argument("--attention-heads", type=int, default=0)
    parser.add_argument("--latent-channels", type=int, default=0)
    parser.add_argument("--diffusion-steps", type=int, default=0)
    parser.add_argument("--beta-start", type=float, default=0.0)
    parser.add_argument("--beta-end", type=float, default=0.0)
    parser.add_argument("--noise-schedule", default="")
    parser.add_argument("--autoencoder-reconstruction-weight", type=float, default=0.0)
    parser.add_argument("--diffusion-reconstruction-weight", type=float, default=0.0)
    parser.add_argument("--edge-reconstruction-weight", type=float, default=0.0)
    parser.add_argument("--sampler-mode", default="")
    parser.add_argument("--selection-metric", default="")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-every-n-steps", type=int, default=0)
    parser.add_argument("--no-jsonl", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    from src.training.train_diffusion import train_model

    config = load_config(args.config)
    smoke_cfg = config.get("smoke", {})
    training_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    conditioning_cfg = config.get("conditioning", {})
    inference_cfg = config.get("inference", {})
    selected_categories_cfg = data_cfg.get("selected_categories", [])
    selected_categories = parse_categories_arg(args.categories) if args.categories else list(selected_categories_cfg or [])

    epochs = args.epochs or int(training_cfg.get("epochs_long" if args.epochs_long else "epochs", 50))
    result = train_model(
        data_root=args.data_root,
        category=args.category or smoke_cfg.get("category", "capsule"),
        categories=selected_categories,
        epochs=epochs,
        batch_size=args.batch_size or int(training_cfg.get("batch_size", smoke_cfg.get("batch_size", 2))),
        target_size=(args.image_size or int(data_cfg.get("image_size", 256)),) * 2,
        device=args.device or training_cfg.get("device", "cpu"),
        device_mode=args.device_mode or str(training_cfg.get("device_mode", "")),
        learning_rate=args.learning_rate or float(training_cfg.get("learning_rate", 1e-4)),
        score_mode=args.score_mode or str(inference_cfg.get("score_mode", "noise_error")),
        anomaly_timestep=args.anomaly_timestep or int(inference_cfg.get("anomaly_timestep", 200)),
        descriptor_weight=args.descriptor_weight or float(inference_cfg.get("descriptor_weight", 0.25)),
        global_descriptor_weight=args.global_descriptor_weight or float(inference_cfg.get("global_descriptor_weight", 0.1)),
        multi_timestep_scoring_enabled=coerce_bool_arg(
            args.multi_timestep_scoring_enabled,
            bool(inference_cfg.get("multi_timestep_scoring_enabled", True)),
        ),
        multi_timestep_fractions=parse_csv_floats(args.multi_timestep_fractions)
        if args.multi_timestep_fractions
        else inference_cfg.get("multi_timestep_fractions", [0.05, 0.10, 0.20, 0.30, 0.40]),
        anomaly_map_gaussian_sigma=args.anomaly_map_gaussian_sigma
        or float(inference_cfg.get("anomaly_map_gaussian_sigma", 3.0)),
        lambda_residual=args.lambda_residual or float(inference_cfg.get("lambda_residual", 1.0)),
        lambda_noise=args.lambda_noise or float(inference_cfg.get("lambda_noise", 0.5)),
        lambda_descriptor_spatial=args.lambda_descriptor_spatial
        or float(inference_cfg.get("lambda_descriptor_spatial", 0.25)),
        lambda_descriptor_global=args.lambda_descriptor_global
        or float(inference_cfg.get("lambda_descriptor_global", 0.3)),
        score_weight_schedule=inference_cfg.get("score_weight_schedule"),
        pixel_threshold_percentile=args.pixel_threshold_percentile or float(inference_cfg.get("pixel_threshold_percentile", 0.9)),
        object_mask_threshold=args.object_mask_threshold or float(inference_cfg.get("object_mask_threshold", 0.02)),
        localization_quantile=float(inference_cfg.get("localization_quantile", 0.995)),
        localization_min_region_area_fraction=float(
            inference_cfg.get("localization_min_region_area_fraction", 0.002)
        ),
        localization_min_region_pixels_floor=int(
            inference_cfg.get("localization_min_region_pixels_floor", 4)
        ),
        object_crop_enabled=coerce_bool_arg(
            args.object_crop_enabled,
            bool(data_cfg.get("object_crop_enabled", False)),
        ),
        object_crop_margin_ratio=args.object_crop_margin_ratio
        or float(data_cfg.get("object_crop_margin_ratio", 0.10)),
        object_crop_mask_source=args.object_crop_mask_source
        or str(data_cfg.get("object_crop_mask_source", "rgb_then_pc_density")),
        rgb_normalization_mode=args.rgb_normalization_mode or str(data_cfg.get("rgb_normalization_mode", "none")),
        output_root=args.output_root or training_cfg.get("output_root", "runs"),
        run_name=args.run_name or None,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        max_visualizations=args.max_visualizations,
        base_channels=args.base_channels or int(model_cfg.get("base_channels", 32)),
        global_embedding_dim=args.global_embedding_dim or int(conditioning_cfg.get("global_embedding_dim", 128)),
        time_embedding_dim=args.time_embedding_dim or int(conditioning_cfg.get("time_embedding_dim", 128)),
        category_embedding_dim=args.category_embedding_dim or int(conditioning_cfg.get("category_embedding_dim", 32)),
        attention_heads=args.attention_heads or int(model_cfg.get("attention_heads", 4)),
        latent_channels=args.latent_channels or int(model_cfg.get("latent_channels", 4)),
        diffusion_steps=args.diffusion_steps or int(model_cfg.get("diffusion_steps", 1000)),
        beta_start=args.beta_start or float(model_cfg.get("beta_start", 1e-4)),
        beta_end=args.beta_end or float(model_cfg.get("beta_end", 2e-2)),
        noise_schedule=args.noise_schedule or str(model_cfg.get("noise_schedule", "cosine")),
        autoencoder_reconstruction_weight=args.autoencoder_reconstruction_weight or float(model_cfg.get("autoencoder_reconstruction_weight", 0.5)),
        diffusion_reconstruction_weight=args.diffusion_reconstruction_weight or float(model_cfg.get("diffusion_reconstruction_weight", 0.1)),
        edge_reconstruction_weight=args.edge_reconstruction_weight or float(model_cfg.get("edge_reconstruction_weight", 0.05)),
        sampler_mode=args.sampler_mode or str(training_cfg.get("sampler_mode", "balanced_by_category")),
        selection_metric=args.selection_metric or str(training_cfg.get("selection_metric", "macro_image_auroc")),
        seed=args.seed if args.seed is not None else int(training_cfg.get("seed", 7)),
        log_every_n_steps=args.log_every_n_steps or int(training_cfg.get("log_every_n_steps", 10)),
        log_to_jsonl=not args.no_jsonl if args.no_jsonl else bool(training_cfg.get("log_to_jsonl", True)),
    )

    print("run_dir:", result["summary"]["run_dir"])
    print("summary:", result["summary"])


if __name__ == "__main__":
    main()
