#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
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


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_categories_arg(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def coerce_bool_arg(value: str, fallback: bool) -> bool:
    if not value:
        return fallback
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _filter_supported_kwargs(fn, kwargs: dict) -> dict:
    signature = inspect.signature(fn)
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a MulSenDiff-X checkpoint in baseline per-category mode "
            "or shared joint mode."
        ),
        epilog=(
            "Examples:\n"
            "  python scripts/run_evaluation.py --checkpoint runs/csdd/train/<run>/checkpoints/best.pt --category capsule\n"
            "  python scripts/run_evaluation.py --checkpoint runs/ccdd/train/<run>/checkpoints/best.pt --categories all"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="config/diffusion.yaml")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--processed-root", default="")
    parser.add_argument("--category", default="", help="Baseline mode: evaluate one category.")
    parser.add_argument("--categories", default="", help="Shared mode: evaluate a joint checkpoint.")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=0)
    parser.add_argument("--device", default="")
    parser.add_argument("--device-mode", default="")
    parser.add_argument(
        "--disable-category-embedding",
        action="store_true",
        help="Override checkpoint config and disable category identity during evaluation.",
    )
    parser.add_argument(
        "--enable-category-modality-gating",
        action="store_true",
        help="Override checkpoint config and enable category-conditioned descriptor/support gating.",
    )
    parser.add_argument("--score-mode", default="")
    parser.add_argument("--anomaly-timestep", type=int, default=0)
    parser.add_argument("--descriptor-weight", type=float, default=0.0)
    parser.add_argument("--global-descriptor-weight", type=float, default=0.0)
    parser.add_argument("--multi-timestep-scoring-enabled", default="")
    parser.add_argument("--multi-timestep-fractions", default="")
    parser.add_argument("--anomaly-map-gaussian-sigma", type=float, default=0.0)
    parser.add_argument("--spatial-topk-percent", type=float, default=0.0)
    parser.add_argument(
        "--object-score-strategy",
        default="legacy_raw",
        choices=["legacy_raw", "exceedance_regions", "exceedance_logreg"],
    )
    parser.add_argument("--lambda-residual", type=float, default=0.0)
    parser.add_argument("--lambda-noise", type=float, default=0.0)
    parser.add_argument("--lambda-descriptor-spatial", type=float, default=0.0)
    parser.add_argument("--lambda-descriptor-global", type=float, default=0.0)
    parser.add_argument("--pixel-threshold-percentile", type=float, default=0.0)
    parser.add_argument("--object-mask-threshold", type=float, default=0.0)
    parser.add_argument(
        "--localization-basis",
        default="",
        choices=[
            "",
            "anomaly_map",
            "score_basis_map",
            "residual_component",
            "noise_component",
            "descriptor_support",
            "noise_residual_blend",
            "noise_residual_geomean",
        ],
    )
    parser.add_argument("--localization-basis-candidates", default="")
    parser.add_argument("--object-mask-dilation-kernel-size", type=int, default=0)
    parser.add_argument("--localization-mask-closing-kernel-size", type=int, default=0)
    parser.add_argument("--localization-fill-holes", action="store_true")
    parser.add_argument(
        "--localization-tune-on-synthetic-validation",
        action="store_true",
        help="Tune localization calibration on synthetic validation without retraining the detector.",
    )
    parser.add_argument("--localization-quantile", type=float, default=0.0)
    parser.add_argument("--localization-quantile-candidates", default="")
    parser.add_argument("--localization-min-region-area-fraction", type=float, default=0.0)
    parser.add_argument("--localization-min-region-pixels-floor", type=int, default=0)
    parser.add_argument("--localization-min-region-pixels-floor-candidates", default="")
    parser.add_argument("--object-crop-enabled", default="")
    parser.add_argument("--object-crop-margin-ratio", type=float, default=0.0)
    parser.add_argument("--object-crop-mask-source", default="")
    parser.add_argument("--rgb-normalization-mode", default="")
    parser.add_argument(
        "--enable-internal-defect-gate",
        action="store_true",
        help="Override checkpoint config and enable the Archetype-B IR-dominant object-score override.",
    )
    parser.add_argument("--eval-manifest-csv", default="")
    parser.add_argument(
        "--conditioning-ablation",
        default="none",
        choices=[
            "none",
            "drop_ir",
            "drop_pointcloud",
            "drop_crossmodal",
            "missing_ir",
            "missing_pointcloud",
        ],
    )
    parser.add_argument("--descriptor-noise-sigma", type=float, default=0.0)
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--max-visualizations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--knowledge-base-root", default="")
    parser.add_argument("--retrieval-top-k", type=int, default=0)
    parser.add_argument(
        "--enable-internal-llm-debug",
        action="store_true",
        help="Write detector-first corrective RAG explanation artifacts during evaluation.",
    )
    parser.add_argument(
        "--disable-llm-explanations",
        action="store_true",
        help="Deprecated compatibility flag. Internal explanation debug artifacts are disabled by default.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    from src.project_layout import default_output_root, resolve_regime_identity, resolve_scope
    from src.training.train_diffusion import evaluate_checkpoint

    config = load_config(args.config)
    smoke_cfg = config.get("smoke", {})
    training_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    inference_cfg = config.get("inference", {})
    default_retrieval_root = REPO_ROOT / "data" / "retrieval"
    selected_categories_cfg = data_cfg.get("selected_categories", [])
    selected_categories = parse_categories_arg(args.categories) if args.categories else list(selected_categories_cfg or [])
    joint_mode = len(selected_categories) > 1 or (len(selected_categories) == 1 and selected_categories[0].lower() == "all") or bool(args.categories)
    regime = resolve_regime_identity(
        joint_mode=joint_mode,
        disable_category_embedding=bool(args.disable_category_embedding),
    )
    scope = resolve_scope(selected_categories, category=args.category or smoke_cfg.get("category", "capsule"))
    resolved_output_root = (
        Path(args.output_root)
        if str(args.output_root).strip() and str(args.output_root).strip() != "runs"
        else default_output_root(
            regime_paper=regime.paper,
            run_type="eval",
            scope=scope,
            label=args.run_name or "baseline",
            repo_root=REPO_ROOT,
        )
    )

    requested_kwargs = dict(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        processed_root=getattr(args, "processed_root", "") or None,
        category=args.category,
        categories=selected_categories,
        batch_size=args.batch_size or int(training_cfg.get("batch_size", smoke_cfg.get("batch_size", 2))),
        target_size=(args.image_size or int(data_cfg.get("image_size", 256)),) * 2,
        device=args.device or training_cfg.get("device", "cpu"),
        device_mode=args.device_mode or str(training_cfg.get("device_mode", "")),
        disable_category_embedding=True if args.disable_category_embedding else None,
        enable_category_modality_gating=True if args.enable_category_modality_gating else None,
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
        spatial_topk_percent=args.spatial_topk_percent or None,
        object_score_strategy=args.object_score_strategy or str(
            inference_cfg.get("object_score_strategy", "legacy_raw")
        ),
        lambda_residual=args.lambda_residual or None,
        lambda_noise=args.lambda_noise or None,
        lambda_descriptor_spatial=args.lambda_descriptor_spatial or None,
        lambda_descriptor_global=args.lambda_descriptor_global or None,
        pixel_threshold_percentile=args.pixel_threshold_percentile or float(inference_cfg.get("pixel_threshold_percentile", 0.9)),
        object_mask_threshold=args.object_mask_threshold or float(inference_cfg.get("object_mask_threshold", 0.02)),
        localization_basis=args.localization_basis or str(inference_cfg.get("localization_basis", "anomaly_map")),
        localization_basis_candidates=parse_categories_arg(args.localization_basis_candidates)
        if args.localization_basis_candidates
        else inference_cfg.get("localization_basis_candidates"),
        object_mask_dilation_kernel_size=args.object_mask_dilation_kernel_size
        or int(inference_cfg.get("object_mask_dilation_kernel_size", 9)),
        localization_mask_closing_kernel_size=args.localization_mask_closing_kernel_size or None,
        localization_fill_holes=True if args.localization_fill_holes else None,
        localization_tune_on_synthetic_validation=True if args.localization_tune_on_synthetic_validation else None,
        localization_quantile=args.localization_quantile
        or float(inference_cfg.get("localization_quantile", 0.995)),
        localization_quantile_candidates=parse_csv_floats(args.localization_quantile_candidates)
        if hasattr(args, "localization_quantile_candidates") and args.localization_quantile_candidates
        else inference_cfg.get("localization_quantile_candidates"),
        localization_min_region_area_fraction=args.localization_min_region_area_fraction
        or float(inference_cfg.get("localization_min_region_area_fraction", 0.002)),
        localization_min_region_pixels_floor=args.localization_min_region_pixels_floor
        or int(inference_cfg.get("localization_min_region_pixels_floor", 4)),
        localization_min_region_pixels_floor_candidates=parse_csv_ints(args.localization_min_region_pixels_floor_candidates)
        if hasattr(args, "localization_min_region_pixels_floor_candidates") and args.localization_min_region_pixels_floor_candidates
        else inference_cfg.get("localization_min_region_pixels_floor_candidates"),
        object_crop_enabled=coerce_bool_arg(
            args.object_crop_enabled,
            bool(data_cfg.get("object_crop_enabled", False)),
        ),
        object_crop_margin_ratio=args.object_crop_margin_ratio
        or float(data_cfg.get("object_crop_margin_ratio", 0.10)),
        object_crop_mask_source=args.object_crop_mask_source
        or str(data_cfg.get("object_crop_mask_source", "rgb_then_pc_density")),
        rgb_normalization_mode=args.rgb_normalization_mode or str(data_cfg.get("rgb_normalization_mode", "none")),
        enable_internal_defect_gate=True if args.enable_internal_defect_gate else None,
        eval_manifest_csv=args.eval_manifest_csv or None,
        conditioning_ablation=args.conditioning_ablation,
        descriptor_noise_sigma=args.descriptor_noise_sigma,
        output_root=resolved_output_root,
        run_name=args.run_name or None,
        max_eval_batches=args.max_eval_batches,
        max_visualizations=args.max_visualizations,
        seed=args.seed if args.seed is not None else int(training_cfg.get("seed", 7)),
        knowledge_base_root=(
            args.knowledge_base_root
            or (str(default_retrieval_root) if default_retrieval_root.exists() else None)
        ),
        retrieval_top_k=args.retrieval_top_k or int(inference_cfg.get("retrieval_top_k", 3)),
        enable_llm_explanations=args.enable_internal_llm_debug and not args.disable_llm_explanations,
    )
    result = evaluate_checkpoint(**_filter_supported_kwargs(evaluate_checkpoint, requested_kwargs))

    print("run_dir:", result["summary"]["run_dir"])
    print("summary:", result["summary"])


if __name__ == "__main__":
    main()
