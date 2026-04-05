from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch

from src.data_loader import (
    GLOBAL_FEATURE_NAMES,
    DescriptorConditioningDataset,
    GlobalVectorNormalizationStats,
    ProcessedSampleRecord,
    RGBNormalizationStats,
)
from src.explainer.evidence_builder import (
    EvidencePackage,
    build_evidence_package,
    build_retrieval_query,
    save_evidence_package,
)
from src.explainer.retriever import (
    RetrievedContextItem,
    retrieve_context_for_evidence,
)
from src.inference.anomaly_scorer import (
    AnomalyScoreOutputs,
    GlobalBranchGateCalibration,
    LogisticObjectScoreCalibration,
    SpatialExceedanceCalibration,
    select_localization_basis_map,
    score_batch,
)
from src.inference.global_descriptor_scoring import GlobalDescriptorCalibration
from src.inference.localization import (
    LocalizationCalibration,
    apply_localization_calibration,
    normalize_anomaly_map,
    threshold_anomaly_map,
)
from src.inference.quantification import MasiCalibration
from src.models.mulsendiffx import MulSenDiffX
from src.training.train_diffusion import build_model, load_gt_mask
from src.utils.checkpoint import load_checkpoint, read_checkpoint_payload
from src.utils.visualization import save_anomaly_panel


@dataclass(frozen=True)
class SharedInferenceSession:
    repo_root: Path
    eval_run_root: Path
    checkpoint_path: Path
    config: Dict[str, Any]
    eval_summary: Dict[str, Any]
    device: torch.device
    model: MulSenDiffX
    target_size: tuple[int, int]
    category_vocabulary: tuple[str, ...]
    rgb_stats_by_category: Dict[str, RGBNormalizationStats]
    global_vector_stats_by_category: Dict[str, GlobalVectorNormalizationStats]
    masi_calibration_by_category: Dict[str, MasiCalibration]
    global_descriptor_calibration_by_category: Dict[str, GlobalDescriptorCalibration]
    global_branch_gate_calibration_by_category: Dict[str, GlobalBranchGateCalibration]
    spatial_exceedance_calibration_by_category: Dict[str, SpatialExceedanceCalibration]
    object_score_calibration: LogisticObjectScoreCalibration | None
    localization_calibration_by_category: Dict[str, LocalizationCalibration]
    score_params: Dict[str, Any]


def _coerce_rgb_stats_by_category(payload: Dict[str, Any] | None) -> Dict[str, RGBNormalizationStats]:
    if not payload:
        return {}
    if "categories" in payload and isinstance(payload["categories"], dict):
        payload = payload["categories"]
    stats_by_category: Dict[str, RGBNormalizationStats] = {}
    for category, stats in payload.items():
        if not isinstance(stats, dict):
            continue
        stats_by_category[str(category)] = RGBNormalizationStats(
            category=str(stats.get("category", category)),
            mode=str(stats.get("mode", "none")),
            mean=tuple(float(item) for item in stats.get("mean", [0.0, 0.0, 0.0])),
            std=tuple(max(float(item), 1e-6) for item in stats.get("std", [1.0, 1.0, 1.0])),
            masked_pixels=int(stats.get("masked_pixels", 0)),
        )
    return stats_by_category


def _coerce_global_vector_stats_by_category(
    payload: Dict[str, Any] | None,
) -> Dict[str, GlobalVectorNormalizationStats]:
    if not payload:
        return {}
    if "categories" in payload and isinstance(payload["categories"], dict):
        payload = payload["categories"]
    stats_by_category: Dict[str, GlobalVectorNormalizationStats] = {}
    for category, stats in payload.items():
        if not isinstance(stats, dict):
            continue
        feature_names = tuple(str(item) for item in stats.get("feature_names", GLOBAL_FEATURE_NAMES))
        stats_by_category[str(category)] = GlobalVectorNormalizationStats(
            category=str(stats.get("category", category)),
            feature_names=feature_names,
            mean=tuple(float(item) for item in stats.get("mean", [0.0] * len(feature_names))),
            std=tuple(max(float(item), 1e-6) for item in stats.get("std", [1.0] * len(feature_names))),
            sample_count=int(stats.get("sample_count", 0)),
        )
    return stats_by_category


def _denormalize_rgb(rgb: torch.Tensor, stats: RGBNormalizationStats | None) -> torch.Tensor:
    if stats is None or stats.mode == "none":
        return rgb
    mean = torch.tensor(stats.mean, dtype=rgb.dtype, device=rgb.device).view(3, 1, 1)
    std = torch.tensor(stats.std, dtype=rgb.dtype, device=rgb.device).view(3, 1, 1)
    return rgb * std + mean


def _write_single_record_manifest(path: Path, record: ProcessedSampleRecord) -> Path:
    fieldnames = [
        "category",
        "split",
        "defect_label",
        "sample_id",
        "sample_name",
        "sample_dir",
        "is_anomalous",
        "gt_mask_path",
        "rgb_source_path",
        "ir_source_path",
        "pc_source_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(asdict(record))
    return path


def _resolve_repo_relative_path(path_like: Path | str, repo_root: Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return repo_root / path


def _resolve_run_or_repo_relative_path(path_like: Path | str, *, run_root: Path, repo_root: Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    run_relative = run_root / path
    if run_relative.exists():
        return run_relative
    return repo_root / path


def _default_retrieval_root(repo_root: Path) -> Path | None:
    candidate = repo_root / "data" / "retrieval"
    return candidate if candidate.exists() else None


def _load_masi_calibration_map(path: Path, *, fallback_category: str) -> Dict[str, MasiCalibration]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and not any(isinstance(value, dict) for value in payload.values()):
        normalized_payload = dict(payload)
        normalized_payload.setdefault("category", fallback_category)
        normalized_payload.setdefault("score_mode", "noise_error")
        normalized_payload.setdefault("sample_count", 0)
        normalized_payload.setdefault("mean", 0.0)
        normalized_payload.setdefault("std", 1.0)
        normalized_payload.setdefault("q50", 0.0)
        normalized_payload.setdefault("q95", float(normalized_payload.get("q95", 0.0)))
        return {fallback_category: MasiCalibration.from_dict(normalized_payload)}
    if isinstance(payload, dict) and "category" in payload and "score_mode" in payload:
        return {fallback_category: MasiCalibration.from_dict(payload)}
    return {
        str(category): MasiCalibration.from_dict(stats)
        for category, stats in payload.items()
        if isinstance(stats, dict)
    }


def _load_global_descriptor_calibration_map(
    path: Path,
    *,
    fallback_category: str,
) -> Dict[str, GlobalDescriptorCalibration]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "category" in payload and "feature_names" in payload:
        return {fallback_category: GlobalDescriptorCalibration.from_dict(payload)}
    return {
        str(category): GlobalDescriptorCalibration.from_dict(stats)
        for category, stats in payload.items()
        if isinstance(stats, dict)
    }


def _load_localization_calibration_map(
    path: Path,
    *,
    fallback_category: str,
) -> Dict[str, LocalizationCalibration]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "category" in payload and "threshold" in payload:
        calibration = LocalizationCalibration.from_dict(payload)
        return {fallback_category: calibration}
    return {
        str(category): LocalizationCalibration.from_dict(stats)
        for category, stats in payload.items()
        if isinstance(stats, dict)
    }


def _load_global_branch_gate_calibration_map(
    path: Path,
    *,
    fallback_category: str,
) -> Dict[str, GlobalBranchGateCalibration]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "category" in payload and "spatial_score_q95" in payload:
        calibration = GlobalBranchGateCalibration.from_dict(payload)
        return {fallback_category: calibration}
    return {
        str(category): GlobalBranchGateCalibration.from_dict(stats)
        for category, stats in payload.items()
        if isinstance(stats, dict)
    }


def _load_spatial_exceedance_calibration_map(
    path: Path,
    *,
    fallback_category: str,
) -> Dict[str, SpatialExceedanceCalibration]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "category" in payload and "tau_pix" in payload:
        calibration = SpatialExceedanceCalibration.from_dict(payload)
        return {fallback_category: calibration}
    return {
        str(category): SpatialExceedanceCalibration.from_dict(stats)
        for category, stats in payload.items()
        if isinstance(stats, dict)
    }


def _load_object_score_calibration(
    path: Path,
) -> LogisticObjectScoreCalibration | None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    if "coefficients" not in payload or "feature_names" not in payload:
        return None
    return LogisticObjectScoreCalibration.from_dict(payload)


def load_shared_inference_session(
    *,
    repo_root: Path | str,
    eval_run_root: Path | str,
) -> SharedInferenceSession:
    repo_root = Path(repo_root)
    eval_run_root = Path(eval_run_root)
    eval_summary = json.loads((eval_run_root / "summary.json").read_text(encoding="utf-8"))
    checkpoint_path = Path(eval_summary["checkpoint_path"])
    if not checkpoint_path.is_absolute():
        checkpoint_path = _resolve_run_or_repo_relative_path(
            checkpoint_path,
            run_root=eval_run_root,
            repo_root=repo_root,
        )

    checkpoint_payload = read_checkpoint_payload(checkpoint_path, map_location=torch.device("cpu"))
    config = dict(checkpoint_payload.get("config", {}))
    category_vocabulary = tuple(
        str(item)
        for item in (
            config.get("category_vocabulary")
            or config.get("selected_categories")
            or eval_summary.get("category_vocabulary")
            or []
        )
    )
    target_size = tuple(int(value) for value in config.get("target_size", [256, 256]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    disable_category_embedding = bool(config.get("disable_category_embedding", False))
    enable_category_modality_gating = bool(config.get("enable_category_modality_gating", False))
    fallback_category = (
        category_vocabulary[0]
        if category_vocabulary
        else str(config.get("category") or eval_summary.get("category") or "")
    )

    rgb_stats_by_category = _coerce_rgb_stats_by_category(config.get("rgb_normalization_stats_by_category"))
    if not rgb_stats_by_category:
        stats_path = eval_summary.get("rgb_normalization_stats_by_category_path", "")
        if str(stats_path).strip():
            resolved_stats_path = _resolve_repo_relative_path(str(stats_path), repo_root)
            if not resolved_stats_path.exists():
                resolved_stats_path = _resolve_run_or_repo_relative_path(
                    str(stats_path),
                    run_root=eval_run_root,
                    repo_root=repo_root,
                )
            if resolved_stats_path.exists():
                rgb_stats_by_category = _coerce_rgb_stats_by_category(
                    json.loads(resolved_stats_path.read_text(encoding="utf-8"))
                )

    global_vector_stats_by_category = _coerce_global_vector_stats_by_category(
        config.get("global_vector_normalization_stats_by_category")
    )
    localization_calibration_by_category: Dict[str, LocalizationCalibration] = {}
    localization_calibration_path = str(
        eval_summary.get("localization_calibration_path")
        or config.get("localization_calibration_path")
        or ""
    ).strip()
    if localization_calibration_path:
        resolved_localization_calibration_path = _resolve_repo_relative_path(
            localization_calibration_path,
            repo_root,
        )
        if not resolved_localization_calibration_path.exists():
            resolved_localization_calibration_path = _resolve_run_or_repo_relative_path(
                localization_calibration_path,
                run_root=eval_run_root,
                repo_root=repo_root,
            )
        if resolved_localization_calibration_path.exists():
            localization_calibration_by_category = _load_localization_calibration_map(
                resolved_localization_calibration_path,
                fallback_category=fallback_category,
            )

    spatial_exceedance_calibration_by_category: Dict[str, SpatialExceedanceCalibration] = {}
    spatial_exceedance_calibration_path = str(
        eval_summary.get("spatial_exceedance_calibration_path")
        or config.get("spatial_exceedance_calibration_path")
        or ""
    ).strip()
    if spatial_exceedance_calibration_path:
        resolved_spatial_exceedance_calibration_path = _resolve_repo_relative_path(
            spatial_exceedance_calibration_path,
            repo_root,
        )
        if not resolved_spatial_exceedance_calibration_path.exists():
            resolved_spatial_exceedance_calibration_path = _resolve_run_or_repo_relative_path(
                spatial_exceedance_calibration_path,
                run_root=eval_run_root,
                repo_root=repo_root,
            )
    else:
        resolved_spatial_exceedance_calibration_path = eval_run_root / "evidence" / "spatial_exceedance_calibration.json"
    if resolved_spatial_exceedance_calibration_path.exists():
        spatial_exceedance_calibration_by_category = _load_spatial_exceedance_calibration_map(
            resolved_spatial_exceedance_calibration_path,
            fallback_category=fallback_category,
        )

    object_score_calibration: LogisticObjectScoreCalibration | None = None
    object_score_calibration_path = str(
        eval_summary.get("object_score_calibration_path")
        or config.get("object_score_calibration_path")
        or ""
    ).strip()
    if object_score_calibration_path:
        resolved_object_score_calibration_path = _resolve_repo_relative_path(
            object_score_calibration_path,
            repo_root,
        )
        if not resolved_object_score_calibration_path.exists():
            resolved_object_score_calibration_path = _resolve_run_or_repo_relative_path(
                object_score_calibration_path,
                run_root=eval_run_root,
                repo_root=repo_root,
            )
    else:
        resolved_object_score_calibration_path = eval_run_root / "evidence" / "object_score_calibration.json"
    if resolved_object_score_calibration_path.exists():
        object_score_calibration = _load_object_score_calibration(
            resolved_object_score_calibration_path,
        )

    model = build_model(
        descriptor_channels=int(config.get("conditioning_channels", config.get("descriptor_channels", 7))),
        global_dim=int(config.get("global_dim", 10)),
        base_channels=int(config.get("base_channels", 32)),
        global_embedding_dim=int(config.get("global_embedding_dim", 128)),
        time_embedding_dim=int(config.get("time_embedding_dim", 128)),
        num_categories=1 if disable_category_embedding else max(len(category_vocabulary), 1),
        category_embedding_dim=int(config.get("category_embedding_dim", 32)),
        enable_category_modality_gating=enable_category_modality_gating,
        attention_heads=int(config.get("attention_heads", 4)),
        latent_channels=int(config.get("latent_channels", 4)),
        diffusion_steps=int(config.get("diffusion_steps", 1000)),
        beta_start=float(config.get("beta_start", 1e-4)),
        beta_end=float(config.get("beta_end", 2e-2)),
        noise_schedule=str(config.get("noise_schedule", "cosine")),
        autoencoder_reconstruction_weight=float(config.get("autoencoder_reconstruction_weight", 0.5)),
        diffusion_reconstruction_weight=float(config.get("diffusion_reconstruction_weight", 0.1)),
        edge_reconstruction_weight=float(config.get("edge_reconstruction_weight", 0.05)),
    ).to(device)
    load_checkpoint(checkpoint_path, model=model, map_location=device)
    model.eval()

    return SharedInferenceSession(
        repo_root=repo_root,
        eval_run_root=eval_run_root,
        checkpoint_path=checkpoint_path,
        config=config,
        eval_summary=eval_summary,
        device=device,
        model=model,
        target_size=target_size,
        category_vocabulary=category_vocabulary,
        rgb_stats_by_category=rgb_stats_by_category,
        global_vector_stats_by_category=global_vector_stats_by_category,
        masi_calibration_by_category=_load_masi_calibration_map(
            eval_run_root / "evidence" / "calibration.json",
            fallback_category=fallback_category,
        ),
        global_descriptor_calibration_by_category=_load_global_descriptor_calibration_map(
            eval_run_root / "evidence" / "global_descriptor_calibration.json",
            fallback_category=fallback_category,
        ),
        global_branch_gate_calibration_by_category=_load_global_branch_gate_calibration_map(
            eval_run_root / "evidence" / "global_branch_gate_calibration.json",
            fallback_category=fallback_category,
        )
        if (eval_run_root / "evidence" / "global_branch_gate_calibration.json").exists()
        else {},
        spatial_exceedance_calibration_by_category=spatial_exceedance_calibration_by_category,
        object_score_calibration=object_score_calibration,
        localization_calibration_by_category=localization_calibration_by_category,
        score_params={
            "score_mode": str(eval_summary.get("score_mode", config.get("score_mode", "noise_error"))),
            "object_score_strategy": str(
                eval_summary.get("object_score_strategy", config.get("object_score_strategy", "legacy_raw"))
            ),
            "anomaly_timestep": int(config.get("anomaly_timestep", 200)),
            "descriptor_weight": float(config.get("descriptor_weight", 0.25)),
            "global_descriptor_weight": float(config.get("global_descriptor_weight", 0.1)),
            "multi_timestep_scoring_enabled": bool(config.get("multi_timestep_scoring_enabled", False)),
            "multi_timestep_fractions": list(config.get("multi_timestep_fractions", [0.05, 0.10, 0.20, 0.30, 0.40])),
            "anomaly_map_gaussian_sigma": float(config.get("anomaly_map_gaussian_sigma", 0.0)),
            "spatial_topk_percent": float(eval_summary.get("spatial_topk_percent", config.get("spatial_topk_percent", 0.02))),
            "lambda_residual": float(eval_summary.get("lambda_residual", config.get("lambda_residual", 0.0))),
            "lambda_noise": float(eval_summary.get("lambda_noise", config.get("lambda_noise", 1.0))),
            "lambda_descriptor_spatial": float(
                eval_summary.get("lambda_descriptor_spatial", config.get("lambda_descriptor_spatial", 0.25))
            ),
            "lambda_descriptor_global": float(
                eval_summary.get("lambda_descriptor_global", config.get("lambda_descriptor_global", 0.1))
            ),
            "pixel_threshold_percentile": float(config.get("pixel_threshold_percentile", 0.9)),
            "object_mask_threshold": float(config.get("object_mask_threshold", 0.02)),
            "localization_quantile": float(config.get("localization_quantile", 0.995)),
            "localization_basis": str(eval_summary.get("localization_basis", config.get("localization_basis", "anomaly_map"))),
            "object_mask_dilation_kernel_size": int(
                eval_summary.get(
                    "object_mask_dilation_kernel_size",
                    config.get("object_mask_dilation_kernel_size", 9),
                )
            ),
            "localization_mask_closing_kernel_size": int(
                eval_summary.get(
                    "localization_mask_closing_kernel_size",
                    config.get("localization_mask_closing_kernel_size", 0),
                )
            ),
            "localization_fill_holes": bool(
                eval_summary.get("localization_fill_holes", config.get("localization_fill_holes", False))
            ),
            "disable_category_embedding": disable_category_embedding,
            "enable_internal_defect_gate": bool(
                eval_summary.get("enable_internal_defect_gate", config.get("enable_internal_defect_gate", False))
            ),
        },
    )


def create_app_session_dir(
    *,
    output_root: Path | str,
    sample_slug: str,
) -> Path:
    output_root = Path(output_root)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    root = output_root / f"{timestamp}_{sample_slug}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _load_sample_from_record(
    session: SharedInferenceSession,
    record: ProcessedSampleRecord,
    *,
    session_dir: Path,
) -> Any:
    manifest_path = _write_single_record_manifest(session_dir / "manifests" / "single_sample.csv", record)
    global_stats_path = _resolve_repo_relative_path(Path(session.config["global_stats_json"]), session.repo_root)
    dataset = DescriptorConditioningDataset(
        manifest_path,
        data_root=session.repo_root,
        target_size=session.target_size,
        global_stats_path=global_stats_path,
        object_crop_enabled=bool(session.config.get("object_crop_enabled", False)),
        object_crop_margin_ratio=float(session.config.get("object_crop_margin_ratio", 0.10)),
        object_crop_mask_source=str(session.config.get("object_crop_mask_source", "rgb_then_pc_density")),
        rgb_normalization_mode=str(session.config.get("rgb_normalization_mode", "none")),
        category_vocabulary=session.category_vocabulary,
    )
    dataset.set_rgb_normalization_stats_by_category(session.rgb_stats_by_category)
    dataset.set_global_vector_normalization_stats_by_category(session.global_vector_stats_by_category)
    return dataset[0]


def run_known_sample_inference(
    session: SharedInferenceSession,
    record: ProcessedSampleRecord,
    *,
    output_root: Path | str,
    knowledge_base_root: Path | str | None = None,
) -> Dict[str, Any]:
    sample_slug = f"{record.category}_{record.defect_label}_{record.sample_id}"
    session_dir = create_app_session_dir(output_root=output_root, sample_slug=sample_slug)
    sample = _load_sample_from_record(session, record, session_dir=session_dir)
    score_params = session.score_params

    x_rgb = sample.x_rgb.unsqueeze(0).to(session.device)
    descriptor_maps = sample.conditioning.descriptor_maps.unsqueeze(0).to(session.device)
    support_maps = sample.conditioning.support_maps.unsqueeze(0).to(session.device)
    global_vector = sample.conditioning.global_vector.unsqueeze(0).to(session.device)
    raw_global_vector = sample.conditioning.raw_global_vector.unsqueeze(0).to(session.device)
    category_indices = torch.tensor([sample.category_index], dtype=torch.long, device=session.device)
    if bool(session.score_params.get("disable_category_embedding", False)):
        category_indices = torch.zeros_like(category_indices)
    sample_localization_calibration = session.localization_calibration_by_category.get(sample.category)
    sample_localization_basis = (
        str(sample_localization_calibration.basis)
        if sample_localization_calibration is not None and str(sample_localization_calibration.basis).strip()
        else str(score_params.get("localization_basis", "anomaly_map"))
    )
    sample_object_mask_threshold = (
        float(sample_localization_calibration.object_mask_threshold)
        if sample_localization_calibration is not None
        else float(score_params["object_mask_threshold"])
    )
    sample_object_mask_dilation_kernel_size = (
        int(sample_localization_calibration.object_mask_dilation_kernel_size)
        if sample_localization_calibration is not None
        else int(score_params.get("object_mask_dilation_kernel_size", 9))
    )
    sample_gate_calibration = session.global_branch_gate_calibration_by_category.get(sample.category)
    feature_index = {name: index for index, name in enumerate(GLOBAL_FEATURE_NAMES)}
    sample_ir_hotspot_area_fraction = float(
        sample.conditioning.raw_global_vector[feature_index["ir_hotspot_area_fraction"]].item()
    )
    sample_ir_hotspot_compactness = float(
        sample.conditioning.raw_global_vector[feature_index["ir_hotspot_compactness"]].item()
    )
    sample_ir_mean_hotspot_intensity = float(
        sample.conditioning.raw_global_vector[feature_index["ir_mean_hotspot_intensity"]].item()
    )
    sample_cross_inconsistency_score = float(
        sample.conditioning.raw_global_vector[feature_index["cross_inconsistency_score"]].item()
    )

    with torch.no_grad():
        scored = score_batch(
            session.model,
            x_rgb,
            descriptor_maps,
            support_maps,
            global_vector,
            raw_global_vector=raw_global_vector,
            category_indices=category_indices,
            categories=[sample.category],
            score_mode=str(score_params["score_mode"]),
            timestep=int(score_params["anomaly_timestep"]),
            descriptor_weight=float(score_params["descriptor_weight"]),
            global_descriptor_weight=float(score_params["global_descriptor_weight"]),
            multi_timestep_scoring_enabled=bool(score_params["multi_timestep_scoring_enabled"]),
            multi_timestep_fractions=score_params["multi_timestep_fractions"],
            anomaly_map_gaussian_sigma=float(score_params["anomaly_map_gaussian_sigma"]),
            spatial_topk_percent=float(score_params["spatial_topk_percent"]),
            lambda_residual=float(score_params["lambda_residual"]),
            lambda_noise=float(score_params["lambda_noise"]),
            lambda_descriptor_spatial=float(score_params["lambda_descriptor_spatial"]),
            lambda_descriptor_global=float(score_params["lambda_descriptor_global"]),
            global_descriptor_calibration=session.global_descriptor_calibration_by_category,
            global_branch_gate_calibration=session.global_branch_gate_calibration_by_category,
            spatial_exceedance_calibration=session.spatial_exceedance_calibration_by_category,
            object_score_calibration=session.object_score_calibration,
            object_score_strategy=str(score_params.get("object_score_strategy", "legacy_raw")),
            enable_internal_defect_gate=bool(score_params.get("enable_internal_defect_gate", False)),
            object_mask_threshold=sample_object_mask_threshold,
            object_mask_dilation_kernel_size=sample_object_mask_dilation_kernel_size,
        )

    raw_localization_map = select_localization_basis_map(
        scored,
        sample_localization_basis,
    ).detach().cpu()
    normalized_map = normalize_anomaly_map(raw_localization_map)
    object_mask = scored.object_mask.detach().cpu()
    localization_method = "per_category_calibrated_threshold"
    if sample_localization_calibration is not None:
        predicted_mask = apply_localization_calibration(
            raw_localization_map,
            object_mask=object_mask,
            calibration=sample_localization_calibration,
            normalized=False,
        )
    else:
        localization_method = "percentile_threshold_fallback"
        predicted_mask = threshold_anomaly_map(
            normalized_map,
            percentile=float(score_params["pixel_threshold_percentile"]),
            object_mask=object_mask,
            normalized=True,
        )
    gt_mask = load_gt_mask(
        sample.gt_mask_path,
        target_size=session.target_size,
        crop_box_xyxy=sample.crop_box_xyxy,
        crop_source_size_hw=sample.crop_source_size_hw,
    )
    calibration = session.masi_calibration_by_category[sample.category]
    package = build_evidence_package(
        category=sample.category,
        split=sample.split,
        defect_label=sample.defect_label,
        sample_id=sample.sample_id,
        is_anomalous=sample.is_anomalous,
        score_mode=str(score_params["score_mode"]),
        score_label=scored.score_basis_label,
        raw_score=float(scored.anomaly_score[0].item()),
        calibration=calibration,
        normalized_anomaly_map=normalized_map[0],
        score_basis_map=scored.score_basis_map[0].detach().cpu(),
        descriptor_support=scored.descriptor_support[0].detach().cpu(),
        object_mask=object_mask[0].detach().cpu(),
        predicted_mask=predicted_mask[0].detach().cpu(),
        descriptor_maps=sample.conditioning.descriptor_maps.detach().cpu(),
        support_maps=sample.conditioning.support_maps.detach().cpu(),
        global_descriptor_score=float(scored.global_descriptor_score[0].item()),
        provenance={
            "app_mode": "inspect_new_rgb",
            "source_eval_run": str(session.eval_run_root),
            "object_score_strategy": scored.object_score_strategy,
            "internal_defect_gate_enabled": bool(score_params.get("enable_internal_defect_gate", False)),
            "internal_defect_gate_fired": bool(scored.internal_defect_gate_fired[0].item()),
            "internal_defect_gate_score_changed": bool(scored.internal_defect_gate_score_changed[0].item()),
            "internal_defect_gate_mode": str(scored.internal_defect_gate_mode[0]),
            "internal_defect_gate_reason": str(scored.internal_defect_gate_reason[0]),
            "spatial_score_pre_gate": float(scored.spatial_score_pre_gate[0].item()),
            "spatial_score_post_gate": float(scored.spatial_score_post_gate[0].item()),
            "rescue_score": float(scored.rescue_score[0].item()),
            "ir_dominant_score": float(scored.ir_dominant_score[0].item()),
            "raw_ir_hotspot_area_fraction": sample_ir_hotspot_area_fraction,
            "raw_ir_hotspot_compactness": sample_ir_hotspot_compactness,
            "raw_ir_mean_hotspot_intensity": sample_ir_mean_hotspot_intensity,
            "raw_cross_inconsistency_score": sample_cross_inconsistency_score,
            "cross_inconsistency_q99": float(scored.cross_inconsistency_q99[0].item()),
            "cross_inconsistency_top1": float(scored.cross_inconsistency_top1[0].item()),
            "cross_geo_support_top1": float(scored.cross_geo_support_top1[0].item()),
            "ir_hotspot_top1": float(scored.ir_hotspot_top1[0].item()),
            "ir_gradient_top1": float(scored.ir_gradient_top1[0].item()),
            "gate_ir_hotspot_area_fraction_threshold": (
                float(sample_gate_calibration.ir_hotspot_area_fraction_threshold)
                if sample_gate_calibration is not None
                else None
            ),
            "gate_ir_hotspot_compactness_threshold": (
                float(sample_gate_calibration.ir_hotspot_compactness_threshold)
                if sample_gate_calibration is not None
                else None
            ),
            "gate_ir_mean_hotspot_intensity_threshold": (
                float(sample_gate_calibration.ir_mean_hotspot_intensity_threshold)
                if sample_gate_calibration is not None
                else None
            ),
            "gate_cross_inconsistency_score_threshold": (
                float(sample_gate_calibration.cross_inconsistency_score_threshold)
                if sample_gate_calibration is not None
                else None
            ),
            "gate_spatial_score_q95_threshold": (
                float(sample_gate_calibration.spatial_score_q95)
                if sample_gate_calibration is not None
                else None
            ),
            "gate_rescue_score_q95_threshold": (
                float(sample_gate_calibration.rescue_score_q95)
                if sample_gate_calibration is not None
                else None
            ),
            "gate_rescue_score_q99_threshold": (
                float(sample_gate_calibration.rescue_score_q99)
                if sample_gate_calibration is not None
                else None
            ),
            "gate_required_cues": (
                list(sample_gate_calibration.required_cues)
                if sample_gate_calibration is not None
                else []
            ),
            "gate_degenerate_cues": (
                list(sample_gate_calibration.degenerate_cues)
                if sample_gate_calibration is not None
                else []
            ),
            "localization_basis": sample_localization_basis,
            "timesteps_used": list(scored.timesteps_used),
            "localization_method": localization_method,
            "localization_threshold": (
                float(sample_localization_calibration.threshold)
                if sample_localization_calibration is not None
                else None
            ),
            "localization_min_region_pixels": (
                int(sample_localization_calibration.min_region_pixels)
                if sample_localization_calibration is not None
                else None
            ),
            "localization_mask_closing_kernel_size": (
                int(sample_localization_calibration.mask_closing_kernel_size)
                if sample_localization_calibration is not None
                else int(score_params.get("localization_mask_closing_kernel_size", 0))
            ),
            "localization_fill_holes": (
                bool(sample_localization_calibration.fill_holes)
                if sample_localization_calibration is not None
                else bool(score_params.get("localization_fill_holes", False))
            ),
            "object_mask_threshold": sample_object_mask_threshold,
            "object_mask_dilation_kernel_size": sample_object_mask_dilation_kernel_size,
            "local_exceedance_score": float(scored.local_exceedance_score[0].item()),
            "local_z_score": float(scored.local_z_score[0].item()),
            "global_z_score": float(scored.global_z_score[0].item()),
            "largest_component_area": float(scored.largest_component_area[0].item()),
            "largest_component_mean_excess": float(scored.largest_component_mean_excess[0].item()),
            "largest_component_peak_excess": float(scored.largest_component_peak_excess[0].item()),
            "largest_component_mass_excess": float(scored.largest_component_mass_excess[0].item()),
            "largest_component_area_fraction": float(scored.largest_component_area_fraction[0].item()),
            "num_exceedance_components": int(round(float(scored.num_exceedance_components[0].item()))),
        },
        source_paths={
            "rgb_path": sample.rgb_path,
            "gt_mask_path": sample.gt_mask_path,
            "source_rgb_path": sample.source_rgb_path,
            "source_ir_path": sample.source_ir_path,
            "source_pointcloud_path": sample.source_pointcloud_path,
            "visualization_path": "",
        },
    )
    retrieved_context = retrieve_context_for_evidence(
        package,
        knowledge_base_root=(
            Path(knowledge_base_root)
            if knowledge_base_root
            else _default_retrieval_root(session.repo_root)
        ),
        top_k=3,
    )
    package_path = save_evidence_package(package, session_dir / "evidence" / "package.json")
    retrieval_query_path = session_dir / "llm" / "retrieval_query.json"
    retrieval_query_path.parent.mkdir(parents=True, exist_ok=True)
    retrieval_query_path.write_text(
        json.dumps(build_retrieval_query(package), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    retrieved_context_path = session_dir / "llm" / "retrieved_context.json"
    retrieved_context_path.write_text(
        json.dumps([item.to_dict() for item in retrieved_context], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    rgb_stats = session.rgb_stats_by_category.get(sample.category)
    display_rgb = _denormalize_rgb(sample.x_rgb.detach().cpu(), rgb_stats).clamp(0.0, 1.0)
    display_reconstructed_rgb = _denormalize_rgb(
        scored.display_reconstructed_rgb[0].detach().cpu(),
        rgb_stats,
    ).clamp(0.0, 1.0)
    diffusion_reconstructed_rgb = _denormalize_rgb(
        scored.diffusion_reconstructed_rgb[0].detach().cpu(),
        rgb_stats,
    ).clamp(0.0, 1.0)
    panel_path = save_anomaly_panel(
        rgb=display_rgb,
        reconstructed_rgb=display_reconstructed_rgb,
        reconstructed_label=scored.display_reconstruction_label,
        comparison_rgb=diffusion_reconstructed_rgb,
        comparison_label=scored.diffusion_reconstruction_label,
        anomaly_map=normalized_map[0].detach().cpu(),
        score_map=scored.score_basis_map[0].detach().cpu(),
        score_map_label=scored.score_basis_label,
        support_map=scored.descriptor_support[0].detach().cpu(),
        gt_mask=gt_mask,
        anomaly_score=float(scored.anomaly_score[0].item()),
        title=f"{sample.category} / {sample.defect_label} / {sample.sample_name}",
        path=session_dir / "predictions" / f"{sample.sample_name}.png",
    )

    return {
        "session_dir": session_dir,
        "sample": sample,
        "scored": scored,
        "display_rgb": display_rgb,
        "display_reconstructed_rgb": display_reconstructed_rgb,
        "diffusion_reconstructed_rgb": diffusion_reconstructed_rgb,
        "normalized_map": normalized_map[0],
        "object_mask": object_mask[0],
        "predicted_mask": predicted_mask[0],
        "gt_mask": gt_mask,
        "package": package,
        "package_path": package_path,
        "retrieval_query": build_retrieval_query(package),
        "retrieval_query_path": str(retrieval_query_path),
        "retrieved_context": retrieved_context,
        "retrieved_context_path": str(retrieved_context_path),
        "panel_path": panel_path,
        "localization_method": localization_method,
        "localization_basis": sample_localization_basis,
        "object_score_strategy": scored.object_score_strategy,
        "internal_defect_gate_fired": bool(scored.internal_defect_gate_fired[0].item()),
        "internal_defect_gate_score_changed": bool(scored.internal_defect_gate_score_changed[0].item()),
        "internal_defect_gate_mode": str(scored.internal_defect_gate_mode[0]),
        "rescue_score": float(scored.rescue_score[0].item()),
    }
