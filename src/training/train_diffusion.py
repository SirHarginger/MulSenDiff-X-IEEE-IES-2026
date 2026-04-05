from __future__ import annotations

import json
import platform
import random
import subprocess
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Mapping, Sequence

import torch
from PIL import Image
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from src.data_loader import (
    GLOBAL_FEATURE_NAMES,
    DescriptorConditioningDataset,
    GlobalVectorNormalizationStats,
    RGBNormalizationStats,
    build_runtime_training_manifests,
    collate_model_samples,
    compute_masked_rgb_normalization_stats,
    compute_masked_rgb_normalization_stats_by_category,
    compute_global_vector_normalization_stats_by_category,
    save_rgb_normalization_stats,
    save_rgb_normalization_stats_by_category,
    save_global_vector_normalization_stats_by_category,
)
from src.evaluation.metrics import (
    aupro,
    binary_f1_score,
    image_average_precision,
    image_level_auroc,
    intersection_over_union,
    pixel_level_auroc,
)
from src.project_layout import (
    default_output_root,
    is_joint_category_request,
    resolve_processed_root,
    resolve_regime_identity,
    resolve_scope,
)
from src.category_policies import (
    is_archetype_b,
    resolve_archetype_b_gate_mode,
    resolve_localization_min_region_area_fraction,
)
from src.explainer.evidence_builder import build_evidence_package, save_evidence_package
from src.explainer.llm_pipeline import (
    GeminiExplanationProvider,
    generate_operator_report,
    load_gemini_provider_config,
    save_explanation_bundle,
)
from src.explainer.report import render_templated_explanation, save_templated_explanation
from src.explainer.retriever import retrieve_context_for_evidence
from src.inference.anomaly_scorer import (
    OBJECT_SCORE_FEATURE_NAMES,
    GlobalBranchGateCalibration,
    LogisticObjectScoreCalibration,
    SpatialExceedanceCalibration,
    build_object_score_feature_matrix,
    score_batch,
    select_localization_basis_map,
)
from src.inference.global_descriptor_scoring import (
    GlobalDescriptorCalibration,
    fit_global_descriptor_calibration,
)
from src.inference.localization import (
    LocalizationCalibration,
    apply_localization_calibration,
    estimate_object_mask,
    fit_localization_calibration,
    normalize_anomaly_map,
    remove_small_connected_components,
    threshold_anomaly_map,
)
from src.inference.quantification import MasiCalibration, fit_masi_calibration
from src.models.mulsendiffx import MulSenDiffX
from src.utils.checkpoint import load_checkpoint, read_checkpoint_payload, save_checkpoint
from src.utils.logger import RunPaths, append_jsonl, create_run_dir, write_history_csv, write_json
from src.utils.visualization import (
    save_anomaly_panel,
    save_evaluation_plot_bundle,
    save_training_curves,
)


@dataclass(frozen=True)
class DiffusionInputBatch:
    x_rgb: torch.Tensor
    M: torch.Tensor
    S: torch.Tensor
    g: torch.Tensor
    g_raw: torch.Tensor
    categories: Sequence[str]
    category_indices: torch.Tensor
    defect_labels: Sequence[str]
    sample_ids: Sequence[str]
    sample_names: Sequence[str]
    is_anomalous: Sequence[bool]
    sample_dirs: Sequence[str]
    rgb_paths: Sequence[str]
    gt_mask_paths: Sequence[str]
    source_rgb_paths: Sequence[str]
    source_ir_paths: Sequence[str]
    source_pointcloud_paths: Sequence[str]
    crop_boxes_xyxy: Sequence[tuple[int, int, int, int] | None]
    crop_source_sizes_hw: Sequence[tuple[int, int]]
    rgb_normalization_modes: Sequence[str]


@dataclass(frozen=True)
class RuntimeSettings:
    requested_mode: str
    resolved_device: torch.device
    use_amp: bool
    fallback_reason: str = ""


@dataclass(frozen=True)
class ScoreWeights:
    lambda_residual: float
    lambda_noise: float
    lambda_descriptor_spatial: float
    lambda_descriptor_global: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "lambda_residual": round(float(self.lambda_residual), 6),
            "lambda_noise": round(float(self.lambda_noise), 6),
            "lambda_descriptor_spatial": round(float(self.lambda_descriptor_spatial), 6),
            "lambda_descriptor_global": round(float(self.lambda_descriptor_global), 6),
        }


def _normalize_selected_categories(
    *,
    category: str = "",
    categories: Sequence[str] | None = None,
    fallback: Sequence[str] | None = None,
) -> list[str]:
    requested = [item.strip() for item in (categories or []) if str(item).strip()]
    if requested:
        return requested
    if str(category).strip():
        return [str(category).strip()]
    return [item.strip() for item in (fallback or []) if str(item).strip()]


def _run_slug_for_categories(selected_categories: Sequence[str]) -> str | None:
    if not selected_categories:
        return None
    if len(selected_categories) == 1:
        return selected_categories[0]
    return f"shared_{len(selected_categories)}cat"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git_commit_hash() -> str:
    try:
        probe = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_repo_root(),
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return ""
    return probe.stdout.strip() if probe.returncode == 0 else ""


def _git_dirty() -> bool:
    try:
        probe = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=_repo_root(),
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return False
    return bool(probe.stdout.strip()) if probe.returncode == 0 else False


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _runtime_provenance(*, resolved_device: torch.device, seed: int) -> Dict[str, object]:
    cuda_available = bool(torch.cuda.is_available())
    cuda_device_count = int(torch.cuda.device_count()) if cuda_available else 0
    cuda_device_name = ""
    if cuda_available and cuda_device_count > 0:
        try:
            device_index = resolved_device.index if resolved_device.type == "cuda" and resolved_device.index is not None else 0
            cuda_device_name = str(torch.cuda.get_device_name(device_index))
        except Exception:
            cuda_device_name = ""
    return {
        "seed": int(seed),
        "git_commit": _git_commit_hash(),
        "git_dirty": _git_dirty(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda or "",
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
        "cuda_device_name": cuda_device_name,
        "resolved_device": str(resolved_device),
        "platform": platform.platform(),
        "executable": sys.executable,
    }


def _per_category_metric_rows(per_category: Dict[str, Dict[str, float | int]]) -> List[Dict[str, object]]:
    return [
        {"category": category, **metrics}
        for category, metrics in sorted(per_category.items())
    ]


def _resolve_shared_preview_selection(loader: DataLoader, *, seed: int) -> Dict[str, object] | None:
    dataset = getattr(loader, "dataset", None)
    if dataset is None:
        return None
    category_vocabulary = list(getattr(dataset, "category_vocabulary", []))
    if len(category_vocabulary) <= 1:
        return None
    records = list(getattr(dataset, "records", []))
    if not records:
        return None

    rng = random.Random(seed)
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for record in records:
        category = str(record.get("category", "")).strip()
        sample_name = str(record.get("sample_name", "")).strip()
        if not category or not sample_name:
            continue
        grouped.setdefault(category, []).append(record)

    selected_samples: List[Dict[str, object]] = []
    for category in sorted(grouped):
        choice = rng.choice(grouped[category])
        selected_samples.append(
            {
                "category": category,
                "sample_name": str(choice.get("sample_name", "")),
                "sample_id": str(choice.get("sample_id", "")),
                "defect_label": str(choice.get("defect_label", "")),
            }
        )

    if not selected_samples:
        return None

    return {
        "mode": "shared_joint",
        "strategy": "one_random_sample_per_category",
        "seed": seed,
        "selected_samples": selected_samples,
        "effective_max_visualizations": len(selected_samples),
    }


def prepare_diffusion_batch(batch: Dict[str, object]) -> DiffusionInputBatch:
    category_indices = batch.get("category_indices")
    if category_indices is None:
        category_indices = torch.zeros((len(batch["sample_ids"]),), dtype=torch.long)
    return DiffusionInputBatch(
        x_rgb=batch["x_rgb"],  # type: ignore[arg-type]
        M=batch["descriptor_maps"],  # type: ignore[arg-type]
        S=batch["support_maps"],  # type: ignore[arg-type]
        g=batch["global_vectors"],  # type: ignore[arg-type]
        g_raw=batch.get("raw_global_vectors", batch["global_vectors"]),  # type: ignore[arg-type]
        categories=batch["categories"],  # type: ignore[arg-type]
        category_indices=category_indices,  # type: ignore[arg-type]
        defect_labels=batch["defect_labels"],  # type: ignore[arg-type]
        sample_ids=batch["sample_ids"],  # type: ignore[arg-type]
        sample_names=batch["sample_names"],  # type: ignore[arg-type]
        is_anomalous=batch["is_anomalous"],  # type: ignore[arg-type]
        sample_dirs=batch["sample_dirs"],  # type: ignore[arg-type]
        rgb_paths=batch["rgb_paths"],  # type: ignore[arg-type]
        gt_mask_paths=batch["gt_mask_paths"],  # type: ignore[arg-type]
        source_rgb_paths=batch["source_rgb_paths"],  # type: ignore[arg-type]
        source_ir_paths=batch["source_ir_paths"],  # type: ignore[arg-type]
        source_pointcloud_paths=batch["source_pointcloud_paths"],  # type: ignore[arg-type]
        crop_boxes_xyxy=batch.get("crop_boxes_xyxy", [None] * len(batch["sample_ids"])),  # type: ignore[arg-type]
        crop_source_sizes_hw=batch.get(
            "crop_source_sizes_hw",
            [(int(batch["x_rgb"].shape[-2]), int(batch["x_rgb"].shape[-1]))] * len(batch["sample_ids"]),  # type: ignore[index]
        ),  # type: ignore[arg-type]
        rgb_normalization_modes=batch.get("rgb_normalization_modes", ["none"] * len(batch["sample_ids"])),  # type: ignore[arg-type]
    )


def _resolve_category_conditioning_indices(
    category_indices: torch.Tensor,
    *,
    disable_category_embedding: bool,
) -> torch.Tensor:
    if not disable_category_embedding:
        return category_indices
    return torch.zeros_like(category_indices)


def _coerce_rgb_normalization_stats(
    value: RGBNormalizationStats | Dict[str, object] | None,
) -> RGBNormalizationStats | None:
    if value is None:
        return None
    if isinstance(value, RGBNormalizationStats):
        return value
    return RGBNormalizationStats(
        category=str(value.get("category", "")),
        mode=str(value.get("mode", "none")),
        mean=tuple(float(component) for component in value.get("mean", [0.0, 0.0, 0.0])),
        std=tuple(max(float(component), 1e-6) for component in value.get("std", [1.0, 1.0, 1.0])),
        masked_pixels=int(value.get("masked_pixels", 0)),
    )


def _coerce_rgb_normalization_stats_by_category(
    value: Dict[str, object] | None,
) -> Dict[str, RGBNormalizationStats] | None:
    if value is None:
        return None
    categories_payload = value.get("categories")
    if isinstance(categories_payload, dict):
        return {
            str(category): _coerce_rgb_normalization_stats(stats_payload if isinstance(stats_payload, dict) else None)  # type: ignore[arg-type]
            for category, stats_payload in categories_payload.items()
            if isinstance(stats_payload, dict)
        }
    single = _coerce_rgb_normalization_stats(value)
    return {single.category: single} if single is not None and single.category else None


def _coerce_global_vector_normalization_stats_by_category(
    value: Dict[str, object] | None,
) -> Dict[str, GlobalVectorNormalizationStats] | None:
    if value is None:
        return None
    categories_payload = value.get("categories")
    if isinstance(categories_payload, dict):
        stats_by_category: Dict[str, GlobalVectorNormalizationStats] = {}
        for category, stats_payload in categories_payload.items():
            if not isinstance(stats_payload, dict):
                continue
            feature_names = tuple(str(item) for item in stats_payload.get("feature_names", GLOBAL_FEATURE_NAMES))
            stats_by_category[str(category)] = GlobalVectorNormalizationStats(
                category=str(stats_payload.get("category", category)),
                feature_names=feature_names,
                mean=tuple(float(item) for item in stats_payload.get("mean", [0.0] * len(feature_names))),
                std=tuple(max(float(item), 1e-6) for item in stats_payload.get("std", [1.0] * len(feature_names))),
                sample_count=int(stats_payload.get("sample_count", 0)),
            )
        return stats_by_category
    return None


def _resolve_requested_mode(device: str, device_mode: str) -> str:
    requested = (device_mode or device or "cpu").strip().lower()
    if requested not in {"cpu", "cuda", "cuda_first"}:
        raise ValueError(f"Unsupported device mode {requested!r}. Expected 'cpu', 'cuda', or 'cuda_first'.")
    return requested


def _autocast_context(device: torch.device, *, use_amp: bool):
    if use_amp and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _validate_model_runtime(
    model: MulSenDiffX,
    batch: DiffusionInputBatch,
    *,
    device: torch.device,
    use_amp: bool,
    run_backward: bool,
    disable_category_embedding: bool = False,
) -> None:
    clean_rgb = batch.x_rgb.to(device)
    descriptor_maps = batch.M.to(device)
    support_maps = batch.S.to(device)
    global_vector = batch.g.to(device)
    category_indices = _resolve_category_conditioning_indices(
        batch.category_indices.to(device),
        disable_category_embedding=disable_category_embedding,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(123)
    model.zero_grad(set_to_none=True)
    if run_backward:
        model.train()
        with _autocast_context(device, use_amp=use_amp):
            outputs = model.training_outputs(
                clean_rgb,
                descriptor_maps,
                global_vector,
                category_indices=category_indices,
                generator=generator,
                support_maps=support_maps,
            )
        outputs.loss.backward()
        model.zero_grad(set_to_none=True)
    else:
        model.eval()
        with torch.no_grad():
            with _autocast_context(device, use_amp=use_amp):
                model.training_outputs(
                    clean_rgb,
                    descriptor_maps,
                    global_vector,
                    category_indices=category_indices,
                    generator=generator,
                    support_maps=support_maps,
                )
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def resolve_runtime(
    *,
    requested_mode: str,
    model: MulSenDiffX,
    validation_batch: DiffusionInputBatch,
    run_backward: bool,
    disable_category_embedding: bool = False,
) -> RuntimeSettings:
    if requested_mode == "cpu":
        model.to(torch.device("cpu"))
        return RuntimeSettings(requested_mode=requested_mode, resolved_device=torch.device("cpu"), use_amp=False)

    static_failures: List[str] = []
    try:
        probe = subprocess.run(
            ["nvidia-smi"],
            check=False,
            capture_output=True,
            text=True,
        )
        if probe.returncode != 0:
            stderr = probe.stderr.strip() or probe.stdout.strip() or "unknown nvidia-smi failure"
            static_failures.append(f"nvidia-smi failed: {stderr}")
    except FileNotFoundError:
        static_failures.append("nvidia-smi not found")

    if not torch.cuda.is_available():
        static_failures.append("torch.cuda.is_available() returned False")
    if torch.cuda.device_count() <= 0:
        static_failures.append("torch.cuda.device_count() returned 0")

    if static_failures:
        reason = "; ".join(static_failures)
        if requested_mode == "cuda":
            raise RuntimeError(reason)
        model.to(torch.device("cpu"))
        return RuntimeSettings(
            requested_mode=requested_mode,
            resolved_device=torch.device("cpu"),
            use_amp=False,
            fallback_reason=reason,
        )

    cuda_device = torch.device("cuda")
    model.to(cuda_device)
    try:
        _validate_model_runtime(
            model,
            validation_batch,
            device=cuda_device,
            use_amp=True,
            run_backward=run_backward,
            disable_category_embedding=disable_category_embedding,
        )
    except Exception as exc:
        reason = f"cuda validation failed: {exc}"
        if requested_mode == "cuda":
            raise RuntimeError(reason) from exc
        model.to(torch.device("cpu"))
        torch.cuda.empty_cache()
        return RuntimeSettings(
            requested_mode=requested_mode,
            resolved_device=torch.device("cpu"),
            use_amp=False,
            fallback_reason=reason,
        )
    return RuntimeSettings(requested_mode=requested_mode, resolved_device=cuda_device, use_amp=True)


def resolve_score_weights(
    *,
    total_epochs: int,
    epoch: int | None,
    lambda_residual: float,
    lambda_noise: float,
    lambda_descriptor_spatial: float,
    lambda_descriptor_global: float,
    score_weight_schedule: Dict[str, object] | None,
) -> ScoreWeights:
    if epoch is not None and total_epochs > 0 and score_weight_schedule:
        warmup_payload = score_weight_schedule.get("warmup")
        main_payload = score_weight_schedule.get("main")
        if isinstance(warmup_payload, dict) and isinstance(main_payload, dict):
            warmup_weights = ScoreWeights(
                lambda_residual=float(warmup_payload.get("lambda_residual", lambda_residual)),
                lambda_noise=float(warmup_payload.get("lambda_noise", lambda_noise)),
                lambda_descriptor_spatial=float(
                    warmup_payload.get("lambda_descriptor_spatial", lambda_descriptor_spatial)
                ),
                lambda_descriptor_global=float(
                    warmup_payload.get("lambda_descriptor_global", lambda_descriptor_global)
                ),
            )
            main_weights = ScoreWeights(
                lambda_residual=float(main_payload.get("lambda_residual", lambda_residual)),
                lambda_noise=float(main_payload.get("lambda_noise", lambda_noise)),
                lambda_descriptor_spatial=float(main_payload.get("lambda_descriptor_spatial", lambda_descriptor_spatial)),
                lambda_descriptor_global=float(main_payload.get("lambda_descriptor_global", lambda_descriptor_global)),
            )
            progress = float(epoch) / max(float(total_epochs), 1.0)
            warmup_fraction = float(score_weight_schedule.get("warmup_fraction", 0.4))
            transition_fraction = max(float(score_weight_schedule.get("transition_fraction", 0.0)), 0.0)
            transition_end = min(warmup_fraction + transition_fraction, 1.0)
            if progress <= warmup_fraction:
                return warmup_weights
            if transition_fraction <= 0.0 or progress >= transition_end:
                return main_weights
            blend = (progress - warmup_fraction) / max(transition_end - warmup_fraction, 1e-6)
            return ScoreWeights(
                lambda_residual=warmup_weights.lambda_residual
                + (main_weights.lambda_residual - warmup_weights.lambda_residual) * blend,
                lambda_noise=warmup_weights.lambda_noise
                + (main_weights.lambda_noise - warmup_weights.lambda_noise) * blend,
                lambda_descriptor_spatial=warmup_weights.lambda_descriptor_spatial
                + (main_weights.lambda_descriptor_spatial - warmup_weights.lambda_descriptor_spatial) * blend,
                lambda_descriptor_global=warmup_weights.lambda_descriptor_global
                + (main_weights.lambda_descriptor_global - warmup_weights.lambda_descriptor_global) * blend,
            )
    return ScoreWeights(
        lambda_residual=float(lambda_residual),
        lambda_noise=float(lambda_noise),
        lambda_descriptor_spatial=float(lambda_descriptor_spatial),
        lambda_descriptor_global=float(lambda_descriptor_global),
    )


def _denormalize_rgb_tensor(
    rgb: torch.Tensor,
    stats: RGBNormalizationStats | None,
) -> torch.Tensor:
    if stats is None or stats.mode == "none":
        return rgb
    mean = torch.tensor(stats.mean, dtype=rgb.dtype, device=rgb.device).view(3, 1, 1)
    std = torch.tensor(stats.std, dtype=rgb.dtype, device=rgb.device).view(3, 1, 1)
    return rgb * std + mean


def _resolve_rgb_stats_for_category(
    category: str,
    *,
    rgb_normalization_stats: RGBNormalizationStats | None = None,
    rgb_normalization_stats_by_category: Dict[str, RGBNormalizationStats] | None = None,
) -> RGBNormalizationStats | None:
    if rgb_normalization_stats_by_category is not None:
        return rgb_normalization_stats_by_category.get(category)
    return rgb_normalization_stats


def _resolve_localization_calibration_for_category(
    category: str,
    *,
    localization_calibration: LocalizationCalibration | None = None,
    localization_calibration_by_category: Dict[str, LocalizationCalibration] | None = None,
) -> LocalizationCalibration | None:
    if localization_calibration_by_category is not None:
        return localization_calibration_by_category.get(category)
    return localization_calibration


def _resolve_global_branch_gate_calibration_for_category(
    category: str,
    *,
    global_branch_gate_calibration: GlobalBranchGateCalibration | None = None,
    global_branch_gate_calibration_by_category: Dict[str, GlobalBranchGateCalibration] | None = None,
) -> GlobalBranchGateCalibration | None:
    if global_branch_gate_calibration_by_category is not None:
        return global_branch_gate_calibration_by_category.get(category)
    return global_branch_gate_calibration


def build_diffusion_dataloaders(
    *,
    data_root: Path | str = "data",
    processed_root: Path | str | None = None,
    descriptor_index_csv: Path | str = "data/processed/manifests/descriptor_index.csv",
    category: str = "capsule",
    categories: Sequence[str] | None = None,
    batch_size: int = 2,
    target_size: tuple[int, int] = (256, 256),
    object_crop_enabled: bool = False,
    object_crop_margin_ratio: float = 0.10,
    object_crop_mask_source: str = "rgb_then_pc_density",
    rgb_normalization_mode: str = "none",
    rgb_normalization_stats: RGBNormalizationStats | Dict[str, object] | None = None,
    rgb_normalization_stats_by_category: Dict[str, object] | None = None,
    global_vector_normalization_stats_by_category: Dict[str, object] | None = None,
    manifests_root: Path | str | None = None,
    sampler_mode: str = "balanced_by_category",
    category_vocabulary: Sequence[str] | None = None,
    seed: int = 7,
) -> Dict[str, object]:
    del descriptor_index_csv
    data_root = Path(data_root)
    resolved_processed_root = resolve_processed_root(data_root=data_root, processed_root=processed_root)
    selected_categories = _normalize_selected_categories(category=category, categories=categories)
    manifests = build_runtime_training_manifests(
        data_root=data_root,
        processed_root=resolved_processed_root,
        categories=selected_categories,
        manifests_root=manifests_root,
    )
    resolved_category_vocabulary = list(category_vocabulary or manifests["selected_categories"])  # type: ignore[index]
    joint_mode = len(resolved_category_vocabulary) > 1

    train_dataset = DescriptorConditioningDataset(
        manifests["train_csv"],
        data_root=".",
        target_size=target_size,
        global_stats_path=manifests["global_stats_json"],
        object_crop_enabled=object_crop_enabled,
        object_crop_margin_ratio=object_crop_margin_ratio,
        object_crop_mask_source=object_crop_mask_source,
        rgb_normalization_mode=rgb_normalization_mode,
        category_vocabulary=resolved_category_vocabulary,
    )
    calibration_dataset = DescriptorConditioningDataset(
        manifests["calibration_csv"],
        data_root=".",
        target_size=target_size,
        global_stats_path=manifests["global_stats_json"],
        object_crop_enabled=object_crop_enabled,
        object_crop_margin_ratio=object_crop_margin_ratio,
        object_crop_mask_source=object_crop_mask_source,
        rgb_normalization_mode=rgb_normalization_mode,
        category_vocabulary=resolved_category_vocabulary,
    )
    synthetic_validation_dataset = DescriptorConditioningDataset(
        manifests["synthetic_csv"],
        data_root=".",
        target_size=target_size,
        global_stats_path=manifests["global_stats_json"],
        object_crop_enabled=object_crop_enabled,
        object_crop_margin_ratio=object_crop_margin_ratio,
        object_crop_mask_source=object_crop_mask_source,
        rgb_normalization_mode=rgb_normalization_mode,
        category_vocabulary=resolved_category_vocabulary,
    )
    eval_dataset = DescriptorConditioningDataset(
        manifests["eval_csv"],
        data_root=".",
        target_size=target_size,
        global_stats_path=manifests["global_stats_json"],
        object_crop_enabled=object_crop_enabled,
        object_crop_margin_ratio=object_crop_margin_ratio,
        object_crop_mask_source=object_crop_mask_source,
        rgb_normalization_mode=rgb_normalization_mode,
        category_vocabulary=resolved_category_vocabulary,
    )

    resolved_rgb_normalization_stats = _coerce_rgb_normalization_stats(rgb_normalization_stats)
    resolved_rgb_normalization_stats_by_category = _coerce_rgb_normalization_stats_by_category(
        rgb_normalization_stats_by_category
    )
    resolved_global_vector_stats_by_category = _coerce_global_vector_normalization_stats_by_category(
        global_vector_normalization_stats_by_category
    )
    if rgb_normalization_mode != "none":
        if joint_mode:
            if resolved_rgb_normalization_stats_by_category is None:
                resolved_rgb_normalization_stats_by_category = compute_masked_rgb_normalization_stats_by_category(
                    train_dataset
                )
            train_dataset.set_rgb_normalization_stats_by_category(resolved_rgb_normalization_stats_by_category)
            calibration_dataset.set_rgb_normalization_stats_by_category(resolved_rgb_normalization_stats_by_category)
            synthetic_validation_dataset.set_rgb_normalization_stats_by_category(
                resolved_rgb_normalization_stats_by_category
            )
            eval_dataset.set_rgb_normalization_stats_by_category(resolved_rgb_normalization_stats_by_category)
        else:
            if resolved_rgb_normalization_stats is None:
                resolved_rgb_normalization_stats = compute_masked_rgb_normalization_stats(
                    train_dataset,
                    category=resolved_category_vocabulary[0] if resolved_category_vocabulary else category,
                )
            train_dataset.set_rgb_normalization_stats(resolved_rgb_normalization_stats)
            calibration_dataset.set_rgb_normalization_stats(resolved_rgb_normalization_stats)
            synthetic_validation_dataset.set_rgb_normalization_stats(resolved_rgb_normalization_stats)
            eval_dataset.set_rgb_normalization_stats(resolved_rgb_normalization_stats)

    if joint_mode:
        if resolved_global_vector_stats_by_category is None:
            resolved_global_vector_stats_by_category = compute_global_vector_normalization_stats_by_category(
                manifests["train_rows"],  # type: ignore[arg-type]
                data_root=".",
            )
        train_dataset.set_global_vector_normalization_stats_by_category(resolved_global_vector_stats_by_category)
        calibration_dataset.set_global_vector_normalization_stats_by_category(resolved_global_vector_stats_by_category)
        synthetic_validation_dataset.set_global_vector_normalization_stats_by_category(
            resolved_global_vector_stats_by_category
        )
        eval_dataset.set_global_vector_normalization_stats_by_category(resolved_global_vector_stats_by_category)

    train_category_counts: Dict[str, int] = {}
    for row in manifests["train_rows"]:  # type: ignore[index]
        train_category_counts[row.category] = train_category_counts.get(row.category, 0) + 1

    train_sampler = None
    should_balance = joint_mode and sampler_mode == "balanced_by_category"
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    if should_balance and len(train_dataset) > 0:
        sample_weights = [
            1.0 / max(float(train_category_counts.get(record["category"], 1)), 1.0)
            for record in train_dataset.records
        ]
        train_sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
            generator=train_generator,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        generator=train_generator if train_sampler is None else None,
        num_workers=0,
        collate_fn=collate_model_samples,
    )
    train_reference_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_model_samples,
    )
    calibration_loader = DataLoader(
        calibration_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_model_samples,
    )
    synthetic_validation_loader = DataLoader(
        synthetic_validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_model_samples,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_model_samples,
    )

    return {
        "train_loader": train_loader,
        "train_reference_loader": train_reference_loader,
        "calibration_loader": calibration_loader,
        "synthetic_validation_loader": synthetic_validation_loader,
        "eval_loader": eval_loader,
        "train_dataset": train_dataset,
        "calibration_dataset": calibration_dataset,
        "synthetic_validation_dataset": synthetic_validation_dataset,
        "eval_dataset": eval_dataset,
        "manifests": manifests,
        "processed_root": str(resolved_processed_root),
        "rgb_normalization_stats": resolved_rgb_normalization_stats,
        "rgb_normalization_stats_by_category": resolved_rgb_normalization_stats_by_category,
        "global_vector_normalization_stats_by_category": resolved_global_vector_stats_by_category,
        "selected_categories": resolved_category_vocabulary,
        "category_vocabulary": resolved_category_vocabulary,
        "train_category_counts": train_category_counts,
        "sampler_mode": sampler_mode if should_balance else "shuffle",
    }


def build_model(
    *,
    descriptor_channels: int,
    global_dim: int,
    base_channels: int = 32,
    global_embedding_dim: int = 128,
    time_embedding_dim: int = 128,
    num_categories: int = 1,
    category_embedding_dim: int = 32,
    attention_heads: int = 4,
    latent_channels: int = 4,
    diffusion_steps: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    noise_schedule: str = "cosine",
    autoencoder_reconstruction_weight: float = 0.5,
    diffusion_reconstruction_weight: float = 0.1,
    edge_reconstruction_weight: float = 0.05,
    enable_category_modality_gating: bool = False,
    spatial_descriptor_channels: int = 7,
    support_descriptor_channels: int = 6,
) -> MulSenDiffX:
    return MulSenDiffX(
        rgb_channels=3,
        descriptor_channels=descriptor_channels,
        global_dim=global_dim,
        base_channels=base_channels,
        global_embedding_dim=global_embedding_dim,
        time_embedding_dim=time_embedding_dim,
        num_categories=num_categories,
        category_embedding_dim=category_embedding_dim,
        attention_heads=attention_heads,
        latent_channels=latent_channels,
        diffusion_steps=diffusion_steps,
        beta_start=beta_start,
        beta_end=beta_end,
        noise_schedule=noise_schedule,
        autoencoder_reconstruction_weight=autoencoder_reconstruction_weight,
        diffusion_reconstruction_weight=diffusion_reconstruction_weight,
        edge_reconstruction_weight=edge_reconstruction_weight,
        enable_category_modality_gating=enable_category_modality_gating,
        spatial_descriptor_channels=spatial_descriptor_channels,
        support_descriptor_channels=support_descriptor_channels,
    )


def train_step(
    model: MulSenDiffX,
    batch: DiffusionInputBatch,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    disable_category_embedding: bool = False,
    use_amp: bool = False,
    scaler: torch.cuda.amp.GradScaler | None = None,
    generator: torch.Generator | None = None,
) -> Dict[str, object]:
    model.train()
    optimizer.zero_grad(set_to_none=True)

    clean_rgb = batch.x_rgb.to(device)
    descriptor_maps = batch.M.to(device)
    global_vector = batch.g.to(device)
    category_indices = _resolve_category_conditioning_indices(
        batch.category_indices.to(device),
        disable_category_embedding=disable_category_embedding,
    )

    with _autocast_context(device, use_amp=use_amp):
        outputs = model.training_outputs(
            clean_rgb,
            descriptor_maps,
            global_vector,
            category_indices=category_indices,
            generator=generator,
            support_maps=batch.S.to(device),
        )
    if use_amp and scaler is not None:
        scaler.scale(outputs.loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        outputs.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    return {
        "loss": float(outputs.loss.detach().item()),
        "noise_loss": float(outputs.noise_loss.detach().item()),
        "autoencoder_reconstruction_loss": float(outputs.autoencoder_reconstruction_loss.detach().item()),
        "diffusion_reconstruction_loss": float(outputs.diffusion_reconstruction_loss.detach().item()),
        "edge_reconstruction_loss": float(outputs.edge_reconstruction_loss.detach().item()),
        "batch_shape": tuple(int(dim) for dim in clean_rgb.shape),
        "latent_shape": tuple(int(dim) for dim in outputs.clean_latent.shape),
        "conditioning_dim": int(model.global_embedding_dim + model.time_embedding_dim),
        "timesteps_mean": float(outputs.timesteps.float().mean().item()),
        "timesteps_max": int(outputs.timesteps.max().item()),
        "sample_ids": list(batch.sample_ids),
    }


def run_training_epoch(
    model: MulSenDiffX,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    disable_category_embedding: bool = False,
    use_amp: bool = False,
    epoch: int,
    max_batches: int = 0,
    seed: int = 7,
    log_every_n_steps: int = 10,
    log_to_jsonl: bool = True,
    step_log_path: Path | None = None,
) -> List[Dict[str, object]]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    batch_reports: List[Dict[str, object]] = []
    running_loss = 0.0
    start_time = perf_counter()
    total_batches = min(len(loader), max_batches) if max_batches else len(loader)

    for batch_index, raw_batch in enumerate(loader):
        batch = prepare_diffusion_batch(raw_batch)
        metrics = train_step(
            model,
            batch,
            optimizer,
            device=device,
            disable_category_embedding=disable_category_embedding,
            use_amp=use_amp,
            scaler=scaler,
            generator=generator,
        )
        running_loss += float(metrics["loss"])
        report = {
            "epoch": epoch,
            "batch_index": batch_index,
            "loss": round(float(metrics["loss"]), 6),
            "noise_loss": round(float(metrics["noise_loss"]), 6),
            "autoencoder_reconstruction_loss": round(float(metrics["autoencoder_reconstruction_loss"]), 6),
            "diffusion_reconstruction_loss": round(float(metrics["diffusion_reconstruction_loss"]), 6),
            "edge_reconstruction_loss": round(float(metrics["edge_reconstruction_loss"]), 6),
            "conditioning_dim": metrics["conditioning_dim"],
            "timesteps_mean": round(float(metrics["timesteps_mean"]), 4),
            "timesteps_max": metrics["timesteps_max"],
            "batch_shape": metrics["batch_shape"],
            "latent_shape": metrics["latent_shape"],
            "sample_ids": metrics["sample_ids"],
        }
        batch_reports.append(report)

        steps_done = batch_index + 1
        if total_batches > 0 and (
            steps_done == total_batches
            or (log_every_n_steps > 0 and steps_done % log_every_n_steps == 0)
        ):
            elapsed = perf_counter() - start_time
            steps_per_second = steps_done / max(elapsed, 1e-6)
            remaining_steps = max(total_batches - steps_done, 0)
            eta_seconds = remaining_steps / max(steps_per_second, 1e-6)
            log_payload = {
                "event": "train_step",
                "epoch": epoch,
                "step": steps_done,
                "total_steps": total_batches,
                "loss": report["loss"],
                "noise_loss": report["noise_loss"],
                "autoencoder_reconstruction_loss": report["autoencoder_reconstruction_loss"],
                "diffusion_reconstruction_loss": report["diffusion_reconstruction_loss"],
                "edge_reconstruction_loss": report["edge_reconstruction_loss"],
                "running_loss": round(running_loss / steps_done, 6),
                "steps_per_second": round(steps_per_second, 4),
                "eta_seconds": round(float(eta_seconds), 2),
            }
            print(
                f"[train] epoch={epoch} step={steps_done}/{total_batches} "
                f"loss={report['loss']:.6f} noise={report['noise_loss']:.6f} "
                f"ae={report['autoencoder_reconstruction_loss']:.6f} "
                f"diff={report['diffusion_reconstruction_loss']:.6f} "
                f"edge={report['edge_reconstruction_loss']:.6f} "
                f"avg={log_payload['running_loss']:.6f} eta={log_payload['eta_seconds']:.1f}s"
            )
            if log_to_jsonl and step_log_path is not None:
                append_jsonl(step_log_path, log_payload)

        if max_batches and len(batch_reports) >= max_batches:
            break
    return batch_reports


def run_eval_scoring(
    model: MulSenDiffX,
    loader: DataLoader,
    *,
    device: torch.device,
    disable_category_embedding: bool = False,
    target_size: tuple[int, int],
    score_mode: str = "noise_error",
    anomaly_timestep: int = 200,
    descriptor_weight: float = 0.25,
    global_descriptor_weight: float = 0.1,
    multi_timestep_scoring_enabled: bool = False,
    multi_timestep_fractions: Sequence[float] | None = None,
    anomaly_map_gaussian_sigma: float = 0.0,
    lambda_residual: float | None = None,
    lambda_noise: float | None = None,
    lambda_descriptor_spatial: float | None = None,
    lambda_descriptor_global: float | None = None,
    global_descriptor_calibration: GlobalDescriptorCalibration | Dict[str, GlobalDescriptorCalibration] | None = None,
    global_branch_gate_calibration: GlobalBranchGateCalibration | Dict[str, GlobalBranchGateCalibration] | None = None,
    spatial_exceedance_calibration: SpatialExceedanceCalibration | Dict[str, SpatialExceedanceCalibration] | None = None,
    object_score_calibration: LogisticObjectScoreCalibration | None = None,
    object_score_strategy: str = "legacy_raw",
    spatial_topk_percent: float = 0.02,
    enable_internal_defect_gate: bool = False,
    pixel_threshold_percentile: float = 0.9,
    object_mask_threshold: float = 0.02,
    localization_basis: str = "anomaly_map",
    object_mask_dilation_kernel_size: int = 9,
    localization_mask_closing_kernel_size: int | None = None,
    localization_fill_holes: bool | None = None,
    max_batches: int = 0,
    visualization_dir: Path | None = None,
    preview_selection_path: Path | None = None,
    image_score_data_path: Path | None = None,
    evidence_dir: Path | None = None,
    calibration: MasiCalibration | Dict[str, MasiCalibration] | None = None,
    localization_calibration: LocalizationCalibration | Dict[str, LocalizationCalibration] | None = None,
    max_visualizations: int = 6,
    knowledge_base_root: Path | None = None,
    retrieval_top_k: int = 3,
    enable_llm_explanations: bool = False,
    rgb_normalization_stats: RGBNormalizationStats | None = None,
    rgb_normalization_stats_by_category: Dict[str, RGBNormalizationStats] | None = None,
    preview_seed: int | None = None,
    seed: int = 17,
) -> Dict[str, Any]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    reports: List[Dict[str, object]] = []
    evidence_index: List[Dict[str, object]] = []
    image_scores: List[torch.Tensor] = []
    image_labels: List[torch.Tensor] = []
    pixel_f1_values: List[float] = []
    pixel_iou_values: List[float] = []
    sample_metrics: List[Dict[str, object]] = []
    visualizations_saved = 0
    llm_explanations_saved = 0
    gemini_provider_config = load_gemini_provider_config() if enable_llm_explanations else None
    explanation_provider = (
        GeminiExplanationProvider(gemini_provider_config)
        if gemini_provider_config is not None and gemini_provider_config.enabled
        else None
    )
    pooled_pixel_score_maps: List[torch.Tensor] = []
    pooled_pixel_target_masks: List[torch.Tensor] = []
    pooled_pixel_object_masks: List[torch.Tensor] = []
    per_category_pixel_inputs: Dict[str, Dict[str, List[torch.Tensor]]] = {}
    image_score_rows: List[Dict[str, object]] = []
    internal_defect_gate_fired_count = 0
    internal_defect_gate_score_changed_count = 0
    preview_selection = (
        _resolve_shared_preview_selection(loader, seed=preview_seed if preview_seed is not None else seed)
        if visualization_dir is not None
        else None
    )
    selected_preview_sample_names = (
        {str(item["sample_name"]) for item in preview_selection["selected_samples"]}  # type: ignore[index]
        if preview_selection is not None
        else None
    )
    saved_preview_sample_names: set[str] = set()
    preview_selection_path_str = ""
    if preview_selection is not None and preview_selection_path is not None:
        preview_selection_path_str = str(write_json(preview_selection_path, preview_selection))

    for batch_index, raw_batch in enumerate(loader):
        batch = prepare_diffusion_batch(raw_batch)
        scored = score_batch(
            model,
            batch.x_rgb.to(device),
            batch.M.to(device),
            batch.S.to(device),
            batch.g.to(device),
            _resolve_category_conditioning_indices(
                batch.category_indices.to(device),
                disable_category_embedding=disable_category_embedding,
            ),
            raw_global_vector=batch.g_raw.to(device),
            categories=batch.categories,
            score_mode=score_mode,
            timestep=anomaly_timestep,
            descriptor_weight=descriptor_weight,
            global_descriptor_weight=global_descriptor_weight,
            multi_timestep_scoring_enabled=multi_timestep_scoring_enabled,
            multi_timestep_fractions=multi_timestep_fractions,
            anomaly_map_gaussian_sigma=anomaly_map_gaussian_sigma,
            spatial_topk_percent=spatial_topk_percent,
            enable_internal_defect_gate=enable_internal_defect_gate,
            lambda_residual=lambda_residual,
            lambda_noise=lambda_noise,
            lambda_descriptor_spatial=lambda_descriptor_spatial,
            lambda_descriptor_global=lambda_descriptor_global,
            global_descriptor_calibration=global_descriptor_calibration,
            global_branch_gate_calibration=global_branch_gate_calibration,
            spatial_exceedance_calibration=spatial_exceedance_calibration,
            object_score_calibration=object_score_calibration,
            object_score_strategy=object_score_strategy,
            object_mask_threshold=object_mask_threshold,
            object_mask_dilation_kernel_size=object_mask_dilation_kernel_size,
            generator=generator,
        )
        image_scores.append(scored.anomaly_score.detach().cpu())
        labels = torch.tensor([1 if value else 0 for value in batch.is_anomalous], dtype=torch.float32)
        image_labels.append(labels)

        object_masks = scored.object_mask.detach().cpu()
        predicted_mask_rows: List[torch.Tensor] = []
        sample_raw_localization_maps: List[torch.Tensor] = []
        sample_normalized_localization_maps: List[torch.Tensor] = []
        sample_localization_bases: List[str] = []
        for sample_offset, category_name in enumerate(batch.categories):
            sample_object_mask = object_masks[sample_offset : sample_offset + 1]
            sample_localization_calibration = (
                localization_calibration.get(category_name)
                if isinstance(localization_calibration, dict)
                else localization_calibration
            )
            sample_localization_basis = (
                str(sample_localization_calibration.basis)
                if sample_localization_calibration is not None
                else str(localization_basis)
            )
            sample_raw_map = select_localization_basis_map(
                scored,
                sample_localization_basis,
            )[sample_offset : sample_offset + 1].detach().cpu()
            sample_normalized_map = normalize_anomaly_map(sample_raw_map)
            sample_raw_localization_maps.append(sample_raw_map)
            sample_normalized_localization_maps.append(sample_normalized_map)
            sample_localization_bases.append(sample_localization_basis)
            if sample_localization_calibration is not None:
                predicted_mask_rows.append(
                    apply_localization_calibration(
                        sample_raw_map,
                        object_mask=sample_object_mask,
                        calibration=sample_localization_calibration,
                        normalized=False,
                    )
                )
            else:
                predicted_mask_rows.append(
                    remove_small_connected_components(
                        threshold_anomaly_map(
                            sample_normalized_map,
                            percentile=pixel_threshold_percentile,
                            object_mask=sample_object_mask,
                            normalized=True,
                        ),
                        object_mask=sample_object_mask,
                        min_region_pixels=0,
                        closing_kernel_size=int(max(localization_mask_closing_kernel_size, 0)),
                        fill_holes=bool(localization_fill_holes),
                    )
                )
        predicted_masks = torch.cat(predicted_mask_rows, dim=0)

        for sample_offset, sample_id in enumerate(batch.sample_ids):
            sample_category_name = batch.categories[sample_offset]
            sample_gate_calibration = _resolve_global_branch_gate_calibration_for_category(
                sample_category_name,
                global_branch_gate_calibration=global_branch_gate_calibration
                if isinstance(global_branch_gate_calibration, GlobalBranchGateCalibration)
                else None,
                global_branch_gate_calibration_by_category=global_branch_gate_calibration
                if isinstance(global_branch_gate_calibration, dict)
                else None,
            )
            sample_ir_hotspot_area_fraction = float(scored.raw_ir_hotspot_area_fraction[sample_offset].item())
            sample_ir_hotspot_compactness = float(scored.raw_ir_hotspot_compactness[sample_offset].item())
            sample_ir_mean_hotspot_intensity = float(scored.raw_ir_mean_hotspot_intensity[sample_offset].item())
            sample_cross_inconsistency_score = float(scored.raw_cross_inconsistency_score[sample_offset].item())
            sample_cross_inconsistency_q99 = float(scored.cross_inconsistency_q99[sample_offset].item())
            sample_cross_inconsistency_top1 = float(scored.cross_inconsistency_top1[sample_offset].item())
            sample_cross_geo_support_top1 = float(scored.cross_geo_support_top1[sample_offset].item())
            sample_ir_hotspot_top1 = float(scored.ir_hotspot_top1[sample_offset].item())
            sample_ir_gradient_top1 = float(scored.ir_gradient_top1[sample_offset].item())
            sample_gate_fired = bool(scored.internal_defect_gate_fired[sample_offset].item())
            sample_gate_score_changed = bool(scored.internal_defect_gate_score_changed[sample_offset].item())
            sample_gate_mode = str(scored.internal_defect_gate_mode[sample_offset])
            sample_gate_reason = str(scored.internal_defect_gate_reason[sample_offset])
            sample_rescue_score = float(scored.rescue_score[sample_offset].item())
            internal_defect_gate_fired_count += int(sample_gate_fired)
            internal_defect_gate_score_changed_count += int(sample_gate_score_changed)
            gt_mask = load_gt_mask(
                batch.gt_mask_paths[sample_offset],
                target_size=target_size,
                crop_box_xyxy=batch.crop_boxes_xyxy[sample_offset],
                crop_source_size_hw=batch.crop_source_sizes_hw[sample_offset],
            )
            sample_pixel_f1 = None
            sample_pixel_iou = None
            if gt_mask is not None:
                pred_mask = predicted_masks[sample_offset]
                sample_pixel_f1 = binary_f1_score(pred_mask, gt_mask)
                sample_pixel_iou = intersection_over_union(pred_mask, gt_mask)
                pixel_f1_values.append(sample_pixel_f1)
                pixel_iou_values.append(sample_pixel_iou)

            pixel_target_mask: torch.Tensor | None = None
            if gt_mask is not None:
                pixel_target_mask = gt_mask.detach().cpu()
            elif not bool(batch.is_anomalous[sample_offset]):
                pixel_target_mask = torch.zeros_like(object_masks[sample_offset].detach().cpu(), dtype=torch.float32)
            if pixel_target_mask is not None:
                score_map_cpu = sample_normalized_localization_maps[sample_offset][0].detach().cpu()
                object_mask_cpu = object_masks[sample_offset].detach().cpu()
                pooled_pixel_score_maps.append(score_map_cpu)
                pooled_pixel_target_masks.append(pixel_target_mask)
                pooled_pixel_object_masks.append(object_mask_cpu)
                category_bucket = per_category_pixel_inputs.setdefault(
                    batch.categories[sample_offset],
                    {"score_maps": [], "target_masks": [], "object_masks": []},
                )
                category_bucket["score_maps"].append(score_map_cpu)
                category_bucket["target_masks"].append(pixel_target_mask)
                category_bucket["object_masks"].append(object_mask_cpu)

            panel_path: Path | None = None
            should_save_preview = False
            if visualization_dir is not None:
                sample_name = batch.sample_names[sample_offset]
                if selected_preview_sample_names is not None:
                    should_save_preview = (
                        sample_name in selected_preview_sample_names
                        and sample_name not in saved_preview_sample_names
                    )
                else:
                    should_save_preview = visualizations_saved < max_visualizations
            if visualization_dir is not None and should_save_preview:
                panel_path = visualization_dir / f"{batch_index:03d}_{batch.sample_names[sample_offset]}.png"
                resolved_rgb_stats = _resolve_rgb_stats_for_category(
                    batch.categories[sample_offset],
                    rgb_normalization_stats=rgb_normalization_stats,
                    rgb_normalization_stats_by_category=rgb_normalization_stats_by_category,
                )
                save_anomaly_panel(
                    rgb=_denormalize_rgb_tensor(
                        batch.x_rgb[sample_offset].detach().cpu(),
                        resolved_rgb_stats,
                    ).clamp(0.0, 1.0),
                    reconstructed_rgb=_denormalize_rgb_tensor(
                        scored.display_reconstructed_rgb[sample_offset].detach().cpu(),
                        resolved_rgb_stats,
                    ).clamp(0.0, 1.0),
                    reconstructed_label=scored.display_reconstruction_label,
                    comparison_rgb=_denormalize_rgb_tensor(
                        scored.diffusion_reconstructed_rgb[sample_offset].detach().cpu(),
                        resolved_rgb_stats,
                    ).clamp(0.0, 1.0),
                    comparison_label=scored.diffusion_reconstruction_label,
                    anomaly_map=sample_normalized_localization_maps[sample_offset][0].detach().cpu(),
                    score_map=scored.score_basis_map[sample_offset].detach().cpu(),
                    score_map_label=scored.score_basis_label,
                    support_map=scored.descriptor_support[sample_offset].detach().cpu(),
                    gt_mask=gt_mask,
                    anomaly_score=float(scored.anomaly_score[sample_offset].item()),
                    title=f"{batch.categories[sample_offset]} / {batch.defect_labels[sample_offset]} / {batch.sample_names[sample_offset]}",
                    path=panel_path,
                )
                visualizations_saved += 1
                saved_preview_sample_names.add(batch.sample_names[sample_offset])

            if evidence_dir is not None and calibration is not None:
                if isinstance(calibration, dict):
                    sample_calibration = calibration.get(batch.categories[sample_offset])
                    if sample_calibration is None:
                        raise KeyError(
                            f"Missing MASI calibration for category={batch.categories[sample_offset]!r}."
                        )
                else:
                    sample_calibration = calibration
                package = build_evidence_package(
                    category=batch.categories[sample_offset],
                    split="test",
                    defect_label=batch.defect_labels[sample_offset],
                    sample_id=sample_id,
                    is_anomalous=bool(batch.is_anomalous[sample_offset]),
                    score_mode=score_mode,
                    score_label=scored.score_basis_label,
                    raw_score=float(scored.anomaly_score[sample_offset].item()),
                    calibration=sample_calibration,
                    normalized_anomaly_map=sample_normalized_localization_maps[sample_offset][0].detach().cpu(),
                    score_basis_map=scored.score_basis_map[sample_offset].detach().cpu(),
                    descriptor_support=scored.descriptor_support[sample_offset].detach().cpu(),
                    object_mask=object_masks[sample_offset].detach().cpu(),
                    predicted_mask=predicted_masks[sample_offset].detach().cpu(),
                    descriptor_maps=batch.M[sample_offset].detach().cpu(),
                    support_maps=batch.S[sample_offset].detach().cpu(),
                    global_descriptor_score=float(scored.global_descriptor_score[sample_offset].item()),
                    provenance={
                        "pipeline_stage": "evaluation",
                        "score_mode": score_mode,
                        "object_score_strategy": scored.object_score_strategy,
                        "internal_defect_gate_enabled": bool(enable_internal_defect_gate),
                        "internal_defect_gate_fired": sample_gate_fired,
                        "internal_defect_gate_score_changed": sample_gate_score_changed,
                        "internal_defect_gate_mode": sample_gate_mode,
                        "internal_defect_gate_reason": sample_gate_reason,
                        "spatial_score_pre_gate": float(scored.spatial_score_pre_gate[sample_offset].item()),
                        "spatial_score_post_gate": float(scored.spatial_score_post_gate[sample_offset].item()),
                        "rescue_score": sample_rescue_score,
                        "ir_dominant_score": float(scored.ir_dominant_score[sample_offset].item()),
                        "raw_ir_hotspot_area_fraction": sample_ir_hotspot_area_fraction,
                        "raw_ir_hotspot_compactness": sample_ir_hotspot_compactness,
                        "raw_ir_mean_hotspot_intensity": sample_ir_mean_hotspot_intensity,
                        "raw_cross_inconsistency_score": sample_cross_inconsistency_score,
                        "cross_inconsistency_q99": sample_cross_inconsistency_q99,
                        "cross_inconsistency_top1": sample_cross_inconsistency_top1,
                        "cross_geo_support_top1": sample_cross_geo_support_top1,
                        "ir_hotspot_top1": sample_ir_hotspot_top1,
                        "ir_gradient_top1": sample_ir_gradient_top1,
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
                        "gate_mode": (
                            str(sample_gate_calibration.mode)
                            if sample_gate_calibration is not None
                            else "inactive"
                        ),
                        "gate_active": (
                            bool(sample_gate_calibration.active)
                            if sample_gate_calibration is not None
                            else False
                        ),
                        "timesteps_used": list(scored.timesteps_used),
                        "lambda_residual": float(lambda_residual if lambda_residual is not None else 0.0),
                        "lambda_noise": float(lambda_noise if lambda_noise is not None else 0.0),
                        "lambda_descriptor_spatial": float(
                            lambda_descriptor_spatial if lambda_descriptor_spatial is not None else descriptor_weight
                        ),
                        "lambda_descriptor_global": float(
                            lambda_descriptor_global if lambda_descriptor_global is not None else global_descriptor_weight
                        ),
                        "local_exceedance_score": float(scored.local_exceedance_score[sample_offset].item()),
                        "local_z_score": float(scored.local_z_score[sample_offset].item()),
                        "global_z_score": float(scored.global_z_score[sample_offset].item()),
                        "largest_component_area": float(scored.largest_component_area[sample_offset].item()),
                        "largest_component_mean_excess": float(
                            scored.largest_component_mean_excess[sample_offset].item()
                        ),
                        "largest_component_peak_excess": float(
                            scored.largest_component_peak_excess[sample_offset].item()
                        ),
                        "largest_component_mass_excess": float(
                            scored.largest_component_mass_excess[sample_offset].item()
                        ),
                        "largest_component_area_fraction": float(
                            scored.largest_component_area_fraction[sample_offset].item()
                        ),
                        "num_exceedance_components": int(scored.num_exceedance_components[sample_offset].item()),
                        "localization_basis": sample_localization_bases[sample_offset],
                        "localization_method": (
                            "per_category_calibrated_threshold"
                            if _resolve_localization_calibration_for_category(
                                batch.categories[sample_offset],
                                localization_calibration=localization_calibration
                                if isinstance(localization_calibration, LocalizationCalibration)
                                else None,
                                localization_calibration_by_category=localization_calibration
                                if isinstance(localization_calibration, dict)
                                else None,
                            )
                            is not None
                            else "percentile_threshold_fallback"
                        ),
                    },
                    source_paths={
                        "rgb_path": batch.rgb_paths[sample_offset],
                        "gt_mask_path": batch.gt_mask_paths[sample_offset],
                        "source_rgb_path": batch.source_rgb_paths[sample_offset],
                        "source_ir_path": batch.source_ir_paths[sample_offset],
                        "source_pointcloud_path": batch.source_pointcloud_paths[sample_offset],
                        "visualization_path": str(panel_path) if panel_path is not None else "",
                    },
                )
                package_path = evidence_dir / "packages" / batch.categories[sample_offset] / batch.defect_labels[sample_offset] / f"{sample_id}.json"
                report_path = evidence_dir / "reports" / batch.categories[sample_offset] / batch.defect_labels[sample_offset] / f"{sample_id}.md"
                save_evidence_package(package, package_path)
                save_templated_explanation(render_templated_explanation(package), report_path)

                llm_bundle_path = ""
                retrieval_query_path = ""
                retrieved_context_path = ""
                context_pack_path = ""
                if enable_llm_explanations:
                    retrieved_context = retrieve_context_for_evidence(
                        package,
                        knowledge_base_root=knowledge_base_root,
                        top_k=retrieval_top_k,
                    )
                    explanation = generate_operator_report(
                        package=package,
                        retrieved_context=retrieved_context,
                        provider=explanation_provider,
                    )
                    llm_root = evidence_dir / "llm" / batch.categories[sample_offset] / batch.defect_labels[sample_offset]
                    retrieval_query_path = str(
                        write_json(
                            llm_root / f"{sample_id}__retrieval_query.json",
                            explanation.get("context_pack", {}).get("retrieval_query", {}),
                        )
                    )
                    retrieved_context_path = str(
                        write_json(
                            llm_root / f"{sample_id}__retrieved_context.json",
                            [item.to_dict() if hasattr(item, "to_dict") else item for item in retrieved_context],
                        )
                    )
                    context_pack_path = str(
                        write_json(
                            llm_root / f"{sample_id}__context_pack.json",
                            explanation.get("context_pack", {}),
                        )
                    )
                    llm_bundle = llm_root / f"{sample_id}.json"
                    llm_report = llm_root / f"{sample_id}.md"
                    save_explanation_bundle(
                        output_path=llm_bundle,
                        package=package,
                        generation=explanation,
                        source_eval_run=evidence_dir.parent if evidence_dir is not None else None,
                        source_evidence_path=package_path,
                    )
                    save_templated_explanation(str(explanation.get("markdown", "")), llm_report)
                    llm_bundle_path = str(llm_bundle)
                    llm_explanations_saved += 1

                evidence_index.append(
                    {
                        "category": batch.categories[sample_offset],
                        "defect_label": batch.defect_labels[sample_offset],
                        "sample_id": sample_id,
                        "is_anomalous": bool(batch.is_anomalous[sample_offset]),
                        "raw_score": round(float(scored.anomaly_score[sample_offset].item()), 6),
                        "object_score_strategy": scored.object_score_strategy,
                        "internal_defect_gate_fired": sample_gate_fired,
                        "internal_defect_gate_score_changed": sample_gate_score_changed,
                        "internal_defect_gate_mode": sample_gate_mode,
                        "internal_defect_gate_reason": sample_gate_reason,
                        "spatial_score_pre_gate": round(float(scored.spatial_score_pre_gate[sample_offset].item()), 6),
                        "spatial_score_post_gate": round(float(scored.spatial_score_post_gate[sample_offset].item()), 6),
                        "rescue_score": round(sample_rescue_score, 6),
                        "ir_dominant_score": round(float(scored.ir_dominant_score[sample_offset].item()), 6),
                        "raw_ir_hotspot_area_fraction": round(sample_ir_hotspot_area_fraction, 6),
                        "raw_ir_hotspot_compactness": round(sample_ir_hotspot_compactness, 6),
                        "raw_ir_mean_hotspot_intensity": round(sample_ir_mean_hotspot_intensity, 6),
                        "raw_cross_inconsistency_score": round(sample_cross_inconsistency_score, 6),
                        "cross_inconsistency_q99": round(sample_cross_inconsistency_q99, 6),
                        "cross_inconsistency_top1": round(sample_cross_inconsistency_top1, 6),
                        "cross_geo_support_top1": round(sample_cross_geo_support_top1, 6),
                        "ir_hotspot_top1": round(sample_ir_hotspot_top1, 6),
                        "ir_gradient_top1": round(sample_ir_gradient_top1, 6),
                        "gate_ir_hotspot_area_fraction_threshold": (
                            round(float(sample_gate_calibration.ir_hotspot_area_fraction_threshold), 6)
                            if sample_gate_calibration is not None
                            else None
                        ),
                        "gate_ir_hotspot_compactness_threshold": (
                            round(float(sample_gate_calibration.ir_hotspot_compactness_threshold), 6)
                            if sample_gate_calibration is not None
                            else None
                        ),
                        "gate_ir_mean_hotspot_intensity_threshold": (
                            round(float(sample_gate_calibration.ir_mean_hotspot_intensity_threshold), 6)
                            if sample_gate_calibration is not None
                            else None
                        ),
                        "gate_cross_inconsistency_score_threshold": (
                            round(float(sample_gate_calibration.cross_inconsistency_score_threshold), 6)
                            if sample_gate_calibration is not None
                            else None
                        ),
                        "gate_spatial_score_q95_threshold": (
                            round(float(sample_gate_calibration.spatial_score_q95), 6)
                            if sample_gate_calibration is not None
                            else None
                        ),
                        "gate_rescue_score_q95_threshold": (
                            round(float(sample_gate_calibration.rescue_score_q95), 6)
                            if sample_gate_calibration is not None
                            else None
                        ),
                        "gate_rescue_score_q99_threshold": (
                            round(float(sample_gate_calibration.rescue_score_q99), 6)
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
                        "local_exceedance_score": round(float(scored.local_exceedance_score[sample_offset].item()), 6),
                        "local_z_score": round(float(scored.local_z_score[sample_offset].item()), 6),
                        "global_z_score": round(float(scored.global_z_score[sample_offset].item()), 6),
                        "largest_component_area": round(float(scored.largest_component_area[sample_offset].item()), 6),
                        "largest_component_mean_excess": round(
                            float(scored.largest_component_mean_excess[sample_offset].item()),
                            6,
                        ),
                        "largest_component_peak_excess": round(
                            float(scored.largest_component_peak_excess[sample_offset].item()),
                            6,
                        ),
                        "largest_component_mass_excess": round(
                            float(scored.largest_component_mass_excess[sample_offset].item()),
                            6,
                        ),
                        "num_exceedance_components": int(scored.num_exceedance_components[sample_offset].item()),
                        "localization_basis": sample_localization_bases[sample_offset],
                        "severity_0_100": round(float(package.severity_0_100), 3),
                        "confidence_0_100": round(float(package.confidence_0_100), 3),
                        "status": package.status,
                        "global_descriptor_score": round(float(scored.global_descriptor_score[sample_offset].item()), 6),
                        "package_path": str(package_path),
                        "report_path": str(report_path),
                        "llm_bundle_path": llm_bundle_path,
                        "retrieval_query_path": retrieval_query_path,
                        "retrieved_context_path": retrieved_context_path,
                        "context_pack_path": context_pack_path,
                    }
                )

            sample_metrics.append(
                {
                    "category": batch.categories[sample_offset],
                    "defect_label": batch.defect_labels[sample_offset],
                    "sample_id": sample_id,
                    "sample_name": batch.sample_names[sample_offset],
                    "is_anomalous": bool(batch.is_anomalous[sample_offset]),
                    "score": float(scored.anomaly_score[sample_offset].item()),
                    "object_score_strategy": scored.object_score_strategy,
                    "internal_defect_gate_fired": sample_gate_fired,
                    "internal_defect_gate_score_changed": sample_gate_score_changed,
                    "internal_defect_gate_mode": sample_gate_mode,
                    "internal_defect_gate_reason": sample_gate_reason,
                    "spatial_score_pre_gate": float(scored.spatial_score_pre_gate[sample_offset].item()),
                    "spatial_score_post_gate": float(scored.spatial_score_post_gate[sample_offset].item()),
                    "rescue_score": sample_rescue_score,
                    "ir_dominant_score": float(scored.ir_dominant_score[sample_offset].item()),
                    "raw_ir_hotspot_area_fraction": sample_ir_hotspot_area_fraction,
                    "raw_ir_hotspot_compactness": sample_ir_hotspot_compactness,
                    "raw_ir_mean_hotspot_intensity": sample_ir_mean_hotspot_intensity,
                    "raw_cross_inconsistency_score": sample_cross_inconsistency_score,
                    "cross_inconsistency_q99": sample_cross_inconsistency_q99,
                    "cross_inconsistency_top1": sample_cross_inconsistency_top1,
                    "cross_geo_support_top1": sample_cross_geo_support_top1,
                    "ir_hotspot_top1": sample_ir_hotspot_top1,
                    "ir_gradient_top1": sample_ir_gradient_top1,
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
                    "local_exceedance_score": float(scored.local_exceedance_score[sample_offset].item()),
                    "local_z_score": float(scored.local_z_score[sample_offset].item()),
                    "global_z_score": float(scored.global_z_score[sample_offset].item()),
                    "largest_component_area": float(scored.largest_component_area[sample_offset].item()),
                    "largest_component_mean_excess": float(
                        scored.largest_component_mean_excess[sample_offset].item()
                    ),
                    "largest_component_peak_excess": float(
                        scored.largest_component_peak_excess[sample_offset].item()
                    ),
                    "largest_component_mass_excess": float(
                        scored.largest_component_mass_excess[sample_offset].item()
                    ),
                    "num_exceedance_components": int(scored.num_exceedance_components[sample_offset].item()),
                    "localization_basis": sample_localization_bases[sample_offset],
                    "pixel_f1": sample_pixel_f1,
                    "pixel_iou": sample_pixel_iou,
                }
            )
            image_score_rows.append(
                {
                    "category": batch.categories[sample_offset],
                    "defect_label": batch.defect_labels[sample_offset],
                    "sample_id": sample_id,
                    "sample_name": batch.sample_names[sample_offset],
                    "label": 1 if bool(batch.is_anomalous[sample_offset]) else 0,
                    "score": float(scored.anomaly_score[sample_offset].item()),
                    "object_score_strategy": scored.object_score_strategy,
                    "internal_defect_gate_fired": sample_gate_fired,
                    "internal_defect_gate_score_changed": sample_gate_score_changed,
                    "internal_defect_gate_mode": sample_gate_mode,
                    "internal_defect_gate_reason": sample_gate_reason,
                    "spatial_score_pre_gate": float(scored.spatial_score_pre_gate[sample_offset].item()),
                    "spatial_score_post_gate": float(scored.spatial_score_post_gate[sample_offset].item()),
                    "rescue_score": sample_rescue_score,
                    "ir_dominant_score": float(scored.ir_dominant_score[sample_offset].item()),
                    "raw_ir_hotspot_area_fraction": sample_ir_hotspot_area_fraction,
                    "raw_ir_hotspot_compactness": sample_ir_hotspot_compactness,
                    "raw_ir_mean_hotspot_intensity": sample_ir_mean_hotspot_intensity,
                    "raw_cross_inconsistency_score": sample_cross_inconsistency_score,
                    "cross_inconsistency_q99": sample_cross_inconsistency_q99,
                    "cross_inconsistency_top1": sample_cross_inconsistency_top1,
                    "cross_geo_support_top1": sample_cross_geo_support_top1,
                    "ir_hotspot_top1": sample_ir_hotspot_top1,
                    "ir_gradient_top1": sample_ir_gradient_top1,
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
                    "local_exceedance_score": float(scored.local_exceedance_score[sample_offset].item()),
                    "local_z_score": float(scored.local_z_score[sample_offset].item()),
                    "global_z_score": float(scored.global_z_score[sample_offset].item()),
                    "largest_component_area": float(scored.largest_component_area[sample_offset].item()),
                    "largest_component_mean_excess": float(
                        scored.largest_component_mean_excess[sample_offset].item()
                    ),
                    "largest_component_peak_excess": float(
                        scored.largest_component_peak_excess[sample_offset].item()
                    ),
                    "largest_component_mass_excess": float(
                        scored.largest_component_mass_excess[sample_offset].item()
                    ),
                    "num_exceedance_components": int(scored.num_exceedance_components[sample_offset].item()),
                    "localization_basis": sample_localization_bases[sample_offset],
                }
            )

        reports.append(
            {
                "batch_index": batch_index,
                "mean_anomaly_score": round(float(scored.anomaly_score.mean().item()), 6),
                "mean_spatial_score": round(float(scored.spatial_score.mean().item()), 6),
                "mean_spatial_score_pre_gate": round(float(scored.spatial_score_pre_gate.mean().item()), 6),
                "mean_spatial_score_post_gate": round(float(scored.spatial_score_post_gate.mean().item()), 6),
                "mean_rescue_score": round(float(scored.rescue_score.mean().item()), 6),
                "mean_ir_dominant_score": round(float(scored.ir_dominant_score.mean().item()), 6),
                "internal_defect_gate_fired_count": int(scored.internal_defect_gate_fired.sum().item()),
                "internal_defect_gate_score_changed_count": int(scored.internal_defect_gate_score_changed.sum().item()),
                "mean_global_descriptor_score": round(float(scored.global_descriptor_score.mean().item()), 6),
                "mean_local_exceedance_score": round(float(scored.local_exceedance_score.mean().item()), 6),
                "mean_local_z_score": round(float(scored.local_z_score.mean().item()), 6),
                "mean_global_z_score": round(float(scored.global_z_score.mean().item()), 6),
                "max_anomaly_score": round(float(scored.anomaly_score.max().item()), 6),
                "score_mode": score_mode,
                "object_score_strategy": scored.object_score_strategy,
                "mean_score_basis": round(float(scored.score_basis_map.mean().item()), 6),
                "mean_residual": round(float(scored.residual_map.mean().item()), 6),
                "mean_noise_error": round(float(scored.noise_error_map.mean().item()), 6),
                "mean_descriptor_support": round(float(scored.descriptor_support.mean().item()), 6),
                "mean_residual_component": round(float(scored.residual_component.mean().item()), 6),
                "mean_noise_component": round(float(scored.noise_component.mean().item()), 6),
                "mean_descriptor_spatial_component": round(float(scored.descriptor_spatial_component.mean().item()), 6),
                "timesteps_used": list(scored.timesteps_used),
                "sample_ids": list(batch.sample_ids),
            }
        )
        if max_batches and len(reports) >= max_batches:
            break

    image_scores_tensor = torch.cat(image_scores) if image_scores else torch.empty(0)
    image_labels_tensor = torch.cat(image_labels) if image_labels else torch.empty(0)
    good_scores = image_scores_tensor[image_labels_tensor == 0]
    anomaly_scores = image_scores_tensor[image_labels_tensor == 1]

    image_auroc = image_level_auroc(image_scores_tensor, image_labels_tensor) if image_scores else 0.0
    image_auprc = image_average_precision(image_scores_tensor, image_labels_tensor) if image_scores else 0.0
    threshold = 0.0
    image_f1 = 0.0
    if len(good_scores) and len(anomaly_scores):
        threshold = float((good_scores.mean() + anomaly_scores.mean()).item() / 2.0)
        image_f1 = binary_f1_score(image_scores_tensor, image_labels_tensor, threshold=threshold)
    pooled_pixel_auroc = pixel_level_auroc(
        pooled_pixel_score_maps,
        pooled_pixel_target_masks,
        pooled_pixel_object_masks,
    )
    pooled_aupro = aupro(
        pooled_pixel_score_maps,
        pooled_pixel_target_masks,
        pooled_pixel_object_masks,
    )

    evidence_index_path = ""
    if evidence_dir is not None:
        evidence_index_path = str(write_json(evidence_dir / "index.json", evidence_index))
    image_score_data_path_str = ""
    if image_score_data_path is not None:
        image_score_data_path_str = str(write_json(image_score_data_path, image_score_rows))

    per_category: Dict[str, Dict[str, float | int]] = {}
    for category in sorted({str(item["category"]) for item in sample_metrics}):
        category_rows = [row for row in sample_metrics if row["category"] == category]
        category_scores = torch.tensor([float(row["score"]) for row in category_rows], dtype=torch.float32)
        category_labels = torch.tensor(
            [1.0 if bool(row["is_anomalous"]) else 0.0 for row in category_rows],
            dtype=torch.float32,
        )
        category_good = category_scores[category_labels == 0]
        category_anomaly = category_scores[category_labels == 1]
        category_threshold = 0.0
        category_image_f1 = 0.0
        if len(category_good) and len(category_anomaly):
            category_threshold = float((category_good.mean() + category_anomaly.mean()).item() / 2.0)
            category_image_f1 = binary_f1_score(category_scores, category_labels, threshold=category_threshold)
        category_pixel_f1_values = [
            float(row["pixel_f1"])
            for row in category_rows
            if row["pixel_f1"] is not None
        ]
        category_pixel_iou_values = [
            float(row["pixel_iou"])
            for row in category_rows
            if row["pixel_iou"] is not None
        ]
        category_pixel_bucket = per_category_pixel_inputs.get(category, {"score_maps": [], "target_masks": [], "object_masks": []})
        per_category[category] = {
            "image_auroc": round(float(image_level_auroc(category_scores, category_labels)), 6),
            "image_auprc": round(float(image_average_precision(category_scores, category_labels)), 6),
            "image_f1": round(float(category_image_f1), 6),
            "mean_good_score": round(float(category_good.mean().item()), 6) if len(category_good) else 0.0,
            "mean_anomalous_score": round(float(category_anomaly.mean().item()), 6) if len(category_anomaly) else 0.0,
            "pixel_auroc": round(
                float(
                    pixel_level_auroc(
                        category_pixel_bucket["score_maps"],
                        category_pixel_bucket["target_masks"],
                        category_pixel_bucket["object_masks"],
                    )
                ),
                6,
            ),
            "aupro": round(
                float(
                    aupro(
                        category_pixel_bucket["score_maps"],
                        category_pixel_bucket["target_masks"],
                        category_pixel_bucket["object_masks"],
                    )
                ),
                6,
            ),
            "pixel_f1": round(sum(category_pixel_f1_values) / max(len(category_pixel_f1_values), 1), 6),
            "pixel_iou": round(sum(category_pixel_iou_values) / max(len(category_pixel_iou_values), 1), 6),
            "eval_samples": len(category_rows),
            "pixel_samples": len(category_pixel_f1_values),
        }

    metric_keys = [
        "image_auroc",
        "image_auprc",
        "image_f1",
        "mean_good_score",
        "mean_anomalous_score",
        "pixel_auroc",
        "aupro",
        "pixel_f1",
        "pixel_iou",
    ]
    macro_metrics = {
        f"macro_{metric}": round(
            sum(float(category_metrics[metric]) for category_metrics in per_category.values()) / max(len(per_category), 1),
            6,
        )
        for metric in metric_keys
    }

    return {
        "batch_reports": reports,
        "image_auroc": round(float(image_auroc), 6),
        "image_auprc": round(float(image_auprc), 6),
        "image_f1": round(float(image_f1), 6),
        "image_threshold": round(float(threshold), 6),
        "mean_good_score": round(float(good_scores.mean().item()), 6) if len(good_scores) else 0.0,
        "mean_anomalous_score": round(float(anomaly_scores.mean().item()), 6) if len(anomaly_scores) else 0.0,
        "internal_defect_gate_fired_count": int(internal_defect_gate_fired_count),
        "internal_defect_gate_score_changed_count": int(internal_defect_gate_score_changed_count),
        "internal_defect_gate_fire_rate": round(
            float(internal_defect_gate_fired_count / max(len(sample_metrics), 1)),
            6,
        ),
        "internal_defect_gate_score_changed_rate": round(
            float(internal_defect_gate_score_changed_count / max(len(sample_metrics), 1)),
            6,
        ),
        "localization_basis": localization_basis,
        "object_mask_dilation_kernel_size": int(object_mask_dilation_kernel_size),
        "pixel_auroc": round(float(pooled_pixel_auroc), 6),
        "aupro": round(float(pooled_aupro), 6),
        "pixel_f1": round(sum(pixel_f1_values) / max(len(pixel_f1_values), 1), 6),
        "pixel_iou": round(sum(pixel_iou_values) / max(len(pixel_iou_values), 1), 6),
        "pixel_samples": len(pixel_f1_values),
        "visualizations_saved": visualizations_saved,
        "preview_selection_path": preview_selection_path_str,
        "image_score_data_path": image_score_data_path_str,
        "evidence_packages_saved": len(evidence_index),
        "llm_explanations_saved": llm_explanations_saved,
        "evidence_index_path": evidence_index_path,
        "per_category": per_category,
        **macro_metrics,
    }


def run_smoke_experiment(
    *,
    data_root: Path | str = "data",
    descriptor_index_csv: Path | str = "data/processed/manifests/descriptor_index.csv",
    category: str = "capsule",
    batch_size: int = 2,
    max_batches: int = 3,
    target_size: tuple[int, int] = (256, 256),
    device: str = "cpu",
    device_mode: str = "",
    learning_rate: float = 1e-4,
    score_mode: str = "noise_error",
    anomaly_timestep: int = 200,
    descriptor_weight: float = 0.25,
    global_descriptor_weight: float = 0.1,
    multi_timestep_scoring_enabled: bool = False,
    multi_timestep_fractions: Sequence[float] | None = None,
    anomaly_map_gaussian_sigma: float = 0.0,
    lambda_residual: float = 0.0,
    lambda_noise: float = 1.0,
    lambda_descriptor_spatial: float = 0.25,
    lambda_descriptor_global: float = 0.1,
    pixel_threshold_percentile: float = 0.9,
    object_mask_threshold: float = 0.02,
    base_channels: int = 32,
    global_embedding_dim: int = 128,
    time_embedding_dim: int = 128,
    category_embedding_dim: int = 32,
    attention_heads: int = 4,
    latent_channels: int = 4,
    diffusion_steps: int = 1000,
    noise_schedule: str = "cosine",
    autoencoder_reconstruction_weight: float = 0.5,
    diffusion_reconstruction_weight: float = 0.1,
    edge_reconstruction_weight: float = 0.05,
    object_crop_enabled: bool = False,
    object_crop_margin_ratio: float = 0.10,
    object_crop_mask_source: str = "rgb_then_pc_density",
    rgb_normalization_mode: str = "none",
) -> Dict[str, object]:
    loaders = build_diffusion_dataloaders(
        data_root=data_root,
        descriptor_index_csv=descriptor_index_csv,
        category=category,
        batch_size=batch_size,
        target_size=target_size,
        object_crop_enabled=object_crop_enabled,
        object_crop_margin_ratio=object_crop_margin_ratio,
        object_crop_mask_source=object_crop_mask_source,
        rgb_normalization_mode=rgb_normalization_mode,
    )

    train_dataset: DescriptorConditioningDataset = loaders["train_dataset"]  # type: ignore[assignment]
    category_vocabulary = list(loaders["category_vocabulary"])  # type: ignore[assignment]
    if len(train_dataset) == 0:
        raise ValueError(f"No training records available for smoke experiment category={category!r}")

    model = build_model(
        descriptor_channels=train_dataset.conditioning_channels,
        global_dim=train_dataset.global_dim,
        base_channels=base_channels,
        global_embedding_dim=global_embedding_dim,
        time_embedding_dim=time_embedding_dim,
        num_categories=max(len(category_vocabulary), 1),
        category_embedding_dim=category_embedding_dim,
        attention_heads=attention_heads,
        latent_channels=latent_channels,
        diffusion_steps=diffusion_steps,
        noise_schedule=noise_schedule,
        autoencoder_reconstruction_weight=autoencoder_reconstruction_weight,
        diffusion_reconstruction_weight=diffusion_reconstruction_weight,
        edge_reconstruction_weight=edge_reconstruction_weight,
    )
    runtime = resolve_runtime(
        requested_mode=_resolve_requested_mode(device, device_mode),
        model=model,
        validation_batch=prepare_diffusion_batch(next(iter(loaders["train_loader"]))),  # type: ignore[arg-type]
        run_backward=True,
    )
    torch_device = runtime.resolved_device
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    batch_reports = run_training_epoch(
        model,
        loaders["train_loader"],  # type: ignore[arg-type]
        optimizer,
        device=torch_device,
        use_amp=runtime.use_amp,
        epoch=1,
        max_batches=max_batches,
        log_every_n_steps=0,
        log_to_jsonl=False,
    )
    global_descriptor_calibration = fit_global_descriptor_reference(
        loaders["train_reference_loader"],  # type: ignore[arg-type]
        category=category,
    )
    eval_reports = run_eval_scoring(
        model,
        loaders["eval_loader"],  # type: ignore[arg-type]
        device=torch_device,
        target_size=target_size,
        score_mode=score_mode,
        anomaly_timestep=anomaly_timestep,
        descriptor_weight=descriptor_weight,
        global_descriptor_weight=global_descriptor_weight,
        multi_timestep_scoring_enabled=multi_timestep_scoring_enabled,
        multi_timestep_fractions=multi_timestep_fractions,
        anomaly_map_gaussian_sigma=anomaly_map_gaussian_sigma,
        lambda_residual=lambda_residual,
        lambda_noise=lambda_noise,
        lambda_descriptor_spatial=lambda_descriptor_spatial,
        lambda_descriptor_global=lambda_descriptor_global,
        global_descriptor_calibration=global_descriptor_calibration,
        pixel_threshold_percentile=pixel_threshold_percentile,
        object_mask_threshold=object_mask_threshold,
        max_batches=1,
        enable_llm_explanations=False,
        rgb_normalization_stats=loaders["rgb_normalization_stats"],  # type: ignore[arg-type]
        preview_seed=17,
    )

    manifests = loaders["manifests"]  # type: ignore[assignment]
    summary = {
        "category": category,
        "batch_size": batch_size,
        "max_batches": max_batches,
        "target_size": target_size,
        "device": str(torch_device),
        "device_mode": runtime.requested_mode,
        "amp_enabled": runtime.use_amp,
        "runtime_fallback_reason": runtime.fallback_reason,
        "learning_rate": learning_rate,
        "score_mode": score_mode,
        "anomaly_timestep": anomaly_timestep,
        "descriptor_weight": descriptor_weight,
        "global_descriptor_weight": global_descriptor_weight,
        "multi_timestep_scoring_enabled": multi_timestep_scoring_enabled,
        "anomaly_map_gaussian_sigma": anomaly_map_gaussian_sigma,
        "lambda_residual": lambda_residual,
        "lambda_noise": lambda_noise,
        "lambda_descriptor_spatial": lambda_descriptor_spatial,
        "lambda_descriptor_global": lambda_descriptor_global,
        "pixel_threshold_percentile": pixel_threshold_percentile,
        "object_mask_threshold": object_mask_threshold,
        "object_crop_enabled": object_crop_enabled,
        "object_crop_margin_ratio": object_crop_margin_ratio,
        "object_crop_mask_source": object_crop_mask_source,
        "rgb_normalization_mode": rgb_normalization_mode,
        "train_manifest_csv": str(manifests["train_csv"]),
        "eval_manifest_csv": str(manifests["eval_csv"]),
        "global_stats_json": str(manifests["global_stats_json"]),
        "train_records": len(train_dataset),
        "eval_records": len(loaders["eval_dataset"]),  # type: ignore[arg-type]
        "batches_run": len(batch_reports),
        "batch_reports": batch_reports,
        "eval_reports": eval_reports["batch_reports"],
        "image_auroc": eval_reports["image_auroc"],
        "image_auprc": eval_reports["image_auprc"],
        "pixel_auroc": eval_reports["pixel_auroc"],
        "aupro": eval_reports["aupro"],
        "pixel_iou": eval_reports["pixel_iou"],
        "mean_loss": round(sum(report["loss"] for report in batch_reports) / max(len(batch_reports), 1), 6),
    }

    out_path = Path(data_root) / "processed" / "reports" / f"smoke_experiment_{category}.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return {"summary": summary, "summary_path": out_path}


def train_model(
    *,
    data_root: Path | str = "data",
    processed_root: Path | str | None = None,
    descriptor_index_csv: Path | str = "data/processed/manifests/descriptor_index.csv",
    category: str = "capsule",
    categories: Sequence[str] | None = None,
    epochs: int = 50,
    batch_size: int = 2,
    target_size: tuple[int, int] = (256, 256),
    device: str = "cpu",
    device_mode: str = "",
    learning_rate: float = 1e-4,
    score_mode: str = "noise_error",
    anomaly_timestep: int = 200,
    descriptor_weight: float = 0.25,
    global_descriptor_weight: float = 0.1,
    multi_timestep_scoring_enabled: bool = True,
    multi_timestep_fractions: Sequence[float] | None = None,
    anomaly_map_gaussian_sigma: float = 3.0,
    spatial_topk_percent: float = 0.02,
    object_score_strategy: str = "legacy_raw",
    lambda_residual: float = 0.6,
    lambda_noise: float = 0.4,
    lambda_descriptor_spatial: float = 0.25,
    lambda_descriptor_global: float = 0.55,
    score_weight_schedule: Dict[str, object] | None = None,
    pixel_threshold_percentile: float = 0.9,
    object_mask_threshold: float = 0.02,
    localization_basis: str = "anomaly_map",
    localization_basis_candidates: Sequence[str] | None = None,
    object_mask_dilation_kernel_size: int = 9,
    localization_mask_closing_kernel_size: int = 0,
    localization_fill_holes: bool = False,
    localization_tune_on_synthetic_validation: bool = False,
    localization_quantile: float | None = None,
    localization_quantile_candidates: Sequence[float] | None = None,
    localization_min_region_area_fraction: float | None = None,
    localization_min_region_pixels_floor: int | None = None,
    localization_min_region_pixels_floor_candidates: Sequence[int] | None = None,
    output_root: Path | str = "runs",
    run_name: str | None = None,
    max_train_batches: int = 0,
    max_eval_batches: int = 0,
    max_visualizations: int = 6,
    base_channels: int = 32,
    global_embedding_dim: int = 128,
    time_embedding_dim: int = 128,
    category_embedding_dim: int = 32,
    disable_category_embedding: bool = False,
    enable_category_modality_gating: bool = False,
    attention_heads: int = 4,
    multiscale_conditioning: bool = True,
    latent_channels: int = 4,
    diffusion_steps: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    noise_schedule: str = "cosine",
    autoencoder_reconstruction_weight: float = 0.5,
    diffusion_reconstruction_weight: float = 0.1,
    edge_reconstruction_weight: float = 0.05,
    object_crop_enabled: bool = False,
    object_crop_margin_ratio: float = 0.10,
    object_crop_mask_source: str = "rgb_then_pc_density",
    rgb_normalization_mode: str = "none",
    enable_internal_defect_gate: bool = False,
    sampler_mode: str = "balanced_by_category",
    selection_metric: str = "macro_image_auroc",
    seed: int = 7,
    log_every_n_steps: int = 10,
    log_to_jsonl: bool = True,
) -> Dict[str, Any]:
    selected_categories = _normalize_selected_categories(category=category, categories=categories)
    scope = resolve_scope(selected_categories, category=category)
    regime = resolve_regime_identity(
        joint_mode=is_joint_category_request(selected_categories, category=category),
        disable_category_embedding=disable_category_embedding,
    )
    resolved_output_root = (
        default_output_root(
            regime_paper=regime.paper,
            run_type="train",
            scope=scope,
            label=run_name or "baseline",
            repo_root=_repo_root(),
        )
        if str(output_root).strip() == "runs"
        else Path(output_root)
    )
    resolved_processed_root = resolve_processed_root(data_root=data_root, processed_root=processed_root)
    _set_global_seed(seed)
    run_paths = create_run_dir(
        output_root=resolved_output_root,
        run_name=run_name,
        category=_run_slug_for_categories(selected_categories or [category]),
        regime_paper=regime.paper,
        scope=scope,
        run_type="train",
        label=run_name or "baseline",
    )
    loaders = build_diffusion_dataloaders(
        data_root=data_root,
        processed_root=resolved_processed_root,
        descriptor_index_csv=descriptor_index_csv,
        category=category,
        categories=selected_categories,
        batch_size=batch_size,
        target_size=target_size,
        object_crop_enabled=object_crop_enabled,
        object_crop_margin_ratio=object_crop_margin_ratio,
        object_crop_mask_source=object_crop_mask_source,
        rgb_normalization_mode=rgb_normalization_mode,
        manifests_root=run_paths.root / "manifests",
        sampler_mode=sampler_mode,
        seed=seed,
    )

    train_dataset: DescriptorConditioningDataset = loaders["train_dataset"]  # type: ignore[assignment]
    eval_dataset: DescriptorConditioningDataset = loaders["eval_dataset"]  # type: ignore[assignment]
    selected_categories = list(loaders["selected_categories"])  # type: ignore[assignment]
    category_vocabulary = list(loaders["category_vocabulary"])  # type: ignore[assignment]
    joint_mode = len(category_vocabulary) > 1
    if len(train_dataset) == 0:
        raise ValueError(
            f"No training records available for categories={selected_categories or [category]!r}"
        )
    rgb_stats: RGBNormalizationStats | None = loaders["rgb_normalization_stats"]  # type: ignore[assignment]
    rgb_stats_by_category: Dict[str, RGBNormalizationStats] | None = loaders["rgb_normalization_stats_by_category"]  # type: ignore[assignment]
    global_vector_stats_by_category: Dict[str, GlobalVectorNormalizationStats] | None = loaders["global_vector_normalization_stats_by_category"]  # type: ignore[assignment]
    if score_weight_schedule is None:
        score_weight_schedule = {
            "warmup_fraction": 0.4,
            "transition_fraction": 0.6,
            "warmup": {
                "lambda_residual": 0.2,
                "lambda_noise": 0.3,
                "lambda_descriptor_spatial": 0.25,
                "lambda_descriptor_global": 0.8,
            },
            "main": {
                "lambda_residual": lambda_residual,
                "lambda_noise": lambda_noise,
                "lambda_descriptor_spatial": lambda_descriptor_spatial,
                "lambda_descriptor_global": lambda_descriptor_global,
            },
        }
    model = build_model(
        descriptor_channels=train_dataset.conditioning_channels,
        global_dim=train_dataset.global_dim,
        base_channels=base_channels,
        global_embedding_dim=global_embedding_dim,
        time_embedding_dim=time_embedding_dim,
        num_categories=1 if disable_category_embedding else max(len(category_vocabulary), 1),
        category_embedding_dim=category_embedding_dim,
        enable_category_modality_gating=enable_category_modality_gating,
        attention_heads=attention_heads,
        latent_channels=latent_channels,
        diffusion_steps=diffusion_steps,
        beta_start=beta_start,
        beta_end=beta_end,
        noise_schedule=noise_schedule,
        autoencoder_reconstruction_weight=autoencoder_reconstruction_weight,
        diffusion_reconstruction_weight=diffusion_reconstruction_weight,
        edge_reconstruction_weight=edge_reconstruction_weight,
    )
    runtime = resolve_runtime(
        requested_mode=_resolve_requested_mode(device, device_mode),
        model=model,
        validation_batch=prepare_diffusion_batch(next(iter(loaders["train_loader"]))),  # type: ignore[arg-type]
        run_backward=True,
        disable_category_embedding=disable_category_embedding,
    )
    torch_device = runtime.resolved_device
    provenance = _runtime_provenance(resolved_device=torch_device, seed=seed)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    manifests = loaders["manifests"]  # type: ignore[assignment]
    rgb_stats_path = ""
    rgb_stats_by_category_path = ""
    global_vector_stats_by_category_path = ""
    if rgb_stats is not None:
        rgb_stats_path = str(save_rgb_normalization_stats(rgb_stats, run_paths.metrics / "rgb_normalization_stats.json"))
    if rgb_stats_by_category is not None:
        rgb_stats_by_category_path = str(
            save_rgb_normalization_stats_by_category(
                rgb_stats_by_category,
                run_paths.metrics / "rgb_normalization_stats_by_category.json",
            )
        )
    if global_vector_stats_by_category is not None:
        global_vector_stats_by_category_path = str(
            save_global_vector_normalization_stats_by_category(
                global_vector_stats_by_category,
                run_paths.metrics / "global_vector_stats_by_category.json",
            )
        )
    localization_tuning_report: Dict[str, Any] = {}
    localization_tuning_report_path = ""

    if joint_mode:
        global_descriptor_calibration = fit_global_descriptor_reference_by_category(
            loaders["train_reference_loader"],  # type: ignore[arg-type]
        )
        global_descriptor_calibration_path = write_json(
            run_paths.metrics / "global_descriptor_calibration.json",
            {
                category_name: calibration.to_dict()
                for category_name, calibration in sorted(global_descriptor_calibration.items())
            },
        )
        if localization_tune_on_synthetic_validation:
            localization_calibration, localization_tuning_report = fit_tuned_localization_reference_by_category(
                model,
                loaders["calibration_loader"],  # type: ignore[arg-type]
                loaders["synthetic_validation_loader"],  # type: ignore[arg-type]
                device=torch_device,
                target_size=target_size,
                disable_category_embedding=disable_category_embedding,
                score_mode=score_mode,
                anomaly_timestep=anomaly_timestep,
                descriptor_weight=descriptor_weight,
                global_descriptor_weight=global_descriptor_weight,
                multi_timestep_scoring_enabled=multi_timestep_scoring_enabled,
                multi_timestep_fractions=multi_timestep_fractions,
                anomaly_map_gaussian_sigma=anomaly_map_gaussian_sigma,
                spatial_topk_percent=spatial_topk_percent,
                enable_internal_defect_gate=enable_internal_defect_gate,
                lambda_residual=lambda_residual,
                lambda_noise=lambda_noise,
                lambda_descriptor_spatial=lambda_descriptor_spatial,
                lambda_descriptor_global=lambda_descriptor_global,
                global_descriptor_calibration=global_descriptor_calibration,
                object_mask_threshold=object_mask_threshold,
                localization_basis=localization_basis,
                localization_basis_candidates=localization_basis_candidates,
                object_mask_dilation_kernel_size=object_mask_dilation_kernel_size,
                localization_mask_closing_kernel_size=localization_mask_closing_kernel_size,
                localization_fill_holes=localization_fill_holes,
                localization_quantile=localization_quantile,
                localization_quantile_candidates=localization_quantile_candidates,
                localization_min_region_area_fraction=localization_min_region_area_fraction,
                localization_min_region_pixels_floor=localization_min_region_pixels_floor,
                localization_min_region_pixels_floor_candidates=localization_min_region_pixels_floor_candidates,
                seed=seed + 4000,
            )
        else:
            localization_calibration = fit_localization_reference_by_category(
                model,
                loaders["train_reference_loader"],  # type: ignore[arg-type]
                device=torch_device,
                disable_category_embedding=disable_category_embedding,
                score_mode=score_mode,
                anomaly_timestep=anomaly_timestep,
                descriptor_weight=descriptor_weight,
                global_descriptor_weight=global_descriptor_weight,
                multi_timestep_scoring_enabled=multi_timestep_scoring_enabled,
                multi_timestep_fractions=multi_timestep_fractions,
                anomaly_map_gaussian_sigma=anomaly_map_gaussian_sigma,
                spatial_topk_percent=spatial_topk_percent,
                enable_internal_defect_gate=enable_internal_defect_gate,
                lambda_residual=lambda_residual,
                lambda_noise=lambda_noise,
                lambda_descriptor_spatial=lambda_descriptor_spatial,
                lambda_descriptor_global=lambda_descriptor_global,
                global_descriptor_calibration=global_descriptor_calibration,
                object_mask_threshold=object_mask_threshold,
                localization_basis=localization_basis,
                object_mask_dilation_kernel_size=object_mask_dilation_kernel_size,
                localization_mask_closing_kernel_size=localization_mask_closing_kernel_size,
                localization_fill_holes=localization_fill_holes,
                localization_quantile=localization_quantile,
                localization_min_region_area_fraction=localization_min_region_area_fraction,
                localization_min_region_pixels_floor=localization_min_region_pixels_floor,
                seed=seed + 4000,
            )
        localization_calibration_path = write_json(
            run_paths.metrics / "localization_calibration.json",
            {
                category_name: calibration.to_dict()
                for category_name, calibration in sorted(localization_calibration.items())
            },
        )
    else:
        global_descriptor_calibration = fit_global_descriptor_reference(
            loaders["train_reference_loader"],  # type: ignore[arg-type]
            category=category_vocabulary[0] if category_vocabulary else category,
        )
        global_descriptor_calibration_path = write_json(
            run_paths.metrics / "global_descriptor_calibration.json",
            global_descriptor_calibration.to_dict(),
        )
        if localization_tune_on_synthetic_validation:
            localization_calibration_map, localization_tuning_report = fit_tuned_localization_reference_by_category(
                model,
                loaders["calibration_loader"],  # type: ignore[arg-type]
                loaders["synthetic_validation_loader"],  # type: ignore[arg-type]
                device=torch_device,
                target_size=target_size,
                disable_category_embedding=disable_category_embedding,
                score_mode=score_mode,
                anomaly_timestep=anomaly_timestep,
                descriptor_weight=descriptor_weight,
                global_descriptor_weight=global_descriptor_weight,
                multi_timestep_scoring_enabled=multi_timestep_scoring_enabled,
                multi_timestep_fractions=multi_timestep_fractions,
                anomaly_map_gaussian_sigma=anomaly_map_gaussian_sigma,
                lambda_residual=lambda_residual,
                lambda_noise=lambda_noise,
                lambda_descriptor_spatial=lambda_descriptor_spatial,
                lambda_descriptor_global=lambda_descriptor_global,
                global_descriptor_calibration=global_descriptor_calibration,
                object_mask_threshold=object_mask_threshold,
                localization_basis=localization_basis,
                localization_basis_candidates=localization_basis_candidates,
                object_mask_dilation_kernel_size=object_mask_dilation_kernel_size,
                localization_mask_closing_kernel_size=localization_mask_closing_kernel_size,
                localization_fill_holes=localization_fill_holes,
                localization_quantile=localization_quantile,
                localization_quantile_candidates=localization_quantile_candidates,
                localization_min_region_area_fraction=localization_min_region_area_fraction,
                localization_min_region_pixels_floor=localization_min_region_pixels_floor,
                localization_min_region_pixels_floor_candidates=localization_min_region_pixels_floor_candidates,
                seed=seed + 4000,
            )
        else:
            localization_calibration_map = fit_localization_reference_by_category(
                model,
                loaders["train_reference_loader"],  # type: ignore[arg-type]
                device=torch_device,
                disable_category_embedding=disable_category_embedding,
                score_mode=score_mode,
                anomaly_timestep=anomaly_timestep,
                descriptor_weight=descriptor_weight,
                global_descriptor_weight=global_descriptor_weight,
                multi_timestep_scoring_enabled=multi_timestep_scoring_enabled,
                multi_timestep_fractions=multi_timestep_fractions,
                anomaly_map_gaussian_sigma=anomaly_map_gaussian_sigma,
                lambda_residual=lambda_residual,
                lambda_noise=lambda_noise,
                lambda_descriptor_spatial=lambda_descriptor_spatial,
                lambda_descriptor_global=lambda_descriptor_global,
                global_descriptor_calibration=global_descriptor_calibration,
                object_mask_threshold=object_mask_threshold,
                localization_basis=localization_basis,
                object_mask_dilation_kernel_size=object_mask_dilation_kernel_size,
                localization_mask_closing_kernel_size=localization_mask_closing_kernel_size,
                localization_fill_holes=localization_fill_holes,
                localization_quantile=localization_quantile,
                localization_min_region_area_fraction=localization_min_region_area_fraction,
                localization_min_region_pixels_floor=localization_min_region_pixels_floor,
                seed=seed + 4000,
            )
        selected_localization_category = category_vocabulary[0] if category_vocabulary else category
        localization_calibration = localization_calibration_map[selected_localization_category]
        localization_calibration_path = write_json(
            run_paths.metrics / "localization_calibration.json",
            localization_calibration.to_dict(),
        )
    if localization_tuning_report:
        localization_tuning_report_path = str(
            write_json(run_paths.metrics / "localization_tuning_report.json", localization_tuning_report)
        )

    config_payload = {
        "regime_internal": regime.internal,
        "regime_paper": regime.paper,
        "regime_expansion": regime.expansion,
        "category": "shared" if joint_mode else selected_categories[0],
        "selected_categories": selected_categories,
        "scope": scope,
        "category_vocabulary": category_vocabulary,
        "training_mode": "shared_joint" if joint_mode else "single_category",
        "disable_category_embedding": disable_category_embedding,
        "ablation_mode": "shared_nocat" if disable_category_embedding and joint_mode else "shared" if joint_mode else "per_category",
        "epochs": epochs,
        "batch_size": batch_size,
        "target_size": target_size,
        "device": str(torch_device),
        "resolved_device": str(torch_device),
        "device_mode": runtime.requested_mode,
        "amp_enabled": runtime.use_amp,
        "runtime_fallback_reason": runtime.fallback_reason,
        "learning_rate": learning_rate,
        "score_mode": score_mode,
        "anomaly_timestep": anomaly_timestep,
        "descriptor_weight": descriptor_weight,
        "global_descriptor_weight": global_descriptor_weight,
        "multi_timestep_scoring_enabled": multi_timestep_scoring_enabled,
        "multi_timestep_fractions": list(multi_timestep_fractions or (0.05, 0.10, 0.20, 0.30, 0.40)),
        "anomaly_map_gaussian_sigma": anomaly_map_gaussian_sigma,
        "spatial_topk_percent": spatial_topk_percent,
        "object_score_strategy": object_score_strategy,
        "lambda_residual": lambda_residual,
        "lambda_noise": lambda_noise,
        "lambda_descriptor_spatial": lambda_descriptor_spatial,
        "lambda_descriptor_global": lambda_descriptor_global,
        "score_weight_schedule": score_weight_schedule,
        "pixel_threshold_percentile": pixel_threshold_percentile,
        "object_mask_threshold": object_mask_threshold,
        "localization_basis": localization_basis,
        "localization_basis_candidates": list(localization_basis_candidates or []),
        "object_mask_dilation_kernel_size": object_mask_dilation_kernel_size,
        "localization_mask_closing_kernel_size": localization_mask_closing_kernel_size,
        "localization_fill_holes": localization_fill_holes,
        "localization_tune_on_synthetic_validation": localization_tune_on_synthetic_validation,
        "localization_quantile": localization_quantile,
        "localization_quantile_candidates": list(localization_quantile_candidates or []),
        "localization_min_region_area_fraction": localization_min_region_area_fraction,
        "localization_min_region_pixels_floor": localization_min_region_pixels_floor,
        "localization_min_region_pixels_floor_candidates": list(
            localization_min_region_pixels_floor_candidates or []
        ),
        "object_crop_enabled": object_crop_enabled,
        "object_crop_margin_ratio": object_crop_margin_ratio,
        "object_crop_mask_source": object_crop_mask_source,
        "rgb_normalization_mode": rgb_normalization_mode,
        "enable_internal_defect_gate": enable_internal_defect_gate,
        "rgb_normalization_stats": rgb_stats.to_dict() if rgb_stats is not None else None,
        "rgb_normalization_stats_by_category": {
            category_name: stats.to_dict()
            for category_name, stats in sorted((rgb_stats_by_category or {}).items())
        }
        if rgb_stats_by_category is not None
        else None,
        "rgb_normalization_stats_path": rgb_stats_path,
        "rgb_normalization_stats_by_category_path": rgb_stats_by_category_path,
        "global_vector_normalization_stats_by_category": {
            category_name: stats.to_dict()
            for category_name, stats in sorted((global_vector_stats_by_category or {}).items())
        }
        if global_vector_stats_by_category is not None
        else None,
        "global_vector_normalization_stats_by_category_path": global_vector_stats_by_category_path,
        "localization_calibration_path": str(localization_calibration_path),
        "training_protocol": "train_good_only",
        "evaluation_protocol": "test_good_plus_anomaly",
        "train_manifest_csv": str(manifests["train_csv"]),
        "eval_manifest_csv": str(manifests["eval_csv"]),
        "global_stats_json": str(manifests["global_stats_json"]),
        "processed_root": str(resolved_processed_root),
        "max_train_batches": max_train_batches,
        "max_eval_batches": max_eval_batches,
        "base_channels": base_channels,
        "global_embedding_dim": global_embedding_dim,
        "time_embedding_dim": time_embedding_dim,
        "category_embedding_dim": category_embedding_dim,
        "enable_category_modality_gating": enable_category_modality_gating,
        "num_categories": 1 if disable_category_embedding else max(len(category_vocabulary), 1),
        "attention_heads": attention_heads,
        "multiscale_conditioning": multiscale_conditioning,
        "sampler_mode": loaders["sampler_mode"],
        "train_category_counts": loaders["train_category_counts"],
        "selection_metric": selection_metric,
        "latent_channels": latent_channels,
        "diffusion_steps": diffusion_steps,
        "beta_start": beta_start,
        "beta_end": beta_end,
        "noise_schedule": noise_schedule,
        "autoencoder_reconstruction_weight": autoencoder_reconstruction_weight,
        "diffusion_reconstruction_weight": diffusion_reconstruction_weight,
        "edge_reconstruction_weight": edge_reconstruction_weight,
        "log_every_n_steps": log_every_n_steps,
        "log_to_jsonl": log_to_jsonl,
        **provenance,
    }
    write_json(run_paths.root / "config.json", config_payload)

    history: List[Dict[str, Any]] = []
    best_auroc = float("-inf")
    best_checkpoint_path: Path | None = None
    step_log_path = run_paths.logs / "train_steps.jsonl"
    event_log_path = run_paths.logs / "events.jsonl"

    print(
        f"[train] categories={selected_categories} epochs={epochs} device={torch_device} "
        f"train_records={len(train_dataset)} eval_records={len(eval_dataset)}"
    )
    append_jsonl(
        event_log_path,
        {
            "event": "run_started",
            "regime_internal": regime.internal,
            "regime_paper": regime.paper,
            "regime_expansion": regime.expansion,
            "category": "shared" if joint_mode else selected_categories[0],
            "selected_categories": selected_categories,
            "scope": scope,
            "training_mode": "shared_joint" if joint_mode else "single_category",
            "disable_category_embedding": disable_category_embedding,
            "ablation_mode": "shared_nocat" if disable_category_embedding and joint_mode else "shared" if joint_mode else "per_category",
            "epochs": epochs,
            "device_mode": runtime.requested_mode,
            "resolved_device": str(torch_device),
            "amp_enabled": runtime.use_amp,
            "fallback_reason": runtime.fallback_reason,
            "spatial_topk_percent": spatial_topk_percent,
            "object_score_strategy": object_score_strategy,
            "enable_category_modality_gating": enable_category_modality_gating,
            "enable_internal_defect_gate": enable_internal_defect_gate,
            "training_protocol": "train_good_only",
            "evaluation_protocol": "test_good_plus_anomaly",
            "seed": seed,
            "git_commit": provenance["git_commit"],
            "git_dirty": provenance["git_dirty"],
        },
    )

    for epoch in range(1, epochs + 1):
        score_weights = resolve_score_weights(
            total_epochs=epochs,
            epoch=epoch,
            lambda_residual=lambda_residual,
            lambda_noise=lambda_noise,
            lambda_descriptor_spatial=lambda_descriptor_spatial,
            lambda_descriptor_global=lambda_descriptor_global,
            score_weight_schedule=score_weight_schedule,
        )
        train_reports = run_training_epoch(
            model,
            loaders["train_loader"],  # type: ignore[arg-type]
            optimizer,
            device=torch_device,
            disable_category_embedding=disable_category_embedding,
            use_amp=runtime.use_amp,
            epoch=epoch,
            max_batches=max_train_batches,
            seed=seed + epoch,
            log_every_n_steps=log_every_n_steps,
            log_to_jsonl=log_to_jsonl,
            step_log_path=step_log_path,
        )
        spatial_exceedance_calibration = None
        object_score_calibration = None
        if object_score_strategy != "legacy_raw":
            spatial_exceedance_calibration = fit_spatial_exceedance_calibration_by_category(
                model,
                loaders["calibration_loader"],  # type: ignore[arg-type]
                device=torch_device,
                disable_category_embedding=disable_category_embedding,
                score_mode=score_mode,
                anomaly_timestep=anomaly_timestep,
                descriptor_weight=descriptor_weight,
                global_descriptor_weight=global_descriptor_weight,
                multi_timestep_scoring_enabled=multi_timestep_scoring_enabled,
                multi_timestep_fractions=multi_timestep_fractions,
                anomaly_map_gaussian_sigma=anomaly_map_gaussian_sigma,
                lambda_residual=score_weights.lambda_residual,
                lambda_noise=score_weights.lambda_noise,
                lambda_descriptor_spatial=score_weights.lambda_descriptor_spatial,
                lambda_descriptor_global=score_weights.lambda_descriptor_global,
                global_descriptor_calibration=global_descriptor_calibration,
                object_mask_threshold=object_mask_threshold,
                localization_quantile=localization_quantile,
                localization_min_region_area_fraction=localization_min_region_area_fraction,
                localization_min_region_pixels_floor=localization_min_region_pixels_floor,
                seed=seed + 5000 + epoch,
            )
            if object_score_strategy == "exceedance_logreg":
                negative_features, negative_labels = collect_object_score_features(
                    model,
                    loaders["calibration_loader"],  # type: ignore[arg-type]
                    device=torch_device,
                    disable_category_embedding=disable_category_embedding,
                    score_mode=score_mode,
                    anomaly_timestep=anomaly_timestep,
                    descriptor_weight=descriptor_weight,
                    global_descriptor_weight=global_descriptor_weight,
                    multi_timestep_scoring_enabled=multi_timestep_scoring_enabled,
                    multi_timestep_fractions=multi_timestep_fractions,
                    anomaly_map_gaussian_sigma=anomaly_map_gaussian_sigma,
                    lambda_residual=score_weights.lambda_residual,
                    lambda_noise=score_weights.lambda_noise,
                    lambda_descriptor_spatial=score_weights.lambda_descriptor_spatial,
                    lambda_descriptor_global=score_weights.lambda_descriptor_global,
                    global_descriptor_calibration=global_descriptor_calibration,
                    spatial_exceedance_calibration=spatial_exceedance_calibration,
                    object_mask_threshold=object_mask_threshold,
                    seed=seed + 6000 + epoch,
                )
                positive_features, positive_labels = collect_object_score_features(
                    model,
                    loaders["synthetic_validation_loader"],  # type: ignore[arg-type]
                    device=torch_device,
                    disable_category_embedding=disable_category_embedding,
                    score_mode=score_mode,
                    anomaly_timestep=anomaly_timestep,
                    descriptor_weight=descriptor_weight,
                    global_descriptor_weight=global_descriptor_weight,
                    multi_timestep_scoring_enabled=multi_timestep_scoring_enabled,
                    multi_timestep_fractions=multi_timestep_fractions,
                    anomaly_map_gaussian_sigma=anomaly_map_gaussian_sigma,
                    lambda_residual=score_weights.lambda_residual,
                    lambda_noise=score_weights.lambda_noise,
                    lambda_descriptor_spatial=score_weights.lambda_descriptor_spatial,
                    lambda_descriptor_global=score_weights.lambda_descriptor_global,
                    global_descriptor_calibration=global_descriptor_calibration,
                    spatial_exceedance_calibration=spatial_exceedance_calibration,
                    object_mask_threshold=object_mask_threshold,
                    seed=seed + 7000 + epoch,
                )
                object_score_calibration = fit_logistic_object_score_calibration(
                    torch.cat([negative_features, positive_features], dim=0),
                    torch.cat([negative_labels, positive_labels], dim=0),
                )
        eval_summary = run_eval_scoring(
            model,
            loaders["eval_loader"],  # type: ignore[arg-type]
            device=torch_device,
            disable_category_embedding=disable_category_embedding,
            target_size=target_size,
            score_mode=score_mode,
            anomaly_timestep=anomaly_timestep,
            descriptor_weight=descriptor_weight,
            global_descriptor_weight=global_descriptor_weight,
            multi_timestep_scoring_enabled=multi_timestep_scoring_enabled,
            multi_timestep_fractions=multi_timestep_fractions,
            anomaly_map_gaussian_sigma=anomaly_map_gaussian_sigma,
            spatial_topk_percent=spatial_topk_percent,
            enable_internal_defect_gate=enable_internal_defect_gate,
            lambda_residual=score_weights.lambda_residual,
            lambda_noise=score_weights.lambda_noise,
            lambda_descriptor_spatial=score_weights.lambda_descriptor_spatial,
            lambda_descriptor_global=score_weights.lambda_descriptor_global,
            global_descriptor_calibration=global_descriptor_calibration,
            spatial_exceedance_calibration=spatial_exceedance_calibration,
            object_score_calibration=object_score_calibration,
            object_score_strategy=object_score_strategy,
            pixel_threshold_percentile=pixel_threshold_percentile,
            object_mask_threshold=object_mask_threshold,
            localization_basis=localization_basis,
            object_mask_dilation_kernel_size=object_mask_dilation_kernel_size,
            max_batches=max_eval_batches,
            visualization_dir=run_paths.predictions / f"epoch_{epoch:03d}",
            preview_selection_path=run_paths.metrics / "preview_selection.json",
            max_visualizations=max_visualizations,
            enable_llm_explanations=False,
            localization_calibration=localization_calibration,
            rgb_normalization_stats=rgb_stats,
            rgb_normalization_stats_by_category=rgb_stats_by_category,
            preview_seed=seed,
            seed=seed + 1000 + epoch,
        )
        selected_metric_value = float(
            eval_summary["macro_image_auroc"] if joint_mode and selection_metric == "macro_image_auroc" else eval_summary["image_auroc"]
        )

        epoch_summary = {
            "epoch": epoch,
            "train_loss": round(sum(report["loss"] for report in train_reports) / max(len(train_reports), 1), 6),
            "noise_loss": round(sum(report["noise_loss"] for report in train_reports) / max(len(train_reports), 1), 6),
            "autoencoder_reconstruction_loss": round(
                sum(report["autoencoder_reconstruction_loss"] for report in train_reports) / max(len(train_reports), 1),
                6,
            ),
            "diffusion_reconstruction_loss": round(
                sum(report["diffusion_reconstruction_loss"] for report in train_reports) / max(len(train_reports), 1),
                6,
            ),
            "edge_reconstruction_loss": round(
                sum(report["edge_reconstruction_loss"] for report in train_reports) / max(len(train_reports), 1),
                6,
            ),
            "image_auroc": eval_summary["image_auroc"],
            "image_auprc": eval_summary["image_auprc"],
            "image_f1": eval_summary["image_f1"],
            "pixel_auroc": eval_summary["pixel_auroc"],
            "aupro": eval_summary["aupro"],
            "pixel_f1": eval_summary["pixel_f1"],
            "pixel_iou": eval_summary["pixel_iou"],
            "macro_image_auroc": eval_summary.get("macro_image_auroc", eval_summary["image_auroc"]),
            "macro_image_auprc": eval_summary.get("macro_image_auprc", eval_summary["image_auprc"]),
            "macro_pixel_auroc": eval_summary.get("macro_pixel_auroc", eval_summary["pixel_auroc"]),
            "macro_aupro": eval_summary.get("macro_aupro", eval_summary["aupro"]),
            "mean_good_score": eval_summary["mean_good_score"],
            "mean_anomalous_score": eval_summary["mean_anomalous_score"],
            "selected_metric": round(selected_metric_value, 6),
            **score_weights.to_dict(),
        }
        history.append(epoch_summary)

        write_json(
            run_paths.metrics / f"epoch_{epoch:03d}.json",
            {
                "epoch": epoch,
                "train_reports": train_reports,
                "eval_summary": eval_summary,
            },
        )
        write_json(run_paths.metrics / "per_category.json", eval_summary.get("per_category", {}))
        write_history_csv(
            run_paths.metrics / "per_category.csv",
            _per_category_metric_rows(eval_summary.get("per_category", {})),
        )

        save_checkpoint(
            run_paths.checkpoints / f"epoch_{epoch:03d}.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=epoch_summary,
                config=config_payload,
            )
        if selected_metric_value >= best_auroc:
            best_auroc = float(selected_metric_value)
            best_checkpoint_path = save_checkpoint(
                run_paths.checkpoints / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=epoch_summary,
                config=config_payload,
            )
            print(
                f"[train] epoch={epoch} new_best {selection_metric}={selected_metric_value:.6f} "
                f"checkpoint={best_checkpoint_path}"
            )
            append_jsonl(
                event_log_path,
                {
                    "event": "new_best_checkpoint",
                    "epoch": epoch,
                    "image_auroc": eval_summary["image_auroc"],
                    "selected_metric": round(selected_metric_value, 6),
                    "checkpoint_path": str(best_checkpoint_path),
                },
            )

        write_history_csv(run_paths.logs / "history.csv", history)
        write_json(run_paths.logs / "history.json", history)
        save_training_curves(history, run_paths.plots / "training_curves.png")
        append_jsonl(
            event_log_path,
            {
                "event": "epoch_completed",
                "epoch": epoch,
                "train_loss": epoch_summary["train_loss"],
                "image_auroc": epoch_summary["image_auroc"],
                "macro_image_auroc": epoch_summary["macro_image_auroc"],
                "pixel_iou": epoch_summary["pixel_iou"],
                **score_weights.to_dict(),
            },
        )

    final_summary = {
        "run_dir": str(run_paths.root),
        "regime_internal": regime.internal,
        "regime_paper": regime.paper,
        "regime_expansion": regime.expansion,
        "category": "shared" if joint_mode else selected_categories[0],
        "selected_categories": selected_categories,
        "scope": scope,
        "training_mode": "shared_joint" if joint_mode else "single_category",
        "disable_category_embedding": disable_category_embedding,
        "ablation_mode": "shared_nocat" if disable_category_embedding and joint_mode else "shared" if joint_mode else "per_category",
        "epochs": epochs,
        "device": str(torch_device),
        "resolved_device": str(torch_device),
        "device_mode": runtime.requested_mode,
        "amp_enabled": runtime.use_amp,
        "runtime_fallback_reason": runtime.fallback_reason,
        "learning_rate": learning_rate,
        "score_mode": score_mode,
        "spatial_topk_percent": spatial_topk_percent,
        "object_score_strategy": object_score_strategy,
        "enable_category_modality_gating": enable_category_modality_gating,
        "enable_internal_defect_gate": enable_internal_defect_gate,
        "localization_basis": localization_basis,
        "object_mask_dilation_kernel_size": object_mask_dilation_kernel_size,
        "localization_tune_on_synthetic_validation": localization_tune_on_synthetic_validation,
        "train_records": len(train_dataset),
        "eval_records": len(eval_dataset),
        "training_protocol": "train_good_only",
        "evaluation_protocol": "test_good_plus_anomaly",
        "processed_root": str(resolved_processed_root),
        "rgb_normalization_stats_path": rgb_stats_path,
        "rgb_normalization_stats_by_category_path": rgb_stats_by_category_path,
        "global_vector_normalization_stats_by_category_path": global_vector_stats_by_category_path,
        "best_image_auroc": round(best_auroc, 6) if best_auroc != float("-inf") else 0.0,
        "best_selected_metric": round(best_auroc, 6) if best_auroc != float("-inf") else 0.0,
        "selection_metric": selection_metric,
        "best_checkpoint": str(best_checkpoint_path) if best_checkpoint_path else "",
        "history_rows": len(history),
        "global_descriptor_calibration_path": str(global_descriptor_calibration_path),
        "localization_calibration_path": str(localization_calibration_path),
        "localization_tuning_report_path": str(localization_tuning_report_path),
        "preview_selection_path": str(run_paths.metrics / "preview_selection.json") if joint_mode else "",
        **provenance,
    }
    write_json(run_paths.root / "summary.json", final_summary)
    append_jsonl(event_log_path, {"event": "run_completed", "summary": final_summary})
    return {
        "run_paths": run_paths,
        "history": history,
        "summary": final_summary,
    }


def evaluate_checkpoint(
    *,
    checkpoint_path: Path | str,
    data_root: Path | str = "data",
    processed_root: Path | str | None = None,
    descriptor_index_csv: Path | str = "data/processed/manifests/descriptor_index.csv",
    category: str = "capsule",
    categories: Sequence[str] | None = None,
    batch_size: int = 2,
    target_size: tuple[int, int] = (256, 256),
    device: str = "cpu",
    device_mode: str = "",
    disable_category_embedding: bool | None = None,
    enable_category_modality_gating: bool | None = None,
    score_mode: str = "noise_error",
    anomaly_timestep: int = 200,
    descriptor_weight: float = 0.25,
    global_descriptor_weight: float = 0.1,
    multi_timestep_scoring_enabled: bool | None = None,
    multi_timestep_fractions: Sequence[float] | None = None,
    anomaly_map_gaussian_sigma: float | None = None,
    spatial_topk_percent: float | None = None,
    object_score_strategy: str = "legacy_raw",
    lambda_residual: float | None = None,
    lambda_noise: float | None = None,
    lambda_descriptor_spatial: float | None = None,
    lambda_descriptor_global: float | None = None,
    pixel_threshold_percentile: float = 0.9,
    object_mask_threshold: float = 0.02,
    localization_basis: str = "anomaly_map",
    localization_basis_candidates: Sequence[str] | None = None,
    object_mask_dilation_kernel_size: int = 9,
    localization_mask_closing_kernel_size: int = 0,
    localization_fill_holes: bool = False,
    localization_tune_on_synthetic_validation: bool | None = None,
    localization_quantile: float = 0.995,
    localization_quantile_candidates: Sequence[float] | None = None,
    localization_min_region_area_fraction: float = 0.002,
    localization_min_region_pixels_floor: int = 4,
    localization_min_region_pixels_floor_candidates: Sequence[int] | None = None,
    output_root: Path | str = "runs",
    run_name: str | None = None,
    max_eval_batches: int = 0,
    max_visualizations: int = 10,
    knowledge_base_root: Path | str | None = None,
    retrieval_top_k: int = 3,
    enable_llm_explanations: bool = False,
    object_crop_enabled: bool | None = None,
    object_crop_margin_ratio: float | None = None,
    object_crop_mask_source: str = "",
    rgb_normalization_mode: str = "",
    enable_internal_defect_gate: bool | None = None,
    seed: int = 7,
) -> Dict[str, Any]:
    _set_global_seed(seed)
    checkpoint_payload = read_checkpoint_payload(checkpoint_path, map_location=torch.device("cpu"))
    checkpoint_config = checkpoint_payload.get("config", {})
    selected_categories = _normalize_selected_categories(
        category=category,
        categories=categories,
        fallback=checkpoint_config.get("selected_categories"),
    )
    resolved_object_crop_enabled = (
        bool(object_crop_enabled)
        if object_crop_enabled is not None
        else bool(checkpoint_config.get("object_crop_enabled", False))
    )
    resolved_object_crop_margin_ratio = float(
        object_crop_margin_ratio
        if object_crop_margin_ratio is not None
        else checkpoint_config.get("object_crop_margin_ratio", 0.10)
    )
    resolved_object_crop_mask_source = object_crop_mask_source or str(
        checkpoint_config.get("object_crop_mask_source", "rgb_then_pc_density")
    )
    resolved_rgb_normalization_mode = rgb_normalization_mode or str(
        checkpoint_config.get("rgb_normalization_mode", "none")
    )
    resolved_rgb_stats = _coerce_rgb_normalization_stats(checkpoint_config.get("rgb_normalization_stats"))
    resolved_rgb_stats_by_category = _coerce_rgb_normalization_stats_by_category(
        checkpoint_config.get("rgb_normalization_stats_by_category")
    )
    resolved_global_vector_stats_by_category = _coerce_global_vector_normalization_stats_by_category(
        checkpoint_config.get("global_vector_normalization_stats_by_category")
    )
    resolved_disable_category_embedding = (
        disable_category_embedding
        if disable_category_embedding is not None
        else bool(checkpoint_config.get("disable_category_embedding", False))
    )
    resolved_enable_category_modality_gating = (
        bool(enable_category_modality_gating)
        if enable_category_modality_gating is not None
        else bool(checkpoint_config.get("enable_category_modality_gating", False))
    )
    resolved_enable_internal_defect_gate = (
        bool(enable_internal_defect_gate)
        if enable_internal_defect_gate is not None
        else bool(checkpoint_config.get("enable_internal_defect_gate", False))
    )
    resolved_spatial_topk_percent = float(
        spatial_topk_percent
        if spatial_topk_percent is not None
        else checkpoint_config.get("spatial_topk_percent", 0.02)
    )
    resolved_object_score_strategy = str(
        object_score_strategy or checkpoint_config.get("object_score_strategy", "legacy_raw")
    ).strip().lower()
    resolved_localization_basis = str(
        localization_basis or checkpoint_config.get("localization_basis", "anomaly_map")
    ).strip().lower()
    resolved_object_mask_dilation_kernel_size = int(
        checkpoint_config.get("object_mask_dilation_kernel_size", object_mask_dilation_kernel_size)
    )
    resolved_localization_mask_closing_kernel_size = int(
        localization_mask_closing_kernel_size
        if localization_mask_closing_kernel_size is not None
        else checkpoint_config.get("localization_mask_closing_kernel_size", 0)
    )
    resolved_localization_fill_holes = (
        bool(localization_fill_holes)
        if localization_fill_holes is not None
        else bool(checkpoint_config.get("localization_fill_holes", False))
    )
    resolved_localization_tune_on_synthetic_validation = (
        bool(localization_tune_on_synthetic_validation)
        if localization_tune_on_synthetic_validation is not None
        else bool(checkpoint_config.get("localization_tune_on_synthetic_validation", False))
    )
    resolved_localization_basis_candidates = list(
        localization_basis_candidates
        if localization_basis_candidates is not None
        else checkpoint_config.get("localization_basis_candidates", [])
    )
    resolved_localization_quantile_candidates = list(
        localization_quantile_candidates
        if localization_quantile_candidates is not None
        else checkpoint_config.get("localization_quantile_candidates", [])
    )
    resolved_localization_min_region_pixels_floor_candidates = list(
        localization_min_region_pixels_floor_candidates
        if localization_min_region_pixels_floor_candidates is not None
        else checkpoint_config.get("localization_min_region_pixels_floor_candidates", [])
    )
    scope = resolve_scope(selected_categories, category=category)
    regime = resolve_regime_identity(
        joint_mode=is_joint_category_request(selected_categories, category=category),
        disable_category_embedding=resolved_disable_category_embedding,
    )
    resolved_output_root = (
        default_output_root(
            regime_paper=regime.paper,
            run_type="eval",
            scope=scope,
            label=run_name or "baseline",
            repo_root=_repo_root(),
        )
        if str(output_root).strip() == "runs"
        else Path(output_root)
    )
    resolved_processed_root = resolve_processed_root(
        data_root=data_root,
        processed_root=processed_root or checkpoint_config.get("processed_root"),
    )
    run_paths = create_run_dir(
        output_root=resolved_output_root,
        run_name=run_name or "eval",
        category=_run_slug_for_categories(selected_categories or [category]),
        regime_paper=regime.paper,
        scope=scope,
        run_type="eval",
        label=run_name or "baseline",
    )
    loaders = build_diffusion_dataloaders(
        data_root=data_root,
        processed_root=resolved_processed_root,
        descriptor_index_csv=descriptor_index_csv,
        category=category,
        categories=selected_categories,
        batch_size=batch_size,
        target_size=target_size,
        object_crop_enabled=resolved_object_crop_enabled,
        object_crop_margin_ratio=resolved_object_crop_margin_ratio,
        object_crop_mask_source=resolved_object_crop_mask_source,
        rgb_normalization_mode=resolved_rgb_normalization_mode,
        rgb_normalization_stats=resolved_rgb_stats,
        rgb_normalization_stats_by_category=checkpoint_config.get("rgb_normalization_stats_by_category"),
        global_vector_normalization_stats_by_category=checkpoint_config.get("global_vector_normalization_stats_by_category"),
        manifests_root=run_paths.root / "manifests",
        sampler_mode=str(checkpoint_config.get("sampler_mode", "balanced_by_category")),
        category_vocabulary=checkpoint_config.get("category_vocabulary"),
        seed=seed,
    )
    eval_dataset: DescriptorConditioningDataset = loaders["eval_dataset"]  # type: ignore[assignment]
    rgb_stats: RGBNormalizationStats | None = loaders["rgb_normalization_stats"]  # type: ignore[assignment]
    rgb_stats_by_category: Dict[str, RGBNormalizationStats] | None = loaders["rgb_normalization_stats_by_category"]  # type: ignore[assignment]
    category_vocabulary = list(loaders["category_vocabulary"])  # type: ignore[assignment]
    joint_mode = len(category_vocabulary) > 1
    reported_selected_categories = category_vocabulary if joint_mode and scope == "all" else selected_categories

    model = build_model(
        descriptor_channels=int(checkpoint_config.get("conditioning_channels", eval_dataset.conditioning_channels)),
        global_dim=eval_dataset.global_dim,
        base_channels=int(checkpoint_config.get("base_channels", 32)),
        global_embedding_dim=int(checkpoint_config.get("global_embedding_dim", 128)),
        time_embedding_dim=int(checkpoint_config.get("time_embedding_dim", 128)),
        num_categories=1 if resolved_disable_category_embedding else max(len(category_vocabulary), 1),
        category_embedding_dim=int(checkpoint_config.get("category_embedding_dim", 32)),
        enable_category_modality_gating=resolved_enable_category_modality_gating,
        attention_heads=int(checkpoint_config.get("attention_heads", 4)),
        latent_channels=int(checkpoint_config.get("latent_channels", 4)),
        diffusion_steps=int(checkpoint_config.get("diffusion_steps", 1000)),
        beta_start=float(checkpoint_config.get("beta_start", 1e-4)),
        beta_end=float(checkpoint_config.get("beta_end", 2e-2)),
        noise_schedule=str(checkpoint_config.get("noise_schedule", "cosine")),
        autoencoder_reconstruction_weight=float(
            checkpoint_config.get("autoencoder_reconstruction_weight", 0.5)
        ),
        diffusion_reconstruction_weight=float(
            checkpoint_config.get("diffusion_reconstruction_weight", 0.1)
        ),
        edge_reconstruction_weight=float(
            checkpoint_config.get("edge_reconstruction_weight", 0.05)
        ),
    )
    runtime = resolve_runtime(
        requested_mode=_resolve_requested_mode(device, device_mode),
        model=model,
        validation_batch=prepare_diffusion_batch(next(iter(loaders["eval_loader"]))),  # type: ignore[arg-type]
        run_backward=False,
        disable_category_embedding=resolved_disable_category_embedding,
    )
    torch_device = runtime.resolved_device
    provenance = _runtime_provenance(resolved_device=torch_device, seed=seed)
    payload = load_checkpoint(checkpoint_path, model=model, map_location=torch_device)
    resolved_score_mode = score_mode or str(checkpoint_config.get("score_mode", "noise_error"))
    resolved_multi_timestep = (
        bool(multi_timestep_scoring_enabled)
        if multi_timestep_scoring_enabled is not None
        else bool(checkpoint_config.get("multi_timestep_scoring_enabled", False))
    )
    resolved_multi_timestep_fractions = (
        list(multi_timestep_fractions)
        if multi_timestep_fractions is not None
        else list(checkpoint_config.get("multi_timestep_fractions", [0.05, 0.10, 0.20, 0.30, 0.40]))
    )
    resolved_anomaly_map_sigma = float(
        anomaly_map_gaussian_sigma
        if anomaly_map_gaussian_sigma is not None
        else checkpoint_config.get("anomaly_map_gaussian_sigma", 0.0)
    )
    resolved_localization_quantile = float(
        localization_quantile
        if localization_quantile is not None
        else checkpoint_config.get("localization_quantile", 0.995)
    )
    resolved_localization_min_region_area_fraction = float(
        localization_min_region_area_fraction
        if localization_min_region_area_fraction is not None
        else checkpoint_config.get("localization_min_region_area_fraction", 0.002)
    )
    resolved_localization_min_region_pixels_floor = int(
        localization_min_region_pixels_floor
        if localization_min_region_pixels_floor is not None
        else checkpoint_config.get("localization_min_region_pixels_floor", 4)
    )
    score_weights = resolve_score_weights(
        total_epochs=int(checkpoint_config.get("epochs", 0)),
        epoch=int(payload.get("epoch", 0)) or int(checkpoint_config.get("epochs", 0) or 0),
        lambda_residual=float(
            lambda_residual if lambda_residual is not None else checkpoint_config.get("lambda_residual", 0.0)
        ),
        lambda_noise=float(
            lambda_noise if lambda_noise is not None else checkpoint_config.get("lambda_noise", 1.0)
        ),
        lambda_descriptor_spatial=float(
            lambda_descriptor_spatial
            if lambda_descriptor_spatial is not None
            else checkpoint_config.get("lambda_descriptor_spatial", descriptor_weight)
        ),
        lambda_descriptor_global=float(
            lambda_descriptor_global
            if lambda_descriptor_global is not None
            else checkpoint_config.get("lambda_descriptor_global", global_descriptor_weight)
        ),
        score_weight_schedule=checkpoint_config.get("score_weight_schedule"),
    )
    evidence_root = run_paths.root / "evidence"
    rgb_stats_path = ""
    rgb_stats_by_category_path = ""
    spatial_exceedance_calibration = None
    spatial_exceedance_calibration_path = ""
    object_score_calibration = None
    object_score_calibration_path = ""
    global_branch_gate_calibration = None
    global_branch_gate_calibration_path = ""
    localization_tuning_report: Dict[str, Any] = {}
    localization_tuning_report_path = ""
    if rgb_stats is not None:
        rgb_stats_path = str(save_rgb_normalization_stats(rgb_stats, evidence_root / "rgb_normalization_stats.json"))
    if rgb_stats_by_category is not None:
        rgb_stats_by_category_path = str(
            save_rgb_normalization_stats_by_category(
                rgb_stats_by_category,
                evidence_root / "rgb_normalization_stats_by_category.json",
            )
        )
    if joint_mode:
        global_descriptor_calibration = fit_global_descriptor_reference_by_category(
            loaders["train_reference_loader"],  # type: ignore[arg-type]
        )
        global_descriptor_calibration_path = write_json(
            evidence_root / "global_descriptor_calibration.json",
            {
                category_name: calibration.to_dict()
                for category_name, calibration in sorted(global_descriptor_calibration.items())
            },
        )
        if resolved_enable_internal_defect_gate:
            global_branch_gate_calibration = fit_global_branch_gate_calibration_by_category(
                model,
                loaders["calibration_loader"],  # type: ignore[arg-type]
                device=torch_device,
                disable_category_embedding=resolved_disable_category_embedding,
                score_mode=resolved_score_mode,
                anomaly_timestep=anomaly_timestep,
                descriptor_weight=descriptor_weight,
                global_descriptor_weight=global_descriptor_weight,
                multi_timestep_scoring_enabled=resolved_multi_timestep,
                multi_timestep_fractions=resolved_multi_timestep_fractions,
                anomaly_map_gaussian_sigma=resolved_anomaly_map_sigma,
                lambda_residual=score_weights.lambda_residual,
                lambda_noise=score_weights.lambda_noise,
                lambda_descriptor_spatial=score_weights.lambda_descriptor_spatial,
                lambda_descriptor_global=score_weights.lambda_descriptor_global,
                global_descriptor_calibration=global_descriptor_calibration,
                object_mask_threshold=object_mask_threshold,
                object_mask_dilation_kernel_size=resolved_object_mask_dilation_kernel_size,
                seed=seed + 2300,
            )
            global_branch_gate_calibration_path = write_json(
                evidence_root / "global_branch_gate_calibration.json",
                {
                    category_name: calibration.to_dict()
                    for category_name, calibration in sorted(global_branch_gate_calibration.items())
                },
            )
        if resolved_localization_tune_on_synthetic_validation:
            localization_calibration, localization_tuning_report = fit_tuned_localization_reference_by_category(
                model,
                loaders["calibration_loader"],  # type: ignore[arg-type]
                loaders["synthetic_validation_loader"],  # type: ignore[arg-type]
                device=torch_device,
                target_size=target_size,
                disable_category_embedding=resolved_disable_category_embedding,
                score_mode=resolved_score_mode,
                anomaly_timestep=anomaly_timestep,
                descriptor_weight=descriptor_weight,
                global_descriptor_weight=global_descriptor_weight,
                multi_timestep_scoring_enabled=resolved_multi_timestep,
                multi_timestep_fractions=resolved_multi_timestep_fractions,
                anomaly_map_gaussian_sigma=resolved_anomaly_map_sigma,
                spatial_topk_percent=resolved_spatial_topk_percent,
                enable_internal_defect_gate=resolved_enable_internal_defect_gate,
                lambda_residual=score_weights.lambda_residual,
                lambda_noise=score_weights.lambda_noise,
                lambda_descriptor_spatial=score_weights.lambda_descriptor_spatial,
                lambda_descriptor_global=score_weights.lambda_descriptor_global,
                global_descriptor_calibration=global_descriptor_calibration,
                global_branch_gate_calibration=global_branch_gate_calibration,
                object_mask_threshold=object_mask_threshold,
                localization_basis=resolved_localization_basis,
                localization_basis_candidates=resolved_localization_basis_candidates,
                object_mask_dilation_kernel_size=resolved_object_mask_dilation_kernel_size,
                localization_mask_closing_kernel_size=resolved_localization_mask_closing_kernel_size,
                localization_fill_holes=resolved_localization_fill_holes,
                localization_quantile=resolved_localization_quantile,
                localization_quantile_candidates=resolved_localization_quantile_candidates,
                localization_min_region_area_fraction=resolved_localization_min_region_area_fraction,
                localization_min_region_pixels_floor=resolved_localization_min_region_pixels_floor,
                localization_min_region_pixels_floor_candidates=resolved_localization_min_region_pixels_floor_candidates,
                seed=seed + 2500,
            )
        else:
            localization_calibration = fit_localization_reference_by_category(
                model,
                loaders["train_reference_loader"],  # type: ignore[arg-type]
                device=torch_device,
                disable_category_embedding=resolved_disable_category_embedding,
                score_mode=resolved_score_mode,
                anomaly_timestep=anomaly_timestep,
                descriptor_weight=descriptor_weight,
                global_descriptor_weight=global_descriptor_weight,
                multi_timestep_scoring_enabled=resolved_multi_timestep,
                multi_timestep_fractions=resolved_multi_timestep_fractions,
                anomaly_map_gaussian_sigma=resolved_anomaly_map_sigma,
                spatial_topk_percent=resolved_spatial_topk_percent,
                enable_internal_defect_gate=resolved_enable_internal_defect_gate,
                lambda_residual=score_weights.lambda_residual,
                lambda_noise=score_weights.lambda_noise,
                lambda_descriptor_spatial=score_weights.lambda_descriptor_spatial,
                lambda_descriptor_global=score_weights.lambda_descriptor_global,
                global_descriptor_calibration=global_descriptor_calibration,
                global_branch_gate_calibration=global_branch_gate_calibration,
                object_mask_threshold=object_mask_threshold,
                localization_basis=resolved_localization_basis,
                object_mask_dilation_kernel_size=resolved_object_mask_dilation_kernel_size,
                localization_mask_closing_kernel_size=resolved_localization_mask_closing_kernel_size,
                localization_fill_holes=resolved_localization_fill_holes,
                localization_quantile=resolved_localization_quantile,
                localization_min_region_area_fraction=resolved_localization_min_region_area_fraction,
                localization_min_region_pixels_floor=resolved_localization_min_region_pixels_floor,
                seed=seed + 2500,
            )
        localization_calibration_path = write_json(
            evidence_root / "localization_calibration.json",
            {
                category_name: calibration.to_dict()
                for category_name, calibration in sorted(localization_calibration.items())
            },
        )
        if resolved_object_score_strategy != "legacy_raw":
            spatial_exceedance_calibration = fit_spatial_exceedance_calibration_by_category(
                model,
                loaders["calibration_loader"],  # type: ignore[arg-type]
                device=torch_device,
                disable_category_embedding=resolved_disable_category_embedding,
                score_mode=resolved_score_mode,
                anomaly_timestep=anomaly_timestep,
                descriptor_weight=descriptor_weight,
                global_descriptor_weight=global_descriptor_weight,
                multi_timestep_scoring_enabled=resolved_multi_timestep,
                multi_timestep_fractions=resolved_multi_timestep_fractions,
                anomaly_map_gaussian_sigma=resolved_anomaly_map_sigma,
                lambda_residual=score_weights.lambda_residual,
                lambda_noise=score_weights.lambda_noise,
                lambda_descriptor_spatial=score_weights.lambda_descriptor_spatial,
                lambda_descriptor_global=score_weights.lambda_descriptor_global,
                global_descriptor_calibration=global_descriptor_calibration,
                object_mask_threshold=object_mask_threshold,
                localization_quantile=resolved_localization_quantile,
                localization_min_region_area_fraction=resolved_localization_min_region_area_fraction,
                localization_min_region_pixels_floor=resolved_localization_min_region_pixels_floor,
                seed=seed + 3200,
            )
            spatial_exceedance_calibration_path = write_json(
                evidence_root / "spatial_exceedance_calibration.json",
                {
                    category_name: calibration.to_dict()
                    for category_name, calibration in sorted(spatial_exceedance_calibration.items())
                },
            )
            object_score_summary_payload = {
                "strategy": resolved_object_score_strategy,
                "calibration_split": "calibration_good",
                "region_mean_weight": 0.7,
                "region_peak_weight": 0.3,
                "local_weight": 0.65,
                "global_weight": 0.35,
                "tau_quantile": 0.999,
                "tail_quantile": 0.99,
            }
            if resolved_object_score_strategy == "exceedance_logreg":
                negative_features, negative_labels = collect_object_score_features(
                    model,
                    loaders["calibration_loader"],  # type: ignore[arg-type]
                    device=torch_device,
                    disable_category_embedding=resolved_disable_category_embedding,
                    score_mode=resolved_score_mode,
                    anomaly_timestep=anomaly_timestep,
                    descriptor_weight=descriptor_weight,
                    global_descriptor_weight=global_descriptor_weight,
                    multi_timestep_scoring_enabled=resolved_multi_timestep,
                    multi_timestep_fractions=resolved_multi_timestep_fractions,
                    anomaly_map_gaussian_sigma=resolved_anomaly_map_sigma,
                    lambda_residual=score_weights.lambda_residual,
                    lambda_noise=score_weights.lambda_noise,
                    lambda_descriptor_spatial=score_weights.lambda_descriptor_spatial,
                    lambda_descriptor_global=score_weights.lambda_descriptor_global,
                    global_descriptor_calibration=global_descriptor_calibration,
                    spatial_exceedance_calibration=spatial_exceedance_calibration,
                    object_mask_threshold=object_mask_threshold,
                    seed=seed + 3300,
                )
                positive_features, positive_labels = collect_object_score_features(
                    model,
                    loaders["synthetic_validation_loader"],  # type: ignore[arg-type]
                    device=torch_device,
                    disable_category_embedding=resolved_disable_category_embedding,
                    score_mode=resolved_score_mode,
                    anomaly_timestep=anomaly_timestep,
                    descriptor_weight=descriptor_weight,
                    global_descriptor_weight=global_descriptor_weight,
                    multi_timestep_scoring_enabled=resolved_multi_timestep,
                    multi_timestep_fractions=resolved_multi_timestep_fractions,
                    anomaly_map_gaussian_sigma=resolved_anomaly_map_sigma,
                    lambda_residual=score_weights.lambda_residual,
                    lambda_noise=score_weights.lambda_noise,
                    lambda_descriptor_spatial=score_weights.lambda_descriptor_spatial,
                    lambda_descriptor_global=score_weights.lambda_descriptor_global,
                    global_descriptor_calibration=global_descriptor_calibration,
                    spatial_exceedance_calibration=spatial_exceedance_calibration,
                    object_mask_threshold=object_mask_threshold,
                    seed=seed + 3400,
                )
                object_score_calibration = fit_logistic_object_score_calibration(
                    torch.cat([negative_features, positive_features], dim=0),
                    torch.cat([negative_labels, positive_labels], dim=0),
                )
                object_score_summary_payload.update(object_score_calibration.to_dict())
            object_score_calibration_path = write_json(
                evidence_root / "object_score_calibration.json",
                object_score_summary_payload,
            )
        calibration_scores_by_category = collect_reference_scores_by_category(
            model,
            loaders["calibration_loader"] if resolved_object_score_strategy != "legacy_raw" else loaders["train_reference_loader"],  # type: ignore[arg-type]
            device=torch_device,
            disable_category_embedding=resolved_disable_category_embedding,
            score_mode=resolved_score_mode,
            anomaly_timestep=anomaly_timestep,
            descriptor_weight=descriptor_weight,
            global_descriptor_weight=global_descriptor_weight,
            multi_timestep_scoring_enabled=resolved_multi_timestep,
            multi_timestep_fractions=resolved_multi_timestep_fractions,
            anomaly_map_gaussian_sigma=resolved_anomaly_map_sigma,
            spatial_topk_percent=resolved_spatial_topk_percent,
            enable_internal_defect_gate=resolved_enable_internal_defect_gate,
            lambda_residual=score_weights.lambda_residual,
            lambda_noise=score_weights.lambda_noise,
            lambda_descriptor_spatial=score_weights.lambda_descriptor_spatial,
            lambda_descriptor_global=score_weights.lambda_descriptor_global,
            global_descriptor_calibration=global_descriptor_calibration,
            global_branch_gate_calibration=global_branch_gate_calibration,
            spatial_exceedance_calibration=spatial_exceedance_calibration,
            object_score_calibration=object_score_calibration,
            object_score_strategy=resolved_object_score_strategy,
            object_mask_threshold=object_mask_threshold,
            seed=seed + 2000,
        )
        calibration = {
            category_name: fit_masi_calibration(
                scores,
                category=category_name,
                score_mode=resolved_score_mode,
            )
            for category_name, scores in sorted(calibration_scores_by_category.items())
        }
        calibration_path = write_json(
            evidence_root / "calibration.json",
            {
                category_name: stats.to_dict()
                for category_name, stats in calibration.items()
            },
        )
    else:
        global_descriptor_calibration = fit_global_descriptor_reference(
            loaders["train_reference_loader"],  # type: ignore[arg-type]
            category=selected_categories[0] if selected_categories else category,
        )
        global_descriptor_calibration_path = write_json(
            evidence_root / "global_descriptor_calibration.json",
            global_descriptor_calibration.to_dict(),
        )
        if resolved_enable_internal_defect_gate:
            global_branch_gate_calibration = fit_global_branch_gate_calibration_by_category(
                model,
                loaders["calibration_loader"],  # type: ignore[arg-type]
                device=torch_device,
                disable_category_embedding=resolved_disable_category_embedding,
                score_mode=resolved_score_mode,
                anomaly_timestep=anomaly_timestep,
                descriptor_weight=descriptor_weight,
                global_descriptor_weight=global_descriptor_weight,
                multi_timestep_scoring_enabled=resolved_multi_timestep,
                multi_timestep_fractions=resolved_multi_timestep_fractions,
                anomaly_map_gaussian_sigma=resolved_anomaly_map_sigma,
                lambda_residual=score_weights.lambda_residual,
                lambda_noise=score_weights.lambda_noise,
                lambda_descriptor_spatial=score_weights.lambda_descriptor_spatial,
                lambda_descriptor_global=score_weights.lambda_descriptor_global,
                global_descriptor_calibration=global_descriptor_calibration,
                object_mask_threshold=object_mask_threshold,
                object_mask_dilation_kernel_size=resolved_object_mask_dilation_kernel_size,
                seed=seed + 2300,
            )
            global_branch_gate_calibration_path = write_json(
                evidence_root / "global_branch_gate_calibration.json",
                {
                    category_name: calibration.to_dict()
                    for category_name, calibration in sorted(global_branch_gate_calibration.items())
                },
            )
        if resolved_localization_tune_on_synthetic_validation:
            localization_calibration_map, localization_tuning_report = fit_tuned_localization_reference_by_category(
                model,
                loaders["calibration_loader"],  # type: ignore[arg-type]
                loaders["synthetic_validation_loader"],  # type: ignore[arg-type]
                device=torch_device,
                target_size=target_size,
                disable_category_embedding=resolved_disable_category_embedding,
                score_mode=resolved_score_mode,
                anomaly_timestep=anomaly_timestep,
                descriptor_weight=descriptor_weight,
                global_descriptor_weight=global_descriptor_weight,
                multi_timestep_scoring_enabled=resolved_multi_timestep,
                multi_timestep_fractions=resolved_multi_timestep_fractions,
                anomaly_map_gaussian_sigma=resolved_anomaly_map_sigma,
                lambda_residual=score_weights.lambda_residual,
                lambda_noise=score_weights.lambda_noise,
                lambda_descriptor_spatial=score_weights.lambda_descriptor_spatial,
                lambda_descriptor_global=score_weights.lambda_descriptor_global,
                global_descriptor_calibration=global_descriptor_calibration,
                global_branch_gate_calibration=global_branch_gate_calibration,
                object_mask_threshold=object_mask_threshold,
                localization_basis=resolved_localization_basis,
                localization_basis_candidates=resolved_localization_basis_candidates,
                object_mask_dilation_kernel_size=resolved_object_mask_dilation_kernel_size,
                localization_mask_closing_kernel_size=resolved_localization_mask_closing_kernel_size,
                localization_fill_holes=resolved_localization_fill_holes,
                localization_quantile=resolved_localization_quantile,
                localization_quantile_candidates=resolved_localization_quantile_candidates,
                localization_min_region_area_fraction=resolved_localization_min_region_area_fraction,
                localization_min_region_pixels_floor=resolved_localization_min_region_pixels_floor,
                localization_min_region_pixels_floor_candidates=resolved_localization_min_region_pixels_floor_candidates,
                seed=seed + 2500,
            )
        else:
            localization_calibration_map = fit_localization_reference_by_category(
                model,
                loaders["train_reference_loader"],  # type: ignore[arg-type]
                device=torch_device,
                disable_category_embedding=resolved_disable_category_embedding,
                score_mode=resolved_score_mode,
                anomaly_timestep=anomaly_timestep,
                descriptor_weight=descriptor_weight,
                global_descriptor_weight=global_descriptor_weight,
                multi_timestep_scoring_enabled=resolved_multi_timestep,
                multi_timestep_fractions=resolved_multi_timestep_fractions,
                anomaly_map_gaussian_sigma=resolved_anomaly_map_sigma,
                spatial_topk_percent=resolved_spatial_topk_percent,
                enable_internal_defect_gate=resolved_enable_internal_defect_gate,
                lambda_residual=score_weights.lambda_residual,
                lambda_noise=score_weights.lambda_noise,
                lambda_descriptor_spatial=score_weights.lambda_descriptor_spatial,
                lambda_descriptor_global=score_weights.lambda_descriptor_global,
                global_descriptor_calibration=global_descriptor_calibration,
                global_branch_gate_calibration=global_branch_gate_calibration,
                object_mask_threshold=object_mask_threshold,
                localization_basis=resolved_localization_basis,
                object_mask_dilation_kernel_size=resolved_object_mask_dilation_kernel_size,
                localization_mask_closing_kernel_size=resolved_localization_mask_closing_kernel_size,
                localization_fill_holes=resolved_localization_fill_holes,
                localization_quantile=resolved_localization_quantile,
                localization_min_region_area_fraction=resolved_localization_min_region_area_fraction,
                localization_min_region_pixels_floor=resolved_localization_min_region_pixels_floor,
                seed=seed + 2500,
            )
        selected_localization_category = selected_categories[0] if selected_categories else category
        localization_calibration = localization_calibration_map[selected_localization_category]
        localization_calibration_path = write_json(
            evidence_root / "localization_calibration.json",
            localization_calibration.to_dict(),
        )
        if resolved_object_score_strategy == "legacy_raw":
            calibration_scores = collect_reference_scores(
                model,
                loaders["train_reference_loader"],  # type: ignore[arg-type]
                device=torch_device,
                disable_category_embedding=resolved_disable_category_embedding,
                score_mode=resolved_score_mode,
                anomaly_timestep=anomaly_timestep,
                descriptor_weight=descriptor_weight,
                global_descriptor_weight=global_descriptor_weight,
                multi_timestep_scoring_enabled=resolved_multi_timestep,
                multi_timestep_fractions=resolved_multi_timestep_fractions,
                anomaly_map_gaussian_sigma=resolved_anomaly_map_sigma,
                spatial_topk_percent=resolved_spatial_topk_percent,
                enable_internal_defect_gate=resolved_enable_internal_defect_gate,
                lambda_residual=score_weights.lambda_residual,
                lambda_noise=score_weights.lambda_noise,
                lambda_descriptor_spatial=score_weights.lambda_descriptor_spatial,
                lambda_descriptor_global=score_weights.lambda_descriptor_global,
                global_descriptor_calibration=global_descriptor_calibration,
                global_branch_gate_calibration=global_branch_gate_calibration,
                spatial_exceedance_calibration=None,
                object_score_calibration=None,
                object_score_strategy="legacy_raw",
                object_mask_threshold=object_mask_threshold,
                seed=seed + 2000,
            )
            calibration = fit_masi_calibration(
                calibration_scores,
                category=selected_categories[0] if selected_categories else category,
                score_mode=resolved_score_mode,
            )
            calibration_path = write_json(evidence_root / "calibration.json", calibration.to_dict())
    if localization_tuning_report:
        localization_tuning_report_path = str(
            write_json(evidence_root / "localization_tuning_report.json", localization_tuning_report)
        )
    if resolved_object_score_strategy != "legacy_raw":
        spatial_exceedance_calibration = fit_spatial_exceedance_calibration_by_category(
            model,
            loaders["calibration_loader"],  # type: ignore[arg-type]
            device=torch_device,
            disable_category_embedding=resolved_disable_category_embedding,
            score_mode=resolved_score_mode,
            anomaly_timestep=anomaly_timestep,
            descriptor_weight=descriptor_weight,
            global_descriptor_weight=global_descriptor_weight,
            multi_timestep_scoring_enabled=resolved_multi_timestep,
            multi_timestep_fractions=resolved_multi_timestep_fractions,
            anomaly_map_gaussian_sigma=resolved_anomaly_map_sigma,
            lambda_residual=score_weights.lambda_residual,
            lambda_noise=score_weights.lambda_noise,
            lambda_descriptor_spatial=score_weights.lambda_descriptor_spatial,
            lambda_descriptor_global=score_weights.lambda_descriptor_global,
            global_descriptor_calibration=global_descriptor_calibration,
            object_mask_threshold=object_mask_threshold,
            localization_quantile=resolved_localization_quantile,
            localization_min_region_area_fraction=resolved_localization_min_region_area_fraction,
            localization_min_region_pixels_floor=resolved_localization_min_region_pixels_floor,
            seed=seed + 3200,
        )
        resolved_single_spatial_calibration = spatial_exceedance_calibration.get(selected_localization_category)
        spatial_exceedance_calibration_path = write_json(
            evidence_root / "spatial_exceedance_calibration.json",
            resolved_single_spatial_calibration.to_dict() if resolved_single_spatial_calibration is not None else {},
        )
        object_score_summary_payload = {
            "strategy": resolved_object_score_strategy,
            "calibration_split": "calibration_good",
            "region_mean_weight": 0.7,
            "region_peak_weight": 0.3,
            "local_weight": 0.65,
            "global_weight": 0.35,
            "tau_quantile": 0.999,
            "tail_quantile": 0.99,
        }
        if resolved_object_score_strategy == "exceedance_logreg":
            negative_features, negative_labels = collect_object_score_features(
                model,
                loaders["calibration_loader"],  # type: ignore[arg-type]
                device=torch_device,
                disable_category_embedding=resolved_disable_category_embedding,
                score_mode=resolved_score_mode,
                anomaly_timestep=anomaly_timestep,
                descriptor_weight=descriptor_weight,
                global_descriptor_weight=global_descriptor_weight,
                multi_timestep_scoring_enabled=resolved_multi_timestep,
                multi_timestep_fractions=resolved_multi_timestep_fractions,
                anomaly_map_gaussian_sigma=resolved_anomaly_map_sigma,
                lambda_residual=score_weights.lambda_residual,
                lambda_noise=score_weights.lambda_noise,
                lambda_descriptor_spatial=score_weights.lambda_descriptor_spatial,
                lambda_descriptor_global=score_weights.lambda_descriptor_global,
                global_descriptor_calibration=global_descriptor_calibration,
                spatial_exceedance_calibration=spatial_exceedance_calibration,
                object_mask_threshold=object_mask_threshold,
                seed=seed + 3300,
            )
            positive_features, positive_labels = collect_object_score_features(
                model,
                loaders["synthetic_validation_loader"],  # type: ignore[arg-type]
                device=torch_device,
                disable_category_embedding=resolved_disable_category_embedding,
                score_mode=resolved_score_mode,
                anomaly_timestep=anomaly_timestep,
                descriptor_weight=descriptor_weight,
                global_descriptor_weight=global_descriptor_weight,
                multi_timestep_scoring_enabled=resolved_multi_timestep,
                multi_timestep_fractions=resolved_multi_timestep_fractions,
                anomaly_map_gaussian_sigma=resolved_anomaly_map_sigma,
                lambda_residual=score_weights.lambda_residual,
                lambda_noise=score_weights.lambda_noise,
                lambda_descriptor_spatial=score_weights.lambda_descriptor_spatial,
                lambda_descriptor_global=score_weights.lambda_descriptor_global,
                global_descriptor_calibration=global_descriptor_calibration,
                spatial_exceedance_calibration=spatial_exceedance_calibration,
                object_mask_threshold=object_mask_threshold,
                seed=seed + 3400,
            )
            object_score_calibration = fit_logistic_object_score_calibration(
                torch.cat([negative_features, positive_features], dim=0),
                torch.cat([negative_labels, positive_labels], dim=0),
            )
            object_score_summary_payload.update(object_score_calibration.to_dict())
        object_score_calibration_path = write_json(
            evidence_root / "object_score_calibration.json",
            object_score_summary_payload,
        )
        calibration_scores = collect_reference_scores(
            model,
            loaders["calibration_loader"] if resolved_object_score_strategy != "legacy_raw" else loaders["train_reference_loader"],  # type: ignore[arg-type]
            device=torch_device,
            disable_category_embedding=resolved_disable_category_embedding,
            score_mode=resolved_score_mode,
            anomaly_timestep=anomaly_timestep,
            descriptor_weight=descriptor_weight,
            global_descriptor_weight=global_descriptor_weight,
            multi_timestep_scoring_enabled=resolved_multi_timestep,
            multi_timestep_fractions=resolved_multi_timestep_fractions,
            anomaly_map_gaussian_sigma=resolved_anomaly_map_sigma,
            spatial_topk_percent=resolved_spatial_topk_percent,
            enable_internal_defect_gate=resolved_enable_internal_defect_gate,
            lambda_residual=score_weights.lambda_residual,
            lambda_noise=score_weights.lambda_noise,
            lambda_descriptor_spatial=score_weights.lambda_descriptor_spatial,
            lambda_descriptor_global=score_weights.lambda_descriptor_global,
            global_descriptor_calibration=global_descriptor_calibration,
            global_branch_gate_calibration=global_branch_gate_calibration,
            spatial_exceedance_calibration=spatial_exceedance_calibration,
            object_score_calibration=object_score_calibration,
            object_score_strategy=resolved_object_score_strategy,
            object_mask_threshold=object_mask_threshold,
            seed=seed + 2000,
        )
        calibration = fit_masi_calibration(
            calibration_scores,
            category=selected_categories[0] if selected_categories else category,
            score_mode=resolved_score_mode,
        )
        calibration_path = write_json(evidence_root / "calibration.json", calibration.to_dict())
    eval_summary = run_eval_scoring(
        model,
        loaders["eval_loader"],  # type: ignore[arg-type]
        device=torch_device,
        disable_category_embedding=resolved_disable_category_embedding,
        target_size=target_size,
        score_mode=resolved_score_mode,
        anomaly_timestep=anomaly_timestep,
        descriptor_weight=descriptor_weight,
        global_descriptor_weight=global_descriptor_weight,
        multi_timestep_scoring_enabled=resolved_multi_timestep,
        multi_timestep_fractions=resolved_multi_timestep_fractions,
        anomaly_map_gaussian_sigma=resolved_anomaly_map_sigma,
        spatial_topk_percent=resolved_spatial_topk_percent,
        enable_internal_defect_gate=resolved_enable_internal_defect_gate,
        lambda_residual=score_weights.lambda_residual,
        lambda_noise=score_weights.lambda_noise,
        lambda_descriptor_spatial=score_weights.lambda_descriptor_spatial,
        lambda_descriptor_global=score_weights.lambda_descriptor_global,
        global_descriptor_calibration=global_descriptor_calibration,
        global_branch_gate_calibration=global_branch_gate_calibration,
        spatial_exceedance_calibration=spatial_exceedance_calibration,
        object_score_calibration=object_score_calibration,
        object_score_strategy=resolved_object_score_strategy,
        pixel_threshold_percentile=pixel_threshold_percentile,
        object_mask_threshold=object_mask_threshold,
        localization_basis=resolved_localization_basis,
        object_mask_dilation_kernel_size=resolved_object_mask_dilation_kernel_size,
        localization_mask_closing_kernel_size=resolved_localization_mask_closing_kernel_size,
        localization_fill_holes=resolved_localization_fill_holes,
        max_batches=max_eval_batches,
        visualization_dir=run_paths.predictions / "eval",
        preview_selection_path=run_paths.metrics / "preview_selection.json",
        image_score_data_path=run_paths.metrics / "image_score_data.json",
        evidence_dir=evidence_root,
        calibration=calibration,
        localization_calibration=localization_calibration,
        max_visualizations=max_visualizations,
        knowledge_base_root=Path(knowledge_base_root) if knowledge_base_root else None,
        retrieval_top_k=retrieval_top_k,
        enable_llm_explanations=enable_llm_explanations,
        rgb_normalization_stats=rgb_stats,
        rgb_normalization_stats_by_category=rgb_stats_by_category,
        preview_seed=seed,
        seed=seed + 3000,
    )
    plot_paths = save_evaluation_plot_bundle(
        evaluation_metrics=eval_summary,
        per_category=eval_summary.get("per_category", {}),
        image_score_data_path=eval_summary["image_score_data_path"],
        plot_dir=run_paths.plots,
    )
    eval_summary["plot_paths"] = plot_paths
    summary = {
        "regime_internal": regime.internal,
        "regime_paper": regime.paper,
        "regime_expansion": regime.expansion,
        "checkpoint_path": str(checkpoint_path),
        "loaded_epoch": int(payload.get("epoch", 0)),
        "category": "shared" if joint_mode else reported_selected_categories[0],
        "selected_categories": reported_selected_categories,
        "scope": scope,
        "training_mode": "shared_joint" if joint_mode else "single_category",
        "disable_category_embedding": resolved_disable_category_embedding,
        "ablation_mode": "shared_nocat" if resolved_disable_category_embedding and joint_mode else "shared" if joint_mode else "per_category",
        "device": str(torch_device),
        "resolved_device": str(torch_device),
        "device_mode": runtime.requested_mode,
        "amp_enabled": runtime.use_amp,
        "runtime_fallback_reason": runtime.fallback_reason,
        "score_mode": resolved_score_mode,
        "spatial_topk_percent": resolved_spatial_topk_percent,
        "object_score_strategy": resolved_object_score_strategy,
        "enable_category_modality_gating": resolved_enable_category_modality_gating,
        "enable_internal_defect_gate": resolved_enable_internal_defect_gate,
        "eval_records": len(eval_dataset),
        "training_protocol": "train_good_only",
        "evaluation_protocol": "test_good_plus_anomaly",
        "processed_root": str(resolved_processed_root),
        "image_auroc": eval_summary["image_auroc"],
        "macro_image_auroc": eval_summary.get("macro_image_auroc", eval_summary["image_auroc"]),
        "image_auprc": eval_summary["image_auprc"],
        "macro_image_auprc": eval_summary.get("macro_image_auprc", eval_summary["image_auprc"]),
        "image_f1": eval_summary["image_f1"],
        "pixel_auroc": eval_summary["pixel_auroc"],
        "macro_pixel_auroc": eval_summary.get("macro_pixel_auroc", eval_summary["pixel_auroc"]),
        "aupro": eval_summary["aupro"],
        "macro_aupro": eval_summary.get("macro_aupro", eval_summary["aupro"]),
        "pixel_f1": eval_summary["pixel_f1"],
        "pixel_iou": eval_summary["pixel_iou"],
        "mean_good_score": eval_summary["mean_good_score"],
        "mean_anomalous_score": eval_summary["mean_anomalous_score"],
        "localization_basis": resolved_localization_basis,
        "object_mask_dilation_kernel_size": resolved_object_mask_dilation_kernel_size,
        "localization_mask_closing_kernel_size": resolved_localization_mask_closing_kernel_size,
        "localization_fill_holes": resolved_localization_fill_holes,
        "localization_tune_on_synthetic_validation": resolved_localization_tune_on_synthetic_validation,
        "localization_quantile": resolved_localization_quantile,
        "localization_min_region_area_fraction": resolved_localization_min_region_area_fraction,
        "localization_min_region_pixels_floor": resolved_localization_min_region_pixels_floor,
        **score_weights.to_dict(),
        "visualizations_saved": eval_summary["visualizations_saved"],
        "preview_selection_path": eval_summary.get("preview_selection_path", ""),
        "image_score_data_path": eval_summary.get("image_score_data_path", ""),
        "plot_paths": plot_paths,
        "evidence_packages_saved": eval_summary["evidence_packages_saved"],
        "llm_explanations_saved": eval_summary["llm_explanations_saved"],
        "calibration_path": str(calibration_path),
        "global_descriptor_calibration_path": str(global_descriptor_calibration_path),
        "global_branch_gate_calibration_path": str(global_branch_gate_calibration_path),
        "localization_calibration_path": str(localization_calibration_path),
        "localization_tuning_report_path": str(localization_tuning_report_path),
        "spatial_exceedance_calibration_path": str(spatial_exceedance_calibration_path),
        "object_score_calibration_path": str(object_score_calibration_path),
        "rgb_normalization_stats_path": rgb_stats_path,
        "rgb_normalization_stats_by_category_path": rgb_stats_by_category_path,
        "evidence_index_path": eval_summary["evidence_index_path"],
        "run_dir": str(run_paths.root),
        **provenance,
    }
    write_json(run_paths.metrics / "evaluation.json", eval_summary)
    write_json(run_paths.metrics / "per_category.json", eval_summary.get("per_category", {}))
    write_history_csv(
        run_paths.metrics / "per_category.csv",
        _per_category_metric_rows(eval_summary.get("per_category", {})),
    )
    write_json(run_paths.root / "summary.json", summary)
    return {"run_paths": run_paths, "summary": summary, "eval_summary": eval_summary}


def fit_global_descriptor_reference(loader: DataLoader, *, category: str, max_batches: int = 0) -> GlobalDescriptorCalibration:
    vectors = collect_reference_global_vectors(loader, max_batches=max_batches)
    return fit_global_descriptor_calibration(
        vectors,
        category=category,
        feature_names=GLOBAL_FEATURE_NAMES,
    )


def fit_global_descriptor_reference_by_category(
    loader: DataLoader,
    *,
    max_batches: int = 0,
) -> Dict[str, GlobalDescriptorCalibration]:
    grouped_vectors = collect_reference_global_vectors_by_category(loader, max_batches=max_batches)
    return {
        category: fit_global_descriptor_calibration(
            vectors,
            category=category,
            feature_names=GLOBAL_FEATURE_NAMES,
        )
        for category, vectors in sorted(grouped_vectors.items())
    }


def collect_reference_global_vectors(loader: DataLoader, *, max_batches: int = 0) -> torch.Tensor:
    collected: List[torch.Tensor] = []
    for batch_index, raw_batch in enumerate(loader):
        batch = prepare_diffusion_batch(raw_batch)
        collected.append(batch.g.detach().cpu())
        if max_batches and batch_index + 1 >= max_batches:
            break
    return torch.cat(collected) if collected else torch.zeros((1, len(GLOBAL_FEATURE_NAMES)), dtype=torch.float32)


def collect_reference_global_vectors_by_category(
    loader: DataLoader,
    *,
    max_batches: int = 0,
) -> Dict[str, torch.Tensor]:
    collected: Dict[str, List[torch.Tensor]] = {}
    for batch_index, raw_batch in enumerate(loader):
        batch = prepare_diffusion_batch(raw_batch)
        for sample_index, category in enumerate(batch.categories):
            collected.setdefault(str(category), []).append(batch.g[sample_index : sample_index + 1].detach().cpu())
        if max_batches and batch_index + 1 >= max_batches:
            break
    return {
        category: torch.cat(vectors, dim=0)
        for category, vectors in collected.items()
    }


def _finite_quantile(values: torch.Tensor, quantile: float, *, default: float = 0.0) -> float:
    flattened = values.detach().flatten().float()
    finite = flattened[torch.isfinite(flattened)]
    if finite.numel() == 0:
        return float(default)
    return float(torch.quantile(finite, float(quantile)).item())


def _cue_is_degenerate(values: torch.Tensor) -> bool:
    flattened = values.detach().flatten().float()
    finite = flattened[torch.isfinite(flattened)]
    if finite.numel() == 0:
        return True
    nonzero_count = int((finite.abs() > 1e-6).sum().item())
    minimum_nonzero = max(3, int(round(float(finite.numel()) * 0.05)))
    if nonzero_count < minimum_nonzero:
        return True
    spread = float((finite.max() - finite.min()).item())
    return spread <= 1e-6


def collect_global_branch_gate_measurements_by_category(
    model: MulSenDiffX,
    loader: DataLoader,
    *,
    device: torch.device,
    disable_category_embedding: bool = False,
    score_mode: str,
    anomaly_timestep: int,
    descriptor_weight: float,
    global_descriptor_weight: float,
    multi_timestep_scoring_enabled: bool = False,
    multi_timestep_fractions: Sequence[float] | None = None,
    anomaly_map_gaussian_sigma: float = 0.0,
    lambda_residual: float | None = None,
    lambda_noise: float | None = None,
    lambda_descriptor_spatial: float | None = None,
    lambda_descriptor_global: float | None = None,
    global_descriptor_calibration: GlobalDescriptorCalibration | Dict[str, GlobalDescriptorCalibration] | None = None,
    object_mask_threshold: float = 0.02,
    object_mask_dilation_kernel_size: int = 9,
    max_batches: int = 0,
    seed: int = 61,
) -> Dict[str, Dict[str, torch.Tensor]]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    collected: Dict[str, Dict[str, List[torch.Tensor]]] = {}

    for batch_index, raw_batch in enumerate(loader):
        batch = prepare_diffusion_batch(raw_batch)
        scored = score_batch(
            model,
            batch.x_rgb.to(device),
            batch.M.to(device),
            batch.S.to(device),
            batch.g.to(device),
            _resolve_category_conditioning_indices(
                batch.category_indices.to(device),
                disable_category_embedding=disable_category_embedding,
            ),
            raw_global_vector=batch.g_raw.to(device),
            categories=batch.categories,
            score_mode=score_mode,
            timestep=anomaly_timestep,
            descriptor_weight=descriptor_weight,
            global_descriptor_weight=global_descriptor_weight,
            multi_timestep_scoring_enabled=multi_timestep_scoring_enabled,
            multi_timestep_fractions=multi_timestep_fractions,
            anomaly_map_gaussian_sigma=anomaly_map_gaussian_sigma,
            lambda_residual=lambda_residual,
            lambda_noise=lambda_noise,
            lambda_descriptor_spatial=lambda_descriptor_spatial,
            lambda_descriptor_global=lambda_descriptor_global,
            global_descriptor_calibration=global_descriptor_calibration,
            global_branch_gate_calibration=None,
            object_score_strategy="legacy_raw",
            enable_internal_defect_gate=False,
            object_mask_threshold=object_mask_threshold,
            object_mask_dilation_kernel_size=object_mask_dilation_kernel_size,
            generator=generator,
        )
        spatial_scores = scored.spatial_score_pre_gate.detach().cpu()
        for sample_index, category_name in enumerate(batch.categories):
            if not is_archetype_b(category_name):
                continue
            bucket = collected.setdefault(
                str(category_name),
                {
                    "ir_hotspot_area_fraction": [],
                    "ir_hotspot_compactness": [],
                    "ir_mean_hotspot_intensity": [],
                    "cross_inconsistency_score": [],
                    "cross_inconsistency_q99": [],
                    "cross_inconsistency_top1": [],
                    "cross_geo_support_top1": [],
                    "ir_hotspot_top1": [],
                    "ir_gradient_top1": [],
                    "rescue_score": [],
                    "spatial_score": [],
                },
            )
            bucket["ir_hotspot_area_fraction"].append(
                scored.raw_ir_hotspot_area_fraction[sample_index : sample_index + 1].detach().cpu()
            )
            bucket["ir_hotspot_compactness"].append(
                scored.raw_ir_hotspot_compactness[sample_index : sample_index + 1].detach().cpu()
            )
            bucket["ir_mean_hotspot_intensity"].append(
                scored.raw_ir_mean_hotspot_intensity[sample_index : sample_index + 1].detach().cpu()
            )
            bucket["cross_inconsistency_score"].append(
                scored.raw_cross_inconsistency_score[sample_index : sample_index + 1].detach().cpu()
            )
            bucket["cross_inconsistency_q99"].append(
                scored.cross_inconsistency_q99[sample_index : sample_index + 1].detach().cpu()
            )
            bucket["cross_inconsistency_top1"].append(
                scored.cross_inconsistency_top1[sample_index : sample_index + 1].detach().cpu()
            )
            bucket["cross_geo_support_top1"].append(
                scored.cross_geo_support_top1[sample_index : sample_index + 1].detach().cpu()
            )
            bucket["ir_hotspot_top1"].append(
                scored.ir_hotspot_top1[sample_index : sample_index + 1].detach().cpu()
            )
            bucket["ir_gradient_top1"].append(
                scored.ir_gradient_top1[sample_index : sample_index + 1].detach().cpu()
            )
            bucket["rescue_score"].append(
                scored.rescue_score[sample_index : sample_index + 1].detach().cpu()
            )
            bucket["spatial_score"].append(spatial_scores[sample_index : sample_index + 1])
        if max_batches and batch_index + 1 >= max_batches:
            break

    return {
        category: {
            metric_name: torch.cat(values, dim=0) if values else torch.zeros(0, dtype=torch.float32)
            for metric_name, values in metrics.items()
        }
        for category, metrics in collected.items()
    }


def fit_global_branch_gate_calibration_by_category(
    model: MulSenDiffX,
    loader: DataLoader,
    *,
    device: torch.device,
    disable_category_embedding: bool = False,
    score_mode: str,
    anomaly_timestep: int,
    descriptor_weight: float,
    global_descriptor_weight: float,
    multi_timestep_scoring_enabled: bool = False,
    multi_timestep_fractions: Sequence[float] | None = None,
    anomaly_map_gaussian_sigma: float = 0.0,
    lambda_residual: float | None = None,
    lambda_noise: float | None = None,
    lambda_descriptor_spatial: float | None = None,
    lambda_descriptor_global: float | None = None,
    global_descriptor_calibration: GlobalDescriptorCalibration | Dict[str, GlobalDescriptorCalibration] | None = None,
    object_mask_threshold: float = 0.02,
    object_mask_dilation_kernel_size: int = 9,
    max_batches: int = 0,
    seed: int = 61,
) -> Dict[str, GlobalBranchGateCalibration]:
    measurements = collect_global_branch_gate_measurements_by_category(
        model,
        loader,
        device=device,
        disable_category_embedding=disable_category_embedding,
        score_mode=score_mode,
        anomaly_timestep=anomaly_timestep,
        descriptor_weight=descriptor_weight,
        global_descriptor_weight=global_descriptor_weight,
        multi_timestep_scoring_enabled=multi_timestep_scoring_enabled,
        multi_timestep_fractions=multi_timestep_fractions,
        anomaly_map_gaussian_sigma=anomaly_map_gaussian_sigma,
        lambda_residual=lambda_residual,
        lambda_noise=lambda_noise,
        lambda_descriptor_spatial=lambda_descriptor_spatial,
        lambda_descriptor_global=lambda_descriptor_global,
        global_descriptor_calibration=global_descriptor_calibration,
        object_mask_threshold=object_mask_threshold,
        object_mask_dilation_kernel_size=object_mask_dilation_kernel_size,
        max_batches=max_batches,
        seed=seed,
    )
    calibrations: Dict[str, GlobalBranchGateCalibration] = {}
    for category_name, category_measurements in sorted(measurements.items()):
        if not is_archetype_b(category_name):
            continue
        sample_count = int(category_measurements["spatial_score"].numel())
        if sample_count <= 0:
            continue
        mode = resolve_archetype_b_gate_mode(category_name)
        required_cues = {
            "cross_inconsistency": ("cross_inconsistency_score",),
            "thermal_hotspot": (
                "ir_hotspot_area_fraction",
                "ir_hotspot_compactness",
                "ir_mean_hotspot_intensity",
            ),
        }.get(mode, ())
        tracked_cues = (
            "ir_hotspot_area_fraction",
            "ir_hotspot_compactness",
            "ir_mean_hotspot_intensity",
            "cross_inconsistency_score",
            "cross_inconsistency_q99",
            "cross_inconsistency_top1",
            "cross_geo_support_top1",
            "ir_hotspot_top1",
            "ir_gradient_top1",
        )
        cue_nonzero_counts = {
            cue_name: int((values.detach().flatten().float().abs() > 1e-6).sum().item())
            for cue_name, values in category_measurements.items()
            if cue_name in tracked_cues
        }
        degenerate_cues = tuple(
            sorted(
                cue_name
                for cue_name, values in category_measurements.items()
                if cue_name in tracked_cues and _cue_is_degenerate(values)
            )
        )
        calibrations[category_name] = GlobalBranchGateCalibration(
            category=category_name,
            mode=mode,
            active=bool(mode in {"cross_inconsistency", "thermal_hotspot"} and not any(cue in degenerate_cues for cue in required_cues)),
            required_cues=tuple(required_cues),
            degenerate_cues=degenerate_cues,
            cue_nonzero_counts=cue_nonzero_counts,
            ir_hotspot_area_fraction_threshold=_finite_quantile(
                category_measurements["ir_hotspot_area_fraction"],
                0.95,
            ),
            ir_hotspot_compactness_threshold=_finite_quantile(
                category_measurements["ir_hotspot_compactness"],
                0.95,
            ),
            ir_mean_hotspot_intensity_threshold=_finite_quantile(
                category_measurements["ir_mean_hotspot_intensity"],
                0.95,
            ),
            cross_inconsistency_score_threshold=_finite_quantile(
                category_measurements["cross_inconsistency_score"],
                0.95,
            ),
            rescue_score_q95=_finite_quantile(
                category_measurements["rescue_score"],
                0.95,
            ),
            rescue_score_q99=_finite_quantile(
                category_measurements["rescue_score"],
                0.99,
            ),
            spatial_score_q95=_finite_quantile(
                category_measurements["spatial_score"],
                0.95,
            ),
            sample_count=sample_count,
        )
    return calibrations


def collect_reference_localization_measurements_by_category(
    model: MulSenDiffX,
    loader: DataLoader,
    *,
    device: torch.device,
    disable_category_embedding: bool = False,
    score_mode: str,
    anomaly_timestep: int,
    descriptor_weight: float,
    global_descriptor_weight: float,
    multi_timestep_scoring_enabled: bool = False,
    multi_timestep_fractions: Sequence[float] | None = None,
    anomaly_map_gaussian_sigma: float = 0.0,
    lambda_residual: float | None = None,
    lambda_noise: float | None = None,
    lambda_descriptor_spatial: float | None = None,
    lambda_descriptor_global: float | None = None,
    global_descriptor_calibration: GlobalDescriptorCalibration | Dict[str, GlobalDescriptorCalibration] | None = None,
    global_branch_gate_calibration: GlobalBranchGateCalibration | Dict[str, GlobalBranchGateCalibration] | None = None,
    spatial_topk_percent: float = 0.02,
    enable_internal_defect_gate: bool = False,
    object_mask_threshold: float = 0.02,
    localization_basis: str = "anomaly_map",
    object_mask_dilation_kernel_size: int = 9,
    max_batches: int = 0,
    seed: int = 29,
) -> Dict[str, Dict[str, torch.Tensor]]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    collected_values: Dict[str, List[torch.Tensor]] = {}
    collected_object_pixels: Dict[str, List[torch.Tensor]] = {}

    for batch_index, raw_batch in enumerate(loader):
        batch = prepare_diffusion_batch(raw_batch)
        scored = score_batch(
            model,
            batch.x_rgb.to(device),
            batch.M.to(device),
            batch.S.to(device),
            batch.g.to(device),
            _resolve_category_conditioning_indices(
                batch.category_indices.to(device),
                disable_category_embedding=disable_category_embedding,
            ),
            raw_global_vector=batch.g_raw.to(device),
            categories=batch.categories,
            score_mode=score_mode,
            timestep=anomaly_timestep,
            descriptor_weight=descriptor_weight,
            global_descriptor_weight=global_descriptor_weight,
            multi_timestep_scoring_enabled=multi_timestep_scoring_enabled,
            multi_timestep_fractions=multi_timestep_fractions,
            anomaly_map_gaussian_sigma=anomaly_map_gaussian_sigma,
            lambda_residual=lambda_residual,
            lambda_noise=lambda_noise,
            lambda_descriptor_spatial=lambda_descriptor_spatial,
            lambda_descriptor_global=lambda_descriptor_global,
            global_descriptor_calibration=global_descriptor_calibration,
            global_branch_gate_calibration=global_branch_gate_calibration,
            spatial_topk_percent=spatial_topk_percent,
            enable_internal_defect_gate=enable_internal_defect_gate,
            object_mask_threshold=object_mask_threshold,
            object_mask_dilation_kernel_size=object_mask_dilation_kernel_size,
            generator=generator,
        )
        anomaly_maps = select_localization_basis_map(scored, localization_basis).detach().cpu()
        object_masks = scored.object_mask.detach().cpu()
        for sample_index, category in enumerate(batch.categories):
            sample_map = anomaly_maps[sample_index : sample_index + 1]
            sample_object_mask = object_masks[sample_index : sample_index + 1]
            mask_bool = sample_object_mask > 0
            values = sample_map[mask_bool]
            if values.numel() == 0:
                values = sample_map.reshape(-1)
            collected_values.setdefault(str(category), []).append(values.detach().cpu().flatten())
            collected_object_pixels.setdefault(str(category), []).append(
                torch.tensor([int(mask_bool.sum().item())], dtype=torch.float32)
            )
        if max_batches and batch_index + 1 >= max_batches:
            break

    return {
        category: {
            "values": torch.cat(values, dim=0) if values else torch.zeros(1, dtype=torch.float32),
            "object_pixels": (
                torch.cat(collected_object_pixels.get(category, []), dim=0)
                if collected_object_pixels.get(category)
                else torch.zeros(1, dtype=torch.float32)
            ),
            "sample_count": torch.tensor([len(values)], dtype=torch.int64),
        }
        for category, values in collected_values.items()
    }


def _build_localization_calibration_map_from_measurements(
    measurements: Dict[str, Dict[str, torch.Tensor]],
    *,
    localization_basis: str,
    object_mask_threshold: float,
    object_mask_dilation_kernel_size: int,
    localization_mask_closing_kernel_size: int,
    localization_fill_holes: bool,
    localization_quantile: float,
    localization_min_region_area_fraction: float,
    localization_min_region_pixels_floor: int,
    tuning_source: str,
) -> Dict[str, LocalizationCalibration]:
    return {
        category: fit_localization_calibration(
            stats["values"],
            category=category,
            object_pixel_counts=stats["object_pixels"],
            quantile=localization_quantile,
            min_region_area_fraction=resolve_localization_min_region_area_fraction(
                category,
                localization_min_region_area_fraction,
            ),
            min_region_pixels_floor=localization_min_region_pixels_floor,
            sample_count=int(stats["sample_count"].item()),
            basis=localization_basis,
            object_mask_threshold=object_mask_threshold,
            object_mask_dilation_kernel_size=object_mask_dilation_kernel_size,
            mask_closing_kernel_size=localization_mask_closing_kernel_size,
            fill_holes=localization_fill_holes,
            tuning_source=tuning_source,
        )
        for category, stats in sorted(measurements.items())
    }


def fit_localization_reference_by_category(
    model: MulSenDiffX,
    loader: DataLoader,
    *,
    device: torch.device,
    disable_category_embedding: bool = False,
    score_mode: str,
    anomaly_timestep: int,
    descriptor_weight: float,
    global_descriptor_weight: float,
    multi_timestep_scoring_enabled: bool = False,
    multi_timestep_fractions: Sequence[float] | None = None,
    anomaly_map_gaussian_sigma: float = 0.0,
    lambda_residual: float | None = None,
    lambda_noise: float | None = None,
    lambda_descriptor_spatial: float | None = None,
    lambda_descriptor_global: float | None = None,
    global_descriptor_calibration: GlobalDescriptorCalibration | Dict[str, GlobalDescriptorCalibration] | None = None,
    global_branch_gate_calibration: GlobalBranchGateCalibration | Dict[str, GlobalBranchGateCalibration] | None = None,
    spatial_topk_percent: float = 0.02,
    enable_internal_defect_gate: bool = False,
    object_mask_threshold: float = 0.02,
    localization_basis: str = "anomaly_map",
    object_mask_dilation_kernel_size: int = 9,
    localization_mask_closing_kernel_size: int = 0,
    localization_fill_holes: bool = False,
    localization_quantile: float = 0.995,
    localization_min_region_area_fraction: float = 0.002,
    localization_min_region_pixels_floor: int = 4,
    max_batches: int = 0,
    seed: int = 31,
) -> Dict[str, LocalizationCalibration]:
    measurements = collect_reference_localization_measurements_by_category(
        model,
        loader,
        device=device,
        disable_category_embedding=disable_category_embedding,
        score_mode=score_mode,
        anomaly_timestep=anomaly_timestep,
        descriptor_weight=descriptor_weight,
        global_descriptor_weight=global_descriptor_weight,
        multi_timestep_scoring_enabled=multi_timestep_scoring_enabled,
        multi_timestep_fractions=multi_timestep_fractions,
        anomaly_map_gaussian_sigma=anomaly_map_gaussian_sigma,
        lambda_residual=lambda_residual,
        lambda_noise=lambda_noise,
        lambda_descriptor_spatial=lambda_descriptor_spatial,
        lambda_descriptor_global=lambda_descriptor_global,
        global_descriptor_calibration=global_descriptor_calibration,
        global_branch_gate_calibration=global_branch_gate_calibration,
        spatial_topk_percent=spatial_topk_percent,
        enable_internal_defect_gate=enable_internal_defect_gate,
        object_mask_threshold=object_mask_threshold,
        localization_basis=localization_basis,
        object_mask_dilation_kernel_size=object_mask_dilation_kernel_size,
        max_batches=max_batches,
        seed=seed,
    )
    return _build_localization_calibration_map_from_measurements(
        measurements,
        localization_basis=localization_basis,
        object_mask_threshold=object_mask_threshold,
        object_mask_dilation_kernel_size=object_mask_dilation_kernel_size,
        localization_mask_closing_kernel_size=localization_mask_closing_kernel_size,
        localization_fill_holes=localization_fill_holes,
        localization_quantile=localization_quantile,
        localization_min_region_area_fraction=localization_min_region_area_fraction,
        localization_min_region_pixels_floor=localization_min_region_pixels_floor,
        tuning_source="calibration_good",
    )


def _evaluate_localization_calibration_by_category(
    model: MulSenDiffX,
    loader: DataLoader,
    *,
    device: torch.device,
    target_size: tuple[int, int],
    localization_calibration_by_category: Dict[str, LocalizationCalibration],
    disable_category_embedding: bool = False,
    score_mode: str,
    anomaly_timestep: int,
    descriptor_weight: float,
    global_descriptor_weight: float,
    multi_timestep_scoring_enabled: bool = False,
    multi_timestep_fractions: Sequence[float] | None = None,
    anomaly_map_gaussian_sigma: float = 0.0,
    lambda_residual: float | None = None,
    lambda_noise: float | None = None,
    lambda_descriptor_spatial: float | None = None,
    lambda_descriptor_global: float | None = None,
    global_descriptor_calibration: GlobalDescriptorCalibration | Dict[str, GlobalDescriptorCalibration] | None = None,
    global_branch_gate_calibration: GlobalBranchGateCalibration | Dict[str, GlobalBranchGateCalibration] | None = None,
    spatial_topk_percent: float = 0.02,
    enable_internal_defect_gate: bool = False,
    object_mask_threshold: float = 0.02,
    object_mask_dilation_kernel_size: int = 9,
    max_batches: int = 0,
    seed: int = 47,
) -> Dict[str, Dict[str, float | int]]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    per_category_pixel_inputs: Dict[str, Dict[str, List[torch.Tensor]]] = {}
    per_category_pixel_f1: Dict[str, List[float]] = {}
    per_category_pixel_iou: Dict[str, List[float]] = {}

    for batch_index, raw_batch in enumerate(loader):
        batch = prepare_diffusion_batch(raw_batch)
        scored = score_batch(
            model,
            batch.x_rgb.to(device),
            batch.M.to(device),
            batch.S.to(device),
            batch.g.to(device),
            _resolve_category_conditioning_indices(
                batch.category_indices.to(device),
                disable_category_embedding=disable_category_embedding,
            ),
            raw_global_vector=batch.g_raw.to(device),
            categories=batch.categories,
            score_mode=score_mode,
            timestep=anomaly_timestep,
            descriptor_weight=descriptor_weight,
            global_descriptor_weight=global_descriptor_weight,
            multi_timestep_scoring_enabled=multi_timestep_scoring_enabled,
            multi_timestep_fractions=multi_timestep_fractions,
            anomaly_map_gaussian_sigma=anomaly_map_gaussian_sigma,
            spatial_topk_percent=spatial_topk_percent,
            enable_internal_defect_gate=enable_internal_defect_gate,
            lambda_residual=lambda_residual,
            lambda_noise=lambda_noise,
            lambda_descriptor_spatial=lambda_descriptor_spatial,
            lambda_descriptor_global=lambda_descriptor_global,
            global_descriptor_calibration=global_descriptor_calibration,
            global_branch_gate_calibration=global_branch_gate_calibration,
            object_score_strategy="legacy_raw",
            object_mask_threshold=object_mask_threshold,
            object_mask_dilation_kernel_size=object_mask_dilation_kernel_size,
            generator=generator,
        )
        object_masks = scored.object_mask.detach().cpu()
        for sample_index, category_name in enumerate(batch.categories):
            calibration = localization_calibration_by_category.get(str(category_name))
            if calibration is None:
                continue
            sample_object_mask = object_masks[sample_index : sample_index + 1]
            sample_raw_map = select_localization_basis_map(
                scored,
                calibration.basis,
            )[sample_index : sample_index + 1].detach().cpu()
            sample_normalized_map = normalize_anomaly_map(sample_raw_map)
            predicted_mask = apply_localization_calibration(
                sample_raw_map,
                object_mask=sample_object_mask,
                calibration=calibration,
                normalized=False,
            )
            gt_mask = load_gt_mask(
                batch.gt_mask_paths[sample_index],
                target_size=target_size,
                crop_box_xyxy=batch.crop_boxes_xyxy[sample_index],
                crop_source_size_hw=batch.crop_source_sizes_hw[sample_index],
            )
            if gt_mask is None:
                continue
            per_category_pixel_f1.setdefault(str(category_name), []).append(
                binary_f1_score(predicted_mask[0], gt_mask)
            )
            per_category_pixel_iou.setdefault(str(category_name), []).append(
                intersection_over_union(predicted_mask[0], gt_mask)
            )
            category_bucket = per_category_pixel_inputs.setdefault(
                str(category_name),
                {"score_maps": [], "target_masks": [], "object_masks": []},
            )
            category_bucket["score_maps"].append(sample_normalized_map[0].detach().cpu())
            category_bucket["target_masks"].append(gt_mask.detach().cpu())
            category_bucket["object_masks"].append(sample_object_mask[0].detach().cpu())
        if max_batches and batch_index + 1 >= max_batches:
            break

    metrics: Dict[str, Dict[str, float | int]] = {}
    for category_name, bucket in sorted(per_category_pixel_inputs.items()):
        metrics[category_name] = {
            "pixel_auroc": float(
                pixel_level_auroc(
                    bucket["score_maps"],
                    bucket["target_masks"],
                    bucket["object_masks"],
                )
            ),
            "aupro": float(
                aupro(
                    bucket["score_maps"],
                    bucket["target_masks"],
                    bucket["object_masks"],
                )
            ),
            "pixel_f1": float(
                sum(per_category_pixel_f1.get(category_name, []))
                / max(len(per_category_pixel_f1.get(category_name, [])), 1)
            ),
            "pixel_iou": float(
                sum(per_category_pixel_iou.get(category_name, []))
                / max(len(per_category_pixel_iou.get(category_name, [])), 1)
            ),
            "pixel_samples": int(len(per_category_pixel_f1.get(category_name, []))),
        }
    return metrics


def _localization_selection_key(metrics: Mapping[str, float | int]) -> tuple[float, float, float, float]:
    pixel_f1 = float(metrics.get("pixel_f1", 0.0))
    aupro_value = float(metrics.get("aupro", 0.0))
    pixel_iou = float(metrics.get("pixel_iou", 0.0))
    pixel_auroc_value = float(metrics.get("pixel_auroc", 0.0))
    return (1.0 if pixel_f1 > 0.0 else 0.0, aupro_value, pixel_iou, pixel_auroc_value)


def fit_tuned_localization_reference_by_category(
    model: MulSenDiffX,
    reference_loader: DataLoader,
    tuning_loader: DataLoader,
    *,
    device: torch.device,
    target_size: tuple[int, int],
    disable_category_embedding: bool = False,
    score_mode: str,
    anomaly_timestep: int,
    descriptor_weight: float,
    global_descriptor_weight: float,
    multi_timestep_scoring_enabled: bool = False,
    multi_timestep_fractions: Sequence[float] | None = None,
    anomaly_map_gaussian_sigma: float = 0.0,
    lambda_residual: float | None = None,
    lambda_noise: float | None = None,
    lambda_descriptor_spatial: float | None = None,
    lambda_descriptor_global: float | None = None,
    global_descriptor_calibration: GlobalDescriptorCalibration | Dict[str, GlobalDescriptorCalibration] | None = None,
    global_branch_gate_calibration: GlobalBranchGateCalibration | Dict[str, GlobalBranchGateCalibration] | None = None,
    spatial_topk_percent: float = 0.02,
    enable_internal_defect_gate: bool = False,
    object_mask_threshold: float = 0.02,
    localization_basis: str = "anomaly_map",
    localization_basis_candidates: Sequence[str] | None = None,
    object_mask_dilation_kernel_size: int = 9,
    localization_mask_closing_kernel_size: int = 0,
    localization_fill_holes: bool = False,
    localization_quantile: float = 0.995,
    localization_quantile_candidates: Sequence[float] | None = None,
    localization_min_region_area_fraction: float = 0.002,
    localization_min_region_pixels_floor: int = 4,
    localization_min_region_pixels_floor_candidates: Sequence[int] | None = None,
    max_batches: int = 0,
    seed: int = 53,
) -> tuple[Dict[str, LocalizationCalibration], Dict[str, Any]]:
    basis_candidates = list(
        dict.fromkeys(
            str(item).strip().lower()
            for item in (localization_basis_candidates or [localization_basis])
            if str(item).strip()
        )
    ) or [str(localization_basis).strip().lower()]
    quantile_candidates = list(
        dict.fromkeys(float(item) for item in (localization_quantile_candidates or [localization_quantile]))
    ) or [float(localization_quantile)]
    min_region_floor_candidates = list(
        dict.fromkeys(
            int(item) for item in (localization_min_region_pixels_floor_candidates or [localization_min_region_pixels_floor])
        )
    ) or [int(localization_min_region_pixels_floor)]

    base_measurements = collect_reference_localization_measurements_by_category(
        model,
        reference_loader,
        device=device,
        disable_category_embedding=disable_category_embedding,
        score_mode=score_mode,
        anomaly_timestep=anomaly_timestep,
        descriptor_weight=descriptor_weight,
        global_descriptor_weight=global_descriptor_weight,
        multi_timestep_scoring_enabled=multi_timestep_scoring_enabled,
        multi_timestep_fractions=multi_timestep_fractions,
        anomaly_map_gaussian_sigma=anomaly_map_gaussian_sigma,
        lambda_residual=lambda_residual,
        lambda_noise=lambda_noise,
        lambda_descriptor_spatial=lambda_descriptor_spatial,
        lambda_descriptor_global=lambda_descriptor_global,
        global_descriptor_calibration=global_descriptor_calibration,
        global_branch_gate_calibration=global_branch_gate_calibration,
        spatial_topk_percent=spatial_topk_percent,
        enable_internal_defect_gate=enable_internal_defect_gate,
        object_mask_threshold=object_mask_threshold,
        localization_basis=localization_basis,
        object_mask_dilation_kernel_size=object_mask_dilation_kernel_size,
        max_batches=max_batches,
        seed=seed,
    )
    fallback = _build_localization_calibration_map_from_measurements(
        base_measurements,
        localization_basis=localization_basis,
        object_mask_threshold=object_mask_threshold,
        object_mask_dilation_kernel_size=object_mask_dilation_kernel_size,
        localization_mask_closing_kernel_size=localization_mask_closing_kernel_size,
        localization_fill_holes=localization_fill_holes,
        localization_quantile=localization_quantile,
        localization_min_region_area_fraction=localization_min_region_area_fraction,
        localization_min_region_pixels_floor=localization_min_region_pixels_floor,
        tuning_source="calibration_good",
    )

    measurements_by_basis: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = {}
    best_calibration_by_category: Dict[str, LocalizationCalibration] = {}
    best_report_by_category: Dict[str, Dict[str, Any]] = {}

    for basis_candidate in basis_candidates:
        measurements_by_basis[basis_candidate] = collect_reference_localization_measurements_by_category(
            model,
            reference_loader,
            device=device,
            disable_category_embedding=disable_category_embedding,
            score_mode=score_mode,
            anomaly_timestep=anomaly_timestep,
            descriptor_weight=descriptor_weight,
            global_descriptor_weight=global_descriptor_weight,
            multi_timestep_scoring_enabled=multi_timestep_scoring_enabled,
            multi_timestep_fractions=multi_timestep_fractions,
            anomaly_map_gaussian_sigma=anomaly_map_gaussian_sigma,
            lambda_residual=lambda_residual,
            lambda_noise=lambda_noise,
            lambda_descriptor_spatial=lambda_descriptor_spatial,
            lambda_descriptor_global=lambda_descriptor_global,
            global_descriptor_calibration=global_descriptor_calibration,
            global_branch_gate_calibration=global_branch_gate_calibration,
            spatial_topk_percent=spatial_topk_percent,
            enable_internal_defect_gate=enable_internal_defect_gate,
            object_mask_threshold=object_mask_threshold,
            localization_basis=basis_candidate,
            object_mask_dilation_kernel_size=object_mask_dilation_kernel_size,
            max_batches=max_batches,
            seed=seed + 100,
        )
        for quantile_candidate in quantile_candidates:
            for min_region_floor_candidate in min_region_floor_candidates:
                calibration_map = _build_localization_calibration_map_from_measurements(
                    measurements_by_basis[basis_candidate],
                    localization_basis=basis_candidate,
                    object_mask_threshold=object_mask_threshold,
                    object_mask_dilation_kernel_size=object_mask_dilation_kernel_size,
                    localization_mask_closing_kernel_size=localization_mask_closing_kernel_size,
                    localization_fill_holes=localization_fill_holes,
                    localization_quantile=float(quantile_candidate),
                    localization_min_region_area_fraction=localization_min_region_area_fraction,
                    localization_min_region_pixels_floor=int(min_region_floor_candidate),
                    tuning_source="calibration_good+synthetic_validation",
                )
                metrics_by_category = _evaluate_localization_calibration_by_category(
                    model,
                    tuning_loader,
                    device=device,
                    target_size=target_size,
                    localization_calibration_by_category=calibration_map,
                    disable_category_embedding=disable_category_embedding,
                    score_mode=score_mode,
                    anomaly_timestep=anomaly_timestep,
                    descriptor_weight=descriptor_weight,
                    global_descriptor_weight=global_descriptor_weight,
                    multi_timestep_scoring_enabled=multi_timestep_scoring_enabled,
                    multi_timestep_fractions=multi_timestep_fractions,
                    anomaly_map_gaussian_sigma=anomaly_map_gaussian_sigma,
                    lambda_residual=lambda_residual,
                    lambda_noise=lambda_noise,
                    lambda_descriptor_spatial=lambda_descriptor_spatial,
                    lambda_descriptor_global=lambda_descriptor_global,
                    global_descriptor_calibration=global_descriptor_calibration,
                    global_branch_gate_calibration=global_branch_gate_calibration,
                    spatial_topk_percent=spatial_topk_percent,
                    enable_internal_defect_gate=enable_internal_defect_gate,
                    object_mask_threshold=object_mask_threshold,
                    object_mask_dilation_kernel_size=object_mask_dilation_kernel_size,
                    max_batches=max_batches,
                    seed=seed + 200,
                )
                for category_name, metrics in metrics_by_category.items():
                    resolved_min_region_area_fraction = resolve_localization_min_region_area_fraction(
                        category_name,
                        localization_min_region_area_fraction,
                    )
                    candidate_report = {
                        "basis": basis_candidate,
                        "quantile": float(quantile_candidate),
                        "min_region_pixels_floor": int(min_region_floor_candidate),
                        "resolved_min_region_area_fraction": float(resolved_min_region_area_fraction),
                        "resolved_min_region_pixels": int(calibration_map[category_name].min_region_pixels),
                        "mask_closing_kernel_size": int(calibration_map[category_name].mask_closing_kernel_size),
                        "fill_holes": bool(calibration_map[category_name].fill_holes),
                        "pixel_auroc": round(float(metrics.get("pixel_auroc", 0.0)), 6),
                        "aupro": round(float(metrics.get("aupro", 0.0)), 6),
                        "pixel_f1": round(float(metrics.get("pixel_f1", 0.0)), 6),
                        "pixel_iou": round(float(metrics.get("pixel_iou", 0.0)), 6),
                        "pixel_samples": int(metrics.get("pixel_samples", 0)),
                    }
                    if (
                        category_name not in best_report_by_category
                        or _localization_selection_key(candidate_report)
                        > _localization_selection_key(best_report_by_category[category_name])
                    ):
                        best_report_by_category[category_name] = candidate_report
                        best_calibration_by_category[category_name] = calibration_map[category_name]

    for category_name, calibration in fallback.items():
        if category_name not in best_calibration_by_category:
            best_calibration_by_category[category_name] = calibration
            best_report_by_category[category_name] = {
                "basis": calibration.basis,
                "quantile": calibration.quantile,
                "min_region_pixels_floor": calibration.min_region_pixels,
                "resolved_min_region_area_fraction": calibration.min_region_area_fraction,
                "resolved_min_region_pixels": calibration.min_region_pixels,
                "mask_closing_kernel_size": int(calibration.mask_closing_kernel_size),
                "fill_holes": bool(calibration.fill_holes),
                "pixel_auroc": 0.0,
                "aupro": 0.0,
                "pixel_f1": 0.0,
                "pixel_iou": 0.0,
                "pixel_samples": 0,
                "fallback": True,
            }

    return best_calibration_by_category, {
        "strategy": "synthetic_validation_grid_search",
        "basis_candidates": basis_candidates,
        "quantile_candidates": [float(item) for item in quantile_candidates],
        "min_region_pixels_floor_candidates": [int(item) for item in min_region_floor_candidates],
        "categories": best_report_by_category,
    }


def collect_reference_scores(
    model: MulSenDiffX,
    loader: DataLoader,
    *,
    device: torch.device,
    disable_category_embedding: bool = False,
    score_mode: str,
    anomaly_timestep: int,
    descriptor_weight: float,
    global_descriptor_weight: float,
    multi_timestep_scoring_enabled: bool = False,
    multi_timestep_fractions: Sequence[float] | None = None,
    anomaly_map_gaussian_sigma: float = 0.0,
    lambda_residual: float | None = None,
    lambda_noise: float | None = None,
    lambda_descriptor_spatial: float | None = None,
    lambda_descriptor_global: float | None = None,
    global_descriptor_calibration: GlobalDescriptorCalibration | Dict[str, GlobalDescriptorCalibration] | None,
    global_branch_gate_calibration: GlobalBranchGateCalibration | Dict[str, GlobalBranchGateCalibration] | None = None,
    spatial_exceedance_calibration: SpatialExceedanceCalibration | Dict[str, SpatialExceedanceCalibration] | None = None,
    object_score_calibration: LogisticObjectScoreCalibration | None = None,
    object_score_strategy: str = "legacy_raw",
    spatial_topk_percent: float = 0.02,
    enable_internal_defect_gate: bool = False,
    object_mask_threshold: float,
    max_batches: int = 0,
    seed: int = 29,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    collected: List[torch.Tensor] = []

    for batch_index, raw_batch in enumerate(loader):
        batch = prepare_diffusion_batch(raw_batch)
        scored = score_batch(
            model,
            batch.x_rgb.to(device),
            batch.M.to(device),
            batch.S.to(device),
            batch.g.to(device),
            _resolve_category_conditioning_indices(
                batch.category_indices.to(device),
                disable_category_embedding=disable_category_embedding,
            ),
            raw_global_vector=batch.g_raw.to(device),
            categories=batch.categories,
            score_mode=score_mode,
            timestep=anomaly_timestep,
            descriptor_weight=descriptor_weight,
            global_descriptor_weight=global_descriptor_weight,
            multi_timestep_scoring_enabled=multi_timestep_scoring_enabled,
            multi_timestep_fractions=multi_timestep_fractions,
            anomaly_map_gaussian_sigma=anomaly_map_gaussian_sigma,
            lambda_residual=lambda_residual,
            lambda_noise=lambda_noise,
            lambda_descriptor_spatial=lambda_descriptor_spatial,
            lambda_descriptor_global=lambda_descriptor_global,
            global_descriptor_calibration=global_descriptor_calibration,
            global_branch_gate_calibration=global_branch_gate_calibration,
            spatial_exceedance_calibration=spatial_exceedance_calibration,
            object_score_calibration=object_score_calibration,
            object_score_strategy=object_score_strategy,
            spatial_topk_percent=spatial_topk_percent,
            enable_internal_defect_gate=enable_internal_defect_gate,
            object_mask_threshold=object_mask_threshold,
            generator=generator,
        )
        collected.append(scored.anomaly_score.detach().cpu())
        if max_batches and batch_index + 1 >= max_batches:
            break

    return torch.cat(collected) if collected else torch.empty(0)


def collect_reference_scores_by_category(
    model: MulSenDiffX,
    loader: DataLoader,
    *,
    device: torch.device,
    disable_category_embedding: bool = False,
    score_mode: str,
    anomaly_timestep: int,
    descriptor_weight: float,
    global_descriptor_weight: float,
    multi_timestep_scoring_enabled: bool = False,
    multi_timestep_fractions: Sequence[float] | None = None,
    anomaly_map_gaussian_sigma: float = 0.0,
    lambda_residual: float | None = None,
    lambda_noise: float | None = None,
    lambda_descriptor_spatial: float | None = None,
    lambda_descriptor_global: float | None = None,
    global_descriptor_calibration: GlobalDescriptorCalibration | Dict[str, GlobalDescriptorCalibration] | None = None,
    global_branch_gate_calibration: GlobalBranchGateCalibration | Dict[str, GlobalBranchGateCalibration] | None = None,
    spatial_exceedance_calibration: SpatialExceedanceCalibration | Dict[str, SpatialExceedanceCalibration] | None = None,
    object_score_calibration: LogisticObjectScoreCalibration | None = None,
    object_score_strategy: str = "legacy_raw",
    spatial_topk_percent: float = 0.02,
    enable_internal_defect_gate: bool = False,
    object_mask_threshold: float = 0.02,
    max_batches: int = 0,
    seed: int = 29,
) -> Dict[str, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    collected: Dict[str, List[torch.Tensor]] = {}

    for batch_index, raw_batch in enumerate(loader):
        batch = prepare_diffusion_batch(raw_batch)
        scored = score_batch(
            model,
            batch.x_rgb.to(device),
            batch.M.to(device),
            batch.S.to(device),
            batch.g.to(device),
            _resolve_category_conditioning_indices(
                batch.category_indices.to(device),
                disable_category_embedding=disable_category_embedding,
            ),
            raw_global_vector=batch.g_raw.to(device),
            categories=batch.categories,
            score_mode=score_mode,
            timestep=anomaly_timestep,
            descriptor_weight=descriptor_weight,
            global_descriptor_weight=global_descriptor_weight,
            multi_timestep_scoring_enabled=multi_timestep_scoring_enabled,
            multi_timestep_fractions=multi_timestep_fractions,
            anomaly_map_gaussian_sigma=anomaly_map_gaussian_sigma,
            lambda_residual=lambda_residual,
            lambda_noise=lambda_noise,
            lambda_descriptor_spatial=lambda_descriptor_spatial,
            lambda_descriptor_global=lambda_descriptor_global,
            global_descriptor_calibration=global_descriptor_calibration,
            global_branch_gate_calibration=global_branch_gate_calibration,
            spatial_exceedance_calibration=spatial_exceedance_calibration,
            object_score_calibration=object_score_calibration,
            object_score_strategy=object_score_strategy,
            spatial_topk_percent=spatial_topk_percent,
            enable_internal_defect_gate=enable_internal_defect_gate,
            object_mask_threshold=object_mask_threshold,
            generator=generator,
        )
        for sample_index, category in enumerate(batch.categories):
            collected.setdefault(str(category), []).append(
                scored.anomaly_score[sample_index : sample_index + 1].detach().cpu()
            )
        if max_batches and batch_index + 1 >= max_batches:
            break

    return {
        category: torch.cat(values, dim=0)
        for category, values in collected.items()
    }


def _robust_tail_scale(values: torch.Tensor) -> float:
    flattened = values.detach().flatten().float()
    if flattened.numel() == 0:
        return 1e-6
    median = torch.median(flattened)
    mad = torch.median((flattened - median).abs())
    mad_scale = float((1.4826 * mad).item())
    if mad_scale > 1e-6:
        return mad_scale
    std_scale = float(flattened.std(unbiased=False).item())
    if std_scale > 1e-6:
        return std_scale
    return 1e-6


def fit_spatial_exceedance_calibration_by_category(
    model: MulSenDiffX,
    loader: DataLoader,
    *,
    device: torch.device,
    disable_category_embedding: bool = False,
    score_mode: str,
    anomaly_timestep: int,
    descriptor_weight: float,
    global_descriptor_weight: float,
    multi_timestep_scoring_enabled: bool = False,
    multi_timestep_fractions: Sequence[float] | None = None,
    anomaly_map_gaussian_sigma: float = 0.0,
    lambda_residual: float | None = None,
    lambda_noise: float | None = None,
    lambda_descriptor_spatial: float | None = None,
    lambda_descriptor_global: float | None = None,
    global_descriptor_calibration: GlobalDescriptorCalibration | Dict[str, GlobalDescriptorCalibration] | None = None,
    object_mask_threshold: float = 0.02,
    localization_quantile: float = 0.995,
    localization_min_region_area_fraction: float = 0.002,
    localization_min_region_pixels_floor: int = 4,
    max_batches: int = 0,
    seed: int = 41,
) -> Dict[str, SpatialExceedanceCalibration]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    pixel_values_by_category: Dict[str, List[torch.Tensor]] = {}
    object_pixels_by_category: Dict[str, List[torch.Tensor]] = {}

    for batch_index, raw_batch in enumerate(loader):
        batch = prepare_diffusion_batch(raw_batch)
        scored = score_batch(
            model,
            batch.x_rgb.to(device),
            batch.M.to(device),
            batch.S.to(device),
            batch.g.to(device),
            _resolve_category_conditioning_indices(
                batch.category_indices.to(device),
                disable_category_embedding=disable_category_embedding,
            ),
            raw_global_vector=batch.g_raw.to(device),
            categories=batch.categories,
            score_mode=score_mode,
            timestep=anomaly_timestep,
            descriptor_weight=descriptor_weight,
            global_descriptor_weight=global_descriptor_weight,
            multi_timestep_scoring_enabled=multi_timestep_scoring_enabled,
            multi_timestep_fractions=multi_timestep_fractions,
            anomaly_map_gaussian_sigma=anomaly_map_gaussian_sigma,
            lambda_residual=lambda_residual,
            lambda_noise=lambda_noise,
            lambda_descriptor_spatial=lambda_descriptor_spatial,
            lambda_descriptor_global=lambda_descriptor_global,
            global_descriptor_calibration=global_descriptor_calibration,
            object_score_strategy="legacy_raw",
            object_mask_threshold=object_mask_threshold,
            generator=generator,
        )
        anomaly_maps = scored.anomaly_map.detach().cpu()
        object_masks = scored.object_mask.detach().cpu()
        for sample_index, category_name in enumerate(batch.categories):
            sample_map = anomaly_maps[sample_index : sample_index + 1]
            sample_mask = object_masks[sample_index : sample_index + 1]
            mask_bool = sample_mask > 0
            values = sample_map[mask_bool]
            if values.numel() == 0:
                values = sample_map.reshape(-1)
            pixel_values_by_category.setdefault(str(category_name), []).append(values.flatten())
            object_pixels_by_category.setdefault(str(category_name), []).append(
                torch.tensor([int(mask_bool.sum().item())], dtype=torch.float32)
            )
        if max_batches and batch_index + 1 >= max_batches:
            break

    initial: Dict[str, SpatialExceedanceCalibration] = {}
    for category_name, values_list in sorted(pixel_values_by_category.items()):
        values = torch.cat(values_list, dim=0).float() if values_list else torch.zeros(1, dtype=torch.float32)
        object_counts = (
            torch.cat(object_pixels_by_category.get(category_name, []), dim=0).float()
            if object_pixels_by_category.get(category_name)
            else torch.zeros(1, dtype=torch.float32)
        )
        tau_pix = float(torch.quantile(values, 0.999).item()) if values.numel() else 0.0
        tail_q99 = float(torch.quantile(values, 0.99).item()) if values.numel() else 0.0
        tail_values = values[values >= tail_q99]
        sigma_tail = _robust_tail_scale(tail_values)
        resolved_min_region_area_fraction = resolve_localization_min_region_area_fraction(
            category_name,
            localization_min_region_area_fraction,
        )
        localization_proxy = fit_localization_calibration(
            values,
            category=category_name,
            object_pixel_counts=object_counts,
            quantile=localization_quantile,
            min_region_area_fraction=resolved_min_region_area_fraction,
            min_region_pixels_floor=localization_min_region_pixels_floor,
            sample_count=int(object_counts.numel()),
        )
        initial[category_name] = SpatialExceedanceCalibration(
            category=category_name,
            tau_pix=tau_pix,
            tail_q99=tail_q99,
            sigma_tail=sigma_tail,
            min_region_pixels=int(localization_proxy.min_region_pixels),
            z_local_mu=0.0,
            z_local_std=1.0,
            z_global_mu=0.0,
            z_global_std=1.0,
            sample_count=int(object_counts.numel()),
            pixel_count=int(values.numel()),
        )

    local_logs_by_category: Dict[str, List[torch.Tensor]] = {}
    global_logs_by_category: Dict[str, List[torch.Tensor]] = {}
    generator.manual_seed(seed + 1)
    for batch_index, raw_batch in enumerate(loader):
        batch = prepare_diffusion_batch(raw_batch)
        scored = score_batch(
            model,
            batch.x_rgb.to(device),
            batch.M.to(device),
            batch.S.to(device),
            batch.g.to(device),
            _resolve_category_conditioning_indices(
                batch.category_indices.to(device),
                disable_category_embedding=disable_category_embedding,
            ),
            raw_global_vector=batch.g_raw.to(device),
            categories=batch.categories,
            score_mode=score_mode,
            timestep=anomaly_timestep,
            descriptor_weight=descriptor_weight,
            global_descriptor_weight=global_descriptor_weight,
            multi_timestep_scoring_enabled=multi_timestep_scoring_enabled,
            multi_timestep_fractions=multi_timestep_fractions,
            anomaly_map_gaussian_sigma=anomaly_map_gaussian_sigma,
            lambda_residual=lambda_residual,
            lambda_noise=lambda_noise,
            lambda_descriptor_spatial=lambda_descriptor_spatial,
            lambda_descriptor_global=lambda_descriptor_global,
            global_descriptor_calibration=global_descriptor_calibration,
            spatial_exceedance_calibration=initial,
            object_score_strategy="exceedance_regions",
            object_mask_threshold=object_mask_threshold,
            generator=generator,
        )
        for sample_index, category_name in enumerate(batch.categories):
            local_logs_by_category.setdefault(str(category_name), []).append(
                torch.log1p(scored.local_exceedance_score[sample_index : sample_index + 1].detach().cpu().clamp_min(0.0))
            )
            global_logs_by_category.setdefault(str(category_name), []).append(
                torch.log1p(scored.global_descriptor_score[sample_index : sample_index + 1].detach().cpu().clamp_min(0.0))
            )
        if max_batches and batch_index + 1 >= max_batches:
            break

    final: Dict[str, SpatialExceedanceCalibration] = {}
    for category_name, calibration in sorted(initial.items()):
        local_logs = torch.cat(local_logs_by_category.get(category_name, [torch.zeros(1)]), dim=0)
        global_logs = torch.cat(global_logs_by_category.get(category_name, [torch.zeros(1)]), dim=0)
        final[category_name] = SpatialExceedanceCalibration(
            category=category_name,
            tau_pix=calibration.tau_pix,
            tail_q99=calibration.tail_q99,
            sigma_tail=calibration.sigma_tail,
            min_region_pixels=calibration.min_region_pixels,
            z_local_mu=float(local_logs.mean().item()),
            z_local_std=max(float(local_logs.std(unbiased=False).item()), 1e-6),
            z_global_mu=float(global_logs.mean().item()),
            z_global_std=max(float(global_logs.std(unbiased=False).item()), 1e-6),
            sample_count=calibration.sample_count,
            pixel_count=calibration.pixel_count,
        )
    return final


def collect_object_score_features(
    model: MulSenDiffX,
    loader: DataLoader,
    *,
    device: torch.device,
    disable_category_embedding: bool = False,
    score_mode: str,
    anomaly_timestep: int,
    descriptor_weight: float,
    global_descriptor_weight: float,
    multi_timestep_scoring_enabled: bool = False,
    multi_timestep_fractions: Sequence[float] | None = None,
    anomaly_map_gaussian_sigma: float = 0.0,
    lambda_residual: float | None = None,
    lambda_noise: float | None = None,
    lambda_descriptor_spatial: float | None = None,
    lambda_descriptor_global: float | None = None,
    global_descriptor_calibration: GlobalDescriptorCalibration | Dict[str, GlobalDescriptorCalibration] | None = None,
    spatial_exceedance_calibration: SpatialExceedanceCalibration | Dict[str, SpatialExceedanceCalibration] | None = None,
    object_mask_threshold: float = 0.02,
    max_batches: int = 0,
    seed: int = 43,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    feature_rows: List[torch.Tensor] = []
    label_rows: List[torch.Tensor] = []

    for batch_index, raw_batch in enumerate(loader):
        batch = prepare_diffusion_batch(raw_batch)
        scored = score_batch(
            model,
            batch.x_rgb.to(device),
            batch.M.to(device),
            batch.S.to(device),
            batch.g.to(device),
            _resolve_category_conditioning_indices(
                batch.category_indices.to(device),
                disable_category_embedding=disable_category_embedding,
            ),
            raw_global_vector=batch.g_raw.to(device),
            categories=batch.categories,
            score_mode=score_mode,
            timestep=anomaly_timestep,
            descriptor_weight=descriptor_weight,
            global_descriptor_weight=global_descriptor_weight,
            multi_timestep_scoring_enabled=multi_timestep_scoring_enabled,
            multi_timestep_fractions=multi_timestep_fractions,
            anomaly_map_gaussian_sigma=anomaly_map_gaussian_sigma,
            lambda_residual=lambda_residual,
            lambda_noise=lambda_noise,
            lambda_descriptor_spatial=lambda_descriptor_spatial,
            lambda_descriptor_global=lambda_descriptor_global,
            global_descriptor_calibration=global_descriptor_calibration,
            spatial_exceedance_calibration=spatial_exceedance_calibration,
            object_score_strategy="exceedance_regions",
            object_mask_threshold=object_mask_threshold,
            generator=generator,
        )
        feature_rows.append(build_object_score_feature_matrix(scored).detach().cpu())
        label_rows.append(
            torch.tensor([1.0 if value else 0.0 for value in batch.is_anomalous], dtype=torch.float32)
        )
        if max_batches and batch_index + 1 >= max_batches:
            break

    features = torch.cat(feature_rows, dim=0) if feature_rows else torch.empty((0, len(OBJECT_SCORE_FEATURE_NAMES)))
    labels = torch.cat(label_rows, dim=0) if label_rows else torch.empty(0, dtype=torch.float32)
    return features, labels


def fit_logistic_object_score_calibration(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    l2_regularization: float = 1e-3,
    max_iter: int = 200,
) -> LogisticObjectScoreCalibration:
    inputs = features.detach().float()
    targets = labels.detach().float().flatten()
    if inputs.ndim != 2:
        raise ValueError("features must be a 2D tensor.")
    if targets.numel() != inputs.shape[0]:
        raise ValueError("labels must align with the number of feature rows.")
    positive_count = int((targets >= 0.5).sum().item())
    negative_count = int((targets < 0.5).sum().item())
    if positive_count == 0 or negative_count == 0:
        raise ValueError("Logistic object-score calibration requires both positive and negative samples.")

    feature_mean = inputs.mean(dim=0)
    feature_std = inputs.std(dim=0, unbiased=False).clamp_min(1e-6)
    standardized = (inputs - feature_mean.unsqueeze(0)) / feature_std.unsqueeze(0)
    weights = torch.zeros((standardized.shape[1],), dtype=torch.float32, requires_grad=True)
    intercept = torch.zeros((), dtype=torch.float32, requires_grad=True)
    positive_weight = torch.tensor(float(negative_count) / max(float(positive_count), 1.0), dtype=torch.float32)
    optimizer = torch.optim.LBFGS(
        [weights, intercept],
        lr=1.0,
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        logits = standardized @ weights + intercept
        loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=positive_weight)
        loss = loss + float(l2_regularization) * weights.pow(2).sum()
        loss.backward()
        return loss

    optimizer.step(closure)

    return LogisticObjectScoreCalibration(
        strategy="exceedance_logreg",
        feature_names=tuple(OBJECT_SCORE_FEATURE_NAMES),
        feature_mean=[float(value) for value in feature_mean.tolist()],
        feature_std=[float(value) for value in feature_std.tolist()],
        coefficients=[float(value) for value in weights.detach().tolist()],
        intercept=float(intercept.detach().item()),
        sample_count=int(inputs.shape[0]),
        positive_count=positive_count,
        negative_count=negative_count,
        l2_regularization=float(l2_regularization),
    )


def load_gt_mask(
    path_like: str,
    *,
    target_size: tuple[int, int],
    crop_box_xyxy: tuple[int, int, int, int] | None = None,
    crop_source_size_hw: tuple[int, int] | None = None,
) -> torch.Tensor | None:
    path = Path(path_like) if path_like else None
    if not path or not path.exists():
        return None
    image = Image.open(path).convert("L")
    if crop_box_xyxy is not None and crop_source_size_hw is not None:
        source_h, source_w = crop_source_size_hw
        image_w, image_h = image.size
        scale_x = image_w / max(source_w, 1)
        scale_y = image_h / max(source_h, 1)
        left, top, right, bottom = crop_box_xyxy
        image = image.crop(
            (
                int(round(left * scale_x)),
                int(round(top * scale_y)),
                int(round(right * scale_x)),
                int(round(bottom * scale_y)),
            )
        )
    image = TF.resize(image, list(target_size), interpolation=InterpolationMode.NEAREST)
    tensor = TF.pil_to_tensor(image).float() / 255.0
    return (tensor > 0.5).float()
