from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping, Sequence

import torch
from scipy import ndimage
from torch.nn import functional as F

from src.data_loader import GLOBAL_FEATURE_NAMES, SPATIAL_DESCRIPTOR_ORDER, SUPPORT_DESCRIPTOR_ORDER
from src.category_policies import is_archetype_b, resolve_archetype_b_gate_mode
from src.inference.global_descriptor_scoring import (
    GlobalDescriptorCalibration,
    score_global_descriptors,
)
from src.inference.localization import estimate_object_mask, gaussian_smooth_map
from src.models.mulsendiffx import MulSenDiffX


OBJECT_SCORE_FEATURE_NAMES = (
    "z_local",
    "z_global",
    "log1p_s_local",
    "log1p_s_global",
    "largest_component_area",
    "largest_component_mean_excess",
    "largest_component_peak_excess",
    "largest_component_mass_excess",
    "num_components",
    "largest_component_area_fraction",
)

LOCALIZATION_BASIS_CHOICES = (
    "anomaly_map",
    "score_basis_map",
    "residual_component",
    "noise_component",
    "descriptor_support",
    "noise_residual_blend",
    "noise_residual_geomean",
)


@dataclass(frozen=True)
class AnomalyScoreOutputs:
    anomaly_score: torch.Tensor
    anomaly_map: torch.Tensor
    score_basis_map: torch.Tensor
    score_basis_label: str
    residual_map: torch.Tensor
    noise_error_map: torch.Tensor
    descriptor_support: torch.Tensor
    descriptor_spatial_component: torch.Tensor
    residual_component: torch.Tensor
    noise_component: torch.Tensor
    reconstructed_rgb: torch.Tensor
    diffusion_reconstructed_rgb: torch.Tensor
    diffusion_reconstruction_label: str
    display_reconstructed_rgb: torch.Tensor
    display_reconstruction_label: str
    object_mask: torch.Tensor
    spatial_score: torch.Tensor
    spatial_score_pre_gate: torch.Tensor
    spatial_score_post_gate: torch.Tensor
    rescue_score: torch.Tensor
    ir_dominant_score: torch.Tensor
    internal_defect_gate_fired: torch.Tensor
    internal_defect_gate_score_changed: torch.Tensor
    internal_defect_gate_mode: tuple[str, ...]
    internal_defect_gate_reason: tuple[str, ...]
    raw_ir_hotspot_area_fraction: torch.Tensor
    raw_ir_hotspot_compactness: torch.Tensor
    raw_ir_mean_hotspot_intensity: torch.Tensor
    raw_cross_inconsistency_score: torch.Tensor
    cross_inconsistency_q99: torch.Tensor
    cross_inconsistency_top1: torch.Tensor
    cross_geo_support_top1: torch.Tensor
    ir_hotspot_top1: torch.Tensor
    ir_gradient_top1: torch.Tensor
    global_descriptor_score: torch.Tensor
    timesteps_used: tuple[int, ...]
    object_score_strategy: str
    local_exceedance_score: torch.Tensor
    local_z_score: torch.Tensor
    global_z_score: torch.Tensor
    largest_component_area: torch.Tensor
    largest_component_mean_excess: torch.Tensor
    largest_component_peak_excess: torch.Tensor
    largest_component_mass_excess: torch.Tensor
    largest_component_area_fraction: torch.Tensor
    num_exceedance_components: torch.Tensor


@dataclass(frozen=True)
class GlobalBranchGateCalibration:
    category: str
    mode: str = "monitor_only"
    active: bool = False
    required_cues: tuple[str, ...] = ()
    degenerate_cues: tuple[str, ...] = ()
    cue_nonzero_counts: dict[str, int] = field(default_factory=dict)
    ir_hotspot_area_fraction_threshold: float = 0.0
    ir_hotspot_compactness_threshold: float = 0.0
    ir_mean_hotspot_intensity_threshold: float = 0.0
    cross_inconsistency_score_threshold: float = 0.0
    rescue_score_q95: float = 0.0
    rescue_score_q99: float = 0.0
    spatial_score_q95: float = 0.0
    sample_count: int = 0

    def to_dict(self) -> dict[str, float | int | str | bool | list[str] | dict[str, int]]:
        return {
            "category": self.category,
            "mode": self.mode,
            "active": bool(self.active),
            "required_cues": list(self.required_cues),
            "degenerate_cues": list(self.degenerate_cues),
            "cue_nonzero_counts": {str(key): int(value) for key, value in self.cue_nonzero_counts.items()},
            "ir_hotspot_area_fraction_threshold": float(self.ir_hotspot_area_fraction_threshold),
            "ir_hotspot_compactness_threshold": float(self.ir_hotspot_compactness_threshold),
            "ir_mean_hotspot_intensity_threshold": float(self.ir_mean_hotspot_intensity_threshold),
            "cross_inconsistency_score_threshold": float(self.cross_inconsistency_score_threshold),
            "rescue_score_q95": float(self.rescue_score_q95),
            "rescue_score_q99": float(self.rescue_score_q99),
            "spatial_score_q95": float(self.spatial_score_q95),
            "sample_count": int(self.sample_count),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "GlobalBranchGateCalibration":
        return cls(
            category=str(payload.get("category", "")),
            mode=str(payload.get("mode", "monitor_only")),
            active=bool(payload.get("active", False)),
            required_cues=tuple(str(item) for item in payload.get("required_cues", [])),
            degenerate_cues=tuple(str(item) for item in payload.get("degenerate_cues", [])),
            cue_nonzero_counts={
                str(key): int(value)
                for key, value in dict(payload.get("cue_nonzero_counts", {})).items()
            }
            if isinstance(payload.get("cue_nonzero_counts", {}), Mapping)
            else {},
            ir_hotspot_area_fraction_threshold=float(payload.get("ir_hotspot_area_fraction_threshold", 0.0)),
            ir_hotspot_compactness_threshold=float(payload.get("ir_hotspot_compactness_threshold", 0.0)),
            ir_mean_hotspot_intensity_threshold=float(payload.get("ir_mean_hotspot_intensity_threshold", 0.0)),
            cross_inconsistency_score_threshold=float(payload.get("cross_inconsistency_score_threshold", 0.0)),
            rescue_score_q95=float(payload.get("rescue_score_q95", 0.0)),
            rescue_score_q99=float(payload.get("rescue_score_q99", 0.0)),
            spatial_score_q95=float(payload.get("spatial_score_q95", 0.0)),
            sample_count=int(payload.get("sample_count", 0)),
        )


@dataclass(frozen=True)
class SpatialExceedanceCalibration:
    category: str
    tau_pix: float
    tail_q99: float
    sigma_tail: float
    min_region_pixels: int
    z_local_mu: float
    z_local_std: float
    z_global_mu: float
    z_global_std: float
    sample_count: int
    pixel_count: int

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "category": self.category,
            "tau_pix": float(self.tau_pix),
            "tail_q99": float(self.tail_q99),
            "sigma_tail": float(self.sigma_tail),
            "min_region_pixels": int(self.min_region_pixels),
            "z_local_mu": float(self.z_local_mu),
            "z_local_std": float(self.z_local_std),
            "z_global_mu": float(self.z_global_mu),
            "z_global_std": float(self.z_global_std),
            "sample_count": int(self.sample_count),
            "pixel_count": int(self.pixel_count),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "SpatialExceedanceCalibration":
        return cls(
            category=str(payload.get("category", "")),
            tau_pix=float(payload.get("tau_pix", 0.0)),
            tail_q99=float(payload.get("tail_q99", 0.0)),
            sigma_tail=max(float(payload.get("sigma_tail", 1e-6)), 1e-6),
            min_region_pixels=int(payload.get("min_region_pixels", 1)),
            z_local_mu=float(payload.get("z_local_mu", 0.0)),
            z_local_std=max(float(payload.get("z_local_std", 1.0)), 1e-6),
            z_global_mu=float(payload.get("z_global_mu", 0.0)),
            z_global_std=max(float(payload.get("z_global_std", 1.0)), 1e-6),
            sample_count=int(payload.get("sample_count", 0)),
            pixel_count=int(payload.get("pixel_count", 0)),
        )


@dataclass(frozen=True)
class LogisticObjectScoreCalibration:
    strategy: str
    feature_names: tuple[str, ...]
    feature_mean: list[float]
    feature_std: list[float]
    coefficients: list[float]
    intercept: float
    sample_count: int
    positive_count: int
    negative_count: int
    l2_regularization: float

    def to_dict(self) -> dict[str, object]:
        return {
            "strategy": self.strategy,
            "feature_names": list(self.feature_names),
            "feature_mean": [float(value) for value in self.feature_mean],
            "feature_std": [float(value) for value in self.feature_std],
            "coefficients": [float(value) for value in self.coefficients],
            "intercept": float(self.intercept),
            "sample_count": int(self.sample_count),
            "positive_count": int(self.positive_count),
            "negative_count": int(self.negative_count),
            "l2_regularization": float(self.l2_regularization),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "LogisticObjectScoreCalibration":
        return cls(
            strategy=str(payload.get("strategy", "exceedance_logreg")),
            feature_names=tuple(str(item) for item in payload.get("feature_names", OBJECT_SCORE_FEATURE_NAMES)),
            feature_mean=[float(value) for value in payload.get("feature_mean", [0.0] * len(OBJECT_SCORE_FEATURE_NAMES))],
            feature_std=[
                max(float(value), 1e-6)
                for value in payload.get("feature_std", [1.0] * len(OBJECT_SCORE_FEATURE_NAMES))
            ],
            coefficients=[float(value) for value in payload.get("coefficients", [0.0] * len(OBJECT_SCORE_FEATURE_NAMES))],
            intercept=float(payload.get("intercept", 0.0)),
            sample_count=int(payload.get("sample_count", 0)),
            positive_count=int(payload.get("positive_count", 0)),
            negative_count=int(payload.get("negative_count", 0)),
            l2_regularization=float(payload.get("l2_regularization", 0.0)),
        )


def descriptor_support_map(
    descriptor_maps: torch.Tensor,
    support_maps: torch.Tensor | None = None,
) -> torch.Tensor:
    if support_maps is not None and support_maps.shape[1] >= 2:
        cross_ir_support = support_maps[:, -3:-2, :, :]
        cross_geo_support = support_maps[:, -2:-1, :, :]
        return (cross_ir_support + cross_geo_support + descriptor_maps[:, -1:, :, :]) / 3.0
    return descriptor_maps[:, -1:, :, :]


def build_object_score_feature_matrix(scored: AnomalyScoreOutputs) -> torch.Tensor:
    return torch.stack(
        [
            scored.local_z_score,
            scored.global_z_score,
            torch.log1p(scored.local_exceedance_score.clamp_min(0.0)),
            torch.log1p(scored.global_descriptor_score.clamp_min(0.0)),
            scored.largest_component_area,
            scored.largest_component_mean_excess,
            scored.largest_component_peak_excess,
            scored.largest_component_mass_excess,
            scored.num_exceedance_components,
            scored.largest_component_area_fraction,
        ],
        dim=1,
    )


def select_localization_basis_map(
    scored: AnomalyScoreOutputs,
    localization_basis: str = "anomaly_map",
) -> torch.Tensor:
    def _normalize_map(sample_map: torch.Tensor) -> torch.Tensor:
        minimum = sample_map.amin(dim=(2, 3), keepdim=True)
        maximum = sample_map.amax(dim=(2, 3), keepdim=True)
        return (sample_map - minimum) / (maximum - minimum + 1e-6)

    resolved_basis = str(localization_basis).strip().lower()
    if resolved_basis == "anomaly_map":
        return scored.anomaly_map
    if resolved_basis == "score_basis_map":
        return scored.score_basis_map
    if resolved_basis == "residual_component":
        return scored.residual_component
    if resolved_basis == "noise_component":
        return scored.noise_component
    if resolved_basis == "descriptor_support":
        return scored.descriptor_support
    if resolved_basis == "noise_residual_blend":
        noise_map = _normalize_map(scored.noise_component)
        residual_map = _normalize_map(scored.residual_component)
        return 0.65 * noise_map + 0.35 * residual_map
    if resolved_basis == "noise_residual_geomean":
        noise_map = _normalize_map(scored.noise_component).clamp_min(0.0)
        residual_map = _normalize_map(scored.residual_component).clamp_min(0.0)
        return torch.sqrt((noise_map * residual_map).clamp_min(0.0) + 1e-8)
    raise ValueError(
        f"Unsupported localization_basis={localization_basis!r}. "
        f"Expected one of {LOCALIZATION_BASIS_CHOICES}."
    )


def _masked_mean(values: torch.Tensor, object_mask: torch.Tensor) -> torch.Tensor:
    mask_pixels = object_mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
    return (values * object_mask).sum(dim=(1, 2, 3)) / mask_pixels


def _masked_quantile(
    values: torch.Tensor,
    object_mask: torch.Tensor,
    quantile: float,
) -> torch.Tensor:
    quantiles = []
    for sample_index in range(values.shape[0]):
        sample_values = values[sample_index].reshape(-1)
        sample_mask = object_mask[sample_index].reshape(-1) > 0
        selected = sample_values[sample_mask] if bool(sample_mask.any()) else sample_values
        selected = selected[torch.isfinite(selected)]
        if selected.numel() == 0:
            quantiles.append(values.new_tensor(0.0))
            continue
        quantiles.append(torch.quantile(selected.float(), float(quantile)).to(dtype=values.dtype))
    return torch.stack(quantiles, dim=0)


def _masked_top_fraction_mean(
    values: torch.Tensor,
    object_mask: torch.Tensor,
    *,
    fraction: float = 0.01,
) -> torch.Tensor:
    means = []
    for sample_index in range(values.shape[0]):
        sample_values = values[sample_index].reshape(-1)
        sample_mask = object_mask[sample_index].reshape(-1) > 0
        selected = sample_values[sample_mask] if bool(sample_mask.any()) else sample_values
        selected = selected[torch.isfinite(selected)]
        if selected.numel() == 0:
            means.append(values.new_tensor(0.0))
            continue
        topk = max(int(math.ceil(float(selected.numel()) * max(float(fraction), 1e-6))), 1)
        means.append(torch.topk(selected.float(), k=min(topk, int(selected.numel()))).values.mean().to(dtype=values.dtype))
    return torch.stack(means, dim=0)


def _extract_internal_defect_gate_cues(
    descriptor_maps: torch.Tensor,
    support_maps: torch.Tensor | None,
    object_mask: torch.Tensor,
    raw_global_vector: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    batch_size = descriptor_maps.shape[0]
    feature_index = {name: index for index, name in enumerate(GLOBAL_FEATURE_NAMES)}
    descriptor_index = {name: index for index, name in enumerate(SPATIAL_DESCRIPTOR_ORDER)}
    support_index = {name: index for index, name in enumerate(SUPPORT_DESCRIPTOR_ORDER)}
    fallback_global = (
        raw_global_vector
        if raw_global_vector is not None
        else descriptor_maps.new_zeros((batch_size, len(GLOBAL_FEATURE_NAMES)))
    )
    ir_hotspot_map = descriptor_maps[:, descriptor_index["ir_hotspot"] : descriptor_index["ir_hotspot"] + 1, :, :]
    ir_gradient_map = descriptor_maps[:, descriptor_index["ir_gradient"] : descriptor_index["ir_gradient"] + 1, :, :]
    if support_maps is None:
        cross_inconsistency_map = descriptor_maps.new_zeros((batch_size, 1, *descriptor_maps.shape[-2:]))
        cross_geo_support_map = descriptor_maps.new_zeros((batch_size, 1, *descriptor_maps.shape[-2:]))
    else:
        cross_inconsistency_map = support_maps[
            :,
            support_index["cross_inconsistency"] : support_index["cross_inconsistency"] + 1,
            :,
            :,
        ]
        cross_geo_support_map = support_maps[
            :,
            support_index["cross_geo_support"] : support_index["cross_geo_support"] + 1,
            :,
            :,
        ]
    return {
        "ir_hotspot_area_fraction": fallback_global[:, feature_index["ir_hotspot_area_fraction"]],
        "ir_hotspot_compactness": fallback_global[:, feature_index["ir_hotspot_compactness"]],
        "ir_mean_hotspot_intensity": fallback_global[:, feature_index["ir_mean_hotspot_intensity"]],
        "cross_inconsistency_score": fallback_global[:, feature_index["cross_inconsistency_score"]],
        "cross_inconsistency_q99": _masked_quantile(cross_inconsistency_map, object_mask, 0.99),
        "cross_inconsistency_top1": _masked_top_fraction_mean(cross_inconsistency_map, object_mask, fraction=0.01),
        "cross_geo_support_top1": _masked_top_fraction_mean(cross_geo_support_map, object_mask, fraction=0.01),
        "ir_hotspot_top1": _masked_top_fraction_mean(ir_hotspot_map, object_mask, fraction=0.01),
        "ir_gradient_top1": _masked_top_fraction_mean(ir_gradient_map, object_mask, fraction=0.01),
    }


def _resolve_archetype_b_rescue_score(
    *,
    mode: str,
    cues: Mapping[str, torch.Tensor],
    sample_index: int,
) -> torch.Tensor:
    if mode == "cross_inconsistency":
        return torch.stack(
            [
                cues["cross_inconsistency_q99"][sample_index],
                cues["cross_inconsistency_top1"][sample_index],
                cues["cross_geo_support_top1"][sample_index],
            ],
            dim=0,
        ).max()
    if mode == "thermal_hotspot":
        return torch.stack(
            [
                cues["ir_hotspot_top1"][sample_index],
                cues["ir_gradient_top1"][sample_index],
                cues["ir_mean_hotspot_intensity"][sample_index],
            ],
            dim=0,
        ).max()
    reference = next(iter(cues.values()))
    return reference.new_tensor(0.0)


def _required_gate_cues_for_mode(mode: str) -> tuple[str, ...]:
    if mode == "cross_inconsistency":
        return ("cross_inconsistency_score",)
    if mode == "thermal_hotspot":
        return (
            "ir_hotspot_area_fraction",
            "ir_hotspot_compactness",
            "ir_mean_hotspot_intensity",
        )
    return ()


def _cue_threshold_from_calibration(
    calibration: GlobalBranchGateCalibration,
    cue_name: str,
) -> float:
    if cue_name == "ir_hotspot_area_fraction":
        return float(calibration.ir_hotspot_area_fraction_threshold)
    if cue_name == "ir_hotspot_compactness":
        return float(calibration.ir_hotspot_compactness_threshold)
    if cue_name == "ir_mean_hotspot_intensity":
        return float(calibration.ir_mean_hotspot_intensity_threshold)
    if cue_name == "cross_inconsistency_score":
        return float(calibration.cross_inconsistency_score_threshold)
    raise KeyError(f"Unsupported gate cue {cue_name!r}.")


def _apply_internal_defect_gate(
    cues: Mapping[str, torch.Tensor],
    rescue_score: torch.Tensor,
    spatial_score: torch.Tensor,
    categories: Sequence[str] | None,
    calibration: GlobalBranchGateCalibration | Mapping[str, GlobalBranchGateCalibration] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[str, ...], tuple[str, ...]]:
    if calibration is None:
        empty_flags = torch.zeros((spatial_score.shape[0],), dtype=torch.bool, device=spatial_score.device)
        return empty_flags, spatial_score, empty_flags, tuple("inactive" for _ in range(spatial_score.shape[0])), tuple(
            "missing_calibration" for _ in range(spatial_score.shape[0])
        )
    flags = []
    post_scores = []
    changed_scores = []
    modes: list[str] = []
    reasons: list[str] = []
    for sample_index in range(spatial_score.shape[0]):
        sample_calibration = calibration
        if isinstance(calibration, Mapping):
            if categories is None:
                raise ValueError("Category names are required when using per-category branch gate calibration.")
            sample_calibration = calibration.get(str(categories[sample_index]))
        sample_category = (
            str(categories[sample_index])
            if categories is not None
            else (str(sample_calibration.category) if sample_calibration is not None else "")
        )
        sample_mode = resolve_archetype_b_gate_mode(sample_category)
        sample_reason = "inactive"
        sample_fired = False
        sample_score_changed = False
        sample_spatial_score = float(spatial_score[sample_index].item())
        sample_post_score = spatial_score[sample_index]
        if sample_calibration is None:
            sample_reason = "missing_calibration"
        elif not is_archetype_b(sample_category):
            sample_reason = "not_archetype_b"
        else:
            sample_mode = str(sample_calibration.mode or sample_mode)
            required_cues = tuple(sample_calibration.required_cues or _required_gate_cues_for_mode(sample_mode))
            degenerate_cues = set(str(cue) for cue in sample_calibration.degenerate_cues)
            if sample_mode == "monitor_only":
                sample_reason = "monitor_only"
            elif not bool(sample_calibration.active):
                sample_reason = "inactive_category"
            elif not math.isfinite(sample_spatial_score):
                sample_reason = "non_finite_spatial_score"
            elif sample_spatial_score > float(sample_calibration.spatial_score_q95):
                sample_reason = "spatial_score_above_q95"
            elif any(cue_name in degenerate_cues for cue_name in required_cues):
                sample_reason = "degenerate_required_cues"
            else:
                cue_pass = True
                for cue_name in required_cues:
                    cue_value = float(cues[cue_name][sample_index].item())
                    if not math.isfinite(cue_value) or cue_value < _cue_threshold_from_calibration(sample_calibration, cue_name):
                        cue_pass = False
                        break
                if not cue_pass:
                    sample_reason = "trigger_cues_below_threshold"
                else:
                    sample_fired = True
                    sample_rescue_score = float(rescue_score[sample_index].item())
                    if math.isfinite(sample_rescue_score) and sample_rescue_score > sample_spatial_score:
                        sample_post_score = rescue_score[sample_index]
                        sample_score_changed = True
                        sample_reason = "rescued"
                    else:
                        sample_reason = "fired_no_change"
        flags.append(sample_fired)
        post_scores.append(sample_post_score)
        changed_scores.append(sample_score_changed)
        modes.append(sample_mode)
        reasons.append(sample_reason)
    return (
        torch.tensor(flags, dtype=torch.bool, device=spatial_score.device),
        torch.stack(post_scores, dim=0),
        torch.tensor(changed_scores, dtype=torch.bool, device=spatial_score.device),
        tuple(modes),
        tuple(reasons),
    )


def _resolve_spatial_exceedance_calibration(
    calibration: SpatialExceedanceCalibration | Mapping[str, SpatialExceedanceCalibration] | None,
    *,
    category: str,
) -> SpatialExceedanceCalibration:
    if calibration is None:
        raise ValueError("Spatial exceedance calibration is required for exceedance-based object scoring.")
    if isinstance(calibration, Mapping):
        resolved = calibration.get(str(category))
        if resolved is None:
            raise KeyError(f"Missing spatial exceedance calibration for category={category!r}.")
        return resolved
    return calibration


def _summarize_exceedance_regions(
    exceedance_map: torch.Tensor,
    object_mask: torch.Tensor,
    *,
    min_region_pixels: int,
) -> tuple[float, float, float, float, float, int, float]:
    exceedance_2d = exceedance_map.detach().float().cpu().squeeze().numpy()
    object_mask_2d = (object_mask.detach().float().cpu().squeeze().numpy() > 0)
    if exceedance_2d.shape != object_mask_2d.shape:
        raise ValueError("exceedance_map and object_mask must share the same spatial shape.")

    valid_mask = (exceedance_2d > 0.0) & object_mask_2d
    if not valid_mask.any():
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0

    labeled, component_count = ndimage.label(valid_mask.astype("uint8"))
    resolved_min_region_pixels = max(int(min_region_pixels), 1)
    local_score = 0.0
    largest_area = 0.0
    largest_mean = 0.0
    largest_peak = 0.0
    largest_mass = 0.0
    largest_area_fraction = 0.0
    valid_component_count = 0
    object_pixels = max(int(object_mask_2d.sum()), 1)

    for component_index in range(1, int(component_count) + 1):
        component = labeled == component_index
        area_pixels = int(component.sum())
        if area_pixels < resolved_min_region_pixels:
            continue
        values = exceedance_2d[component]
        if values.size == 0:
            continue
        valid_component_count += 1
        mean_excess = float(values.mean())
        peak_excess = float(values.max())
        mass_excess = float(values.sum())
        region_score = 0.7 * mean_excess * math.log1p(area_pixels) + 0.3 * peak_excess
        local_score = max(local_score, region_score)
        if area_pixels > largest_area:
            largest_area = float(area_pixels)
            largest_mean = mean_excess
            largest_peak = peak_excess
            largest_mass = mass_excess
            largest_area_fraction = float(area_pixels / max(object_pixels, 1))

    if valid_component_count == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0
    return (
        float(local_score),
        float(largest_area),
        float(largest_mean),
        float(largest_peak),
        float(largest_mass),
        int(valid_component_count),
        float(largest_area_fraction),
    )


def _apply_logistic_object_score_calibration(
    feature_matrix: torch.Tensor,
    calibration: LogisticObjectScoreCalibration,
) -> torch.Tensor:
    mean = torch.tensor(calibration.feature_mean, dtype=feature_matrix.dtype, device=feature_matrix.device)
    std = torch.tensor(calibration.feature_std, dtype=feature_matrix.dtype, device=feature_matrix.device).clamp_min(1e-6)
    coefficients = torch.tensor(calibration.coefficients, dtype=feature_matrix.dtype, device=feature_matrix.device)
    intercept = torch.tensor(float(calibration.intercept), dtype=feature_matrix.dtype, device=feature_matrix.device)
    standardized = (feature_matrix - mean.unsqueeze(0)) / std.unsqueeze(0)
    logits = standardized @ coefficients + intercept
    return torch.sigmoid(logits)


def score_batch(
    model: MulSenDiffX,
    x_rgb: torch.Tensor,
    descriptor_maps: torch.Tensor,
    support_maps: torch.Tensor | None,
    global_vector: torch.Tensor,
    category_indices: torch.Tensor | None = None,
    *,
    raw_global_vector: torch.Tensor | None = None,
    categories: Sequence[str] | None = None,
    score_mode: str = "noise_error",
    timestep: int = 200,
    descriptor_weight: float = 0.25,
    global_descriptor_weight: float = 0.0,
    multi_timestep_scoring_enabled: bool = False,
    multi_timestep_fractions: Sequence[float] | None = None,
    anomaly_map_gaussian_sigma: float = 0.0,
    lambda_residual: float | None = None,
    lambda_noise: float | None = None,
    lambda_descriptor_spatial: float | None = None,
    lambda_descriptor_global: float | None = None,
    global_descriptor_calibration: GlobalDescriptorCalibration | Mapping[str, GlobalDescriptorCalibration] | None = None,
    global_branch_gate_calibration: GlobalBranchGateCalibration | Mapping[str, GlobalBranchGateCalibration] | None = None,
    spatial_exceedance_calibration: SpatialExceedanceCalibration | Mapping[str, SpatialExceedanceCalibration] | None = None,
    object_score_calibration: LogisticObjectScoreCalibration | None = None,
    object_score_strategy: str = "legacy_raw",
    spatial_topk_percent: float = 0.02,
    enable_internal_defect_gate: bool = False,
    object_mask_threshold: float = 0.02,
    object_mask_dilation_kernel_size: int = 9,
    generator: torch.Generator | None = None,
) -> AnomalyScoreOutputs:
    del spatial_topk_percent
    if score_mode not in {"noise_error", "residual"}:
        raise ValueError(f"Unsupported score_mode={score_mode!r}. Expected 'noise_error' or 'residual'.")
    if object_score_strategy not in {"legacy_raw", "exceedance_regions", "exceedance_logreg"}:
        raise ValueError(
            f"Unsupported object_score_strategy={object_score_strategy!r}. "
            "Expected 'legacy_raw', 'exceedance_regions', or 'exceedance_logreg'."
        )

    device = x_rgb.device
    if category_indices is None:
        category_indices = torch.zeros((x_rgb.shape[0],), dtype=torch.long, device=device)
    clean_latent = model.encode_rgb(x_rgb)
    noise = torch.randn(clean_latent.shape, generator=generator, device=device, dtype=clean_latent.dtype)
    if any(value is not None for value in (lambda_residual, lambda_noise, lambda_descriptor_spatial, lambda_descriptor_global)):
        resolved_lambda_residual = float(lambda_residual if lambda_residual is not None else 0.0)
        resolved_lambda_noise = float(lambda_noise if lambda_noise is not None else 0.0)
        resolved_lambda_descriptor_spatial = float(
            lambda_descriptor_spatial if lambda_descriptor_spatial is not None else descriptor_weight
        )
        resolved_lambda_descriptor_global = float(
            lambda_descriptor_global if lambda_descriptor_global is not None else global_descriptor_weight
        )
    elif score_mode == "noise_error":
        resolved_lambda_residual = 0.0
        resolved_lambda_noise = 1.0
        resolved_lambda_descriptor_spatial = float(descriptor_weight)
        resolved_lambda_descriptor_global = float(global_descriptor_weight)
    else:
        resolved_lambda_residual = 1.0
        resolved_lambda_noise = 0.0
        resolved_lambda_descriptor_spatial = float(descriptor_weight)
        resolved_lambda_descriptor_global = float(global_descriptor_weight)

    if multi_timestep_scoring_enabled:
        fractions = tuple(multi_timestep_fractions or (0.05, 0.10, 0.20, 0.30, 0.40))
        timestep_values = sorted(
            {
                min(max(int(round(model.diffusion_steps * fraction)), 1), max(model.diffusion_steps - 1, 0))
                for fraction in fractions
            }
        )
    else:
        timestep_values = [min(int(timestep), max(model.diffusion_steps - 1, 0))]

    with torch.no_grad():
        autoencoded_rgb = model.decode_latent(clean_latent)
        reconstructed_rgb_values = []
        residual_map_values = []
        noise_error_values = []
        for timestep_value in timestep_values:
            timesteps = torch.full((x_rgb.shape[0],), timestep_value, device=device, dtype=torch.long)
            noisy_latent = model.q_sample(clean_latent, timesteps, noise)
            predicted_noise = model(
                noisy_latent,
                descriptor_maps,
                global_vector,
                category_indices,
                timesteps,
                support_maps=support_maps,
            )
            predicted_clean_latent = model.predict_x0_from_noise(noisy_latent, timesteps, predicted_noise)
            reconstructed_rgb_values.append(model.decode_latent(predicted_clean_latent))
            residual_map_values.append((reconstructed_rgb_values[-1] - x_rgb).abs().mean(dim=1, keepdim=True))
            per_step_noise_error = (noise - predicted_noise).pow(2).mean(dim=1, keepdim=True)
            noise_error_values.append(
                F.interpolate(
                    per_step_noise_error,
                    size=x_rgb.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            )
        reconstructed_rgb = torch.stack(reconstructed_rgb_values, dim=0).mean(dim=0)
        residual_map = torch.stack(residual_map_values, dim=0).mean(dim=0)
        noise_error_map = torch.stack(noise_error_values, dim=0).mean(dim=0)
        support = descriptor_support_map(descriptor_maps, support_maps)
        object_mask = estimate_object_mask(
            x_rgb,
            threshold=object_mask_threshold,
            dilation_kernel_size=max(int(object_mask_dilation_kernel_size), 1),
            support_maps=support_maps,
        )

        residual_component = resolved_lambda_residual * residual_map
        noise_component = resolved_lambda_noise * noise_error_map
        descriptor_component = resolved_lambda_descriptor_spatial * support
        score_basis_map = residual_component + noise_component
        if resolved_lambda_residual > 0.0 and resolved_lambda_noise > 0.0:
            score_basis_label = "Weighted Residual + Noise"
        elif resolved_lambda_noise > 0.0:
            score_basis_label = "Latent Noise Error"
        else:
            score_basis_label = "Decoded RGB Residual"

        anomaly_map = (score_basis_map + descriptor_component) * object_mask
        if anomaly_map_gaussian_sigma > 0.0:
            anomaly_map = gaussian_smooth_map(anomaly_map, sigma=anomaly_map_gaussian_sigma) * object_mask
        gate_cues = _extract_internal_defect_gate_cues(
            descriptor_maps,
            support_maps,
            object_mask,
            raw_global_vector,
        )
        gate_modes = tuple(
            resolve_archetype_b_gate_mode(str(category_name))
            for category_name in (categories or ("shared",) * x_rgb.shape[0])
        )
        rescue_score = torch.stack(
            [
                _resolve_archetype_b_rescue_score(
                    mode=gate_modes[sample_index],
                    cues=gate_cues,
                    sample_index=sample_index,
                )
                for sample_index in range(x_rgb.shape[0])
            ],
            dim=0,
        )

        if global_descriptor_calibration is not None and resolved_lambda_descriptor_global > 0.0:
            if isinstance(global_descriptor_calibration, Mapping):
                if categories is None:
                    raise ValueError("Category names are required when using per-category global descriptor calibration.")
                scores = []
                for sample_index, category_name in enumerate(categories):
                    calibration = global_descriptor_calibration.get(str(category_name))
                    if calibration is None:
                        raise KeyError(f"Missing global descriptor calibration for category={category_name!r}.")
                    sample_score = score_global_descriptors(
                        global_vector[sample_index : sample_index + 1],
                        calibration,
                    )
                    scores.append(sample_score.squeeze(0))
                global_score = torch.stack(scores, dim=0) if scores else torch.zeros((x_rgb.shape[0],), device=device)
            else:
                global_score = score_global_descriptors(global_vector, global_descriptor_calibration)
        else:
            global_score = torch.zeros((x_rgb.shape[0],), dtype=anomaly_map.dtype, device=device)

        if object_score_strategy == "legacy_raw":
            spatial_score_pre_gate = _masked_mean(anomaly_map, object_mask)
            internal_defect_gate_fired = torch.zeros(
                (x_rgb.shape[0],),
                dtype=torch.bool,
                device=device,
            )
            internal_defect_gate_score_changed = torch.zeros(
                (x_rgb.shape[0],),
                dtype=torch.bool,
                device=device,
            )
            spatial_score_post_gate = spatial_score_pre_gate
            internal_defect_gate_reason = tuple("disabled" for _ in range(x_rgb.shape[0]))
            if enable_internal_defect_gate:
                (
                    internal_defect_gate_fired,
                    spatial_score_post_gate,
                    internal_defect_gate_score_changed,
                    gate_modes,
                    internal_defect_gate_reason,
                ) = _apply_internal_defect_gate(
                    gate_cues,
                    rescue_score,
                    spatial_score_pre_gate,
                    categories,
                    global_branch_gate_calibration,
                )
            spatial_score = spatial_score_post_gate
            anomaly_score = spatial_score_post_gate + resolved_lambda_descriptor_global * global_score
            ir_dominant_score = rescue_score
            local_exceedance_score = torch.zeros_like(spatial_score)
            local_z_score = torch.zeros_like(spatial_score)
            global_z_score = torch.zeros_like(spatial_score)
            largest_component_area = torch.zeros_like(spatial_score)
            largest_component_mean_excess = torch.zeros_like(spatial_score)
            largest_component_peak_excess = torch.zeros_like(spatial_score)
            largest_component_mass_excess = torch.zeros_like(spatial_score)
            largest_component_area_fraction = torch.zeros_like(spatial_score)
            num_exceedance_components = torch.zeros_like(spatial_score)
        else:
            if categories is None:
                raise ValueError("Category names are required when using exceedance-based object scoring.")
            local_scores = []
            local_z_scores = []
            global_z_scores = []
            largest_area_values = []
            largest_mean_values = []
            largest_peak_values = []
            largest_mass_values = []
            largest_area_fraction_values = []
            component_count_values = []
            anomaly_scores = []
            for sample_index, category_name in enumerate(categories):
                calibration = _resolve_spatial_exceedance_calibration(
                    spatial_exceedance_calibration,
                    category=str(category_name),
                )
                sample_map = anomaly_map[sample_index : sample_index + 1]
                sample_mask = object_mask[sample_index : sample_index + 1]
                exceedance_map = torch.relu(
                    (sample_map - calibration.tau_pix)
                    / max(float(calibration.sigma_tail), 1e-6)
                ) * sample_mask
                (
                    local_score_value,
                    largest_area_value,
                    largest_mean_value,
                    largest_peak_value,
                    largest_mass_value,
                    component_count_value,
                    largest_area_fraction_value,
                ) = _summarize_exceedance_regions(
                    exceedance_map,
                    sample_mask,
                    min_region_pixels=calibration.min_region_pixels,
                )
                local_tensor = anomaly_map.new_tensor(local_score_value)
                global_tensor = global_score[sample_index]
                local_z_tensor = (
                    torch.log1p(local_tensor.clamp_min(0.0)) - float(calibration.z_local_mu)
                ) / max(float(calibration.z_local_std), 1e-6)
                global_z_tensor = (
                    torch.log1p(global_tensor.clamp_min(0.0)) - float(calibration.z_global_mu)
                ) / max(float(calibration.z_global_std), 1e-6)
                has_region = component_count_value > 0
                if object_score_strategy == "exceedance_logreg":
                    if object_score_calibration is None:
                        raise ValueError(
                            "Logistic object-score calibration is required for object_score_strategy='exceedance_logreg'."
                        )
                    feature_matrix = torch.stack(
                        [
                            local_z_tensor,
                            global_z_tensor,
                            torch.log1p(local_tensor.clamp_min(0.0)),
                            torch.log1p(global_tensor.clamp_min(0.0)),
                            anomaly_map.new_tensor(float(largest_area_value)),
                            anomaly_map.new_tensor(float(largest_mean_value)),
                            anomaly_map.new_tensor(float(largest_peak_value)),
                            anomaly_map.new_tensor(float(largest_mass_value)),
                            anomaly_map.new_tensor(float(component_count_value)),
                            anomaly_map.new_tensor(float(largest_area_fraction_value)),
                        ],
                        dim=0,
                    ).unsqueeze(0)
                    final_score = _apply_logistic_object_score_calibration(feature_matrix, object_score_calibration).squeeze(0)
                else:
                    final_score = global_z_tensor if not has_region else 0.65 * local_z_tensor + 0.35 * global_z_tensor
                local_scores.append(local_tensor)
                local_z_scores.append(local_z_tensor)
                global_z_scores.append(global_z_tensor)
                largest_area_values.append(anomaly_map.new_tensor(float(largest_area_value)))
                largest_mean_values.append(anomaly_map.new_tensor(float(largest_mean_value)))
                largest_peak_values.append(anomaly_map.new_tensor(float(largest_peak_value)))
                largest_mass_values.append(anomaly_map.new_tensor(float(largest_mass_value)))
                largest_area_fraction_values.append(anomaly_map.new_tensor(float(largest_area_fraction_value)))
                component_count_values.append(anomaly_map.new_tensor(float(component_count_value)))
                anomaly_scores.append(final_score)
            spatial_score = torch.stack(local_scores, dim=0)
            spatial_score_pre_gate = spatial_score
            spatial_score_post_gate = spatial_score
            ir_dominant_score = rescue_score
            internal_defect_gate_fired = torch.zeros(
                (x_rgb.shape[0],),
                dtype=torch.bool,
                device=device,
            )
            internal_defect_gate_score_changed = torch.zeros(
                (x_rgb.shape[0],),
                dtype=torch.bool,
                device=device,
            )
            internal_defect_gate_reason = tuple("unsupported_object_score_strategy" for _ in range(x_rgb.shape[0]))
            local_exceedance_score = spatial_score
            local_z_score = torch.stack(local_z_scores, dim=0)
            global_z_score = torch.stack(global_z_scores, dim=0)
            largest_component_area = torch.stack(largest_area_values, dim=0)
            largest_component_mean_excess = torch.stack(largest_mean_values, dim=0)
            largest_component_peak_excess = torch.stack(largest_peak_values, dim=0)
            largest_component_mass_excess = torch.stack(largest_mass_values, dim=0)
            largest_component_area_fraction = torch.stack(largest_area_fraction_values, dim=0)
            num_exceedance_components = torch.stack(component_count_values, dim=0)
            anomaly_score = torch.stack(anomaly_scores, dim=0)

    return AnomalyScoreOutputs(
        anomaly_score=anomaly_score,
        anomaly_map=anomaly_map,
        score_basis_map=score_basis_map,
        score_basis_label=score_basis_label,
        residual_map=residual_map,
        noise_error_map=noise_error_map,
        descriptor_support=support,
        descriptor_spatial_component=descriptor_component,
        residual_component=residual_component,
        noise_component=noise_component,
        reconstructed_rgb=reconstructed_rgb,
        diffusion_reconstructed_rgb=reconstructed_rgb,
        diffusion_reconstruction_label="Diffusion Reconstruction",
        display_reconstructed_rgb=autoencoded_rgb,
        display_reconstruction_label="Autoencoded RGB",
        object_mask=object_mask,
        spatial_score=spatial_score,
        spatial_score_pre_gate=spatial_score_pre_gate,
        spatial_score_post_gate=spatial_score_post_gate,
        rescue_score=rescue_score,
        ir_dominant_score=ir_dominant_score,
        internal_defect_gate_fired=internal_defect_gate_fired,
        internal_defect_gate_score_changed=internal_defect_gate_score_changed,
        internal_defect_gate_mode=gate_modes,
        internal_defect_gate_reason=internal_defect_gate_reason,
        raw_ir_hotspot_area_fraction=gate_cues["ir_hotspot_area_fraction"],
        raw_ir_hotspot_compactness=gate_cues["ir_hotspot_compactness"],
        raw_ir_mean_hotspot_intensity=gate_cues["ir_mean_hotspot_intensity"],
        raw_cross_inconsistency_score=gate_cues["cross_inconsistency_score"],
        cross_inconsistency_q99=gate_cues["cross_inconsistency_q99"],
        cross_inconsistency_top1=gate_cues["cross_inconsistency_top1"],
        cross_geo_support_top1=gate_cues["cross_geo_support_top1"],
        ir_hotspot_top1=gate_cues["ir_hotspot_top1"],
        ir_gradient_top1=gate_cues["ir_gradient_top1"],
        global_descriptor_score=global_score,
        timesteps_used=tuple(timestep_values),
        object_score_strategy=object_score_strategy,
        local_exceedance_score=local_exceedance_score,
        local_z_score=local_z_score,
        global_z_score=global_z_score,
        largest_component_area=largest_component_area,
        largest_component_mean_excess=largest_component_mean_excess,
        largest_component_peak_excess=largest_component_peak_excess,
        largest_component_mass_excess=largest_component_mass_excess,
        largest_component_area_fraction=largest_component_area_fraction,
        num_exceedance_components=num_exceedance_components,
    )
