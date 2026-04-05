from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import torch

from src.data_loader import SPATIAL_DESCRIPTOR_ORDER, SUPPORT_DESCRIPTOR_ORDER
from src.inference.quantification import MasiCalibration, MasiQuantification, quantify_masi


@dataclass(frozen=True)
class RetrievalFeatures:
    category: str
    defect_label: str
    confidence_band: str
    severity_band: str
    distribution: str
    dominant_modalities: List[str]
    sensor_profile_tags: List[str]
    defect_profile_tags: List[str]
    query_text: str


@dataclass(frozen=True)
class EvidencePackage:
    category: str
    split: str
    defect_label: str
    sample_id: str
    is_anomalous: bool
    score_mode: str
    score_label: str
    raw_score: float
    severity_0_100: float
    confidence_0_100: float
    status: str
    affected_area_pct: float
    peak_anomaly_0_100: float
    top_regions: List[Dict[str, Any]]
    rgb_observations: List[str]
    thermal_observations: List[str]
    geometric_observations: List[str]
    cross_modal_support: List[str]
    evidence_breakdown: Dict[str, float]
    confidence_notes: List[str]
    global_descriptor_score: float
    retrieval_features: RetrievalFeatures
    retrieved_context: List[Dict[str, str]] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)
    source_paths: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_evidence_package(
    *,
    category: str,
    split: str,
    defect_label: str,
    sample_id: str,
    is_anomalous: bool,
    score_mode: str,
    score_label: str,
    raw_score: float,
    calibration: MasiCalibration,
    normalized_anomaly_map: torch.Tensor,
    score_basis_map: torch.Tensor,
    descriptor_support: torch.Tensor,
    object_mask: torch.Tensor,
    predicted_mask: torch.Tensor,
    descriptor_maps: torch.Tensor,
    support_maps: torch.Tensor,
    global_descriptor_score: float = 0.0,
    provenance: Dict[str, Any] | None = None,
    source_paths: Dict[str, str] | None = None,
) -> EvidencePackage:
    masi = quantify_masi(
        raw_score=raw_score,
        calibration=calibration,
        normalized_anomaly_map=normalized_anomaly_map,
        object_mask=object_mask,
        predicted_mask=predicted_mask,
    )
    region_mask = _ensure_region_mask(predicted_mask, normalized_anomaly_map, object_mask)
    thermal_strength = _masked_mean(
        _stack_named_maps(descriptor_maps, ["ir_hotspot", "ir_gradient"]),
        region_mask,
    )
    geometric_strength = _masked_mean(
        _stack_named_maps(
            descriptor_maps,
            ["pc_curvature", "pc_roughness", "pc_normal_deviation"],
        ),
        region_mask,
    )
    agreement_strength = _masked_mean(
        torch.stack(
            [
                descriptor_maps[SPATIAL_DESCRIPTOR_ORDER.index("cross_agreement")],
                support_maps[SUPPORT_DESCRIPTOR_ORDER.index("cross_geo_support")],
            ],
            dim=0,
        ),
        region_mask,
    )
    inconsistency_strength = _masked_mean(
        _stack_named_support_maps(support_maps, ["cross_inconsistency"]),
        region_mask,
    )
    score_basis_strength = _masked_mean(score_basis_map, region_mask)
    support_strength = _masked_mean(descriptor_support, region_mask)

    evidence_breakdown = _normalize_contributions(
        {
            "score_basis_pct": score_basis_strength,
            "thermal_pct": thermal_strength,
            "geometric_pct": geometric_strength,
            "crossmodal_pct": max(agreement_strength + support_strength - inconsistency_strength, 0.0),
            "global_descriptor_pct": max(float(global_descriptor_score), 0.0),
        }
    )

    resolved_provenance = provenance or {}
    confidence_notes = _build_confidence_notes(
        quantification=masi,
        agreement_strength=agreement_strength,
        inconsistency_strength=inconsistency_strength,
        global_descriptor_score=global_descriptor_score,
    )
    thermal_sentence = _strength_sentence(
        "Thermal evidence",
        thermal_strength,
        strong="strong localized hotspot/gradient activity",
        medium="moderate localized heating cues",
    )
    geometric_sentence = _strength_sentence(
        "Geometric evidence",
        geometric_strength,
        strong="clear local surface or shape deviation",
        medium="moderate local geometric irregularity",
    )
    cross_modal_sentence = _cross_modal_sentence(agreement_strength, inconsistency_strength)
    retrieval_features = _build_retrieval_features(
        category=category,
        defect_label=defect_label,
        score_label=score_label,
        severity_0_100=masi.severity_0_100,
        confidence_0_100=masi.confidence_0_100,
        affected_area_pct=masi.affected_area_pct,
        evidence_breakdown=evidence_breakdown,
        thermal_observations=[thermal_sentence],
        geometric_observations=[geometric_sentence],
        cross_modal_support=[cross_modal_sentence],
        provenance=resolved_provenance,
        global_descriptor_score=float(global_descriptor_score),
    )

    return EvidencePackage(
        category=category,
        split=split,
        defect_label=defect_label,
        sample_id=sample_id,
        is_anomalous=is_anomalous,
        score_mode=score_mode,
        score_label=score_label,
        raw_score=float(raw_score),
        severity_0_100=masi.severity_0_100,
        confidence_0_100=masi.confidence_0_100,
        status=masi.status,
        affected_area_pct=masi.affected_area_pct,
        peak_anomaly_0_100=masi.peak_anomaly_0_100,
        top_regions=[masi.top_region.to_dict()],
        rgb_observations=[
            f"{score_label} concentrates around the dominant region with mean intensity {score_basis_strength:.3f}.",
            f"Predicted anomalous area covers {masi.affected_area_pct:.2f}% of the detected object region.",
        ],
        thermal_observations=[thermal_sentence],
        geometric_observations=[geometric_sentence],
        cross_modal_support=[cross_modal_sentence],
        evidence_breakdown=evidence_breakdown,
        confidence_notes=confidence_notes,
        global_descriptor_score=float(global_descriptor_score),
        retrieval_features=retrieval_features,
        provenance=resolved_provenance,
        source_paths=source_paths or {},
    )


def save_evidence_package(package: EvidencePackage, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(package.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return path


def _stack_named_maps(descriptor_maps: torch.Tensor, names: List[str]) -> torch.Tensor:
    channels = [descriptor_maps[SPATIAL_DESCRIPTOR_ORDER.index(name)] for name in names]
    return torch.stack(channels, dim=0)


def _stack_named_support_maps(support_maps: torch.Tensor, names: List[str]) -> torch.Tensor:
    channels = [support_maps[SUPPORT_DESCRIPTOR_ORDER.index(name)] for name in names]
    return torch.stack(channels, dim=0)


def _ensure_region_mask(
    predicted_mask: torch.Tensor,
    normalized_anomaly_map: torch.Tensor,
    object_mask: torch.Tensor,
) -> torch.Tensor:
    region_mask = predicted_mask.detach().float().squeeze()
    if region_mask.sum().item() > 0:
        return region_mask

    object_mask_2d = object_mask.detach().float().squeeze()
    anomaly_map_2d = normalized_anomaly_map.detach().float().squeeze()
    if object_mask_2d.sum().item() > 0:
        peak_value = anomaly_map_2d[object_mask_2d > 0].amax()
        return ((anomaly_map_2d >= peak_value) & (object_mask_2d > 0)).float()

    peak_index = int(torch.argmax(anomaly_map_2d).item())
    height, width = anomaly_map_2d.shape
    y = peak_index // width
    x = peak_index % width
    fallback = torch.zeros_like(anomaly_map_2d)
    fallback[y, x] = 1.0
    return fallback


def _masked_mean(tensor: torch.Tensor, region_mask: torch.Tensor) -> float:
    if tensor.ndim == 2:
        values = tensor[region_mask > 0]
    else:
        expanded_mask = region_mask.unsqueeze(0).expand(tensor.shape[0], -1, -1)
        values = tensor[expanded_mask > 0]
    if values.numel() == 0:
        return 0.0
    return float(values.float().mean().item())


def _normalize_contributions(raw_values: Dict[str, float]) -> Dict[str, float]:
    positive_values = {key: max(0.0, value) for key, value in raw_values.items()}
    total = sum(positive_values.values())
    if total <= 1e-8:
        equal_share = round(100.0 / max(len(positive_values), 1), 3)
        return {key: equal_share for key in positive_values}
    return {key: round(100.0 * value / total, 3) for key, value in positive_values.items()}


def build_retrieval_query(package: EvidencePackage) -> Dict[str, Any]:
    return asdict(package.retrieval_features)


def _strength_sentence(label: str, strength: float, *, strong: str, medium: str) -> str:
    if strength >= 0.6:
        detail = strong
    elif strength >= 0.3:
        detail = medium
    else:
        detail = "weak supporting signal"
    return f"{label} is {detail} (regional mean {strength:.3f})."


def _cross_modal_sentence(agreement_strength: float, inconsistency_strength: float) -> str:
    if agreement_strength >= 0.55 and inconsistency_strength <= 0.30:
        return (
            f"Cross-modal support is strong, with agreement {agreement_strength:.3f} "
            f"and low inconsistency {inconsistency_strength:.3f}."
        )
    if agreement_strength >= 0.35:
        return (
            f"Cross-modal support is moderate; agreement {agreement_strength:.3f} "
            f"exceeds inconsistency {inconsistency_strength:.3f}."
        )
    return (
        f"Cross-modal support is weak or mixed; agreement {agreement_strength:.3f}, "
        f"inconsistency {inconsistency_strength:.3f}."
    )


def _build_confidence_notes(
    *,
    quantification: MasiQuantification,
    agreement_strength: float,
    inconsistency_strength: float,
    global_descriptor_score: float,
) -> List[str]:
    notes = [f"Severity band: {quantification.status} ({quantification.severity_0_100:.1f}/100)."]
    if quantification.confidence_0_100 < 35.0:
        notes.append("Confidence is limited because the score remains close to the normal-reference boundary.")
    else:
        notes.append("Confidence is elevated because the score sits above the normal-reference tail.")

    if quantification.affected_area_pct < 1.0:
        notes.append("The predicted anomalous region is very small, so localisation should be interpreted cautiously.")
    if agreement_strength < inconsistency_strength:
        notes.append("Cross-modal evidence is mixed, so the explanation should emphasize uncertainty.")
    if global_descriptor_score > 0.5:
        notes.append(
            f"Sample-level global descriptor deviation is elevated ({global_descriptor_score:.3f}), reinforcing the anomaly hypothesis."
        )
    return notes


def _build_retrieval_features(
    *,
    category: str,
    defect_label: str,
    score_label: str,
    severity_0_100: float,
    confidence_0_100: float,
    affected_area_pct: float,
    evidence_breakdown: Dict[str, float],
    thermal_observations: List[str],
    geometric_observations: List[str],
    cross_modal_support: List[str],
    provenance: Dict[str, Any],
    global_descriptor_score: float,
) -> RetrievalFeatures:
    distribution = _distribution_label(affected_area_pct)
    dominant_modalities = _dominant_modalities(evidence_breakdown)
    sensor_profile_tags = _sensor_profile_tags(
        dominant_modalities=dominant_modalities,
        thermal_observations=thermal_observations,
        geometric_observations=geometric_observations,
        cross_modal_support=cross_modal_support,
        provenance=provenance,
    )
    defect_profile_tags = _defect_profile_tags(
        score_label=score_label,
        distribution=distribution,
        thermal_observations=thermal_observations,
        geometric_observations=geometric_observations,
        cross_modal_support=cross_modal_support,
        provenance=provenance,
        global_descriptor_score=global_descriptor_score,
    )
    confidence_band = _confidence_band(confidence_0_100)
    severity_band = _severity_band(severity_0_100)
    query_parts = [
        f"category={category}",
        f"defect_label={defect_label}",
        f"confidence={confidence_band}",
        f"severity={severity_band}",
        f"distribution={distribution}",
        f"dominant_modalities={','.join(dominant_modalities)}",
        f"sensor_profile={','.join(sensor_profile_tags)}",
        f"defect_profile={','.join(defect_profile_tags)}",
    ]
    return RetrievalFeatures(
        category=category,
        defect_label=defect_label,
        confidence_band=confidence_band,
        severity_band=severity_band,
        distribution=distribution,
        dominant_modalities=dominant_modalities,
        sensor_profile_tags=sensor_profile_tags,
        defect_profile_tags=defect_profile_tags,
        query_text=" | ".join(query_parts),
    )


def _confidence_band(value: float) -> str:
    if value >= 70.0:
        return "high"
    if value >= 40.0:
        return "medium"
    return "low"


def _severity_band(value: float) -> str:
    if value >= 70.0:
        return "high"
    if value >= 35.0:
        return "medium"
    return "low"


def _distribution_label(affected_area_pct: float) -> str:
    if affected_area_pct >= 20.0:
        return "broad"
    if affected_area_pct >= 7.5:
        return "moderately_concentrated"
    return "compact"


def _dominant_modalities(evidence_breakdown: Dict[str, float]) -> List[str]:
    modality_names = {
        "score_basis_pct": "score_basis",
        "thermal_pct": "thermal",
        "geometric_pct": "geometric",
        "crossmodal_pct": "crossmodal",
        "global_descriptor_pct": "global_descriptor",
    }
    ranked = sorted(
        evidence_breakdown.items(),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    dominant = [modality_names[key] for key, value in ranked if float(value) > 0.0][:2]
    return dominant or ["score_basis"]


def _sensor_profile_tags(
    *,
    dominant_modalities: List[str],
    thermal_observations: List[str],
    geometric_observations: List[str],
    cross_modal_support: List[str],
    provenance: Dict[str, Any],
) -> List[str]:
    tags: List[str] = []
    thermal_text = " ".join(thermal_observations).lower()
    geometric_text = " ".join(geometric_observations).lower()
    cross_text = " ".join(cross_modal_support).lower()

    if "thermal" in dominant_modalities or "hotspot" in thermal_text or "heating" in thermal_text:
        tags.append("thermal_hotspot")
    if "geometric" in dominant_modalities or "surface" in geometric_text or "shape" in geometric_text:
        tags.append("geometric_shift")
    if "crossmodal" in dominant_modalities:
        tags.append("cross_modal_signal")
    if "mixed" in cross_text or "inconsistency" in cross_text or float(provenance.get("raw_cross_inconsistency_score", 0.0)) > 0.0:
        tags.append("cross_inconsistency")

    gate_mode = str(provenance.get("internal_defect_gate_mode", "")).strip()
    if gate_mode == "cross_inconsistency":
        tags.append("internal_defect_signal")
    elif gate_mode == "thermal_hotspot":
        tags.append("internal_thermal_signal")
    if bool(provenance.get("internal_defect_gate_fired", False)):
        tags.append("gate_fired")
    return _dedupe_tags(tags)


def _defect_profile_tags(
    *,
    score_label: str,
    distribution: str,
    thermal_observations: List[str],
    geometric_observations: List[str],
    cross_modal_support: List[str],
    provenance: Dict[str, Any],
    global_descriptor_score: float,
) -> List[str]:
    tags: List[str] = [_slugify_tag(score_label), f"{distribution}_anomaly"]
    thermal_text = " ".join(thermal_observations).lower()
    geometric_text = " ".join(geometric_observations).lower()
    cross_text = " ".join(cross_modal_support).lower()

    if "strong" in thermal_text:
        tags.append("strong_thermal_support")
    elif "moderate" in thermal_text:
        tags.append("moderate_thermal_support")
    else:
        tags.append("weak_thermal_support")

    if "strong" in geometric_text or "clear" in geometric_text:
        tags.append("strong_geometric_support")
    elif "moderate" in geometric_text:
        tags.append("moderate_geometric_support")
    else:
        tags.append("weak_geometric_support")

    if "mixed" in cross_text or "inconsistency" in cross_text:
        tags.append("cross_modal_inconsistency")
    else:
        tags.append("cross_modal_agreement")

    gate_mode = str(provenance.get("internal_defect_gate_mode", "")).strip()
    if gate_mode == "cross_inconsistency":
        tags.append("internal_defect_profile")
    elif gate_mode == "thermal_hotspot":
        tags.append("thermal_internal_profile")
    if global_descriptor_score >= 0.5:
        tags.append("global_descriptor_drift")
    return _dedupe_tags(tags)


def _slugify_tag(value: str) -> str:
    return "_".join(part for part in str(value).lower().replace("-", "_").split() if part)


def _dedupe_tags(values: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        resolved = value.strip().lower()
        if not resolved or resolved in seen:
            continue
        seen.add(resolved)
        ordered.append(resolved)
    return ordered
