from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

import torch


@dataclass(frozen=True)
class MasiCalibration:
    category: str
    score_mode: str
    sample_count: int
    mean: float
    std: float
    q50: float
    q90: float
    q95: float
    q99: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RegionSummary:
    area_pixels: int
    area_fraction: float
    bbox_xyxy: list[int]
    centroid_xy: list[float]
    mean_score: float
    peak_score: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MasiQuantification:
    raw_score: float
    severity_0_100: float
    confidence_0_100: float
    status: str
    affected_area_pct: float
    peak_anomaly_0_100: float
    top_region: RegionSummary

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["top_region"] = self.top_region.to_dict()
        return payload


def fit_masi_calibration(
    scores: torch.Tensor,
    *,
    category: str,
    score_mode: str,
) -> MasiCalibration:
    flattened = scores.detach().flatten().float()
    if flattened.numel() == 0:
        flattened = torch.tensor([0.0], dtype=torch.float32)

    return MasiCalibration(
        category=category,
        score_mode=score_mode,
        sample_count=int(flattened.numel()),
        mean=float(flattened.mean().item()),
        std=float(flattened.std(unbiased=False).item()),
        q50=float(torch.quantile(flattened, 0.50).item()),
        q90=float(torch.quantile(flattened, 0.90).item()),
        q95=float(torch.quantile(flattened, 0.95).item()),
        q99=float(torch.quantile(flattened, 0.99).item()),
    )


def quantify_masi(
    *,
    raw_score: float,
    calibration: MasiCalibration,
    normalized_anomaly_map: torch.Tensor,
    object_mask: torch.Tensor,
    predicted_mask: torch.Tensor,
) -> MasiQuantification:
    severity = _bounded_scale(raw_score, lower=calibration.q50, upper=calibration.q99)
    confidence = _bounded_scale(raw_score, lower=calibration.q95, upper=calibration.q99)
    top_region = summarize_top_region(
        normalized_anomaly_map=normalized_anomaly_map,
        object_mask=object_mask,
        predicted_mask=predicted_mask,
    )

    object_pixels = int((object_mask > 0).sum().item())
    predicted_pixels = int((predicted_mask > 0).sum().item())
    affected_area_pct = 100.0 * predicted_pixels / max(object_pixels, 1)
    peak_anomaly = 100.0 * top_region.peak_score

    return MasiQuantification(
        raw_score=float(raw_score),
        severity_0_100=severity,
        confidence_0_100=confidence,
        status=severity_status(severity),
        affected_area_pct=affected_area_pct,
        peak_anomaly_0_100=peak_anomaly,
        top_region=top_region,
    )


def severity_status(severity_0_100: float) -> str:
    if severity_0_100 < 25.0:
        return "Normal"
    if severity_0_100 < 50.0:
        return "Mild deviation"
    if severity_0_100 < 75.0:
        return "Suspicious"
    if severity_0_100 < 90.0:
        return "High anomaly"
    return "Critical anomaly"


def summarize_top_region(
    *,
    normalized_anomaly_map: torch.Tensor,
    object_mask: torch.Tensor,
    predicted_mask: torch.Tensor,
) -> RegionSummary:
    anomaly_map = normalized_anomaly_map.detach().float().squeeze()
    object_mask_2d = object_mask.detach().float().squeeze()
    predicted_mask_2d = predicted_mask.detach().float().squeeze()

    region_mask = predicted_mask_2d > 0
    if not region_mask.any():
        fallback_mask = object_mask_2d > 0
        if not fallback_mask.any():
            peak_index = int(torch.argmax(anomaly_map).item())
            height, width = anomaly_map.shape
            y = peak_index // width
            x = peak_index % width
            region_mask = torch.zeros_like(anomaly_map, dtype=torch.bool)
            region_mask[y, x] = True
        else:
            peak_value = anomaly_map[fallback_mask].amax()
            region_mask = (anomaly_map >= peak_value) & fallback_mask

    coordinates = torch.nonzero(region_mask, as_tuple=False)
    ys = coordinates[:, 0]
    xs = coordinates[:, 1]
    values = anomaly_map[region_mask]
    object_pixels = int((object_mask_2d > 0).sum().item())

    return RegionSummary(
        area_pixels=int(region_mask.sum().item()),
        area_fraction=float(region_mask.sum().item() / max(object_pixels, 1)),
        bbox_xyxy=[int(xs.min().item()), int(ys.min().item()), int(xs.max().item()), int(ys.max().item())],
        centroid_xy=[float(xs.float().mean().item()), float(ys.float().mean().item())],
        mean_score=float(values.mean().item()),
        peak_score=float(values.max().item()),
    )


def _bounded_scale(value: float, *, lower: float, upper: float) -> float:
    if upper <= lower + 1e-8:
        return 100.0 if value > upper else 0.0
    scaled = 100.0 * (value - lower) / (upper - lower)
    return float(min(100.0, max(0.0, scaled)))
