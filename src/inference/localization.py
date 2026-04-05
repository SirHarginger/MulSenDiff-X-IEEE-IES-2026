from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from src.data_loader import SUPPORT_DESCRIPTOR_ORDER


@dataclass(frozen=True)
class LocalizationCalibration:
    category: str
    sample_count: int
    value_count: int
    quantile: float
    threshold: float
    q95: float
    q99: float
    median_object_pixels: float
    min_region_area_fraction: float
    min_region_pixels: int
    basis: str = "anomaly_map"
    object_mask_threshold: float = 0.02
    object_mask_dilation_kernel_size: int = 9
    mask_closing_kernel_size: int = 0
    fill_holes: bool = False
    tuning_source: str = "calibration_good"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "LocalizationCalibration":
        return cls(
            category=str(payload["category"]),
            sample_count=int(payload["sample_count"]),
            value_count=int(payload["value_count"]),
            quantile=float(payload["quantile"]),
            threshold=float(payload["threshold"]),
            q95=float(payload["q95"]),
            q99=float(payload["q99"]),
            median_object_pixels=float(payload["median_object_pixels"]),
            min_region_area_fraction=float(payload["min_region_area_fraction"]),
            min_region_pixels=int(payload["min_region_pixels"]),
            basis=str(payload.get("basis", "anomaly_map")),
            object_mask_threshold=float(payload.get("object_mask_threshold", 0.02)),
            object_mask_dilation_kernel_size=int(payload.get("object_mask_dilation_kernel_size", 9)),
            mask_closing_kernel_size=int(payload.get("mask_closing_kernel_size", 0)),
            fill_holes=bool(payload.get("fill_holes", False)),
            tuning_source=str(payload.get("tuning_source", "calibration_good")),
        )


def normalize_anomaly_map(anomaly_map: torch.Tensor) -> torch.Tensor:
    minimum = anomaly_map.amin(dim=(2, 3), keepdim=True)
    maximum = anomaly_map.amax(dim=(2, 3), keepdim=True)
    return (anomaly_map - minimum) / (maximum - minimum + 1e-6)


def estimate_object_mask(
    x_rgb: torch.Tensor,
    *,
    threshold: float = 0.02,
    dilation_kernel_size: int = 9,
    support_maps: torch.Tensor | None = None,
    support_threshold: float = 0.05,
    min_mask_pixels: int = 16,
) -> torch.Tensor:
    brightness = x_rgb.mean(dim=1, keepdim=True)
    rgb_mask = (brightness > threshold).float()
    mask = rgb_mask
    if support_maps is not None and support_maps.shape[1] >= len(SUPPORT_DESCRIPTOR_ORDER):
        density_index = SUPPORT_DESCRIPTOR_ORDER.index("pc_density")
        density_mask = (support_maps[:, density_index : density_index + 1, :, :] > support_threshold).float()
        rgb_pixels = rgb_mask.sum(dim=(1, 2, 3), keepdim=True)
        density_pixels = density_mask.sum(dim=(1, 2, 3), keepdim=True)
        use_density = rgb_pixels < float(min_mask_pixels)
        mask = torch.where(use_density, density_mask, rgb_mask)
        mask = torch.where((mask.sum(dim=(1, 2, 3), keepdim=True) < float(min_mask_pixels)), rgb_mask + density_mask, mask)
    if dilation_kernel_size > 1:
        padding = dilation_kernel_size // 2
        mask = F.max_pool2d(mask, kernel_size=dilation_kernel_size, stride=1, padding=padding)
    return (mask > 0).float()


def gaussian_smooth_map(anomaly_map: torch.Tensor, *, sigma: float = 3.0) -> torch.Tensor:
    if sigma <= 0.0:
        return anomaly_map
    radius = max(int(round(3.0 * sigma)), 1)
    kernel_size = radius * 2 + 1
    coords = torch.arange(kernel_size, dtype=anomaly_map.dtype, device=anomaly_map.device) - radius
    kernel_1d = torch.exp(-(coords**2) / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum().clamp_min(1e-6)
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel_2d = kernel_2d / kernel_2d.sum().clamp_min(1e-6)
    kernel = kernel_2d.view(1, 1, kernel_size, kernel_size).expand(anomaly_map.shape[1], 1, kernel_size, kernel_size)
    return F.conv2d(
        anomaly_map,
        kernel,
        padding=radius,
        groups=anomaly_map.shape[1],
    )


def threshold_anomaly_map(
    anomaly_map: torch.Tensor,
    threshold: float | None = 0.5,
    *,
    percentile: float | None = None,
    object_mask: torch.Tensor | None = None,
    normalized: bool = False,
) -> torch.Tensor:
    maps = anomaly_map if normalized else normalize_anomaly_map(anomaly_map)
    predicted_masks = []

    for sample_index in range(maps.shape[0]):
        sample_map = maps[sample_index : sample_index + 1]
        sample_mask = None
        if object_mask is not None:
            sample_mask = object_mask[sample_index : sample_index + 1]
            sample_map = sample_map * sample_mask

        if percentile is not None:
            values = sample_map[sample_mask.bool()] if sample_mask is not None else sample_map.reshape(-1)
            if values.numel() == 0:
                cutoff = torch.tensor(1.0, dtype=sample_map.dtype, device=sample_map.device)
            else:
                cutoff = torch.quantile(values, percentile)
        else:
            cutoff = torch.tensor(float(threshold if threshold is not None else 0.5), dtype=sample_map.dtype, device=sample_map.device)

        predicted = (sample_map >= cutoff).float()
        if sample_mask is not None:
            predicted = predicted * sample_mask
        predicted_masks.append(predicted)

    return torch.cat(predicted_masks, dim=0)


def fit_localization_calibration(
    values: torch.Tensor,
    *,
    category: str,
    object_pixel_counts: torch.Tensor | None = None,
    quantile: float = 0.995,
    min_region_area_fraction: float = 0.002,
    min_region_pixels_floor: int = 4,
    sample_count: int | None = None,
    basis: str = "anomaly_map",
    object_mask_threshold: float = 0.02,
    object_mask_dilation_kernel_size: int = 9,
    mask_closing_kernel_size: int = 0,
    fill_holes: bool = False,
    tuning_source: str = "calibration_good",
) -> LocalizationCalibration:
    flattened = values.detach().flatten().float()
    if flattened.numel() == 0:
        flattened = torch.tensor([0.0], dtype=torch.float32)

    object_counts = (
        object_pixel_counts.detach().flatten().float()
        if object_pixel_counts is not None and object_pixel_counts.numel() > 0
        else torch.tensor([0.0], dtype=torch.float32)
    )
    median_object_pixels = float(torch.median(object_counts).item()) if object_counts.numel() > 0 else 0.0
    resolved_min_region_pixels = max(
        int(min_region_pixels_floor),
        int(math.ceil(max(median_object_pixels, 0.0) * max(min_region_area_fraction, 0.0))),
    )
    resolved_quantile = min(max(float(quantile), 0.0), 1.0)

    return LocalizationCalibration(
        category=category,
        sample_count=int(sample_count if sample_count is not None else object_counts.numel()),
        value_count=int(flattened.numel()),
        quantile=resolved_quantile,
        threshold=float(torch.quantile(flattened, resolved_quantile).item()),
        q95=float(torch.quantile(flattened, 0.95).item()),
        q99=float(torch.quantile(flattened, 0.99).item()),
        median_object_pixels=median_object_pixels,
        min_region_area_fraction=float(min_region_area_fraction),
        min_region_pixels=resolved_min_region_pixels,
        basis=str(basis),
        object_mask_threshold=float(object_mask_threshold),
        object_mask_dilation_kernel_size=int(max(object_mask_dilation_kernel_size, 1)),
        mask_closing_kernel_size=int(max(mask_closing_kernel_size, 0)),
        fill_holes=bool(fill_holes),
        tuning_source=str(tuning_source),
    )


def apply_localization_calibration(
    anomaly_map: torch.Tensor,
    *,
    object_mask: torch.Tensor | None,
    calibration: LocalizationCalibration,
    normalized: bool = False,
) -> torch.Tensor:
    predicted = threshold_anomaly_map(
        anomaly_map,
        threshold=calibration.threshold,
        percentile=None,
        object_mask=object_mask,
        normalized=normalized,
    )
    return remove_small_connected_components(
        predicted,
        object_mask=object_mask,
        min_region_pixels=calibration.min_region_pixels,
        closing_kernel_size=calibration.mask_closing_kernel_size,
        fill_holes=calibration.fill_holes,
    )


def remove_small_connected_components(
    predicted_mask: torch.Tensor,
    *,
    object_mask: torch.Tensor | None = None,
    min_region_pixels: int = 0,
    closing_kernel_size: int = 0,
    fill_holes: bool = False,
) -> torch.Tensor:
    masks = predicted_mask.detach().cpu().float()
    if masks.ndim == 3:
        masks = masks.unsqueeze(0)
    object_masks = None
    if object_mask is not None:
        object_masks = object_mask.detach().cpu().float()
        if object_masks.ndim == 3:
            object_masks = object_masks.unsqueeze(0)

    cleaned_masks: list[torch.Tensor] = []
    for sample_index in range(masks.shape[0]):
        sample_mask = masks[sample_index].squeeze(0).numpy() > 0
        sample_object_mask = None
        if object_masks is not None:
            sample_object_mask = object_masks[sample_index].squeeze(0).numpy() > 0
            sample_mask &= sample_object_mask

        if closing_kernel_size > 1:
            structure = np.ones((int(closing_kernel_size), int(closing_kernel_size)), dtype=bool)
            sample_mask = ndimage.binary_closing(sample_mask, structure=structure)

        if fill_holes:
            sample_mask = ndimage.binary_fill_holes(sample_mask)

        if sample_object_mask is not None:
            sample_mask &= sample_object_mask

        if min_region_pixels > 1:
            labeled, component_count = ndimage.label(sample_mask)
            filtered = np.zeros_like(sample_mask, dtype=bool)
            for component_index in range(1, component_count + 1):
                component = labeled == component_index
                if int(component.sum()) >= int(min_region_pixels):
                    filtered |= component
            sample_mask = filtered

        cleaned_masks.append(torch.from_numpy(sample_mask.astype(np.float32)).unsqueeze(0))
    return torch.stack(cleaned_masks, dim=0)
