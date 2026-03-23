from __future__ import annotations

import torch
import torch.nn.functional as F

from src.data_loader import SUPPORT_DESCRIPTOR_ORDER


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
