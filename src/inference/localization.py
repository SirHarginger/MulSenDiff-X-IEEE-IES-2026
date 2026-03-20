from __future__ import annotations

import torch
import torch.nn.functional as F


def normalize_anomaly_map(anomaly_map: torch.Tensor) -> torch.Tensor:
    minimum = anomaly_map.amin(dim=(2, 3), keepdim=True)
    maximum = anomaly_map.amax(dim=(2, 3), keepdim=True)
    return (anomaly_map - minimum) / (maximum - minimum + 1e-6)


def estimate_object_mask(
    x_rgb: torch.Tensor,
    *,
    threshold: float = 0.02,
    dilation_kernel_size: int = 9,
) -> torch.Tensor:
    brightness = x_rgb.mean(dim=1, keepdim=True)
    mask = (brightness > threshold).float()
    if dilation_kernel_size > 1:
        padding = dilation_kernel_size // 2
        mask = F.max_pool2d(mask, kernel_size=dilation_kernel_size, stride=1, padding=padding)
    return (mask > 0).float()


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
