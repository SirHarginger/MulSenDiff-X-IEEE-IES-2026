from __future__ import annotations

from dataclasses import dataclass

import torch
from src.inference.localization import estimate_object_mask
from src.models.mulsendiffx import MulSenDiffX


@dataclass(frozen=True)
class AnomalyScoreOutputs:
    anomaly_score: torch.Tensor
    anomaly_map: torch.Tensor
    score_basis_map: torch.Tensor
    score_basis_label: str
    residual_map: torch.Tensor
    noise_error_map: torch.Tensor
    descriptor_support: torch.Tensor
    reconstructed_rgb: torch.Tensor


def descriptor_support_map(descriptor_maps: torch.Tensor) -> torch.Tensor:
    crossmodal_maps = descriptor_maps[:, -4:, :, :]
    return crossmodal_maps.mean(dim=1, keepdim=True)


def score_batch(
    model: MulSenDiffX,
    x_rgb: torch.Tensor,
    descriptor_maps: torch.Tensor,
    global_vector: torch.Tensor,
    *,
    score_mode: str = "noise_error",
    timestep: int = 200,
    descriptor_weight: float = 0.25,
    object_mask_threshold: float = 0.02,
    generator: torch.Generator | None = None,
) -> AnomalyScoreOutputs:
    if score_mode not in {"noise_error", "residual"}:
        raise ValueError(f"Unsupported score_mode={score_mode!r}. Expected 'noise_error' or 'residual'.")

    device = x_rgb.device
    timesteps = torch.full((x_rgb.shape[0],), timestep, device=device, dtype=torch.long)
    noise = torch.randn(x_rgb.shape, generator=generator, device=device, dtype=x_rgb.dtype)

    with torch.no_grad():
        noisy_rgb = model.q_sample(x_rgb, timesteps, noise)
        predicted_noise = model(noisy_rgb, descriptor_maps, global_vector, timesteps)
        reconstructed_rgb = model.predict_x0_from_noise(noisy_rgb, timesteps, predicted_noise)
        residual_map = (reconstructed_rgb - x_rgb).abs().mean(dim=1, keepdim=True)
        noise_error_map = (noise - predicted_noise).pow(2).mean(dim=1, keepdim=True)
        support = descriptor_support_map(descriptor_maps)
        object_mask = estimate_object_mask(x_rgb, threshold=object_mask_threshold)

        if score_mode == "noise_error":
            score_basis_map = noise_error_map
            score_basis_label = "Noise Error"
        else:
            score_basis_map = residual_map
            score_basis_label = "Residual"

        anomaly_map = (score_basis_map + descriptor_weight * support) * object_mask
        mask_pixels = object_mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
        anomaly_score = anomaly_map.sum(dim=(1, 2, 3)) / mask_pixels

    return AnomalyScoreOutputs(
        anomaly_score=anomaly_score,
        anomaly_map=anomaly_map,
        score_basis_map=score_basis_map,
        score_basis_label=score_basis_label,
        residual_map=residual_map,
        noise_error_map=noise_error_map,
        descriptor_support=support,
        reconstructed_rgb=reconstructed_rgb,
    )
