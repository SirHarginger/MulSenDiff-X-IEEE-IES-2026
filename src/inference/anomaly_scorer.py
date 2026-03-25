from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch.nn import functional as F

from src.inference.global_descriptor_scoring import (
    GlobalDescriptorCalibration,
    score_global_descriptors,
)
from src.inference.localization import estimate_object_mask, gaussian_smooth_map
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
    global_descriptor_score: torch.Tensor
    timesteps_used: tuple[int, ...]


def descriptor_support_map(
    descriptor_maps: torch.Tensor,
    support_maps: torch.Tensor | None = None,
) -> torch.Tensor:
    if support_maps is not None and support_maps.shape[1] >= 2:
        cross_ir_support = support_maps[:, -3:-2, :, :]
        cross_geo_support = support_maps[:, -2:-1, :, :]
        return (cross_ir_support + cross_geo_support + descriptor_maps[:, -1:, :, :]) / 3.0
    return descriptor_maps[:, -1:, :, :]


def score_batch(
    model: MulSenDiffX,
    x_rgb: torch.Tensor,
    descriptor_maps: torch.Tensor,
    support_maps: torch.Tensor | None,
    global_vector: torch.Tensor,
    category_indices: torch.Tensor | None = None,
    *,
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
    object_mask_threshold: float = 0.02,
    generator: torch.Generator | None = None,
) -> AnomalyScoreOutputs:
    if score_mode not in {"noise_error", "residual"}:
        raise ValueError(f"Unsupported score_mode={score_mode!r}. Expected 'noise_error' or 'residual'.")

    device = x_rgb.device
    if category_indices is None:
        category_indices = torch.zeros((x_rgb.shape[0],), dtype=torch.long, device=device)
    clean_latent = model.encode_rgb(x_rgb)
    noise = torch.randn(clean_latent.shape, generator=generator, device=device, dtype=clean_latent.dtype)
    resolved_lambda_residual: float
    resolved_lambda_noise: float
    resolved_lambda_descriptor_spatial: float
    resolved_lambda_descriptor_global: float
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
            predicted_noise = model(noisy_latent, descriptor_maps, global_vector, category_indices, timesteps)
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
        mask_pixels = object_mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
        spatial_score = anomaly_map.sum(dim=(1, 2, 3)) / mask_pixels
        if global_descriptor_calibration is not None and resolved_lambda_descriptor_global > 0.0:
            if isinstance(global_descriptor_calibration, Mapping):
                if categories is None:
                    raise ValueError("Category names are required when using per-category global descriptor calibration.")
                scores = []
                for sample_index, category in enumerate(categories):
                    calibration = global_descriptor_calibration.get(str(category))
                    if calibration is None:
                        raise KeyError(f"Missing global descriptor calibration for category={category!r}.")
                    sample_score = score_global_descriptors(
                        global_vector[sample_index : sample_index + 1],
                        calibration,
                    )
                    scores.append(sample_score.squeeze(0))
                global_score = torch.stack(scores, dim=0) if scores else torch.zeros_like(spatial_score)
            else:
                global_score = score_global_descriptors(global_vector, global_descriptor_calibration)
        else:
            global_score = torch.zeros_like(spatial_score)
        anomaly_score = spatial_score + resolved_lambda_descriptor_global * global_score

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
        global_descriptor_score=global_score,
        timesteps_used=tuple(timestep_values),
    )
