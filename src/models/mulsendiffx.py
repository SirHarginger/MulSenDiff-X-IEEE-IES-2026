from __future__ import annotations

from dataclasses import dataclass

import math
import torch
from torch import nn

from src.models.diffusion_unet import ConditionedUNet, SinusoidalTimeEmbedding
from src.models.encoders import GlobalConditionEncoder, RGBAutoencoder, SpatialDescriptorEncoder
from src.training.losses import mixed_reconstruction_loss, mse_loss, sobel_edge_loss


@dataclass(frozen=True)
class DiffusionOutputs:
    predicted_noise: torch.Tensor
    noisy_latent: torch.Tensor
    target_noise: torch.Tensor
    timesteps: torch.Tensor
    loss: torch.Tensor
    noise_loss: torch.Tensor
    autoencoder_reconstruction_loss: torch.Tensor
    diffusion_reconstruction_loss: torch.Tensor
    edge_reconstruction_loss: torch.Tensor
    clean_latent: torch.Tensor
    predicted_clean_latent: torch.Tensor
    reconstructed_rgb: torch.Tensor
    autoencoded_rgb: torch.Tensor


def _cosine_beta_schedule(steps: int, *, s: float = 0.008, max_beta: float = 0.999) -> torch.Tensor:
    time = torch.linspace(0, steps, steps + 1, dtype=torch.float32)
    alpha_bar = torch.cos(((time / steps) + s) / (1 + s) * math.pi * 0.5).pow(2)
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])
    return betas.clamp(1e-5, max_beta)


def _linear_beta_schedule(steps: int, beta_start: float, beta_end: float) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, steps, dtype=torch.float32)


class MulSenDiffX(nn.Module):
    def __init__(
        self,
        *,
        rgb_channels: int = 3,
        descriptor_channels: int = 7,
        global_dim: int = 10,
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
    ) -> None:
        super().__init__()
        self.rgb_autoencoder = RGBAutoencoder(
            in_channels=rgb_channels,
            base_channels=base_channels,
            latent_channels=latent_channels,
        )
        self.descriptor_encoder = SpatialDescriptorEncoder(
            in_channels=descriptor_channels,
            base_channels=base_channels,
            context_channels=global_embedding_dim,
        )
        self.global_encoder = GlobalConditionEncoder(global_dim, embedding_dim=global_embedding_dim)
        self.category_embedding = nn.Embedding(max(num_categories, 1), category_embedding_dim)
        self.enable_category_modality_gating = bool(enable_category_modality_gating)
        self.spatial_descriptor_channels = int(spatial_descriptor_channels)
        self.support_descriptor_channels = int(support_descriptor_channels)
        self.total_descriptor_channels = self.spatial_descriptor_channels + self.support_descriptor_channels
        if self.enable_category_modality_gating:
            hidden_dim = max(32, category_embedding_dim)
            self.category_modality_gate = nn.Sequential(
                nn.Linear(category_embedding_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, self.total_descriptor_channels),
            )
        self.category_global_fusion = nn.Sequential(
            nn.Linear(global_embedding_dim + category_embedding_dim, global_embedding_dim),
            nn.LayerNorm(global_embedding_dim),
            nn.GELU(),
            nn.Linear(global_embedding_dim, global_embedding_dim),
        )
        self.time_encoder = SinusoidalTimeEmbedding(time_embedding_dim)
        self.unet = ConditionedUNet(
            latent_channels=latent_channels,
            base_channels=base_channels,
            context_channels=global_embedding_dim,
            global_dim=global_embedding_dim,
            time_dim=time_embedding_dim,
            attention_heads=attention_heads,
        )
        self.global_embedding_dim = global_embedding_dim
        self.time_embedding_dim = time_embedding_dim
        self.category_embedding_dim = category_embedding_dim
        self.num_categories = max(int(num_categories), 1)
        self.latent_channels = latent_channels
        self.noise_schedule = noise_schedule
        self.autoencoder_reconstruction_weight = autoencoder_reconstruction_weight
        self.diffusion_reconstruction_weight = diffusion_reconstruction_weight
        self.edge_reconstruction_weight = edge_reconstruction_weight

        if noise_schedule == "cosine":
            betas = _cosine_beta_schedule(diffusion_steps)
        elif noise_schedule == "linear":
            betas = _linear_beta_schedule(diffusion_steps, beta_start=beta_start, beta_end=beta_end)
        else:
            raise ValueError(f"Unsupported noise_schedule={noise_schedule!r}. Expected 'cosine' or 'linear'.")

        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))

    @property
    def diffusion_steps(self) -> int:
        return int(self.betas.shape[0])

    def encode_rgb(self, clean_rgb: torch.Tensor) -> torch.Tensor:
        return self.rgb_autoencoder.encode(clean_rgb)

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return self.rgb_autoencoder.decode(latent)

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.diffusion_steps, (batch_size,), device=device, dtype=torch.long)

    def q_sample(self, clean_latent: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        alpha = self.sqrt_alpha_bars[timesteps].view(-1, 1, 1, 1)
        sigma = self.sqrt_one_minus_alpha_bars[timesteps].view(-1, 1, 1, 1)
        return alpha * clean_latent + sigma * noise

    def predict_x0_from_noise(
        self,
        noisy_latent: torch.Tensor,
        timesteps: torch.Tensor,
        predicted_noise: torch.Tensor,
    ) -> torch.Tensor:
        alpha = self.sqrt_alpha_bars[timesteps].view(-1, 1, 1, 1)
        sigma = self.sqrt_one_minus_alpha_bars[timesteps].view(-1, 1, 1, 1)
        return (noisy_latent - sigma * predicted_noise) / alpha.clamp_min(1e-6)

    def forward(
        self,
        noisy_latent: torch.Tensor,
        descriptor_maps: torch.Tensor,
        global_vector: torch.Tensor,
        category_indices: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        support_maps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if timesteps is None:
            if category_indices is None:
                raise ValueError("timesteps are required for MulSenDiffX.forward().")
            timesteps = category_indices
            category_indices = None
        if category_indices is None:
            category_indices = torch.zeros((noisy_latent.shape[0],), dtype=torch.long, device=noisy_latent.device)
        category_embedding = self.category_embedding(category_indices.clamp_min(0))
        if self.enable_category_modality_gating:
            modality_gates = torch.sigmoid(self.category_modality_gate(category_embedding))
            if support_maps is None and descriptor_maps.shape[1] == self.total_descriptor_channels:
                descriptor_maps = descriptor_maps * modality_gates.unsqueeze(-1).unsqueeze(-1)
            elif (
                support_maps is not None
                and descriptor_maps.shape[1] == self.spatial_descriptor_channels
                and support_maps.shape[1] == self.support_descriptor_channels
            ):
                spatial_gates = modality_gates[:, : self.spatial_descriptor_channels].unsqueeze(-1).unsqueeze(-1)
                support_gates = modality_gates[:, self.spatial_descriptor_channels :].unsqueeze(-1).unsqueeze(-1)
                descriptor_maps = descriptor_maps * spatial_gates
                support_maps = support_maps * support_gates
        expected_descriptor_channels = int(self.descriptor_encoder.network[0].block[0].in_channels)
        resolved_descriptor_maps = descriptor_maps
        if descriptor_maps.shape[1] != expected_descriptor_channels:
            if (
                support_maps is not None
                and descriptor_maps.shape[1] + support_maps.shape[1] == expected_descriptor_channels
            ):
                resolved_descriptor_maps = torch.cat([descriptor_maps, support_maps], dim=1)
            else:
                raise ValueError(
                    "MulSenDiffX received descriptor maps with incompatible channel count: "
                    f"got {descriptor_maps.shape[1]}, expected {expected_descriptor_channels}."
                )
        descriptor_context = self.descriptor_encoder(resolved_descriptor_maps)
        global_embedding = self.global_encoder(global_vector)
        fused_global_embedding = self.category_global_fusion(
            torch.cat([global_embedding, category_embedding], dim=1)
        )
        time_embedding = self.time_encoder(timesteps)
        return self.unet(noisy_latent, descriptor_context, fused_global_embedding, time_embedding)

    def category_modality_gates_for_indices(self, category_indices: torch.Tensor) -> torch.Tensor:
        if not self.enable_category_modality_gating:
            raise RuntimeError("Category modality gating is disabled for this checkpoint.")
        category_embedding = self.category_embedding(category_indices.clamp_min(0))
        return torch.sigmoid(self.category_modality_gate(category_embedding))

    def training_outputs(
        self,
        clean_rgb: torch.Tensor,
        descriptor_maps: torch.Tensor,
        global_vector: torch.Tensor,
        category_indices: torch.Tensor | None = None,
        *,
        timesteps: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        support_maps: torch.Tensor | None = None,
    ) -> DiffusionOutputs:
        clean_latent = self.encode_rgb(clean_rgb)
        if category_indices is None:
            category_indices = torch.zeros((clean_rgb.shape[0],), dtype=torch.long, device=clean_rgb.device)
        if timesteps is None:
            timesteps = self.sample_timesteps(clean_rgb.shape[0], clean_rgb.device)
        if noise is None:
            noise = torch.randn(clean_latent.shape, generator=generator, device=clean_rgb.device, dtype=clean_latent.dtype)
        noisy_latent = self.q_sample(clean_latent, timesteps, noise)
        predicted_noise = self.forward(
            noisy_latent,
            descriptor_maps,
            global_vector,
            category_indices,
            timesteps,
            support_maps=support_maps,
        )
        predicted_clean_latent = self.predict_x0_from_noise(noisy_latent, timesteps, predicted_noise)
        reconstructed_rgb = self.decode_latent(predicted_clean_latent)
        autoencoded_rgb = self.decode_latent(clean_latent)
        noise_loss = mse_loss(predicted_noise, noise)
        autoencoder_reconstruction_loss = mixed_reconstruction_loss(autoencoded_rgb, clean_rgb)
        diffusion_reconstruction_loss = mixed_reconstruction_loss(reconstructed_rgb, clean_rgb)
        edge_reconstruction_loss = 0.5 * (
            sobel_edge_loss(autoencoded_rgb, clean_rgb) + sobel_edge_loss(reconstructed_rgb, clean_rgb)
        )
        loss = (
            noise_loss
            + self.autoencoder_reconstruction_weight * autoencoder_reconstruction_loss
            + self.diffusion_reconstruction_weight * diffusion_reconstruction_loss
            + self.edge_reconstruction_weight * edge_reconstruction_loss
        )
        return DiffusionOutputs(
            predicted_noise=predicted_noise,
            noisy_latent=noisy_latent,
            target_noise=noise,
            timesteps=timesteps,
            loss=loss,
            noise_loss=noise_loss,
            autoencoder_reconstruction_loss=autoencoder_reconstruction_loss,
            diffusion_reconstruction_loss=diffusion_reconstruction_loss,
            edge_reconstruction_loss=edge_reconstruction_loss,
            clean_latent=clean_latent,
            predicted_clean_latent=predicted_clean_latent,
            reconstructed_rgb=reconstructed_rgb,
            autoencoded_rgb=autoencoded_rgb,
        )
