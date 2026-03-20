from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.models.diffusion_unet import ConditionedUNet, SinusoidalTimeEmbedding
from src.models.encoders import GlobalConditionEncoder, RGBAppearanceEncoder, SpatialDescriptorEncoder
from src.models.fusion import DescriptorConditionFusion
from src.training.losses import mse_loss


@dataclass(frozen=True)
class DiffusionOutputs:
    predicted_noise: torch.Tensor
    noisy_rgb: torch.Tensor
    target_noise: torch.Tensor
    timesteps: torch.Tensor
    loss: torch.Tensor


class MulSenDiffX(nn.Module):
    def __init__(
        self,
        *,
        rgb_channels: int = 3,
        descriptor_channels: int = 13,
        global_dim: int = 21,
        base_channels: int = 32,
        global_embedding_dim: int = 64,
        time_embedding_dim: int = 64,
        diffusion_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ) -> None:
        super().__init__()
        self.rgb_encoder = RGBAppearanceEncoder(in_channels=rgb_channels, base_channels=base_channels)
        self.descriptor_encoder = SpatialDescriptorEncoder(in_channels=descriptor_channels, base_channels=base_channels)
        self.global_encoder = GlobalConditionEncoder(global_dim, embedding_dim=global_embedding_dim)
        self.time_encoder = SinusoidalTimeEmbedding(time_embedding_dim)
        condition_dim = global_embedding_dim + time_embedding_dim
        self.fusion = DescriptorConditionFusion(base_channels * 4, condition_dim)
        self.unet = ConditionedUNet(
            rgb_channels=rgb_channels,
            base_channels=base_channels,
            condition_dim=condition_dim,
        )

        betas = torch.linspace(beta_start, beta_end, diffusion_steps, dtype=torch.float32)
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

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.diffusion_steps, (batch_size,), device=device, dtype=torch.long)

    def q_sample(self, clean_rgb: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        alpha = self.sqrt_alpha_bars[timesteps].view(-1, 1, 1, 1)
        sigma = self.sqrt_one_minus_alpha_bars[timesteps].view(-1, 1, 1, 1)
        return alpha * clean_rgb + sigma * noise

    def predict_x0_from_noise(
        self,
        noisy_rgb: torch.Tensor,
        timesteps: torch.Tensor,
        predicted_noise: torch.Tensor,
    ) -> torch.Tensor:
        alpha = self.sqrt_alpha_bars[timesteps].view(-1, 1, 1, 1)
        sigma = self.sqrt_one_minus_alpha_bars[timesteps].view(-1, 1, 1, 1)
        return (noisy_rgb - sigma * predicted_noise) / alpha.clamp_min(1e-6)

    def forward(
        self,
        noisy_rgb: torch.Tensor,
        descriptor_maps: torch.Tensor,
        global_vector: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        rgb_features = self.rgb_encoder(noisy_rgb)
        descriptor_features = self.descriptor_encoder(descriptor_maps)
        global_embedding = self.global_encoder(global_vector)
        time_embedding = self.time_encoder(timesteps)
        condition_vector = torch.cat([global_embedding, time_embedding], dim=1)
        fused_bottleneck = self.fusion(
            rgb_bottleneck=rgb_features.bottleneck,
            descriptor_bottleneck=descriptor_features.bottleneck,
            condition_vector=condition_vector,
        )
        return self.unet(fused_bottleneck, tuple(rgb_features.skips), condition_vector)

    def training_outputs(
        self,
        clean_rgb: torch.Tensor,
        descriptor_maps: torch.Tensor,
        global_vector: torch.Tensor,
        *,
        timesteps: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> DiffusionOutputs:
        if timesteps is None:
            timesteps = self.sample_timesteps(clean_rgb.shape[0], clean_rgb.device)
        if noise is None:
            noise = torch.randn(clean_rgb.shape, generator=generator, device=clean_rgb.device, dtype=clean_rgb.dtype)
        noisy_rgb = self.q_sample(clean_rgb, timesteps, noise)
        predicted_noise = self.forward(noisy_rgb, descriptor_maps, global_vector, timesteps)
        loss = mse_loss(predicted_noise, noise)
        return DiffusionOutputs(
            predicted_noise=predicted_noise,
            noisy_rgb=noisy_rgb,
            target_noise=noise,
            timesteps=timesteps,
            loss=loss,
        )
