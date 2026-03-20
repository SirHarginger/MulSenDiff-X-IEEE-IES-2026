from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.embedding_dim // 2
        factor = math.log(10000) / max(half_dim - 1, 1)
        frequencies = torch.exp(
            torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -factor
        )
        args = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if embedding.shape[1] < self.embedding_dim:
            padding = torch.zeros((embedding.shape[0], self.embedding_dim - embedding.shape[1]), device=timesteps.device)
            embedding = torch.cat([embedding, padding], dim=1)
        return self.projection(embedding)


class ConditionalResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, condition_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels)
        self.activation = nn.SiLU()
        self.cond = nn.Linear(condition_dim, out_channels * 2)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, condition_vector: torch.Tensor) -> torch.Tensor:
        scale, shift = self.cond(condition_vector).chunk(2, dim=1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)

        out = self.conv1(x)
        out = self.norm1(out)
        out = out * (1.0 + torch.tanh(scale)) + shift
        out = self.activation(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.activation(out)
        return out + self.skip(x)


class ConditionedUNet(nn.Module):
    def __init__(self, rgb_channels: int = 3, base_channels: int = 32, condition_dim: int = 128) -> None:
        super().__init__()
        bottleneck_channels = base_channels * 4
        self.bottleneck = ConditionalResidualBlock(bottleneck_channels, bottleneck_channels, condition_dim)
        self.up1 = nn.ConvTranspose2d(bottleneck_channels, base_channels * 2, kernel_size=2, stride=2)
        self.block1 = ConditionalResidualBlock(base_channels * 4, base_channels * 2, condition_dim)
        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.block2 = ConditionalResidualBlock(base_channels * 2, base_channels, condition_dim)
        self.out = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, rgb_channels, kernel_size=1),
        )

    def forward(
        self,
        fused_bottleneck: torch.Tensor,
        skips: tuple[torch.Tensor, torch.Tensor],
        condition_vector: torch.Tensor,
    ) -> torch.Tensor:
        stem_skip, downsampled_skip = skips
        x = self.bottleneck(fused_bottleneck, condition_vector)
        x = self.up1(x)
        x = torch.cat([x, downsampled_skip], dim=1)
        x = self.block1(x, condition_vector)
        x = self.up2(x)
        x = torch.cat([x, stem_skip], dim=1)
        x = self.block2(x, condition_vector)
        return self.out(x)
