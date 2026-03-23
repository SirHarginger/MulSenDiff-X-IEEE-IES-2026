from __future__ import annotations

import torch
from torch import nn


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
            nn.SiLU(),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.skip(x)


class RGBAutoencoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 3,
        base_channels: int = 32,
        latent_channels: int = 4,
    ) -> None:
        super().__init__()
        hidden_channels = base_channels * 4
        self.encoder = nn.Sequential(
            ConvNormAct(in_channels, base_channels),
            ConvNormAct(base_channels, base_channels * 2, stride=2),
            ConvNormAct(base_channels * 2, hidden_channels, stride=2),
            ConvNormAct(hidden_channels, hidden_channels, stride=2),
        )
        self.to_latent = nn.Conv2d(hidden_channels, latent_channels, kernel_size=1)

        self.from_latent = nn.Sequential(
            nn.Conv2d(latent_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, hidden_channels), num_channels=hidden_channels),
            nn.SiLU(),
        )
        self.up1 = nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=2, stride=2)
        self.dec1 = ConvNormAct(hidden_channels, hidden_channels)
        self.up2 = nn.ConvTranspose2d(hidden_channels, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = ConvNormAct(base_channels * 2, base_channels * 2)
        self.up3 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec3 = ConvNormAct(base_channels, base_channels)
        self.output = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, base_channels), num_channels=base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, in_channels, kernel_size=1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_latent(self.encoder(x))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        x = self.from_latent(latent)
        x = self.dec1(self.up1(x))
        x = self.dec2(self.up2(x))
        x = self.dec3(self.up3(x))
        return self.output(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


class SpatialDescriptorEncoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        base_channels: int = 32,
        context_channels: int = 128,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            ConvNormAct(in_channels, base_channels),
            ConvNormAct(base_channels, base_channels * 2, stride=2),
            ConvNormAct(base_channels * 2, base_channels * 4, stride=2),
            ConvNormAct(base_channels * 4, context_channels, stride=2),
        )

    def forward(self, descriptor_maps: torch.Tensor) -> torch.Tensor:
        return self.network(descriptor_maps)


class GlobalConditionEncoder(nn.Module):
    def __init__(self, in_features: int, embedding_dim: int = 128) -> None:
        super().__init__()
        hidden_dim = max(64, embedding_dim // 2)
        self.network = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, global_vector: torch.Tensor) -> torch.Tensor:
        return self.network(global_vector)
