from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from src.models.fusion import DescriptorConditionFusion


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
            padding = torch.zeros(
                (embedding.shape[0], self.embedding_dim - embedding.shape[1]),
                device=timesteps.device,
            )
            embedding = torch.cat([embedding, padding], dim=1)
        return self.projection(embedding)


class TimeConditionedResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels)
        self.time_modulation = nn.Linear(time_dim, out_channels * 2)
        self.activation = nn.SiLU()
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        scale, shift = self.time_modulation(time_embedding).chunk(2, dim=1)
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


def _resize_like(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if x.shape[-2:] == reference.shape[-2:]:
        return x
    return F.interpolate(x, size=reference.shape[-2:], mode="bilinear", align_corners=False)


class ConditionedUNet(nn.Module):
    def __init__(
        self,
        *,
        latent_channels: int = 4,
        base_channels: int = 32,
        context_channels: int = 128,
        global_dim: int = 128,
        time_dim: int = 128,
        attention_heads: int = 4,
    ) -> None:
        super().__init__()
        bottleneck_channels = base_channels * 4
        self.in_projection = nn.Conv2d(latent_channels, base_channels, kernel_size=3, padding=1)

        self.enc0 = TimeConditionedResidualBlock(base_channels, base_channels, time_dim)
        self.inject0 = DescriptorConditionFusion(
            base_channels,
            global_dim,
            context_channels=context_channels,
            num_heads=attention_heads,
        )
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1)

        self.enc1 = TimeConditionedResidualBlock(base_channels * 2, base_channels * 2, time_dim)
        self.inject1 = DescriptorConditionFusion(
            base_channels * 2,
            global_dim,
            context_channels=context_channels,
            num_heads=attention_heads,
        )
        self.down2 = nn.Conv2d(base_channels * 2, bottleneck_channels, kernel_size=3, stride=2, padding=1)

        self.enc2 = TimeConditionedResidualBlock(bottleneck_channels, bottleneck_channels, time_dim)
        self.inject2 = DescriptorConditionFusion(
            bottleneck_channels,
            global_dim,
            context_channels=context_channels,
            num_heads=attention_heads,
        )
        self.down3 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=2, padding=1)

        self.bottleneck = TimeConditionedResidualBlock(bottleneck_channels, bottleneck_channels, time_dim)
        self.bottleneck_inject = DescriptorConditionFusion(
            bottleneck_channels,
            global_dim,
            context_channels=context_channels,
            num_heads=attention_heads,
        )

        self.up2 = nn.ConvTranspose2d(bottleneck_channels, bottleneck_channels, kernel_size=2, stride=2)
        self.dec2 = TimeConditionedResidualBlock(bottleneck_channels * 2, bottleneck_channels, time_dim)
        self.dec2_inject = DescriptorConditionFusion(
            bottleneck_channels,
            global_dim,
            context_channels=context_channels,
            num_heads=attention_heads,
        )

        self.up1 = nn.ConvTranspose2d(bottleneck_channels, base_channels * 2, kernel_size=2, stride=2)
        self.dec1 = TimeConditionedResidualBlock(base_channels * 4, base_channels * 2, time_dim)
        self.dec1_inject = DescriptorConditionFusion(
            base_channels * 2,
            global_dim,
            context_channels=context_channels,
            num_heads=attention_heads,
        )

        self.up0 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec0 = TimeConditionedResidualBlock(base_channels * 2, base_channels, time_dim)
        self.dec0_inject = DescriptorConditionFusion(
            base_channels,
            global_dim,
            context_channels=context_channels,
            num_heads=attention_heads,
        )

        self.out = nn.Sequential(
            nn.GroupNorm(num_groups=min(8, base_channels), num_channels=base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, latent_channels, kernel_size=1),
        )

    def forward(
        self,
        noisy_latent: torch.Tensor,
        descriptor_context: torch.Tensor,
        global_embedding: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:
        x0 = self.in_projection(noisy_latent)
        x0 = self.enc0(x0, time_embedding)
        x0 = self.inject0(x0, descriptor_context, global_embedding)

        x1 = self.down1(x0)
        x1 = self.enc1(x1, time_embedding)
        x1 = self.inject1(x1, descriptor_context, global_embedding)

        x2 = self.down2(x1)
        x2 = self.enc2(x2, time_embedding)
        x2 = self.inject2(x2, descriptor_context, global_embedding)

        bottleneck = self.down3(x2)
        bottleneck = self.bottleneck(bottleneck, time_embedding)
        bottleneck = self.bottleneck_inject(bottleneck, descriptor_context, global_embedding)

        y2 = self.up2(bottleneck)
        y2 = _resize_like(y2, x2)
        y2 = torch.cat([y2, x2], dim=1)
        y2 = self.dec2(y2, time_embedding)
        y2 = self.dec2_inject(y2, descriptor_context, global_embedding)

        y1 = self.up1(y2)
        y1 = _resize_like(y1, x1)
        y1 = torch.cat([y1, x1], dim=1)
        y1 = self.dec1(y1, time_embedding)
        y1 = self.dec1_inject(y1, descriptor_context, global_embedding)

        y0 = self.up0(y1)
        y0 = _resize_like(y0, x0)
        y0 = torch.cat([y0, x0], dim=1)
        y0 = self.dec0(y0, time_embedding)
        y0 = self.dec0_inject(y0, descriptor_context, global_embedding)
        return self.out(y0)
