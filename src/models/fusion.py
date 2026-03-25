from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.models.cross_modal_attention import CrossModalAttention2d


class GlobalAdaLN2d(nn.Module):
    def __init__(self, channels: int, global_dim: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
        self.modulation = nn.Linear(global_dim, channels * 2)

    def forward(self, x: torch.Tensor, global_embedding: torch.Tensor) -> torch.Tensor:
        scale, shift = self.modulation(global_embedding).chunk(2, dim=1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)
        normalized = self.norm(x)
        return normalized * (1.0 + torch.tanh(scale)) + shift


class DescriptorConditionFusion(nn.Module):
    def __init__(
        self,
        channels: int,
        global_dim: int,
        *,
        context_channels: int | None = None,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.cross_attention = CrossModalAttention2d(
            query_channels=channels,
            context_channels=context_channels or channels,
            num_heads=num_heads,
        )
        self.gate = nn.Sequential(
            nn.Linear(global_dim, channels),
            nn.Sigmoid(),
        )
        self.global_adaln = GlobalAdaLN2d(channels, global_dim)
        self.post = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, channels), num_channels=channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, channels), num_channels=channels),
            nn.SiLU(),
        )

    def forward(
        self,
        query_map: torch.Tensor,
        context_map: torch.Tensor,
        global_embedding: torch.Tensor,
    ) -> torch.Tensor:
        if context_map.shape[-2:] != query_map.shape[-2:]:
            context_map = F.interpolate(
                context_map,
                size=query_map.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        attended = self.cross_attention(query_map, context_map)
        gate = self.gate(global_embedding).unsqueeze(-1).unsqueeze(-1)
        fused = query_map + gate * attended
        fused = self.global_adaln(fused, global_embedding)
        return fused + self.post(fused)
