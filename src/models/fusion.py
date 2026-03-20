from __future__ import annotations

import torch
from torch import nn

from src.models.cross_modal_attention import CrossModalAttention2d


class DescriptorConditionFusion(nn.Module):
    def __init__(self, channels: int, condition_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.cross_attention = CrossModalAttention2d(channels, num_heads=num_heads)
        self.gate = nn.Sequential(
            nn.Linear(condition_dim, channels),
            nn.Sigmoid(),
        )
        self.film = nn.Linear(condition_dim, channels * 2)
        self.post = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, channels), num_channels=channels),
            nn.SiLU(),
        )

    def forward(
        self,
        rgb_bottleneck: torch.Tensor,
        descriptor_bottleneck: torch.Tensor,
        condition_vector: torch.Tensor,
    ) -> torch.Tensor:
        attended = self.cross_attention(rgb_bottleneck, descriptor_bottleneck)
        gate = self.gate(condition_vector).unsqueeze(-1).unsqueeze(-1)
        scale, shift = self.film(condition_vector).chunk(2, dim=1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)

        fused = rgb_bottleneck + gate * attended
        fused = fused * (1.0 + torch.tanh(scale)) + shift
        return self.post(fused)
