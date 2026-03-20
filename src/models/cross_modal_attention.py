from __future__ import annotations

import torch
from torch import nn


class CrossModalAttention2d(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(channels)
        self.context_norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True)
        self.out_projection = nn.Linear(channels, channels)

    def forward(self, query_map: torch.Tensor, context_map: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = query_map.shape
        query = query_map.flatten(2).transpose(1, 2)
        context = context_map.flatten(2).transpose(1, 2)

        query = self.query_norm(query)
        context = self.context_norm(context)
        attended, _ = self.attention(query=query, key=context, value=context, need_weights=False)
        attended = self.out_projection(attended)
        return attended.transpose(1, 2).reshape(batch, channels, height, width)
