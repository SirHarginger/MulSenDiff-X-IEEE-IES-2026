from __future__ import annotations

import torch
from torch import nn


def _resolve_attention_heads(embed_dim: int, requested_heads: int) -> int:
    heads = max(1, min(requested_heads, embed_dim))
    while embed_dim % heads != 0 and heads > 1:
        heads -= 1
    return max(1, heads)


class CrossModalAttention2d(nn.Module):
    def __init__(
        self,
        query_channels: int,
        context_channels: int | None = None,
        *,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.query_channels = query_channels
        self.context_channels = context_channels or query_channels
        self.query_norm = nn.LayerNorm(query_channels)
        self.context_norm = nn.LayerNorm(self.context_channels)
        attention_heads = _resolve_attention_heads(query_channels, num_heads)
        self.attention = nn.MultiheadAttention(
            embed_dim=query_channels,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
            kdim=self.context_channels,
            vdim=self.context_channels,
        )
        self.out_projection = nn.Linear(query_channels, query_channels)

    def forward(self, query_map: torch.Tensor, context_map: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = query_map.shape
        query = query_map.flatten(2).transpose(1, 2)
        context = context_map.flatten(2).transpose(1, 2)

        query = self.query_norm(query)
        context = self.context_norm(context)
        attended, _ = self.attention(query=query, key=context, value=context, need_weights=False)
        attended = self.out_projection(attended)
        return attended.transpose(1, 2).reshape(batch, channels, height, width)
