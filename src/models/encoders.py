from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

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


@dataclass(frozen=True)
class EncoderOutputs:
    stem: torch.Tensor
    downsampled: torch.Tensor
    bottleneck: torch.Tensor
    skips: Sequence[torch.Tensor]


class RGBAppearanceEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        self.stem = ConvNormAct(in_channels, base_channels)
        self.down1 = ConvNormAct(base_channels, base_channels * 2, stride=2)
        self.down2 = ConvNormAct(base_channels * 2, base_channels * 4, stride=2)

    def forward(self, x: torch.Tensor) -> EncoderOutputs:
        stem = self.stem(x)
        downsampled = self.down1(stem)
        bottleneck = self.down2(downsampled)
        return EncoderOutputs(stem=stem, downsampled=downsampled, bottleneck=bottleneck, skips=(stem, downsampled))


class SpatialDescriptorEncoder(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 32) -> None:
        super().__init__()
        self.stem = ConvNormAct(in_channels, base_channels)
        self.down1 = ConvNormAct(base_channels, base_channels * 2, stride=2)
        self.down2 = ConvNormAct(base_channels * 2, base_channels * 4, stride=2)

    def forward(self, descriptor_maps: torch.Tensor) -> EncoderOutputs:
        stem = self.stem(descriptor_maps)
        downsampled = self.down1(stem)
        bottleneck = self.down2(downsampled)
        return EncoderOutputs(stem=stem, downsampled=downsampled, bottleneck=bottleneck, skips=(stem, downsampled))


class GlobalConditionEncoder(nn.Module):
    def __init__(self, in_features: int, embedding_dim: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, global_vector: torch.Tensor) -> torch.Tensor:
        return self.network(global_vector)
