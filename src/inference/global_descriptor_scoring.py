from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Sequence

import torch


@dataclass(frozen=True)
class GlobalDescriptorCalibration:
    category: str
    feature_names: Sequence[str]
    sample_count: int
    regularization: float
    mean: list[float]
    inverse_covariance: list[list[float]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "GlobalDescriptorCalibration":
        return cls(
            category=str(payload["category"]),
            feature_names=tuple(payload["feature_names"]),
            sample_count=int(payload["sample_count"]),
            regularization=float(payload["regularization"]),
            mean=[float(value) for value in payload["mean"]],
            inverse_covariance=[
                [float(item) for item in row]
                for row in payload["inverse_covariance"]
            ],
        )


def fit_global_descriptor_calibration(
    vectors: torch.Tensor,
    *,
    category: str,
    feature_names: Sequence[str],
    regularization: float = 1e-3,
) -> GlobalDescriptorCalibration:
    data = vectors.detach().float()
    if data.ndim == 1:
        data = data.unsqueeze(0)
    if data.numel() == 0:
        data = torch.zeros((1, len(feature_names)), dtype=torch.float32)

    mean = data.mean(dim=0)
    centered = data - mean
    if data.shape[0] <= 1:
        covariance = torch.eye(data.shape[1], dtype=torch.float32)
    else:
        covariance = centered.T @ centered / float(max(data.shape[0] - 1, 1))
    covariance = covariance + regularization * torch.eye(covariance.shape[0], dtype=torch.float32)
    inverse_covariance = torch.linalg.pinv(covariance)

    return GlobalDescriptorCalibration(
        category=category,
        feature_names=tuple(feature_names),
        sample_count=int(data.shape[0]),
        regularization=float(regularization),
        mean=[float(value) for value in mean.tolist()],
        inverse_covariance=[
            [float(item) for item in row]
            for row in inverse_covariance.tolist()
        ],
    )


def score_global_descriptors(
    vectors: torch.Tensor,
    calibration: GlobalDescriptorCalibration,
) -> torch.Tensor:
    if vectors.ndim == 1:
        vectors = vectors.unsqueeze(0)

    mean = torch.tensor(calibration.mean, dtype=vectors.dtype, device=vectors.device)
    inverse_covariance = torch.tensor(
        calibration.inverse_covariance,
        dtype=vectors.dtype,
        device=vectors.device,
    )
    centered = vectors - mean.unsqueeze(0)
    mahalanobis_squared = torch.einsum("bi,ij,bj->b", centered, inverse_covariance, centered)
    normalized = torch.sqrt(mahalanobis_squared.clamp_min(0.0))
    return normalized / max(float(len(calibration.feature_names)) ** 0.5, 1.0)
