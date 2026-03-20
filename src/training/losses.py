from __future__ import annotations

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency guard
    raise ModuleNotFoundError(
        "torch is required for src.training.losses. Activate the project venv before running training code."
    ) from exc


def mse_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(f"MSE shape mismatch: {prediction.shape} != {target.shape}")
    return torch.mean((prediction - target) ** 2)
