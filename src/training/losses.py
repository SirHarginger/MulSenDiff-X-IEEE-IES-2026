from __future__ import annotations

try:
    import torch
    from torch.nn import functional as F
except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency guard
    raise ModuleNotFoundError(
        "torch is required for src.training.losses. Activate the project venv before running training code."
    ) from exc


def mse_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(f"MSE shape mismatch: {prediction.shape} != {target.shape}")
    return torch.mean((prediction - target) ** 2)


def l1_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(f"L1 shape mismatch: {prediction.shape} != {target.shape}")
    return torch.mean(torch.abs(prediction - target))


def mixed_reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    l1_weight: float = 0.5,
    mse_weight: float = 0.5,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(f"Reconstruction shape mismatch: {prediction.shape} != {target.shape}")
    total_weight = float(l1_weight + mse_weight)
    if total_weight <= 0.0:
        raise ValueError("At least one reconstruction loss weight must be positive.")
    normalized_l1 = float(l1_weight) / total_weight
    normalized_mse = float(mse_weight) / total_weight
    return normalized_l1 * l1_loss(prediction, target) + normalized_mse * mse_loss(prediction, target)


def sobel_edge_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(f"Sobel edge shape mismatch: {prediction.shape} != {target.shape}")
    if prediction.ndim != 4:
        raise ValueError(f"Sobel edge loss expects BCHW tensors, got shape {prediction.shape}")

    channels = prediction.shape[1]
    sobel_x = torch.tensor(
        [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]],
        dtype=prediction.dtype,
        device=prediction.device,
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]],
        dtype=prediction.dtype,
        device=prediction.device,
    ).view(1, 1, 3, 3)
    sobel_x = sobel_x.repeat(channels, 1, 1, 1)
    sobel_y = sobel_y.repeat(channels, 1, 1, 1)

    pred_grad_x = F.conv2d(prediction, sobel_x, padding=1, groups=channels)
    pred_grad_y = F.conv2d(prediction, sobel_y, padding=1, groups=channels)
    target_grad_x = F.conv2d(target, sobel_x, padding=1, groups=channels)
    target_grad_y = F.conv2d(target, sobel_y, padding=1, groups=channels)

    pred_grad_mag = torch.sqrt(pred_grad_x.pow(2) + pred_grad_y.pow(2) + 1e-6)
    target_grad_mag = torch.sqrt(target_grad_x.pow(2) + target_grad_y.pow(2) + 1e-6)
    return l1_loss(pred_grad_mag, target_grad_mag)
