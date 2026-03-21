from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


IR_STATS_JSON = "ir_stats.json"


def load_ir_grayscale(path: Path | str, *, target_size: tuple[int, int] = (256, 256)) -> tuple[np.ndarray, float]:
    image = Image.open(path).convert("L")
    sigma = 1.0 if min(image.size) >= 256 else 1.5
    image = image.resize(target_size, Image.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0, sigma


def normalize_ir_image(image: np.ndarray, *, minimum: float, maximum: float) -> np.ndarray:
    if maximum <= minimum + 1e-8:
        return np.clip(image, 0.0, 1.0).astype(np.float32)
    normalized = (image - minimum) / (maximum - minimum)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def compute_ir_gradient(image: np.ndarray, *, sigma: float) -> np.ndarray:
    tensor = _to_tensor(image)
    smoothed = _gaussian_blur(tensor, sigma=sigma)

    sobel_x = torch.tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
        dtype=torch.float32,
    ).unsqueeze(0)
    sobel_y = torch.tensor(
        [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
        dtype=torch.float32,
    ).unsqueeze(0)

    grad_x = F.conv2d(smoothed, sobel_x, padding=1)
    grad_y = F.conv2d(smoothed, sobel_y, padding=1)
    magnitude = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-12)
    return magnitude.squeeze().cpu().numpy().astype(np.float32)


def compute_ir_local_variance(image: np.ndarray, *, window_size: int = 9) -> np.ndarray:
    tensor = _to_tensor(image)
    padding = window_size // 2
    mean = F.avg_pool2d(tensor, kernel_size=window_size, stride=1, padding=padding)
    mean_sq = F.avg_pool2d(tensor.pow(2), kernel_size=window_size, stride=1, padding=padding)
    variance = torch.clamp(mean_sq - mean.pow(2), min=0.0)
    return variance.squeeze().cpu().numpy().astype(np.float32)


def compute_ir_category_stats(
    ir_paths: Sequence[Path | str],
    *,
    target_size: tuple[int, int] = (256, 256),
) -> Dict[str, object]:
    if not ir_paths:
        raise ValueError("At least one IR training path is required to compute category statistics.")

    resized_images: list[np.ndarray] = []
    sigmas: list[float] = []
    global_min = float("inf")
    global_max = float("-inf")

    for path in ir_paths:
        raw_image, sigma = load_ir_grayscale(path, target_size=target_size)
        resized_images.append(raw_image)
        sigmas.append(sigma)
        global_min = min(global_min, float(raw_image.min()))
        global_max = max(global_max, float(raw_image.max()))

    normalized_images = [
        normalize_ir_image(image, minimum=global_min, maximum=global_max) for image in resized_images
    ]
    stacked = np.stack(normalized_images, axis=0)
    ir_mean = stacked.mean(axis=0).astype(np.float32)
    ir_std = stacked.std(axis=0).astype(np.float32)
    global_mean_std = float(ir_std.mean()) if ir_std.size else 0.0
    std_floor = max(global_mean_std * 0.01, 1e-4)
    ir_std_regularized = np.clip(ir_std, std_floor, None).astype(np.float32)

    gradient_maps = [
        compute_ir_gradient(image, sigma=sigma)
        for image, sigma in zip(normalized_images, sigmas)
    ]
    gradient_mean = np.stack(gradient_maps, axis=0).mean(axis=0).astype(np.float32)

    variance_maps = [
        compute_ir_local_variance(image - ir_mean, window_size=9)
        for image in normalized_images
    ]
    variance_mean = np.stack(variance_maps, axis=0).mean(axis=0).astype(np.float32)

    gradient_residual_max = max(
        float(np.clip(gradient - gradient_mean, 0.0, None).max()) for gradient in gradient_maps
    )
    variance_residual_max = max(
        float(np.clip(variance - variance_mean, 0.0, None).max()) for variance in variance_maps
    )

    return {
        "target_size": list(target_size),
        "global_min": float(global_min),
        "global_max": float(global_max),
        "global_mean_std": global_mean_std,
        "std_floor": std_floor,
        "gradient_residual_max": max(gradient_residual_max, 1e-6),
        "variance_residual_max": max(variance_residual_max, 1e-6),
        "ir_mean": ir_mean,
        "ir_std": ir_std_regularized,
        "ir_gradient_mean": gradient_mean,
        "ir_variance_mean": variance_mean,
    }


def save_ir_category_stats(stats: Dict[str, object], out_dir: Path | str) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "ir_mean.npy", stats["ir_mean"])
    np.save(out_dir / "ir_std.npy", stats["ir_std"])
    np.save(out_dir / "ir_gradient_mean.npy", stats["ir_gradient_mean"])
    np.save(out_dir / "ir_variance_mean.npy", stats["ir_variance_mean"])
    json_payload = {
        "target_size": stats["target_size"],
        "global_min": stats["global_min"],
        "global_max": stats["global_max"],
        "global_mean_std": stats["global_mean_std"],
        "std_floor": stats["std_floor"],
        "gradient_residual_max": stats["gradient_residual_max"],
        "variance_residual_max": stats["variance_residual_max"],
    }
    (out_dir / IR_STATS_JSON).write_text(json.dumps(json_payload, indent=2, sort_keys=True), encoding="utf-8")


def load_ir_category_stats(stats_dir: Path | str) -> Dict[str, object]:
    stats_dir = Path(stats_dir)
    json_payload = json.loads((stats_dir / IR_STATS_JSON).read_text(encoding="utf-8"))
    return {
        **json_payload,
        "ir_mean": np.load(stats_dir / "ir_mean.npy"),
        "ir_std": np.load(stats_dir / "ir_std.npy"),
        "ir_gradient_mean": np.load(stats_dir / "ir_gradient_mean.npy"),
        "ir_variance_mean": np.load(stats_dir / "ir_variance_mean.npy"),
    }


def generate_ir_sample_descriptors(
    ir_path: Path | str,
    *,
    stats: Dict[str, object],
    target_size: tuple[int, int] = (256, 256),
) -> Dict[str, object]:
    raw_image, sigma = load_ir_grayscale(ir_path, target_size=target_size)
    normalized = normalize_ir_image(
        raw_image,
        minimum=float(stats["global_min"]),
        maximum=float(stats["global_max"]),
    )
    ir_mean = np.asarray(stats["ir_mean"], dtype=np.float32)
    ir_std = np.asarray(stats["ir_std"], dtype=np.float32)
    gradient_mean = np.asarray(stats["ir_gradient_mean"], dtype=np.float32)
    variance_mean = np.asarray(stats["ir_variance_mean"], dtype=np.float32)

    gradient_raw = compute_ir_gradient(normalized, sigma=sigma)
    gradient_residual = np.clip(gradient_raw - gradient_mean, 0.0, None)
    gradient_norm = np.clip(
        gradient_residual / max(float(stats["gradient_residual_max"]), 1e-6),
        0.0,
        1.0,
    ).astype(np.float32)

    variance_raw = compute_ir_local_variance(normalized - ir_mean, window_size=9)
    variance_residual = np.clip(variance_raw - variance_mean, 0.0, None)
    variance_norm = np.clip(
        variance_residual / max(float(stats["variance_residual_max"]), 1e-6),
        0.0,
        1.0,
    ).astype(np.float32)

    z_map = (normalized - ir_mean) / np.clip(ir_std, 1e-6, None)
    hotspot_z = np.where(z_map >= 2.0, np.clip(z_map, 0.0, 5.0), 0.0).astype(np.float32)
    hotspot_norm = np.clip(hotspot_z / 5.0, 0.0, 1.0).astype(np.float32)

    hotspot_mask = hotspot_z > 0.0
    hotspot_area_fraction = float(hotspot_mask.mean())
    mean_hotspot_intensity = float(hotspot_z[hotspot_mask].mean()) if hotspot_mask.any() else 0.0

    return {
        "ir_normalized": normalized,
        "ir_gradient": gradient_norm,
        "ir_variance": variance_norm,
        "ir_hotspot": hotspot_norm,
        "global": {
            "ir_hotspot_area_fraction": round(hotspot_area_fraction, 6),
            "ir_max_gradient": round(float(gradient_norm.max()), 6),
            "ir_mean_hotspot_intensity": round(mean_hotspot_intensity, 6),
            "ir_hotspot_compactness": round(_hotspot_compactness(hotspot_mask), 6),
        },
    }


def save_grayscale_png(array: np.ndarray, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(array, 0.0, 1.0) * 255.0).round().astype(np.uint8), mode="L").save(path)


def _to_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array.astype(np.float32, copy=False)).unsqueeze(0).unsqueeze(0)


def _gaussian_blur(tensor: torch.Tensor, *, sigma: float) -> torch.Tensor:
    radius = max(1, int(round(sigma * 3)))
    kernel_size = radius * 2 + 1
    positions = torch.arange(kernel_size, dtype=torch.float32) - radius
    kernel_1d = torch.exp(-(positions**2) / max(2.0 * sigma * sigma, 1e-6))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)
    return F.conv2d(tensor, kernel_2d, padding=radius)


def _hotspot_compactness(mask: np.ndarray) -> float:
    if not mask.any():
        return 0.0

    binary = mask.astype(np.uint8)
    area = int(binary.sum())
    padded = np.pad(binary, 1, mode="constant")
    perimeter = 0
    perimeter += int(np.sum((padded[1:-1, 1:-1] == 1) & (padded[:-2, 1:-1] == 0)))
    perimeter += int(np.sum((padded[1:-1, 1:-1] == 1) & (padded[2:, 1:-1] == 0)))
    perimeter += int(np.sum((padded[1:-1, 1:-1] == 1) & (padded[1:-1, :-2] == 0)))
    perimeter += int(np.sum((padded[1:-1, 1:-1] == 1) & (padded[1:-1, 2:] == 0)))
    if perimeter <= 0:
        return 1.0
    return float((4.0 * np.pi * area) / float(perimeter * perimeter))
