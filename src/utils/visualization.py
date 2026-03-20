from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import torch


def save_training_curves(history: Sequence[dict], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    train_loss = [row.get("train_loss") for row in history]
    image_auroc = [row.get("image_auroc") for row in history]
    pixel_iou = [row.get("pixel_iou") for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, train_loss, marker="o", color="#1f77b4")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, image_auroc, marker="o", label="Image AUROC", color="#2ca02c")
    axes[1].plot(epochs, pixel_iou, marker="o", label="Pixel IoU", color="#d62728")
    axes[1].set_title("Evaluation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def save_anomaly_panel(
    *,
    rgb: torch.Tensor,
    reconstructed_rgb: torch.Tensor,
    anomaly_map: torch.Tensor,
    score_map: torch.Tensor,
    score_map_label: str,
    support_map: torch.Tensor,
    gt_mask: torch.Tensor | None,
    anomaly_score: float,
    title: str,
    path: Path | str,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 6 if gt_mask is not None else 5, figsize=(18, 4))
    axes = list(axes)

    def _show_rgb(ax, tensor: torch.Tensor, label: str) -> None:
        image = tensor.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0)
        ax.imshow(image)
        ax.set_title(label)
        ax.axis("off")

    def _show_map(ax, tensor: torch.Tensor, label: str, cmap: str = "magma") -> None:
        image = tensor.detach().cpu().squeeze(0)
        ax.imshow(image, cmap=cmap)
        ax.set_title(label)
        ax.axis("off")

    _show_rgb(axes[0], rgb, "RGB")
    _show_rgb(axes[1], reconstructed_rgb, "Reconstructed")
    _show_map(axes[2], anomaly_map, "Anomaly Map")
    _show_map(axes[3], score_map, score_map_label)
    _show_map(axes[4], support_map, "Descriptor Support")
    if gt_mask is not None:
        _show_map(axes[5], gt_mask, "GT Mask", cmap="gray")

    fig.suptitle(f"{title} | score={anomaly_score:.4f}")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path
