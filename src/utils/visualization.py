from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

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
    reconstructed_label: str = "Reconstructed",
    comparison_rgb: torch.Tensor | None = None,
    comparison_label: str = "Comparison Reconstruction",
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

    num_cols = 6 if gt_mask is not None else 5
    if comparison_rgb is not None:
        num_cols += 1
    fig, axes = plt.subplots(1, num_cols, figsize=(3.2 * num_cols, 4))
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
    _show_rgb(axes[1], reconstructed_rgb, reconstructed_label)
    next_index = 2
    if comparison_rgb is not None:
        _show_rgb(axes[next_index], comparison_rgb, comparison_label)
        next_index += 1
    _show_map(axes[next_index], anomaly_map, "Anomaly Map")
    _show_map(axes[next_index + 1], score_map, score_map_label)
    _show_map(axes[next_index + 2], support_map, "Descriptor Support")
    if gt_mask is not None:
        _show_map(axes[next_index + 3], gt_mask, "GT Mask", cmap="gray")

    fig.suptitle(f"{title} | score={anomaly_score:.4f}")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _image_score_tensors(image_score_records: Sequence[Mapping[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.tensor(
        [float(record.get("score", 0.0)) for record in image_score_records],
        dtype=torch.float32,
    )
    labels = torch.tensor(
        [
            int(record.get("label", 1 if bool(record.get("is_anomalous", False)) else 0))
            for record in image_score_records
        ],
        dtype=torch.int64,
    )
    return scores, labels


def _roc_curve_points(
    scores: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[list[float], list[float]] | None:
    flattened_scores = scores.flatten().float()
    flattened_labels = labels.flatten().to(torch.int64)
    positives = int(flattened_labels.sum().item())
    negatives = int((flattened_labels == 0).sum().item())
    if positives == 0 or negatives == 0:
        return None

    order = torch.argsort(flattened_scores, descending=True)
    sorted_labels = flattened_labels[order]
    tpr_values = [0.0]
    fpr_values = [0.0]
    tp = 0
    fp = 0
    for label in sorted_labels.tolist():
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr_values.append(tp / positives)
        fpr_values.append(fp / negatives)
    return fpr_values, tpr_values


def _pr_curve_points(
    scores: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[list[float], list[float]] | None:
    flattened_scores = scores.flatten().float()
    flattened_labels = labels.flatten().to(torch.int64)
    positives = int(flattened_labels.sum().item())
    if positives == 0:
        return None

    order = torch.argsort(flattened_scores, descending=True)
    sorted_labels = flattened_labels[order].float()
    true_positive_cumsum = torch.cumsum(sorted_labels, dim=0)
    false_positive_cumsum = torch.cumsum(1.0 - sorted_labels, dim=0)
    precision = true_positive_cumsum / torch.clamp(true_positive_cumsum + false_positive_cumsum, min=1.0)
    recall = true_positive_cumsum / positives
    recall_values = [0.0] + recall.tolist()
    precision_values = [1.0] + precision.tolist()
    return recall_values, precision_values


def save_evaluation_summary_plot(
    *,
    evaluation_metrics: Mapping[str, Any],
    path: Path | str,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_keys = ["image_auroc", "image_auprc", "pixel_auroc", "aupro"]
    metric_labels = ["Image AUROC", "Image AUPRC", "Pixel AUROC", "AUPRO"]
    pooled_values = [float(evaluation_metrics.get(metric, 0.0)) for metric in metric_keys]
    macro_values = [float(evaluation_metrics.get(f"macro_{metric}", pooled)) for metric, pooled in zip(metric_keys, pooled_values)]
    macro_present = any(f"macro_{metric}" in evaluation_metrics for metric in metric_keys)

    positions = list(range(len(metric_keys)))
    width = 0.35 if macro_present else 0.6
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if macro_present:
        ax.bar([position - width / 2 for position in positions], pooled_values, width=width, label="Pooled", color="#4c72b0")
        ax.bar([position + width / 2 for position in positions], macro_values, width=width, label="Macro", color="#dd8452")
        ax.legend(frameon=False)
    else:
        ax.bar(positions, pooled_values, width=width, color="#4c72b0")
    ax.set_xticks(positions)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Evaluation Summary")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def save_per_category_core_metrics_plot(
    *,
    per_category: Mapping[str, Mapping[str, Any]],
    path: Path | str,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(
        per_category.items(),
        key=lambda item: float(item[1].get("image_auroc", 0.0)),
        reverse=True,
    )
    categories = [category for category, _ in sorted_rows]
    metric_keys = ["image_auroc", "image_auprc", "pixel_auroc", "aupro"]
    metric_labels = ["Image AUROC", "Image AUPRC", "Pixel AUROC", "AUPRO"]
    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b3"]

    positions = list(range(len(categories)))
    width = 0.18
    fig_width = max(10.0, 0.9 * max(len(categories), 1))
    fig, ax = plt.subplots(figsize=(fig_width, 5.5))
    for index, (metric_key, label, color) in enumerate(zip(metric_keys, metric_labels, colors)):
        offset = (index - 1.5) * width
        metric_values = [float(metrics.get(metric_key, 0.0)) for _, metrics in sorted_rows]
        ax.bar([position + offset for position in positions], metric_values, width=width, label=label, color=color)
    ax.set_xticks(positions)
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Per-Category Core Metrics")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def save_image_roc_curve_plot(
    *,
    image_score_records: Sequence[Mapping[str, Any]],
    image_auroc: float,
    path: Path | str,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scores, labels = _image_score_tensors(image_score_records)
    curve = _roc_curve_points(scores, labels)

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    if curve is None:
        ax.text(0.5, 0.5, "Insufficient class variation", ha="center", va="center")
    else:
        fpr_values, tpr_values = curve
        ax.plot(fpr_values, tpr_values, color="#4c72b0", linewidth=2.0, label=f"AUROC = {image_auroc:.3f}")
        ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="#888888", linewidth=1.0)
        ax.legend(frameon=False, loc="lower right")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Image ROC Curve")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def save_image_pr_curve_plot(
    *,
    image_score_records: Sequence[Mapping[str, Any]],
    image_auprc: float,
    path: Path | str,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scores, labels = _image_score_tensors(image_score_records)
    curve = _pr_curve_points(scores, labels)

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    if curve is None:
        ax.text(0.5, 0.5, "Insufficient positive labels", ha="center", va="center")
    else:
        recall_values, precision_values = curve
        ax.plot(recall_values, precision_values, color="#55a868", linewidth=2.0, label=f"AUPRC = {image_auprc:.3f}")
        ax.legend(frameon=False, loc="lower left")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Image PR Curve")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def save_image_score_distribution_plot(
    *,
    image_score_records: Sequence[Mapping[str, Any]],
    path: Path | str,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scores, labels = _image_score_tensors(image_score_records)
    good_scores = scores[labels == 0]
    anomalous_scores = scores[labels == 1]
    transformed_good = torch.log10(torch.clamp(good_scores, min=0.0) + 1.0)
    transformed_anomalous = torch.log10(torch.clamp(anomalous_scores, min=0.0) + 1.0)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    if transformed_good.numel():
        ax.hist(
            transformed_good.tolist(),
            bins=20,
            alpha=0.65,
            color="#4c72b0",
            label=f"Good (n={transformed_good.numel()})",
        )
    if transformed_anomalous.numel():
        ax.hist(
            transformed_anomalous.tolist(),
            bins=20,
            alpha=0.55,
            color="#c44e52",
            label=f"Anomalous (n={transformed_anomalous.numel()})",
        )
    ax.set_xlabel("log10(1 + anomaly score)")
    ax.set_ylabel("Count")
    ax.set_title("Image Score Distribution")
    ax.grid(alpha=0.3)
    if transformed_good.numel() or transformed_anomalous.numel():
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def save_evaluation_plot_bundle(
    *,
    evaluation_metrics: Mapping[str, Any],
    per_category: Mapping[str, Mapping[str, Any]],
    image_score_data_path: Path | str,
    plot_dir: Path | str,
) -> dict[str, str]:
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    image_score_data_path = Path(image_score_data_path)
    image_score_records = json.loads(image_score_data_path.read_text(encoding="utf-8"))

    summary_path = save_evaluation_summary_plot(
        evaluation_metrics=evaluation_metrics,
        path=plot_dir / "evaluation_summary.png",
    )
    per_category_path = save_per_category_core_metrics_plot(
        per_category=per_category,
        path=plot_dir / "per_category_core_metrics.png",
    )
    roc_path = save_image_roc_curve_plot(
        image_score_records=image_score_records,
        image_auroc=float(evaluation_metrics.get("image_auroc", 0.0)),
        path=plot_dir / "image_roc_curve.png",
    )
    pr_path = save_image_pr_curve_plot(
        image_score_records=image_score_records,
        image_auprc=float(evaluation_metrics.get("image_auprc", 0.0)),
        path=plot_dir / "image_pr_curve.png",
    )
    distribution_path = save_image_score_distribution_plot(
        image_score_records=image_score_records,
        path=plot_dir / "image_score_distribution.png",
    )
    return {
        "evaluation_summary": str(summary_path),
        "per_category_core_metrics": str(per_category_path),
        "image_roc_curve": str(roc_path),
        "image_pr_curve": str(pr_path),
        "image_score_distribution": str(distribution_path),
    }
