from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from scipy.ndimage import label as connected_components


def _binary_ranking_inputs(scores: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    flattened_scores = scores.flatten().float()
    flattened_labels = (labels.flatten() >= 0.5).to(torch.int64)
    return flattened_scores, flattened_labels


def binary_f1_score(prediction: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred = (prediction >= threshold).to(torch.int64)
    truth = (target >= threshold).to(torch.int64)
    tp = ((pred == 1) & (truth == 1)).sum().item()
    fp = ((pred == 1) & (truth == 0)).sum().item()
    fn = ((pred == 0) & (truth == 1)).sum().item()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def intersection_over_union(prediction: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred = prediction >= threshold
    truth = target >= threshold
    intersection = (pred & truth).sum().item()
    union = (pred | truth).sum().item()
    return intersection / max(union, 1)


def image_level_auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    scores, labels = _binary_ranking_inputs(scores, labels)
    positives = int(labels.sum().item())
    negatives = int((labels == 0).sum().item())
    if positives == 0 or negatives == 0:
        return 0.0

    order = torch.argsort(scores, descending=True)
    sorted_labels = labels[order]
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

    auc = 0.0
    for idx in range(1, len(tpr_values)):
        auc += (fpr_values[idx] - fpr_values[idx - 1]) * (tpr_values[idx] + tpr_values[idx - 1]) * 0.5
    return auc


def image_average_precision(scores: torch.Tensor, labels: torch.Tensor) -> float:
    scores, labels = _binary_ranking_inputs(scores, labels)
    positives = int(labels.sum().item())
    if positives == 0:
        return 0.0

    order = torch.argsort(scores, descending=True)
    sorted_labels = labels[order]
    true_positive_cumsum = torch.cumsum(sorted_labels, dim=0).float()
    ranks = torch.arange(1, len(sorted_labels) + 1, dtype=torch.float32)
    precision = true_positive_cumsum / ranks
    average_precision = (precision * sorted_labels.float()).sum().item() / positives
    return float(average_precision)


def _flatten_masked_pixel_data(
    score_maps: Sequence[torch.Tensor],
    target_masks: Sequence[torch.Tensor],
    object_masks: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    flattened_scores: list[torch.Tensor] = []
    flattened_labels: list[torch.Tensor] = []

    for score_map, target_mask, object_mask in zip(score_maps, target_masks, object_masks):
        scores = torch.as_tensor(score_map).detach().float().cpu().squeeze()
        targets = (torch.as_tensor(target_mask).detach().float().cpu().squeeze() >= 0.5).to(torch.int64)
        mask = torch.as_tensor(object_mask).detach().bool().cpu().squeeze()
        if scores.shape != targets.shape or scores.shape != mask.shape or not bool(mask.any()):
            continue
        flattened_scores.append(scores[mask])
        flattened_labels.append(targets[mask])

    if not flattened_scores:
        return torch.empty(0), torch.empty(0, dtype=torch.int64)
    return torch.cat(flattened_scores), torch.cat(flattened_labels)


def pixel_level_auroc(
    score_maps: Sequence[torch.Tensor],
    target_masks: Sequence[torch.Tensor],
    object_masks: Sequence[torch.Tensor],
) -> float:
    scores, labels = _flatten_masked_pixel_data(score_maps, target_masks, object_masks)
    if scores.numel() == 0:
        return 0.0
    return image_level_auroc(scores, labels)


def pixel_level_average_precision(
    score_maps: Sequence[torch.Tensor],
    target_masks: Sequence[torch.Tensor],
    object_masks: Sequence[torch.Tensor],
) -> float:
    scores, labels = _flatten_masked_pixel_data(score_maps, target_masks, object_masks)
    if scores.numel() == 0:
        return 0.0
    return image_average_precision(scores, labels)


def pixel_f1_max(
    score_maps: Sequence[torch.Tensor],
    target_masks: Sequence[torch.Tensor],
    object_masks: Sequence[torch.Tensor],
    *,
    num_thresholds: int = 256,
) -> float:
    scores, labels = _flatten_masked_pixel_data(score_maps, target_masks, object_masks)
    if scores.numel() == 0:
        return 0.0

    thresholds = torch.linspace(0.0, 1.0, steps=max(int(num_thresholds), 2), dtype=torch.float32)
    best_f1 = 0.0
    labels_float = labels.float()
    for threshold in thresholds.tolist():
        best_f1 = max(best_f1, binary_f1_score(scores, labels_float, threshold=float(threshold)))
    return float(best_f1)


def aupro(
    score_maps: Sequence[torch.Tensor],
    target_masks: Sequence[torch.Tensor],
    object_masks: Sequence[torch.Tensor],
    *,
    max_fpr: float = 0.30,
    num_thresholds: int = 256,
) -> float:
    prepared: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]] = []
    total_negative_pixels = 0
    total_regions = 0

    for score_map, target_mask, object_mask in zip(score_maps, target_masks, object_masks):
        scores = torch.as_tensor(score_map).detach().float().cpu().squeeze().numpy()
        targets = (torch.as_tensor(target_mask).detach().float().cpu().squeeze().numpy() >= 0.5)
        mask = torch.as_tensor(object_mask).detach().bool().cpu().squeeze().numpy()
        if scores.shape != targets.shape or scores.shape != mask.shape or not mask.any():
            continue
        masked_targets = targets & mask
        labeled_targets, num_regions = connected_components(masked_targets.astype(np.uint8))
        total_negative_pixels += int((mask & ~masked_targets).sum())
        total_regions += int(num_regions)
        prepared.append((scores, masked_targets, mask, labeled_targets, int(num_regions)))

    if not prepared or total_negative_pixels <= 0 or total_regions <= 0:
        return 0.0

    thresholds = np.concatenate(([1.01], np.linspace(1.0, 0.0, num=num_thresholds)))
    curve: list[tuple[float, float]] = []

    for threshold in thresholds:
        false_positives = 0
        region_overlaps: list[float] = []
        for scores, masked_targets, mask, labeled_targets, num_regions in prepared:
            prediction = (scores >= threshold) & mask
            false_positives += int((prediction & ~masked_targets).sum())
            for region_index in range(1, num_regions + 1):
                region = labeled_targets == region_index
                region_size = int(region.sum())
                if region_size <= 0:
                    continue
                region_overlaps.append(float((prediction & region).sum()) / region_size)
        fpr = false_positives / max(total_negative_pixels, 1)
        pro = float(np.mean(region_overlaps)) if region_overlaps else 0.0
        curve.append((fpr, pro))

    dedup: dict[float, float] = {}
    for fpr, pro in curve:
        key = round(float(fpr), 8)
        dedup[key] = max(dedup.get(key, 0.0), float(pro))
    sorted_curve = sorted((float(fpr), float(pro)) for fpr, pro in dedup.items())

    clipped_curve: list[tuple[float, float]] = []
    previous_fpr = 0.0
    previous_pro = 0.0
    for fpr, pro in sorted_curve:
        if not clipped_curve:
            clipped_curve.append((0.0, 0.0))
        if fpr <= max_fpr:
            clipped_curve.append((fpr, pro))
            previous_fpr, previous_pro = fpr, pro
            continue
        if fpr > previous_fpr:
            slope = (pro - previous_pro) / (fpr - previous_fpr)
            interpolated_pro = previous_pro + slope * (max_fpr - previous_fpr)
            clipped_curve.append((max_fpr, float(interpolated_pro)))
        break

    if clipped_curve[-1][0] < max_fpr:
        clipped_curve.append((max_fpr, clipped_curve[-1][1]))

    area = 0.0
    for idx in range(1, len(clipped_curve)):
        x0, y0 = clipped_curve[idx - 1]
        x1, y1 = clipped_curve[idx]
        area += (x1 - x0) * (y0 + y1) * 0.5
    return float(area / max(max_fpr, 1e-8))
