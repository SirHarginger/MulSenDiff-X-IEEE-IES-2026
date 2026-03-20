from __future__ import annotations

import torch


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
    scores = scores.flatten().float()
    labels = labels.flatten().to(torch.int64)
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
