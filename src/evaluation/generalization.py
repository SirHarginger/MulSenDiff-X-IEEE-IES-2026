from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import torch

from src.category_policies import ALL_CATEGORIES
from src.evaluation.metrics import image_average_precision, image_level_auroc
from src.utils.logger import write_history_csv, write_json


@dataclass(frozen=True)
class HeldOutFold:
    name: str
    held_out_categories: tuple[str, ...]

    @property
    def training_categories(self) -> tuple[str, ...]:
        return tuple(category for category in ALL_CATEGORIES if category not in set(self.held_out_categories))

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["training_categories"] = list(self.training_categories)
        return payload


HELD_OUT_FOLDS: Dict[str, HeldOutFold] = {
    "fold_a": HeldOutFold(
        name="fold_a",
        held_out_categories=("button_cell", "capsule", "nut", "screen", "zipper"),
    ),
    "fold_b": HeldOutFold(
        name="fold_b",
        held_out_categories=("cube", "flat_pad", "screw", "spring_pad", "toothbrush"),
    ),
    "fold_c": HeldOutFold(
        name="fold_c",
        held_out_categories=("cotton", "light", "piggy", "plastic_cylinder", "solar_panel"),
    ),
}

GENERALIZATION_METRICS = (
    "image_auroc",
    "pixel_auroc",
    "pixel_aupr",
    "pixel_f1_max",
    "aupro",
)


def resolve_held_out_folds(folds: Sequence[str] | None = None) -> List[HeldOutFold]:
    requested = [str(item).strip().lower() for item in (folds or []) if str(item).strip()]
    if not requested:
        return [HELD_OUT_FOLDS[name] for name in sorted(HELD_OUT_FOLDS)]
    resolved: List[HeldOutFold] = []
    for name in requested:
        if name not in HELD_OUT_FOLDS:
            raise KeyError(f"Unknown held-out fold: {name}")
        resolved.append(HELD_OUT_FOLDS[name])
    return resolved


def summarize_held_out_generalization(
    *,
    closed_set_eval_run: Path | str,
    fold_eval_runs: Mapping[str, Path | str],
    output_dir: Path | str,
) -> Dict[str, Any]:
    closed_set_root = Path(closed_set_eval_run)
    closed_set_per_category = _load_per_category_metrics(closed_set_root)

    fold_rows: List[Dict[str, Any]] = []
    category_rows: List[Dict[str, Any]] = []
    pooled_scores: List[float] = []
    pooled_labels: List[float] = []
    all_held_out_categories: List[str] = []

    for fold_name, eval_run_root_like in sorted(fold_eval_runs.items()):
        fold = HELD_OUT_FOLDS.get(str(fold_name).strip().lower())
        if fold is None:
            raise KeyError(f"Unknown held-out fold in summary input: {fold_name}")
        eval_run_root = Path(eval_run_root_like)
        held_out_per_category = _load_per_category_metrics(eval_run_root)
        held_out_categories = [category for category in fold.held_out_categories if category in held_out_per_category]
        if not held_out_categories:
            raise RuntimeError(f"No held-out categories from {fold.name} found in {eval_run_root}")

        held_out_macro = _macro_metrics(held_out_per_category, held_out_categories)
        closed_set_macro = _macro_metrics(closed_set_per_category, held_out_categories)
        fold_payload: Dict[str, Any] = {
            "fold": fold.name,
            "held_out_categories": ",".join(held_out_categories),
            "training_categories": ",".join(fold.training_categories),
        }
        for metric in GENERALIZATION_METRICS:
            held_out_value = float(held_out_macro.get(metric, 0.0))
            closed_set_value = float(closed_set_macro.get(metric, 0.0))
            fold_payload[f"heldout_{metric}"] = round(held_out_value, 6)
            fold_payload[f"closedset_{metric}"] = round(closed_set_value, 6)
            fold_payload[f"delta_{metric}"] = round(held_out_value - closed_set_value, 6)
        fold_rows.append(fold_payload)

        for category in held_out_categories:
            held_out_metrics = held_out_per_category.get(category, {})
            closed_metrics = closed_set_per_category.get(category, {})
            category_payload: Dict[str, Any] = {
                "fold": fold.name,
                "category": category,
            }
            for metric in GENERALIZATION_METRICS:
                held_out_value = float(held_out_metrics.get(metric, 0.0))
                closed_set_value = float(closed_metrics.get(metric, 0.0))
                category_payload[f"heldout_{metric}"] = round(held_out_value, 6)
                category_payload[f"closedset_{metric}"] = round(closed_set_value, 6)
                category_payload[f"delta_{metric}"] = round(held_out_value - closed_set_value, 6)
            category_rows.append(category_payload)

        pooled_rows = _load_image_score_rows(eval_run_root)
        pooled_scores.extend(float(row.get("score", 0.0)) for row in pooled_rows)
        pooled_labels.extend(float(row.get("label", 0.0)) for row in pooled_rows)
        all_held_out_categories.extend(held_out_categories)

    overall_held_out = _macro_metrics_from_rows(category_rows, prefix="heldout")
    overall_closed_set = _macro_metrics_from_rows(category_rows, prefix="closedset")

    pooled_scores_tensor = torch.tensor(pooled_scores, dtype=torch.float32)
    pooled_labels_tensor = torch.tensor(pooled_labels, dtype=torch.float32)
    pooled_object_auroc = (
        round(float(image_level_auroc(pooled_scores_tensor, pooled_labels_tensor)), 6)
        if pooled_scores
        else 0.0
    )
    pooled_object_auprc = (
        round(float(image_average_precision(pooled_scores_tensor, pooled_labels_tensor)), 6)
        if pooled_scores
        else 0.0
    )
    overall_summary = {
        "held_out_categories_count": len(set(all_held_out_categories)),
        "pooled_object_auroc": pooled_object_auroc,
        "pooled_object_auprc": pooled_object_auprc,
        "heldout_macro_image_auroc": round(float(overall_held_out.get("image_auroc", 0.0)), 6),
        "closedset_macro_image_auroc": round(float(overall_closed_set.get("image_auroc", 0.0)), 6),
        "delta_macro_image_auroc": round(
            float(overall_held_out.get("image_auroc", 0.0) - overall_closed_set.get("image_auroc", 0.0)),
            6,
        ),
        "heldout_macro_pixel_auroc": round(float(overall_held_out.get("pixel_auroc", 0.0)), 6),
        "heldout_macro_pixel_aupr": round(float(overall_held_out.get("pixel_aupr", 0.0)), 6),
        "heldout_macro_pixel_f1_max": round(float(overall_held_out.get("pixel_f1_max", 0.0)), 6),
        "heldout_macro_aupro": round(float(overall_held_out.get("aupro", 0.0)), 6),
        "closedset_macro_pixel_auroc": round(float(overall_closed_set.get("pixel_auroc", 0.0)), 6),
        "closedset_macro_pixel_aupr": round(float(overall_closed_set.get("pixel_aupr", 0.0)), 6),
        "closedset_macro_pixel_f1_max": round(float(overall_closed_set.get("pixel_f1_max", 0.0)), 6),
        "closedset_macro_aupro": round(float(overall_closed_set.get("aupro", 0.0)), 6),
        "delta_macro_pixel_auroc": round(
            float(overall_held_out.get("pixel_auroc", 0.0) - overall_closed_set.get("pixel_auroc", 0.0)),
            6,
        ),
        "delta_macro_pixel_aupr": round(
            float(overall_held_out.get("pixel_aupr", 0.0) - overall_closed_set.get("pixel_aupr", 0.0)),
            6,
        ),
        "delta_macro_pixel_f1_max": round(
            float(overall_held_out.get("pixel_f1_max", 0.0) - overall_closed_set.get("pixel_f1_max", 0.0)),
            6,
        ),
        "delta_macro_aupro": round(
            float(overall_held_out.get("aupro", 0.0) - overall_closed_set.get("aupro", 0.0)),
            6,
        ),
    }

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    write_history_csv(output_root / "fold_summary.csv", fold_rows)
    write_history_csv(output_root / "category_comparison.csv", category_rows)
    write_json(output_root / "overall_summary.json", overall_summary)
    write_json(
        output_root / "fold_definitions.json",
        {name: fold.to_dict() for name, fold in HELD_OUT_FOLDS.items()},
    )
    return {
        "fold_rows": fold_rows,
        "category_rows": category_rows,
        "overall_summary": overall_summary,
    }


def _load_per_category_metrics(eval_run_root: Path) -> Dict[str, Dict[str, float]]:
    path = eval_run_root / "metrics" / "per_category.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(category): {str(metric): float(value) for metric, value in metrics.items()}
        for category, metrics in payload.items()
    }


def _load_image_score_rows(eval_run_root: Path) -> List[Dict[str, Any]]:
    path = eval_run_root / "metrics" / "image_score_data.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [dict(row) for row in payload]


def _macro_metrics(
    per_category: Mapping[str, Mapping[str, float]],
    categories: Sequence[str],
) -> Dict[str, float]:
    if not categories:
        return {metric: 0.0 for metric in GENERALIZATION_METRICS}
    summary: Dict[str, float] = {}
    for metric in GENERALIZATION_METRICS:
        values = [float(per_category.get(category, {}).get(metric, 0.0)) for category in categories]
        summary[metric] = sum(values) / max(len(values), 1)
    return summary


def _macro_metrics_from_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    prefix: str,
) -> Dict[str, float]:
    if not rows:
        return {metric: 0.0 for metric in GENERALIZATION_METRICS}
    summary: Dict[str, float] = {}
    for metric in GENERALIZATION_METRICS:
        values = [float(row.get(f"{prefix}_{metric}", 0.0)) for row in rows]
        summary[metric] = sum(values) / max(len(values), 1)
    return summary
