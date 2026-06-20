#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.category_policies import ALL_CATEGORIES


METRIC_KEYS = (
    "image_auroc",
    "image_auprc",
    "aupro",
    "pixel_auroc",
    "pixel_aupr",
    "pixel_f1",
    "pixel_iou",
)

TABLE_METRICS = (
    ("image_auroc", "Img AUROC"),
    ("image_auprc", "Img AUPRC"),
    ("aupro", "AUPRO"),
    ("pixel_auroc", "PxAUROC"),
    ("pixel_aupr", "PxAUPR"),
)

REGIME_LABELS = {
    "ccdd": "CCDD",
    "cadd": "CADD",
    "csdd": "CSDD",
}


def _read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path | str, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _write_csv(path: Path | str, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return path


def _coerce_seed_mapping(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    return {str(seed): value for seed, value in payload.items() if str(seed).strip()}


def _seed_sort_key(value: Any) -> tuple[int, int | str]:
    text = str(value).strip()
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


def _as_path(value: Any) -> Path:
    return Path(str(value)).expanduser()


def _load_eval_artifacts(eval_run: Path | str) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    root = _as_path(eval_run)
    evaluation_path = root / "metrics" / "evaluation.json"
    per_category_path = root / "metrics" / "per_category.json"
    missing = [str(path) for path in (root / "summary.json", evaluation_path, per_category_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing expected eval artifact(s) for {root}: {', '.join(missing)}")
    evaluation = dict(_read_json(evaluation_path))
    per_category_raw = _read_json(per_category_path)
    per_category = {
        str(category): {
            str(metric): float(value)
            for metric, value in dict(metrics).items()
            if _is_number(value)
        }
        for category, metrics in dict(per_category_raw).items()
    }
    return evaluation, per_category


def _load_eval_summary(eval_run: Path | str) -> dict[str, Any]:
    summary_path = _as_path(eval_run) / "summary.json"
    if not summary_path.exists():
        return {}
    payload = _read_json(summary_path)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _is_number(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _macro_from_per_category(per_category: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    macro: dict[str, float] = {}
    for metric in METRIC_KEYS:
        values = [
            float(metrics.get(metric, 0.0))
            for _, metrics in sorted(per_category.items())
            if metric in metrics
        ]
        macro[metric] = _mean(values)
    return macro


def _macro_from_evaluation(evaluation: Mapping[str, Any], per_category: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    fallback = _macro_from_per_category(per_category)
    macro: dict[str, float] = {}
    for metric in METRIC_KEYS:
        macro_key = f"macro_{metric}"
        if macro_key in evaluation and _is_number(evaluation[macro_key]):
            macro[metric] = float(evaluation[macro_key])
        elif metric in evaluation and _is_number(evaluation[metric]):
            macro[metric] = float(evaluation[metric])
        else:
            macro[metric] = fallback.get(metric, 0.0)
    return macro


def _require_all_categories(per_category: Mapping[str, Mapping[str, float]], *, context: str) -> None:
    present = set(per_category)
    expected = set(ALL_CATEGORIES)
    missing = sorted(expected - present)
    extra = sorted(present - expected)
    if missing or extra:
        detail_parts = []
        if missing:
            detail_parts.append(f"missing={','.join(missing)}")
        if extra:
            detail_parts.append(f"unexpected={','.join(extra)}")
        raise ValueError(f"{context} must cover the {len(ALL_CATEGORIES)} MulSen-AD categories ({'; '.join(detail_parts)})")


def _mean(values: Sequence[float]) -> float:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    return sum(clean) / max(len(clean), 1)


def _std(values: Sequence[float]) -> float:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if len(clean) <= 1:
        return 0.0
    mu = _mean(clean)
    return math.sqrt(sum((value - mu) ** 2 for value in clean) / (len(clean) - 1))


def _format_metric(value: float) -> str:
    return f"{float(value):.4f}"


def _format_mean_std(mean_value: float, std_value: float) -> str:
    return f"{_format_metric(mean_value)} +- {_format_metric(std_value)}"


def _format_latex_mean_std(mean_value: float, std_value: float) -> str:
    return f"{_format_metric(mean_value)} $\\pm$ {_format_metric(std_value)}"


def _metric_values_for_summary(rows: Sequence[Mapping[str, Any]], metric: str) -> list[float]:
    return [float(row[metric]) for row in rows if metric in row and _is_number(row[metric])]


def _summarize_metric_rows(rows: Sequence[Mapping[str, Any]], *, group_key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_key, ""))].append(row)
    summary_rows: list[dict[str, Any]] = []
    for group, group_rows in sorted(grouped.items()):
        seeds = sorted(
            {str(row.get("seed", "")) for row in group_rows if str(row.get("seed", "")).strip()},
            key=_seed_sort_key,
        )
        payload: dict[str, Any] = {
            group_key: group,
            "seeds": ",".join(seeds),
            "n": len(group_rows),
        }
        for metric in METRIC_KEYS:
            values = _metric_values_for_summary(group_rows, metric)
            payload[f"{metric}_mean"] = round(_mean(values), 6)
            payload[f"{metric}_std"] = round(_std(values), 6)
            payload[f"{metric}_mean_std"] = _format_mean_std(_mean(values), _std(values))
        summary_rows.append(payload)
    return summary_rows


def _load_shared_or_joint_seed(
    *,
    regime: str,
    seed: str,
    eval_run: Path | str,
) -> tuple[dict[str, Any], dict[str, dict[str, float]], dict[str, Any]]:
    evaluation, per_category = _load_eval_artifacts(eval_run)
    _require_all_categories(per_category, context=f"{REGIME_LABELS.get(regime, regime.upper())} seed {seed}")
    macro = _macro_from_evaluation(evaluation, per_category)
    row: dict[str, Any] = {
        "regime": REGIME_LABELS.get(regime, regime.upper()),
        "regime_key": regime,
        "seed": seed,
        "source_run": str(eval_run),
        "category_count": len(per_category),
    }
    row.update({metric: round(macro.get(metric, 0.0), 6) for metric in METRIC_KEYS})
    source = {
        "kind": "main",
        "regime": REGIME_LABELS.get(regime, regime.upper()),
        "seed": seed,
        "category": "",
        "eval_run": str(eval_run),
        **_source_provenance(eval_run),
    }
    return row, per_category, source


def _resolve_csdd_seed_paths(seed_payload: Any) -> dict[str, Path]:
    if isinstance(seed_payload, Mapping):
        return {str(category): _as_path(path) for category, path in seed_payload.items() if str(path).strip()}
    if isinstance(seed_payload, Sequence) and not isinstance(seed_payload, (str, bytes)):
        resolved: dict[str, Path] = {}
        for item in seed_payload:
            _, per_category = _load_eval_artifacts(_as_path(item))
            for category in per_category:
                resolved[str(category)] = _as_path(item)
        return resolved
    return {}


def _load_csdd_seed(
    *,
    seed: str,
    seed_payload: Any,
) -> tuple[dict[str, Any], dict[str, dict[str, float]], list[dict[str, Any]]]:
    category_paths = _resolve_csdd_seed_paths(seed_payload)
    per_category: dict[str, dict[str, float]] = {}
    source_rows: list[dict[str, Any]] = []
    for category, eval_run in sorted(category_paths.items()):
        _, category_metrics = _load_eval_artifacts(eval_run)
        if category in category_metrics:
            per_category[category] = category_metrics[category]
        elif len(category_metrics) == 1:
            only_category, only_metrics = next(iter(category_metrics.items()))
            per_category[str(only_category)] = only_metrics
            category = str(only_category)
        else:
            raise KeyError(f"Could not resolve CSDD category={category!r} from {eval_run}")
        source_rows.append(
            {
                "kind": "main",
                "regime": "CSDD",
                "seed": seed,
                "category": category,
                "eval_run": str(eval_run),
                **_source_provenance(eval_run),
            }
        )
    _require_all_categories(per_category, context=f"CSDD seed {seed}")
    macro = _macro_from_per_category(per_category)
    row: dict[str, Any] = {
        "regime": "CSDD",
        "regime_key": "csdd",
        "seed": seed,
        "source_run": ";".join(str(path) for _, path in sorted(category_paths.items())),
        "category_count": len(per_category),
    }
    row.update({metric: round(macro.get(metric, 0.0), 6) for metric in METRIC_KEYS})
    return row, per_category, source_rows


def _collect_main_runs(manifest: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, dict[str, float]]], list[dict[str, Any]]]:
    main = dict(manifest.get("main", {}))
    rows: list[dict[str, Any]] = []
    per_category_by_regime_seed: dict[str, dict[str, dict[str, dict[str, float]]]] = defaultdict(dict)
    source_rows: list[dict[str, Any]] = []

    for regime in ("ccdd", "cadd"):
        for seed, eval_run in sorted(_coerce_seed_mapping(main.get(regime, {})).items(), key=lambda item: _seed_sort_key(item[0])):
            row, per_category, source = _load_shared_or_joint_seed(regime=regime, seed=seed, eval_run=eval_run)
            rows.append(row)
            per_category_by_regime_seed[regime][seed] = per_category
            source_rows.append(source)

    for seed, seed_payload in sorted(_coerce_seed_mapping(main.get("csdd", {})).items(), key=lambda item: _seed_sort_key(item[0])):
        row, per_category, sources = _load_csdd_seed(seed=seed, seed_payload=seed_payload)
        rows.append(row)
        per_category_by_regime_seed["csdd"][seed] = per_category
        source_rows.extend(sources)

    return rows, per_category_by_regime_seed, source_rows


def _paired_ccdd_cadd_rows(main_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_regime_seed = {
        (str(row.get("regime_key", "")), str(row.get("seed", ""))): row
        for row in main_rows
    }
    seeds = sorted(
        {
            str(row.get("seed", ""))
            for row in main_rows
            if str(row.get("regime_key", "")) in {"ccdd", "cadd"}
        },
        key=_seed_sort_key,
    )
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        ccdd = by_regime_seed.get(("ccdd", seed))
        cadd = by_regime_seed.get(("cadd", seed))
        if ccdd is None or cadd is None:
            continue
        payload: dict[str, Any] = {"comparison": "CCDD-CADD", "seed": seed}
        for metric in METRIC_KEYS:
            payload[metric] = round(float(ccdd.get(metric, 0.0)) - float(cadd.get(metric, 0.0)), 6)
        rows.append(payload)
    return rows


def _merge_per_category_runs(value: Any) -> dict[str, dict[str, float]]:
    if not value:
        return {}
    if isinstance(value, Mapping):
        merged: dict[str, dict[str, float]] = {}
        for _, item in sorted(value.items()):
            _, per_category = _load_eval_artifacts(_as_path(item))
            merged.update(per_category)
        return merged
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        merged: dict[str, dict[str, float]] = {}
        for item in value:
            _, per_category = _load_eval_artifacts(_as_path(item))
            merged.update(per_category)
        return merged
    _, per_category = _load_eval_artifacts(_as_path(value))
    return per_category


def _affected_macro(per_category: Mapping[str, Mapping[str, float]], categories: Sequence[str]) -> dict[str, float]:
    subset = {
        category: dict(per_category[category])
        for category in categories
        if category in per_category
    }
    return _macro_from_per_category(subset)


def _collect_ablation_rows(
    manifest: Mapping[str, Any],
    per_category_by_regime_seed: Mapping[str, Mapping[str, Mapping[str, Mapping[str, float]]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ablations = dict(manifest.get("ablations", {}))
    rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    for ablation_key, ablation_payload_raw in sorted(ablations.items()):
        if not isinstance(ablation_payload_raw, Mapping):
            continue
        ablation_payload = dict(ablation_payload_raw)
        label = str(ablation_payload.get("label", ablation_key))
        baseline_regime = str(ablation_payload.get("baseline_regime", "ccdd")).strip().lower()
        affected_categories = [
            str(category)
            for category in ablation_payload.get("replace_categories", [])
            if str(category).strip()
        ]
        eval_runs = _coerce_seed_mapping(ablation_payload.get("eval_runs", {}))
        for seed, eval_run_value in sorted(eval_runs.items(), key=lambda item: _seed_sort_key(item[0])):
            baseline_pc = per_category_by_regime_seed.get(baseline_regime, {}).get(seed)
            if baseline_pc is None:
                continue
            ablation_pc = _merge_per_category_runs(eval_run_value)
            replacement_categories = affected_categories or sorted(ablation_pc)
            reconstructed = {category: dict(metrics) for category, metrics in baseline_pc.items()}
            replaced = []
            for category in replacement_categories:
                if category in ablation_pc:
                    reconstructed[category] = dict(ablation_pc[category])
                    replaced.append(category)
            baseline_macro = _macro_from_per_category(baseline_pc)
            intervention_macro = _macro_from_per_category(reconstructed)
            baseline_affected = _affected_macro(baseline_pc, replaced)
            intervention_affected = _affected_macro(ablation_pc, replaced)
            row: dict[str, Any] = {
                "ablation_key": ablation_key,
                "ablation": label,
                "baseline_regime": REGIME_LABELS.get(baseline_regime, baseline_regime.upper()),
                "seed": seed,
                "replaced_categories": ",".join(replaced),
                "source_run": _source_run_string(eval_run_value),
            }
            for metric in METRIC_KEYS:
                base = baseline_macro.get(metric, 0.0)
                intervention = intervention_macro.get(metric, 0.0)
                affected_base = baseline_affected.get(metric, 0.0)
                affected_intervention = intervention_affected.get(metric, 0.0)
                row[f"baseline_{metric}"] = round(base, 6)
                row[f"intervention_{metric}"] = round(intervention, 6)
                row[f"delta_{metric}"] = round(intervention - base, 6)
                row[f"affected_baseline_{metric}"] = round(affected_base, 6)
                row[f"affected_intervention_{metric}"] = round(affected_intervention, 6)
                row[f"affected_delta_{metric}"] = round(affected_intervention - affected_base, 6)
            rows.append(row)
            for category in replaced:
                source_rows.append(
                    {
                        "kind": "ablation",
                        "regime": REGIME_LABELS.get(baseline_regime, baseline_regime.upper()),
                        "seed": seed,
                        "category": category,
                        "ablation": label,
                        "eval_run": row["source_run"],
                        **_source_provenance_for_value(eval_run_value),
                    }
                )
    return rows, source_rows


def _source_run_string(value: Any) -> str:
    if isinstance(value, Mapping):
        return ";".join(str(item) for _, item in sorted(value.items()))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return ";".join(str(item) for item in value)
    return str(value)


def _source_provenance(eval_run: Path | str) -> dict[str, Any]:
    summary = _load_eval_summary(eval_run)
    return {
        "reported_seed": summary.get("seed", ""),
        "enable_internal_defect_gate": summary.get("enable_internal_defect_gate", ""),
        "object_score_strategy": summary.get("object_score_strategy", ""),
    }


def _source_provenance_for_value(value: Any) -> dict[str, Any]:
    paths: list[Any]
    if isinstance(value, Mapping):
        paths = [item for _, item in sorted(value.items())]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        paths = list(value)
    else:
        paths = [value]
    summaries = [_source_provenance(path) for path in paths if str(path).strip()]
    if not summaries:
        return {
            "reported_seed": "",
            "enable_internal_defect_gate": "",
            "object_score_strategy": "",
        }
    return {
        "reported_seed": ",".join(
            sorted({str(item.get("reported_seed", "")) for item in summaries}, key=_seed_sort_key)
        ),
        "enable_internal_defect_gate": ",".join(
            sorted({str(item.get("enable_internal_defect_gate", "")) for item in summaries})
        ),
        "object_score_strategy": ",".join(sorted({str(item.get("object_score_strategy", "")) for item in summaries})),
    }


def _summarize_ablation_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("ablation", ""))].append(row)
    summary: list[dict[str, Any]] = []
    for ablation, group_rows in sorted(grouped.items()):
        payload: dict[str, Any] = {
            "ablation": ablation,
            "baseline_regime": str(group_rows[0].get("baseline_regime", "")) if group_rows else "",
            "replaced_categories": str(group_rows[0].get("replaced_categories", "")) if group_rows else "",
            "seeds": ",".join(sorted({str(row.get("seed", "")) for row in group_rows}, key=_seed_sort_key)),
            "n": len(group_rows),
        }
        for metric in METRIC_KEYS:
            full_values = [float(row.get(f"delta_{metric}", 0.0)) for row in group_rows]
            affected_values = [float(row.get(f"affected_delta_{metric}", 0.0)) for row in group_rows]
            payload[f"delta_{metric}_mean"] = round(_mean(full_values), 6)
            payload[f"delta_{metric}_std"] = round(_std(full_values), 6)
            payload[f"affected_delta_{metric}_mean"] = round(_mean(affected_values), 6)
            payload[f"affected_delta_{metric}_std"] = round(_std(affected_values), 6)
            payload[f"delta_{metric}_mean_std"] = _format_mean_std(_mean(full_values), _std(full_values))
            payload[f"affected_delta_{metric}_mean_std"] = _format_mean_std(_mean(affected_values), _std(affected_values))
        summary.append(payload)
    return summary


def _write_latex_table(path: Path | str, headers: Sequence[str], rows: Sequence[Sequence[str]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    column_spec = "l" + "c" * max(len(headers) - 1, 0)
    lines = [
        "\\begin{tabular}{" + column_spec + "}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_main_latex(summary_rows: Sequence[Mapping[str, Any]], output_root: Path) -> None:
    headers = ["Regime", *[label for _, label in TABLE_METRICS]]
    order = {"CCDD": 0, "CADD": 1, "CSDD": 2}
    rows = []
    for row in sorted(summary_rows, key=lambda item: order.get(str(item.get("regime", "")), 99)):
        rows.append(
            [
                str(row.get("regime", "")),
                *[
                    _format_latex_mean_std(
                        float(row.get(f"{metric}_mean", 0.0)),
                        float(row.get(f"{metric}_std", 0.0)),
                    )
                    for metric, _ in TABLE_METRICS
                ],
            ]
        )
    _write_latex_table(output_root / "tables" / "main_multiseed_table.tex", headers, rows)


def _write_delta_latex(delta_summary_rows: Sequence[Mapping[str, Any]], output_root: Path) -> None:
    headers = ["Comparison", *[label for _, label in TABLE_METRICS]]
    rows = []
    for row in delta_summary_rows:
        rows.append(
            [
                str(row.get("comparison", "")),
                *[
                    _format_latex_mean_std(
                        float(row.get(f"{metric}_mean", 0.0)),
                        float(row.get(f"{metric}_std", 0.0)),
                    )
                    for metric, _ in TABLE_METRICS
                ],
            ]
        )
    _write_latex_table(output_root / "tables" / "ccdd_cadd_delta_table.tex", headers, rows)


def _write_ablation_latex(ablation_summary_rows: Sequence[Mapping[str, Any]], output_root: Path) -> None:
    headers = ["Intervention", "Cats.", "Full AUPRO", "Full PxAUROC", "Affected AUPRO", "Affected PxAUROC"]
    rows = []
    for row in ablation_summary_rows:
        rows.append(
            [
                str(row.get("ablation", "")),
                str(row.get("replaced_categories", "")).replace(",", ", "),
                _format_latex_mean_std(
                    float(row.get("delta_aupro_mean", 0.0)),
                    float(row.get("delta_aupro_std", 0.0)),
                ),
                _format_latex_mean_std(
                    float(row.get("delta_pixel_auroc_mean", 0.0)),
                    float(row.get("delta_pixel_auroc_std", 0.0)),
                ),
                _format_latex_mean_std(
                    float(row.get("affected_delta_aupro_mean", 0.0)),
                    float(row.get("affected_delta_aupro_std", 0.0)),
                ),
                _format_latex_mean_std(
                    float(row.get("affected_delta_pixel_auroc_mean", 0.0)),
                    float(row.get("affected_delta_pixel_auroc_std", 0.0)),
                ),
            ]
        )
    _write_latex_table(output_root / "tables" / "mechanism_ablation_table.tex", headers, rows)


def _average_per_category(seed_payloads: Mapping[str, Mapping[str, Mapping[str, float]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    categories = sorted(set(ALL_CATEGORIES) | {category for per_seed in seed_payloads.values() for category in per_seed})
    for category in categories:
        category_rows = [per_seed[category] for per_seed in seed_payloads.values() if category in per_seed]
        if not category_rows:
            continue
        payload: dict[str, Any] = {"category": category, "n": len(category_rows)}
        for metric in METRIC_KEYS:
            values = [float(row.get(metric, 0.0)) for row in category_rows if metric in row]
            payload[f"{metric}_mean"] = round(_mean(values), 6)
            payload[f"{metric}_std"] = round(_std(values), 6)
        rows.append(payload)
    return rows


def _plot_grouped_bars(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric_specs: Sequence[tuple[str, str]],
    sort_metric: str,
    title: str,
    output_path: Path,
) -> None:
    if not rows:
        return
    sorted_rows = sorted(rows, key=lambda row: float(row.get(f"{sort_metric}_mean", 0.0)), reverse=True)
    categories = [str(row["category"]) for row in sorted_rows]
    positions = list(range(len(categories)))
    width = min(0.8 / max(len(metric_specs), 1), 0.26)
    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b3", "#937860"]
    fig_width = max(10.0, 0.78 * len(categories))
    fig, ax = plt.subplots(figsize=(fig_width, 5.2))
    for index, (metric, label) in enumerate(metric_specs):
        offset = (index - (len(metric_specs) - 1) / 2.0) * width
        values = [float(row.get(f"{metric}_mean", 0.0)) for row in sorted_rows]
        ax.bar(
            [position + offset for position in positions],
            values,
            width=width,
            label=label,
            color=colors[index % len(colors)],
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, ncol=min(len(metric_specs), 3))
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_split_figures(
    per_category_by_regime_seed: Mapping[str, Mapping[str, Mapping[str, Mapping[str, float]]]],
    output_root: Path,
) -> None:
    ccdd = per_category_by_regime_seed.get("ccdd", {})
    averaged_rows = _average_per_category(ccdd)
    fieldnames = ["category", "n"]
    for metric in METRIC_KEYS:
        fieldnames.extend([f"{metric}_mean", f"{metric}_std"])
    _write_csv(output_root / "figures" / "ccdd_per_category_mean_metrics.csv", averaged_rows, fieldnames)
    _plot_grouped_bars(
        averaged_rows,
        metric_specs=(("image_auroc", "Image AUROC"), ("image_auprc", "Image AUPRC")),
        sort_metric="image_auroc",
        title="CCDD Per-Category Detection Metrics",
        output_path=output_root / "figures" / "ccdd_detection_metrics.png",
    )
    _plot_grouped_bars(
        averaged_rows,
        metric_specs=(("pixel_auroc", "Pixel AUROC"), ("pixel_aupr", "Pixel AUPR"), ("aupro", "AUPRO")),
        sort_metric="aupro",
        title="CCDD Per-Category Localization Metrics",
        output_path=output_root / "figures" / "ccdd_localization_metrics.png",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate camera-ready MulSenDiff-X reruns, targeted ablations, tables, and split figures.",
    )
    parser.add_argument("--manifest", required=True, help="JSON file with explicit seed-to-run paths.")
    parser.add_argument("--output-root", default="runs/camera_ready_summary")
    parser.add_argument(
        "--copy-manifest",
        action="store_true",
        help="Copy the input manifest into the output root for provenance.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    manifest_path = Path(args.manifest)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, Mapping):
        raise SystemExit("manifest must be a JSON object")

    if args.copy_manifest:
        shutil.copy2(manifest_path, output_root / "run_manifest.json")

    main_rows, per_category_by_regime_seed, source_rows = _collect_main_runs(manifest)
    main_fieldnames = ["regime", "regime_key", "seed", "source_run", "category_count", *METRIC_KEYS]
    _write_csv(output_root / "main_runs.csv", main_rows, main_fieldnames)

    main_summary = _summarize_metric_rows(main_rows, group_key="regime")
    summary_fieldnames = ["regime", "seeds", "n"]
    for metric in METRIC_KEYS:
        summary_fieldnames.extend([f"{metric}_mean", f"{metric}_std", f"{metric}_mean_std"])
    _write_csv(output_root / "main_multiseed_summary.csv", main_summary, summary_fieldnames)
    _write_main_latex(main_summary, output_root)

    delta_rows = _paired_ccdd_cadd_rows(main_rows)
    delta_fieldnames = ["comparison", "seed", *METRIC_KEYS]
    _write_csv(output_root / "ccdd_cadd_paired_deltas.csv", delta_rows, delta_fieldnames)
    delta_summary = _summarize_metric_rows(delta_rows, group_key="comparison")
    _write_csv(output_root / "ccdd_cadd_paired_delta_summary.csv", delta_summary, ["comparison", "seeds", "n", *summary_fieldnames[3:]])
    _write_delta_latex(delta_summary, output_root)

    ablation_rows, ablation_sources = _collect_ablation_rows(manifest, per_category_by_regime_seed)
    ablation_fieldnames = ["ablation_key", "ablation", "baseline_regime", "seed", "replaced_categories", "source_run"]
    for metric in METRIC_KEYS:
        ablation_fieldnames.extend(
            [
                f"baseline_{metric}",
                f"intervention_{metric}",
                f"delta_{metric}",
                f"affected_baseline_{metric}",
                f"affected_intervention_{metric}",
                f"affected_delta_{metric}",
            ]
        )
    _write_csv(output_root / "mechanism_ablation_runs.csv", ablation_rows, ablation_fieldnames)
    ablation_summary = _summarize_ablation_rows(ablation_rows)
    ablation_summary_fieldnames = ["ablation", "baseline_regime", "replaced_categories", "seeds", "n"]
    for metric in METRIC_KEYS:
        ablation_summary_fieldnames.extend(
            [
                f"delta_{metric}_mean",
                f"delta_{metric}_std",
                f"affected_delta_{metric}_mean",
                f"affected_delta_{metric}_std",
                f"delta_{metric}_mean_std",
                f"affected_delta_{metric}_mean_std",
            ]
        )
    _write_csv(output_root / "mechanism_ablation_summary.csv", ablation_summary, ablation_summary_fieldnames)
    _write_ablation_latex(ablation_summary, output_root)

    _write_split_figures(per_category_by_regime_seed, output_root)
    _write_csv(
        output_root / "source_runs.csv",
        [*source_rows, *ablation_sources],
        [
            "kind",
            "regime",
            "seed",
            "reported_seed",
            "enable_internal_defect_gate",
            "object_score_strategy",
            "category",
            "ablation",
            "eval_run",
        ],
    )
    _write_json(
        output_root / "summary.json",
        {
            "manifest": str(manifest_path),
            "main_rows": len(main_rows),
            "main_summary_rows": len(main_summary),
            "paired_delta_rows": len(delta_rows),
            "ablation_rows": len(ablation_rows),
            "output_root": str(output_root),
        },
    )
    print("camera_ready_summary:", output_root)


if __name__ == "__main__":
    main()
