from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from src.data_loader import ProcessedSampleRecord, select_processed_sample_records


@dataclass(frozen=True)
class AvailableRun:
    root: Path
    summary_path: Path
    evidence_index_path: Path
    calibration_path: Path
    name: str


@dataclass(frozen=True)
class AvailableTrainingRun:
    root: Path
    config_path: Path
    checkpoints_dir: Path
    best_checkpoint_path: Path
    name: str


def find_available_evaluation_runs(runs_root: Path | str = "runs") -> List[AvailableRun]:
    runs_root = Path(runs_root)
    candidates: List[AvailableRun] = []
    if not runs_root.exists():
        return candidates

    for run_root in sorted((path for path in runs_root.iterdir() if path.is_dir()), reverse=True):
        summary_path = run_root / "summary.json"
        evidence_index_path = run_root / "evidence" / "index.json"
        calibration_path = run_root / "evidence" / "calibration.json"
        if summary_path.exists() and evidence_index_path.exists() and calibration_path.exists():
            candidates.append(
                AvailableRun(
                    root=run_root,
                    summary_path=summary_path,
                    evidence_index_path=evidence_index_path,
                    calibration_path=calibration_path,
                    name=run_root.name,
                )
            )
    return candidates


def find_available_training_runs(runs_root: Path | str = "runs") -> List[AvailableTrainingRun]:
    runs_root = Path(runs_root)
    candidates: List[AvailableTrainingRun] = []
    if not runs_root.exists():
        return candidates

    for run_root in sorted((path for path in runs_root.iterdir() if path.is_dir()), reverse=True):
        config_path = run_root / "config.json"
        checkpoints_dir = run_root / "checkpoints"
        best_checkpoint_path = checkpoints_dir / "best.pt"
        if config_path.exists() and checkpoints_dir.exists() and best_checkpoint_path.exists():
            candidates.append(
                AvailableTrainingRun(
                    root=run_root,
                    config_path=config_path,
                    checkpoints_dir=checkpoints_dir,
                    best_checkpoint_path=best_checkpoint_path,
                    name=run_root.name,
                )
            )
    return candidates


def load_json(path: Path | str) -> Dict[str, Any] | List[Dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_run_bundle(run_root: Path | str) -> Dict[str, Any]:
    run_root = Path(run_root)
    summary = load_json(run_root / "summary.json")
    checkpoint_config = _load_checkpoint_config(run_root, summary)
    if checkpoint_config:
        actual_categories = list(
            checkpoint_config.get("category_vocabulary")
            or checkpoint_config.get("selected_categories")
            or ([checkpoint_config["category"]] if str(checkpoint_config.get("category", "")).strip() else [])
            or []
        )
        if actual_categories and summary.get("selected_categories") == ["all"]:
            summary = dict(summary)
            summary["selected_categories"] = actual_categories
            summary["category_vocabulary"] = actual_categories
    calibration = load_json(run_root / "evidence" / "calibration.json")
    evidence_index = load_json(run_root / "evidence" / "index.json")
    evaluation = load_json(run_root / "metrics" / "evaluation.json") if (run_root / "metrics" / "evaluation.json").exists() else {}
    per_category = load_json(run_root / "metrics" / "per_category.json") if (run_root / "metrics" / "per_category.json").exists() else {}
    preview_selection = (
        load_json(run_root / "metrics" / "preview_selection.json")
        if (run_root / "metrics" / "preview_selection.json").exists()
        else {}
    )
    image_score_data = (
        load_json(run_root / "metrics" / "image_score_data.json")
        if (run_root / "metrics" / "image_score_data.json").exists()
        else []
    )
    return {
        "run_root": run_root,
        "summary": summary,
        "calibration": calibration,
        "evidence_index": evidence_index,
        "evaluation": evaluation,
        "per_category": per_category,
        "preview_selection": preview_selection,
        "image_score_data": image_score_data,
        "plot_paths": list_plot_files(run_root),
    }


def _load_checkpoint_config(run_root: Path, summary: Dict[str, Any]) -> Dict[str, Any]:
    checkpoint_path = Path(str(summary.get("checkpoint_path", "")).strip())
    if checkpoint_path:
        if not checkpoint_path.is_absolute():
            checkpoint_path = run_root.parent.parent / checkpoint_path
        checkpoint_config_path = checkpoint_path.parent.parent / "config.json"
        if checkpoint_config_path.exists():
            payload = load_json(checkpoint_config_path)
            if isinstance(payload, dict):
                return payload
    return {}


def summarize_evidence_index(entries: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    by_status: Dict[str, int] = {}
    by_defect: Dict[str, int] = {}
    by_status_defect: Dict[str, int] = {}

    for entry in entries:
        status = str(entry.get("status", "Unknown"))
        defect = str(entry.get("defect_label", "unknown"))
        by_status[status] = by_status.get(status, 0) + 1
        by_defect[defect] = by_defect.get(defect, 0) + 1
        key = f"{status}::{defect}"
        by_status_defect[key] = by_status_defect.get(key, 0) + 1

    return {
        "by_status": dict(sorted(by_status.items())),
        "by_defect": dict(sorted(by_defect.items())),
        "by_status_defect": dict(sorted(by_status_defect.items())),
    }


def resolve_repo_relative_path(path_like: str | Path, repo_root: Path | str) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return Path(repo_root) / path


def load_sample_artifacts(
    *,
    repo_root: Path | str,
    entry: Dict[str, Any],
) -> Dict[str, Any]:
    repo_root = Path(repo_root)
    package_path = resolve_repo_relative_path(entry["package_path"], repo_root)
    report_path = resolve_repo_relative_path(entry["report_path"], repo_root) if str(entry.get("report_path", "")).strip() else None
    package = load_json(package_path)
    report_text = report_path.read_text(encoding="utf-8") if report_path is not None and report_path.exists() else ""
    llm_bundle_path = ""
    llm_bundle = None
    llm_markdown = ""
    if str(entry.get("llm_bundle_path", "")).strip():
        llm_bundle_path = str(entry["llm_bundle_path"])
        resolved_llm_bundle_path = resolve_repo_relative_path(llm_bundle_path, repo_root)
        if resolved_llm_bundle_path.exists():
            llm_bundle = load_json(resolved_llm_bundle_path)
            if isinstance(llm_bundle, dict):
                llm_markdown = str(llm_bundle.get("rendered_markdown", ""))
    visualization_path = ""
    if isinstance(package, dict):
        visualization_path = str(package.get("source_paths", {}).get("visualization_path", ""))
    visualization_resolved = resolve_repo_relative_path(visualization_path, repo_root) if visualization_path else None
    return {
        "package_path": package_path,
        "report_path": report_path,
        "package": package,
        "report_text": report_text,
        "llm_bundle_path": resolve_repo_relative_path(llm_bundle_path, repo_root) if llm_bundle_path else None,
        "llm_bundle": llm_bundle,
        "llm_markdown": llm_markdown,
        "visualization_path": visualization_resolved,
    }


def format_entry_label(entry: Dict[str, Any]) -> str:
    return (
        f"{entry.get('status', 'Unknown')} | "
        f"{entry.get('defect_label', 'unknown')} / {entry.get('sample_id', '?')} | "
        f"severity {float(entry.get('severity_0_100', 0.0)):.1f}"
    )


def list_checkpoint_files(run_root: Path | str) -> List[Path]:
    run_root = Path(run_root)
    return sorted((run_root / "checkpoints").glob("*.pt"))


def list_plot_files(run_root: Path | str) -> Dict[str, Path]:
    run_root = Path(run_root)
    plots_root = run_root / "plots"
    if not plots_root.exists():
        return {}
    return {
        path.name: path
        for path in sorted(plots_root.glob("*.png"))
    }


def filter_evidence_entries(
    entries: Sequence[Dict[str, Any]],
    *,
    category: str = "All",
    defect_label: str = "All",
    status: str = "All",
) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for entry in entries:
        if category != "All" and str(entry.get("category", "")) != category:
            continue
        if defect_label != "All" and str(entry.get("defect_label", "")) != defect_label:
            continue
        if status != "All" and str(entry.get("status", "")) != status:
            continue
        filtered.append(entry)
    return filtered


def find_known_processed_samples(
    repo_root: Path | str,
    *,
    categories: Sequence[str] | None = None,
) -> List[ProcessedSampleRecord]:
    repo_root = Path(repo_root)
    data_root = repo_root / "data"
    return select_processed_sample_records(data_root=data_root, categories=categories)


def format_known_sample_label(record: ProcessedSampleRecord) -> str:
    source_name = Path(record.rgb_source_path).name if record.rgb_source_path else Path(record.sample_dir).name
    return (
        f"{record.category} / {record.split} / {record.defect_label} / "
        f"{record.sample_id} | {source_name}"
    )


def match_uploaded_rgb_name(
    upload_name: str,
    records: Sequence[ProcessedSampleRecord],
) -> List[ProcessedSampleRecord]:
    normalized_name = Path(upload_name).name.strip().lower()
    normalized_stem = Path(upload_name).stem.strip().lower()
    if not normalized_name and not normalized_stem:
        return []

    matches: List[ProcessedSampleRecord] = []
    for record in records:
        candidates = {
            Path(record.rgb_source_path).name.strip().lower(),
            Path(record.rgb_source_path).stem.strip().lower(),
            Path(record.sample_dir).name.strip().lower(),
            record.sample_name.strip().lower(),
        }
        candidates.discard("")
        if normalized_name in candidates or normalized_stem in candidates:
            matches.append(record)
    return matches


def default_app_sessions_root(repo_root: Path | str) -> Path:
    return Path(repo_root) / "runs" / "app_sessions"
