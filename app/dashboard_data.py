from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List


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
    calibration = load_json(run_root / "evidence" / "calibration.json")
    evidence_index = load_json(run_root / "evidence" / "index.json")
    return {
        "run_root": run_root,
        "summary": summary,
        "calibration": calibration,
        "evidence_index": evidence_index,
    }


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
    report_path = resolve_repo_relative_path(entry["report_path"], repo_root)
    package = load_json(package_path)
    report_text = report_path.read_text(encoding="utf-8")
    visualization_path = ""
    if isinstance(package, dict):
        visualization_path = str(package.get("source_paths", {}).get("visualization_path", ""))
    visualization_resolved = resolve_repo_relative_path(visualization_path, repo_root) if visualization_path else None
    return {
        "package_path": package_path,
        "report_path": report_path,
        "package": package,
        "report_text": report_text,
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
