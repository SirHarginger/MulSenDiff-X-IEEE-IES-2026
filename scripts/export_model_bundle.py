#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.project_layout import resolve_regime_output_dir


SUMMARY_PATH_KEYS = [
    "image_score_data_path",
    "preview_selection_path",
    "calibration_path",
    "global_descriptor_calibration_path",
    "global_branch_gate_calibration_path",
    "localization_calibration_path",
    "localization_tuning_report_path",
    "spatial_exceedance_calibration_path",
    "object_score_calibration_path",
    "rgb_normalization_stats_path",
    "rgb_normalization_stats_by_category_path",
    "evidence_index_path",
]
LOCAL_ARTIFACT_ANCHORS = ("metrics", "evidence", "plots", "predictions")


def _infer_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "src").exists() and (candidate / "scripts").exists():
            return candidate
    return start


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _resolve_checkpoint_path(eval_run: Path, summary: Dict[str, Any], repo_root: Path) -> Path:
    checkpoint_path = Path(str(summary.get("checkpoint_path", "")).strip())
    if not checkpoint_path:
        raise FileNotFoundError("summary.json does not contain checkpoint_path")
    if checkpoint_path.is_absolute():
        if checkpoint_path.exists():
            return checkpoint_path
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    candidates = [
        eval_run / checkpoint_path,
        repo_root / checkpoint_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"checkpoint not found for relative path: {checkpoint_path}")


def _resolve_config_path(eval_run: Path, checkpoint_path: Path) -> Path:
    local_config = eval_run / "config.json"
    if local_config.exists():
        return local_config
    checkpoint_config = checkpoint_path.parent.parent / "config.json"
    if checkpoint_config.exists():
        return checkpoint_config
    raise FileNotFoundError(f"config.json not found for checkpoint: {checkpoint_path}")


def _infer_regime_dir(summary: Dict[str, Any]) -> str:
    regime_paper = str(summary.get("regime_paper", "")).strip()
    if regime_paper:
        return resolve_regime_output_dir(regime_paper)
    regime_internal = str(summary.get("regime_internal", "")).strip().lower()
    if regime_internal == "shared_nocat":
        return "cadd"
    if regime_internal == "per_category":
        return "csdd"
    return "ccdd"


def _artifact_relative_path(path_like: str | Path) -> Path | None:
    value = str(path_like).strip()
    if not value:
        return None
    path = Path(value)
    if path.name in {"checkpoint.pt", "best.pt"} or path.suffix == ".pt":
        return Path("checkpoint.pt")
    if path.name == "config.json":
        return Path("config.json")
    parts = list(path.parts)
    for anchor in LOCAL_ARTIFACT_ANCHORS:
        if anchor in parts:
            return Path(*parts[parts.index(anchor) :])
    return None


def _remap_summary_paths(summary: Dict[str, Any], target_root: Path) -> Dict[str, Any]:
    updated = dict(summary)
    updated["checkpoint_path"] = "checkpoint.pt"
    updated["run_dir"] = str(target_root)
    for key in SUMMARY_PATH_KEYS:
        relative = _artifact_relative_path(updated.get(key, ""))
        updated[key] = str(relative) if relative is not None and (target_root / relative).exists() else ""
    plot_paths = updated.get("plot_paths", {})
    if isinstance(plot_paths, dict):
        updated["plot_paths"] = {
            str(name): str(relative)
            for name, relative in (
                (name, _artifact_relative_path(path))
                for name, path in plot_paths.items()
            )
            if relative is not None and (target_root / relative).exists()
        }
    return updated


def _rewrite_evidence_artifacts(target_root: Path) -> None:
    evidence_index_path = target_root / "evidence" / "index.json"
    if not evidence_index_path.exists():
        return
    payload = json.loads(evidence_index_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return

    for entry in payload:
        if not isinstance(entry, dict):
            continue
        for key in ("package_path", "report_path", "llm_bundle_path"):
            relative = _artifact_relative_path(entry.get(key, ""))
            entry[key] = str(relative) if relative is not None and (target_root / relative).exists() else ""
        package_path = Path(str(entry.get("package_path", "")).strip())
        if not package_path:
            continue
        resolved_package_path = target_root / package_path
        if not resolved_package_path.exists():
            continue
        package_payload = json.loads(resolved_package_path.read_text(encoding="utf-8"))
        if not isinstance(package_payload, dict):
            continue
        source_paths = package_payload.get("source_paths", {})
        if isinstance(source_paths, dict):
            visualization_relative = _artifact_relative_path(source_paths.get("visualization_path", ""))
            source_paths["visualization_path"] = (
                str(visualization_relative)
                if visualization_relative is not None and (target_root / visualization_relative).exists()
                else ""
            )
            package_payload["source_paths"] = source_paths
            resolved_package_path.write_text(
                json.dumps(package_payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )

    evidence_index_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _copy_tree_if_present(source: Path, target: Path) -> None:
    if source.exists():
        shutil.copytree(source, target, dirs_exist_ok=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export an evaluation run into an app-ready bundle under model/.")
    parser.add_argument("--eval-run", required=True)
    parser.add_argument("--model-root", default="model")
    parser.add_argument("--regime", choices=["ccdd", "cadd", "csdd"], default="")
    parser.add_argument("--target-name", default="")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    eval_run = Path(args.eval_run).resolve()
    repo_root = _infer_repo_root(eval_run)
    summary = _load_json(eval_run / "summary.json")
    checkpoint_path = _resolve_checkpoint_path(eval_run, summary, repo_root)
    config_path = _resolve_config_path(eval_run, checkpoint_path)
    regime_dir = args.regime or _infer_regime_dir(summary)
    bundle_name = args.target_name.strip() or eval_run.name

    if regime_dir == "csdd":
        category = str(summary.get("category") or summary.get("scope") or "bundle").strip().lower()
        target_root = (repo_root / args.model_root / regime_dir / category / bundle_name).resolve()
    else:
        target_root = (repo_root / args.model_root / regime_dir / bundle_name).resolve()

    if target_root.exists():
        if not args.force:
            raise SystemExit(f"target bundle already exists: {target_root} (use --force to replace)")
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    for source_dir_name in ("metrics", "evidence", "plots", "predictions"):
        _copy_tree_if_present(eval_run / source_dir_name, target_root / source_dir_name)

    shutil.copy2(checkpoint_path, target_root / "checkpoint.pt")
    shutil.copy2(config_path, target_root / "config.json")

    exported_summary = _remap_summary_paths(summary, target_root)
    exported_summary["source_eval_run_dir"] = str(eval_run)
    exported_summary["source_checkpoint_path"] = str(checkpoint_path)
    (target_root / "summary.json").write_text(
        json.dumps(exported_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (target_root / "bundle_manifest.json").write_text(
        json.dumps(
            {
                "bundle_root": str(target_root),
                "source_eval_run_dir": str(eval_run),
                "source_checkpoint_path": str(checkpoint_path),
                "source_config_path": str(config_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    _rewrite_evidence_artifacts(target_root)

    print("bundle_root:", target_root)
    print("source_eval_run:", eval_run)
    print("checkpoint:", checkpoint_path)


if __name__ == "__main__":
    main()
